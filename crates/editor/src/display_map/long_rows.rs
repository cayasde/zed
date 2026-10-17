use std::{
    ops::Range,
    sync::{
        Arc, OnceLock,
        atomic::{self, AtomicUsize},
    },
};

use collections::HashMap;

use gpui::{
    FontId, FontRun, LineLayout, LineWrapper, LineWrapperHandle, Pixels, ShapedGlyph, ShapedRun,
    SharedString, TextAlign, TextSystem, point, px,
};
use language::LanguageAwareStyling;
use parking_lot::Mutex;
use smallvec::{SmallVec, smallvec};
use sum_tree::Bias;
use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};

use super::{
    ChunkRendererId, Highlights, LongRowSource,
    invisibles::{is_invisible, replacement},
    tab_map::{TabChunks, TabEdit, TabPoint, TabSnapshot},
};
use crate::{MAX_LINE_LEN, scroll::ScrollPixelOffset};

#[derive(Clone)]
pub struct LongRow {
    snapshot: TabSnapshot,
    row: u32,
    source: Option<Arc<LongRowSource>>,
    geometry: Arc<RowGeometryState>,
    widths: Arc<CharWidths>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RowWindow {
    segments: SmallVec<[Range<u32>; 1]>,
}

#[derive(Clone)]
pub(crate) struct TextPiece {
    pub(crate) columns: Range<u32>,
    pub(crate) x: ScrollPixelOffset,
    pub(crate) text: SharedString,
    pub(crate) layout: Arc<LineLayout>,
    pub(crate) runs: SmallVec<[FontRun; 1]>,
    index: Arc<LayoutIndex>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RowMetrics {
    pub(crate) ascent: Pixels,
    pub(crate) descent: Pixels,
}

#[derive(Clone, Default)]
pub(crate) struct LongRows {
    widths: Option<Arc<CharWidths>>,
    rows: Arc<Mutex<HashMap<(usize, u32), LongRow>>>,
    width_index: Arc<OnceLock<Arc<RowWidthIndex>>>,
    source: Option<Arc<LongRowSource>>,
    epoch: usize,
    version: usize,
}

struct RowWidthIndex {
    version: usize,
    rows: Box<[(u32, ScrollPixelOffset)]>,
    max_width: ScrollPixelOffset,
}

struct FontRunsCursor<'a> {
    runs: &'a [FontRun],
    start: u32,
}

struct CharWidths {
    wrapper: LineWrapperHandle,
    text_system: Arc<TextSystem>,
    base_font: FontId,
    em_width: ScrollPixelOffset,
    metrics: RowMetrics,
    #[cfg(test)]
    shaped_rows: AtomicUsize,
}

#[derive(Default)]
struct RowGeometryState {
    value: OnceLock<Arc<RowGeometry>>,
    previous: Mutex<Option<Arc<RowGeometry>>>,
}

struct RowGeometry {
    len: u32,
    width: ScrollPixelOffset,
    content_end: u32,
    atoms: AtomIndex,
    pieces: Vec<TextPiece>,
    paint_overhang: Range<ScrollPixelOffset>,
    font_runs: Arc<[FontRun]>,
    metrics: RowMetrics,
}

#[derive(Default)]
struct AtomIndex {
    columns: Arc<[u32]>,
    positions: Arc<[ScrollPixelOffset]>,
}

#[derive(Default)]
struct AtomBuilder {
    columns: Vec<u32>,
    positions: Vec<ScrollPixelOffset>,
}

struct AtomCursor<'a> {
    columns: &'a [u32],
    end: usize,
}

struct Atom {
    start: u32,
    end: u32,
    start_x: ScrollPixelOffset,
    end_x: ScrollPixelOffset,
}

struct InputRun {
    start: u32,
    text: String,
    renderer: Option<(ChunkRendererId, Option<Pixels>)>,
}

static NEXT_EPOCH: AtomicUsize = AtomicUsize::new(1);

impl LongRow {
    pub(crate) fn len(&self) -> u32 {
        self.snapshot.line_len(self.row)
    }

    pub(crate) fn width(&self) -> ScrollPixelOffset {
        self.geometry().width
    }

    pub(crate) fn trailing_whitespace_start(&self) -> Option<u32> {
        Some(self.geometry().content_end)
    }

    pub(crate) fn metrics(&self) -> RowMetrics {
        self.geometry().metrics
    }

    pub(crate) fn x_for_column(&self, column: u32) -> ScrollPixelOffset {
        let columns = self.atom_range(column);
        self.geometry().x_for_boundary(if columns.start == column {
            column
        } else {
            columns.end
        })
    }

    pub(crate) fn closest_column_for_x(&self, x: ScrollPixelOffset) -> u32 {
        if x <= 0. {
            return 0;
        }
        if x >= self.width() {
            return self.len();
        }
        let atom = self.geometry().atom_at_x(x);
        if x - atom.start_x <= atom.end_x - x {
            atom.start
        } else {
            atom.end
        }
    }

    pub(crate) fn column_at_x(&self, x: ScrollPixelOffset) -> u32 {
        if x <= 0. {
            return 0;
        }
        if x >= self.width() {
            return self.len();
        }
        self.geometry().atom_at_x(x).start
    }

    pub(crate) fn visible_window(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> RowWindow {
        let visible = self.visible_x(scroll_x, text_align, viewport_width);
        let geometry = self.geometry();
        let start_x = visible.start - geometry.paint_overhang.end;
        let end_x = visible.end - geometry.paint_overhang.start;
        let start = geometry
            .atoms
            .positions
            .partition_point(|x| *x < start_x)
            .saturating_sub(1);
        let end = geometry.atoms.positions.partition_point(|x| *x <= end_x);
        let start = geometry
            .atoms
            .columns
            .get(start)
            .copied()
            .unwrap_or(geometry.len);
        let end = geometry
            .atoms
            .columns
            .get(end)
            .copied()
            .unwrap_or(geometry.len);
        self.chunk_aligned(&RowWindow::from_iter([start..end.max(start)]))
    }

    pub(crate) fn window_layout(
        &self,
        columns: Range<u32>,
        origin_x: ScrollPixelOffset,
    ) -> Vec<TextPiece> {
        let geometry = self.geometry();
        let columns = columns.start.min(geometry.len)..columns.end.min(geometry.len);

        let first = geometry
            .pieces
            .partition_point(|piece| piece.columns.end <= columns.start);
        geometry.pieces[first..]
            .iter()
            .take_while(|piece| piece.columns.start < columns.end)
            .filter_map(|piece| piece.window(&columns, origin_x))
            .collect()
    }

    pub(crate) fn chunk_aligned(&self, window: &RowWindow) -> RowWindow {
        let geometry = self.geometry();
        window
            .segments()
            .iter()
            .map(|columns| {
                geometry.boundary(columns.start, Bias::Left)
                    ..geometry.boundary(columns.end, Bias::Right)
            })
            .collect()
    }

    pub(crate) fn floor_column(&self, column: u32) -> u32 {
        self.atom_range(column).start
    }

    pub(crate) fn atom_columns(&self, column: u32) -> Range<u32> {
        self.atom_range(column)
    }

    fn geometry(&self) -> &Arc<RowGeometry> {
        self.geometry.value.get_or_init(|| self.shape())
    }

    fn shape(&self) -> Arc<RowGeometry> {
        let previous = self.geometry.previous.lock().take();
        let font_runs = self.resolved_font_runs();
        if let Some(previous) = previous
            && previous.font_runs == font_runs
        {
            return previous;
        }
        #[cfg(test)]
        self.widths
            .shaped_rows
            .fetch_add(1, atomic::Ordering::Relaxed);
        Arc::new(RowGeometry::shape(
            &self.snapshot,
            self.row,
            font_runs,
            &self.widths.wrapper,
            self.widths.metrics,
            &self.widths.text_system,
        ))
    }

    fn resolved_font_runs(&self) -> Arc<[FontRun]> {
        let Some(source) = &self.source else {
            return Arc::from([FontRun {
                len: self.len() as usize,
                font_id: self.widths.base_font,
            }]);
        };
        let mut runs = SmallVec::<[FontRun; 1]>::new();
        let mut previous_decoration = None;
        for chunk in source.highlighted_chunks(&self.snapshot, self.row, 0..self.len(), usize::MAX)
        {
            let run = chunk.text_run(&source.style);
            #[cfg(test)]
            source
                .resolved_bytes
                .fetch_add(run.len, atomic::Ordering::Relaxed);
            if run.len == 0 {
                continue;
            }
            let font_id = source.text_system.resolve_font(&run.font);
            let decoration = (run.color, run.underline, run.strikethrough);
            if let Some(previous) = runs.last_mut()
                && previous.font_id == font_id
                && previous_decoration == Some(decoration)
            {
                previous.len += run.len;
            } else {
                runs.push(FontRun {
                    len: run.len,
                    font_id,
                });
            }
            previous_decoration = Some(decoration);
        }
        Arc::from(runs.into_vec())
    }

    fn atom_range(&self, column: u32) -> Range<u32> {
        let geometry = self.geometry();
        if column >= geometry.len {
            return geometry.len..geometry.len;
        }
        let index = geometry
            .atoms
            .columns
            .partition_point(|start| *start <= column)
            .saturating_sub(1);
        let start = geometry.atoms.columns.get(index).copied().unwrap_or(0);
        let end = geometry
            .atoms
            .columns
            .get(index + 1)
            .copied()
            .unwrap_or(geometry.len);
        start..end
    }

    fn visible_x(
        &self,
        scroll_x: ScrollPixelOffset,
        text_align: TextAlign,
        viewport_width: ScrollPixelOffset,
    ) -> Range<ScrollPixelOffset> {
        let viewport_width = viewport_width.max(0.);
        let free_width = viewport_width - self.width();
        let alignment_offset = match text_align {
            TextAlign::Left => 0.,
            TextAlign::Center => free_width / 2.,
            TextAlign::Right => free_width,
        };
        let start = scroll_x - alignment_offset;
        start..start + viewport_width
    }
}

impl LongRows {
    pub(crate) fn new(
        mut wrapper: LineWrapperHandle,
        text_system: Arc<TextSystem>,
        version: usize,
    ) -> Self {
        let em_width = f64::from(wrapper.width_for_char('m'));
        let base_font = wrapper.font_id();
        let layout = wrapper.layout_line(
            " ",
            &[FontRun {
                len: 1,
                font_id: base_font,
            }],
        );
        Self {
            widths: Some(Arc::new(CharWidths {
                metrics: RowMetrics {
                    ascent: layout.ascent,
                    descent: layout.descent,
                },
                wrapper,
                text_system,
                base_font,
                em_width,
                #[cfg(test)]
                shaped_rows: AtomicUsize::new(0),
            })),
            version,
            epoch: next_epoch(),
            ..Self::default()
        }
    }

    pub(crate) fn epoch(&self) -> usize {
        self.epoch
    }

    pub(crate) fn max_width_in(
        &self,
        snapshot: &TabSnapshot,
        rows: Range<u32>,
    ) -> ScrollPixelOffset {
        if self.widths.is_none() {
            return 0.;
        }
        let row_count = snapshot.max_point().row() + 1;
        let rows = rows.start.min(row_count)..rows.end.min(row_count);
        if rows.is_empty() {
            return 0.;
        }
        if let Some(index) = self
            .width_index
            .get()
            .filter(|index| index.version == snapshot.version)
        {
            return index.max_width_in(rows);
        }
        let mut measured = Vec::new();
        let mut scan = LongRowScan::new(0..row_count);
        loop {
            match scan.step(snapshot) {
                ScanStep::Row { row } => {
                    if let Some(long_row) = self.row(snapshot, row) {
                        measured.push((row, long_row.width()));
                    }
                }
                ScanStep::Continue => {}
                ScanStep::Done => break,
            }
        }
        self.finish_width_index(snapshot.version, measured)
            .max_width_in(rows)
    }

    pub(crate) fn em_width(&self) -> Option<ScrollPixelOffset> {
        Some(self.widths.as_ref()?.em_width)
    }

    pub(crate) fn row(&self, snapshot: &TabSnapshot, row: u32) -> Option<LongRow> {
        let widths = self.widths.as_ref()?;
        if snapshot.line_len(row) as usize <= MAX_LINE_LEN {
            return None;
        }
        Some(
            self.rows
                .lock()
                .entry((snapshot.version, row))
                .or_insert_with(|| LongRow {
                    snapshot: snapshot.clone(),
                    row,
                    source: self.source.clone(),
                    geometry: Arc::default(),
                    widths: widths.clone(),
                })
                .clone(),
        )
    }

    pub(crate) fn interpolate(&self, snapshot: &TabSnapshot, edits: &[TabEdit]) -> Self {
        if edits.is_empty() && self.version == snapshot.version {
            return self.clone();
        }
        let mut rows = HashMap::default();
        let previous_rows = self
            .rows
            .lock()
            .iter()
            .filter(|((version, _), _)| *version == self.version)
            .map(|((_, row), previous)| (*row, previous.clone()))
            .collect::<Vec<_>>();
        for (row, previous) in previous_rows {
            let next_edit = edits.partition_point(|edit| edit.old.end.row() < row);
            if edits
                .get(next_edit)
                .is_some_and(|edit| edit.old.start.row() <= row)
            {
                continue;
            }
            let delta = next_edit.checked_sub(1).map_or(0, |index| {
                let edit = &edits[index];
                i64::from(edit.new.end.row()) - i64::from(edit.old.end.row())
            });
            let Ok(row) = u32::try_from(i64::from(row) + delta) else {
                continue;
            };
            if row > snapshot.max_point().row() || snapshot.line_len(row) != previous.len() {
                continue;
            }
            rows.insert(
                (snapshot.version, row),
                LongRow {
                    snapshot: snapshot.clone(),
                    row,
                    geometry: Arc::new(previous.geometry.invalidated()),
                    ..previous
                },
            );
        }
        Self {
            rows: Arc::new(Mutex::new(rows)),
            width_index: Arc::default(),
            epoch: next_epoch(),
            version: snapshot.version,
            ..self.clone()
        }
    }

    pub(super) fn set_shaping_source(&mut self, source: Arc<LongRowSource>) {
        let rows = self
            .rows
            .lock()
            .iter()
            .map(|(key, row)| {
                (
                    *key,
                    LongRow {
                        source: Some(source.clone()),
                        geometry: Arc::new(row.geometry.invalidated()),
                        ..row.clone()
                    },
                )
            })
            .collect();
        self.rows = Arc::new(Mutex::new(rows));
        self.source = Some(source);
        self.epoch = next_epoch();
        self.width_index = Arc::default();
    }

    fn finish_width_index(
        &self,
        version: usize,
        rows: Vec<(u32, ScrollPixelOffset)>,
    ) -> Arc<RowWidthIndex> {
        let index = Arc::new(RowWidthIndex {
            version,
            max_width: rows.iter().fold(0_f64, |width, row| width.max(row.1)),
            rows: rows.into_boxed_slice(),
        });
        let cached = self.width_index.get_or_init(|| index.clone());
        if cached.version == version {
            cached.clone()
        } else {
            index
        }
    }
}

impl RowWidthIndex {
    fn max_width_in(&self, rows: Range<u32>) -> ScrollPixelOffset {
        if rows.is_empty() {
            return 0.;
        }
        let start = self.rows.partition_point(|(row, _)| *row < rows.start);
        let end = self.rows.partition_point(|(row, _)| *row < rows.end);
        if start == 0 && end == self.rows.len() {
            self.max_width
        } else {
            self.rows[start..end]
                .iter()
                .fold(0_f64, |width, row| width.max(row.1))
        }
    }
}

impl<'a> FontRunsCursor<'a> {
    fn new(runs: &'a [FontRun]) -> Self {
        Self { runs, start: 0 }
    }

    fn slice(&mut self, columns: Range<u32>) -> SmallVec<[FontRun; 1]> {
        while let Some(run) = self.runs.first()
            && self.start + run.len as u32 <= columns.start
        {
            self.start += run.len as u32;
            self.runs = &self.runs[1..];
        }
        slice_font_runs(self.runs.iter().copied(), self.start, columns)
    }
}

impl RowGeometryState {
    fn invalidated(&self) -> Self {
        Self {
            value: OnceLock::new(),
            previous: Mutex::new(
                self.value
                    .get()
                    .cloned()
                    .or_else(|| self.previous.lock().clone()),
            ),
        }
    }
}

impl<'a> AtomCursor<'a> {
    fn new(columns: &'a [u32]) -> Self {
        Self { columns, end: 0 }
    }

    fn at(&mut self, column: usize) -> Option<usize> {
        while self
            .columns
            .get(self.end)
            .is_some_and(|start| *start as usize <= column)
        {
            self.end += 1;
        }
        let last = self.end.checked_sub(1)?;
        if self.columns[last] as usize <= column {
            Some(last)
        } else {
            self.columns[..self.end]
                .partition_point(|start| *start as usize <= column)
                .checked_sub(1)
        }
    }
}

impl AtomBuilder {
    fn push(&mut self, column: u32, x: ScrollPixelOffset) {
        self.columns.push(column);
        self.positions.push(x);
    }

    fn finish(self, piece: Option<&TextPiece>) -> AtomIndex {
        let columns = match piece {
            Some(piece) if piece.index.partition_columns.as_ref() == self.columns.as_slice() => {
                piece.index.partition_columns.clone()
            }
            _ => Arc::from(self.columns),
        };
        let positions = match piece {
            Some(piece) if piece.index.positions.as_ref() == self.positions.as_slice() => {
                piece.index.positions.clone()
            }
            _ => Arc::from(self.positions),
        };
        AtomIndex { columns, positions }
    }
}

impl RowGeometry {
    fn shape(
        snapshot: &TabSnapshot,
        row: u32,
        font_runs: Arc<[FontRun]>,
        wrapper: &LineWrapper,
        metrics: RowMetrics,
        text_system: &TextSystem,
    ) -> Self {
        let len = snapshot.line_len(row);
        let mut geometry = Self {
            len,
            width: 0.,
            content_end: 0,
            atoms: AtomIndex::default(),
            pieces: Vec::new(),
            paint_overhang: 0.0..0.0,
            font_runs,
            metrics,
        };
        let mut atoms = AtomBuilder::default();
        let mut input = Vec::<InputRun>::new();
        let mut column = 0;
        for chunk in row_chunks(snapshot, row, 0..len).1 {
            let renderer = chunk
                .renderer
                .as_ref()
                .map(|renderer| (renderer.id, renderer.measured_width));
            if renderer.is_some() || chunk.is_inlay {
                geometry.content_end = column + chunk.text.len() as u32;
            } else if let Some((offset, character)) = chunk
                .text
                .char_indices()
                .rev()
                .find(|(_, character)| !character.is_whitespace() || is_invisible(*character))
            {
                geometry.content_end = column + (offset + character.len_utf8()) as u32;
            }
            if let Some(last) = input.last_mut()
                && last.renderer.as_ref().map(|renderer| renderer.0)
                    == renderer.as_ref().map(|renderer| renderer.0)
            {
                last.text.push_str(chunk.text);
            } else {
                input.push(InputRun {
                    start: column,
                    text: chunk.text.to_string(),
                    renderer,
                });
            }
            column += chunk.text.len() as u32;
        }
        let font_runs = geometry.font_runs.clone();
        let mut fonts = FontRunsCursor::new(&font_runs);
        for run in input {
            if let Some((_, measured_width)) = run.renderer {
                atoms.push(run.start, geometry.width);
                geometry.width += match measured_width {
                    Some(width) => f64::from(width),
                    None => {
                        let fonts = fonts.slice(run.start..run.start + run.text.len() as u32);
                        let (layout, positions, width) =
                            wrapper.layout_line_with_precise_positions(&run.text, &fonts);
                        let overhang = glyph_overhang(&layout, &positions, text_system, |_| 0.);
                        geometry.paint_overhang.start =
                            geometry.paint_overhang.start.min(overhang.start);
                        geometry.paint_overhang.end = geometry.paint_overhang.end.max(overhang.end);
                        width
                    }
                };
                continue;
            }
            if run.text.bytes().all(|byte| (b' '..=b'~').contains(&byte)) {
                geometry.push_text(run.start, &run.text, &mut fonts, wrapper, &mut atoms);
                continue;
            }
            let mut start = 0;
            for (offset, grapheme) in run.text.grapheme_indices(true) {
                let Some(shown) = single_invisible(grapheme).and_then(replacement) else {
                    continue;
                };
                geometry.push_text(
                    run.start + start as u32,
                    &run.text[start..offset],
                    &mut fonts,
                    wrapper,
                    &mut atoms,
                );
                let column = run.start + offset as u32;
                let end = column + grapheme.len() as u32;

                atoms.push(column, geometry.width);
                let font_id = fonts
                    .slice(column..end)
                    .first()
                    .map_or(wrapper.font_id(), |run| run.font_id);
                let mut bytes = [0; 4];
                let shown = shown.encode_utf8(&mut bytes);
                let (layout, positions, width) = wrapper.layout_line_with_precise_positions(
                    shown,
                    &[FontRun {
                        len: shown.len(),
                        font_id,
                    }],
                );
                geometry.metrics.ascent = geometry.metrics.ascent.max(layout.ascent);
                geometry.metrics.descent = geometry.metrics.descent.max(layout.descent);
                let overhang = glyph_overhang(&layout, &positions, text_system, |_| 0.);
                geometry.paint_overhang.start = geometry.paint_overhang.start.min(overhang.start);
                geometry.paint_overhang.end = geometry.paint_overhang.end.max(overhang.end);
                geometry.width += width;
                start = offset + grapheme.len();
            }
            geometry.push_text(
                run.start + start as u32,
                &run.text[start..],
                &mut fonts,
                wrapper,
                &mut atoms,
            );
        }
        if geometry.atoms.columns.is_empty() {
            geometry.atoms = atoms.finish(
                geometry
                    .pieces
                    .first()
                    .filter(|piece| geometry.pieces.len() == 1 && piece.columns == (0..len)),
            );
        }
        let mut cursor = AtomCursor::new(&geometry.atoms.columns);
        for piece in &geometry.pieces {
            let overhang = glyph_overhang(
                &piece.layout,
                &piece.index.positions,
                text_system,
                |column| {
                    let atom = cursor
                        .at(piece.columns.start as usize + column)
                        .unwrap_or(0);
                    geometry
                        .atoms
                        .positions
                        .get(atom)
                        .copied()
                        .unwrap_or(piece.x)
                        - piece.x
                },
            );
            geometry.paint_overhang.start = geometry.paint_overhang.start.min(overhang.start);
            geometry.paint_overhang.end = geometry.paint_overhang.end.max(overhang.end);
        }
        geometry
    }

    fn push_text(
        &mut self,
        start: u32,
        text: &str,
        fonts: &mut FontRunsCursor<'_>,
        wrapper: &LineWrapper,
        row_atoms: &mut AtomBuilder,
    ) {
        if text.is_empty() {
            return;
        }
        let columns = start..start + text.len() as u32;
        let fonts = fonts.slice(columns.clone());
        let (layout, positions, width) = wrapper.layout_line_with_precise_positions(text, &fonts);
        let index = Arc::new(LayoutIndex::new(&layout, &fonts, positions, width));
        let end_x = self.width + width;
        let printable_ascii = text.bytes().all(|byte| (b' '..=b'~').contains(&byte));
        let shared = start == 0
            && self.width == 0.
            && row_atoms.columns.is_empty()
            && text.len() == self.len as usize
            && index.partition_glyphs.is_none()
            && index.partition_columns.len() == index.positions.len()
            && width.is_finite()
            && (if printable_ascii {
                index
                    .partition_columns
                    .iter()
                    .copied()
                    .eq(0..text.len() as u32)
            } else {
                text.grapheme_indices(true)
                    .map(|(offset, _)| offset as u32)
                    .eq(index.partition_columns.iter().copied())
            })
            && index
                .positions
                .iter()
                .try_fold(0., |previous, position| {
                    (*position >= previous && *position <= width).then_some(*position)
                })
                .is_some();
        if shared {
            self.atoms = AtomIndex {
                columns: index.partition_columns.clone(),
                positions: index.positions.clone(),
            };
        } else {
            let atoms = if printable_ascii {
                (0..text.len() as u32).collect::<Vec<_>>()
            } else {
                text.grapheme_indices(true)
                    .map(|(offset, _)| offset as u32)
                    .collect::<Vec<_>>()
            };
            let positions = atom_offsets_x(&layout, &atoms, &index);
            for (column, position) in atoms.into_iter().zip(positions) {
                let previous = row_atoms.positions.last().copied().unwrap_or(self.width);
                row_atoms.push(
                    start + column,
                    (self.width + position).clamp(previous, end_x.max(previous)),
                );
            }
        }
        self.metrics.ascent = self.metrics.ascent.max(layout.ascent);
        self.metrics.descent = self.metrics.descent.max(layout.descent);
        self.pieces.push(TextPiece {
            columns,
            x: self.width,
            text: SharedString::new(text),
            layout: Arc::new(layout),
            runs: fonts,
            index,
        });
        self.width = end_x;
    }

    fn x_for_boundary(&self, column: u32) -> ScrollPixelOffset {
        if column == 0 {
            return 0.;
        }
        self.atoms
            .positions
            .get(self.atoms.columns.partition_point(|start| *start < column))
            .copied()
            .unwrap_or(self.width)
    }

    fn atom_at_x(&self, x: ScrollPixelOffset) -> Atom {
        let mut index = self
            .atoms
            .positions
            .partition_point(|position| *position <= x)
            .saturating_sub(1);
        if let Some(piece) = self
            .pieces
            .partition_point(|piece| piece.x <= x)
            .checked_sub(1)
            .and_then(|index| self.pieces.get(index))
            && self
                .atoms
                .columns
                .get(index)
                .is_some_and(|column| *column < piece.columns.start)
        {
            index = self
                .atoms
                .columns
                .partition_point(|column| *column < piece.columns.start);
        }
        let start = self.atoms.columns.get(index).copied().unwrap_or(self.len);
        let start_x = self
            .atoms
            .positions
            .get(index)
            .copied()
            .unwrap_or(self.width);
        let end = self
            .atoms
            .columns
            .get(index + 1)
            .copied()
            .unwrap_or(self.len);
        let end_x = self
            .atoms
            .positions
            .get(index + 1)
            .copied()
            .unwrap_or(self.width);
        Atom {
            start,
            end,
            start_x,
            end_x,
        }
    }

    fn boundary(&self, column: u32, bias: Bias) -> u32 {
        if column >= self.len {
            return self.len;
        }
        let index = self
            .pieces
            .partition_point(|piece| piece.columns.end <= column);
        if let Some(piece) = self
            .pieces
            .get(index)
            .filter(|piece| piece.columns.start <= column)
        {
            return piece.columns.start
                + piece
                    .index
                    .boundary((column - piece.columns.start) as usize, bias)
                    .0 as u32;
        }
        let index = match bias {
            Bias::Left => self
                .atoms
                .columns
                .partition_point(|start| *start <= column)
                .saturating_sub(1),
            Bias::Right => self.atoms.columns.partition_point(|start| *start < column),
        };
        self.atoms.columns.get(index).copied().unwrap_or(self.len)
    }
}

fn glyph_overhang(
    layout: &LineLayout,
    positions: &[f64],
    text_system: &TextSystem,
    mut origin_for_index: impl FnMut(usize) -> f64,
) -> Range<f64> {
    let mut overhang = 0.0_f64..0.0_f64;
    let mut positions = positions.iter();
    for run in &layout.runs {
        let width = f64::from(
            text_system
                .bounding_box(run.font_id, layout.font_size)
                .size
                .width,
        );
        for (glyph, x) in run.glyphs.iter().zip(positions.by_ref()) {
            let x = x - origin_for_index(glyph.index);
            overhang.start = overhang.start.min(x);
            overhang.end = overhang.end.max(x + width);
        }
    }
    overhang
}

fn next_epoch() -> usize {
    NEXT_EPOCH.fetch_add(1, atomic::Ordering::Relaxed)
}

fn cluster_prefix(snapshot: &TabSnapshot, row: u32, column: u32) -> String {
    let len = snapshot.line_len(row);
    let column = clip_row_columns(snapshot, row, column..column).start;
    if column == 0 || column >= len {
        return String::new();
    }
    let mut context_len = 64;
    loop {
        let (mut start, chunks) = row_chunks(
            snapshot,
            row,
            column.saturating_sub(context_len)..column + 1,
        );
        let mut complete = start == 0;
        let mut position = start;
        let mut text = String::new();
        for chunk in chunks {
            let end = position + chunk.text.len() as u32;
            if chunk.renderer.is_some() {
                if position <= column && column <= end {
                    return String::new();
                }
                if column < position {
                    break;
                }
                text.clear();
                start = end;
                complete = true;
            } else {
                text.push_str(chunk.text);
            }
            position = end;
        }
        let offset = (column - start) as usize;
        if offset >= text.len() {
            return String::new();
        }
        let cluster_start = if complete {
            Some(
                text.grapheme_indices(true)
                    .take_while(|(start, _)| *start <= offset)
                    .last()
                    .map_or(0, |(start, _)| start),
            )
        } else {
            let mut cursor = GraphemeCursor::new(column as usize, len as usize, true);
            match cursor.is_boundary(&text, start as usize) {
                Ok(true) => return String::new(),
                Ok(false) => match cursor.prev_boundary(&text, start as usize) {
                    Ok(Some(boundary)) => Some(boundary - start as usize),
                    Ok(None) | Err(_) => None,
                },
                Err(_) => None,
            }
        };
        if let Some(cluster_start) = cluster_start {
            return text[cluster_start..offset].to_string();
        }
        context_len = context_len.saturating_mul(2);
    }
}

impl TextPiece {
    pub(crate) fn is_shaped_with(&self, runs: impl IntoIterator<Item = FontRun>) -> bool {
        runs.into_iter()
            .filter(|run| run.len > 0)
            .eq(self.runs.iter().copied())
    }

    pub(crate) fn window(&self, columns: &Range<u32>, origin_x: ScrollPixelOffset) -> Option<Self> {
        let start = columns.start.max(self.columns.start);
        let end = columns.end.min(self.columns.end);
        if start >= end {
            return None;
        }
        let index = &self.index;
        let start = index.boundary((start - self.columns.start) as usize, Bias::Left);
        let end = index.boundary((end - self.columns.start) as usize, Bias::Right);
        let origin_x = (origin_x - self.x).max(start.2).min(end.2);
        if start.0 == 0 && end.0 == self.layout.len && origin_x == 0. {
            return Some(self.clone());
        }
        let first_run = index
            .fonts
            .partition_point(|offset| *offset <= start.0)
            .saturating_sub(1);
        let runs = slice_font_runs(
            self.runs[first_run..].iter().copied(),
            index.fonts[first_run] as u32,
            start.0 as u32..end.0 as u32,
        );
        let layout = index.slice(&self.layout, start, end, origin_x);
        let positions = index.positions[start.1..end.1]
            .iter()
            .map(|position| position - origin_x)
            .collect();
        let index = Arc::new(LayoutIndex::new(
            &layout,
            &runs,
            positions,
            end.2 - origin_x,
        ));
        Some(Self {
            columns: self.columns.start + start.0 as u32..self.columns.start + end.0 as u32,
            x: self.x + origin_x,
            text: if start.0 == 0 && end.0 == self.layout.len {
                self.text.clone()
            } else {
                SharedString::new(&self.text[start.0..end.0])
            },
            layout,
            runs,
            index,
        })
    }
}

struct LayoutIndex {
    partition_columns: Arc<[u32]>,
    partition_glyphs: Option<Box<[usize]>>,
    runs: Box<[usize]>,
    fonts: Box<[usize]>,
    positions: Arc<[f64]>,
    len: usize,
    width: f64,
}

impl LayoutIndex {
    fn new(layout: &LineLayout, fonts: &[FontRun], positions: Vec<f64>, width: f64) -> Self {
        let mut partition_columns = Vec::new();
        let mut partition_glyphs = None::<Vec<usize>>;
        for (column, offset) in [(0, 0)].into_iter().chain(glyph_partitions(layout)) {
            let column = column as u32;
            if partition_columns.last() == Some(&column) {
                continue;
            }
            if let Some(offsets) = &mut partition_glyphs {
                offsets.push(offset);
            } else if offset != partition_columns.len() {
                let mut offsets = (0..partition_columns.len()).collect::<Vec<_>>();
                offsets.push(offset);
                partition_glyphs = Some(offsets);
            }
            partition_columns.push(column);
        }
        let mut offset = 0;
        let runs = layout
            .runs
            .iter()
            .map(|run| {
                let start = offset;
                offset += run.glyphs.len();
                start
            })
            .chain([layout.runs.iter().map(|run| run.glyphs.len()).sum()])
            .collect();
        let mut offset = 0;
        let fonts = fonts
            .iter()
            .map(|run| {
                let start = offset;
                offset += run.len;
                start
            })
            .collect();
        Self {
            partition_columns: Arc::from(partition_columns),
            partition_glyphs: partition_glyphs.map(Vec::into_boxed_slice),
            runs,
            fonts,
            positions: Arc::from(positions),
            len: layout.len,
            width,
        }
    }

    fn boundary(&self, column: usize, bias: Bias) -> (usize, usize, ScrollPixelOffset) {
        if column >= self.len {
            return (self.len, self.positions.len(), self.width);
        }
        let index = match bias {
            Bias::Left => self
                .partition_columns
                .partition_point(|start| *start as usize <= column)
                .saturating_sub(1),
            Bias::Right => self
                .partition_columns
                .partition_point(|start| (*start as usize) < column),
        };
        let Some(column) = self.partition_columns.get(index) else {
            return (self.len, self.positions.len(), self.width);
        };
        let offset = self
            .partition_glyphs
            .as_ref()
            .map_or(index, |offsets| offsets[index]);
        let x = if *column == 0 {
            0.
        } else {
            self.positions.get(offset).copied().unwrap_or(self.width)
        };
        (*column as usize, offset, x)
    }

    fn slice(
        &self,
        layout: &LineLayout,
        start: (usize, usize, ScrollPixelOffset),
        end: (usize, usize, ScrollPixelOffset),
        origin_x: ScrollPixelOffset,
    ) -> Arc<LineLayout> {
        let first = self
            .runs
            .partition_point(|offset| *offset <= start.1)
            .saturating_sub(1);
        let last = self.runs.partition_point(|offset| *offset < end.1);
        let runs = layout
            .runs
            .iter()
            .enumerate()
            .skip(first)
            .take(last.saturating_sub(first))
            .map(|(index, run)| {
                let run_start = self.runs[index];
                let glyphs = run
                    .glyphs
                    .iter()
                    .zip(&self.positions[run_start..])
                    .skip(start.1.saturating_sub(run_start))
                    .take(end.1 - run_start.max(start.1))
                    .map(|(glyph, x)| ShapedGlyph {
                        index: glyph.index - start.0,
                        position: point(px((x - origin_x) as f32), glyph.position.y),
                        ..glyph.clone()
                    })
                    .collect();
                ShapedRun {
                    font_id: run.font_id,
                    glyphs,
                }
            })
            .collect();
        Arc::new(LineLayout {
            font_size: layout.font_size,
            width: px((end.2 - origin_x) as f32),
            ascent: layout.ascent,
            descent: layout.descent,
            runs,
            len: end.0 - start.0,
        })
    }
}

impl RowWindow {
    pub(crate) fn prefix(end: u32) -> Self {
        Self {
            segments: smallvec![0..end],
        }
    }

    pub(crate) fn segments(&self) -> &[Range<u32>] {
        &self.segments
    }

    pub(crate) fn end(&self) -> u32 {
        self.segments.last().map_or(0, |segment| segment.end)
    }
}

impl FromIterator<Range<u32>> for RowWindow {
    fn from_iter<I: IntoIterator<Item = Range<u32>>>(segments: I) -> Self {
        let mut merged = SmallVec::<[Range<u32>; 1]>::new();
        for segment in segments {
            match merged.last_mut() {
                Some(previous) if segment.start <= previous.end => {
                    previous.end = previous.end.max(segment.end)
                }
                _ => merged.push(segment),
            }
        }
        Self { segments: merged }
    }
}

struct LongRowScan {
    stack: Vec<Range<u32>>,
}

enum ScanStep {
    Row { row: u32 },
    Continue,
    Done,
}

impl LongRowScan {
    fn new(rows: Range<u32>) -> Self {
        Self { stack: vec![rows] }
    }

    fn step(&mut self, snapshot: &TabSnapshot) -> ScanStep {
        let Some(rows) = self.stack.pop() else {
            return ScanStep::Done;
        };
        if rows.is_empty() || snapshot.max_line_len_bound(rows.clone()) as usize <= MAX_LINE_LEN {
            return ScanStep::Continue;
        }
        if rows.len() == 1 {
            let len = snapshot.line_len(rows.start);
            return if len as usize > MAX_LINE_LEN {
                ScanStep::Row { row: rows.start }
            } else {
                ScanStep::Continue
            };
        }
        let middle = rows.start + (rows.end - rows.start) / 2;
        self.stack.push(middle..rows.end);
        self.stack.push(rows.start..middle);
        ScanStep::Continue
    }
}

fn atom_offsets_x(
    layout: &LineLayout,
    atoms: &[u32],
    index: &LayoutIndex,
) -> Vec<ScrollPixelOffset> {
    let mut left_edges = vec![None::<ScrollPixelOffset>; atoms.len()];
    let mut cursor = AtomCursor::new(atoms);
    for (offset, glyph) in layout.runs.iter().flat_map(|run| &run.glyphs).enumerate() {
        let Some(atom) = cursor.at(glyph.index) else {
            continue;
        };
        let x = index.positions[offset];
        let left_edge = &mut left_edges[atom];
        *left_edge = Some(left_edge.map_or(x, |left_edge| left_edge.min(x)));
    }
    let mut edges = left_edges
        .into_iter()
        .enumerate()
        .filter_map(|(atom, x)| x.map(|x| (atom, x)))
        .peekable();
    let mut positions = Vec::with_capacity(atoms.len());
    let (first, x) = edges.peek().copied().unwrap_or((atoms.len(), index.width));
    positions.resize(first, x);
    while let Some((atom, cluster_x)) = edges.next() {
        let (end, end_x) = edges.peek().copied().unwrap_or((atoms.len(), index.width));
        let parts = (end - atom) as f64;
        for rank in 0..end - atom {
            positions.push(cluster_x + (end_x - cluster_x) * rank as f64 / parts);
        }
    }
    positions
}

fn single_invisible(text: &str) -> Option<char> {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(character), None) if is_invisible(character) => Some(character),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomBoundary {
    Cluster,
    Interior,
}

pub(super) struct StandaloneInvisibles {
    atom: String,
    last: AtomBoundary,
}

impl Default for StandaloneInvisibles {
    fn default() -> Self {
        Self {
            atom: String::new(),
            last: AtomBoundary::Cluster,
        }
    }
}

impl StandaloneInvisibles {
    pub(super) fn for_columns(snapshot: &TabSnapshot, row: u32, columns: &Range<u32>) -> Self {
        let atom = cluster_prefix(snapshot, row, columns.start);
        let last = if cluster_prefix(snapshot, row, columns.end).is_empty() {
            AtomBoundary::Cluster
        } else {
            AtomBoundary::Interior
        };
        Self { atom, last }
    }

    pub(super) fn reset(&mut self) {
        self.atom.clear();
    }

    pub(super) fn find(
        &mut self,
        text: &str,
        next: Option<char>,
        is_last: bool,
    ) -> SmallVec<[Range<usize>; 1]> {
        if text.len() > 1 && text.bytes().all(|byte| (b' '..=b'~').contains(&byte)) {
            self.atom.clear();
            self.atom.push_str(&text[text.len() - 1..]);
            return SmallVec::new();
        }
        let mut found = SmallVec::new();
        let mut candidate = None::<Range<usize>>;
        let mut confirm = |candidate: &mut Option<Range<usize>>, boundary: Option<AtomBoundary>| {
            if let Some(range) = candidate.take()
                && boundary == Some(AtomBoundary::Cluster)
            {
                found.push(range);
            }
        };
        let ascii_suffix = text.bytes().rev().take_while(u8::is_ascii).count();
        let general_end = (text.len() - ascii_suffix + 1).min(text.len());
        for (offset, character) in text[..general_end].char_indices() {
            let boundary = self.push(character);
            confirm(&mut candidate, boundary);
            if boundary == Some(AtomBoundary::Cluster) && is_invisible(character) {
                candidate = Some(offset..offset + character.len_utf8());
            }
        }
        let bytes = text.as_bytes();
        let mut last_atom_start = None;
        for offset in general_end..text.len() {
            let boundary = (!(bytes[offset - 1] == b'\r' && bytes[offset] == b'\n'))
                .then_some(AtomBoundary::Cluster);
            confirm(&mut candidate, boundary);
            if boundary.is_some() {
                last_atom_start = Some(offset);
                if is_invisible(char::from(bytes[offset])) {
                    candidate = Some(offset..offset + 1);
                }
            }
        }
        match last_atom_start {
            Some(start) => {
                self.atom.clear();
                self.atom.push_str(&text[start..]);
            }
            None => self.atom.push_str(&text[general_end..]),
        }
        if candidate.is_some() {
            let boundary = match next {
                Some(next) => self.would_start_atom(next),
                None if is_last => Some(self.last),
                None => Some(AtomBoundary::Cluster),
            };
            confirm(&mut candidate, boundary);
        }
        found
    }

    fn push(&mut self, character: char) -> Option<AtomBoundary> {
        let len = self.atom.len();
        self.atom.push(character);
        if len == 0 {
            return Some(AtomBoundary::Cluster);
        }
        let boundary = is_cluster_boundary(&self.atom, len).then_some(AtomBoundary::Cluster);
        if boundary.is_some() {
            self.atom.clear();
            self.atom.push(character);
        }
        boundary
    }

    fn would_start_atom(&mut self, character: char) -> Option<AtomBoundary> {
        let len = self.atom.len();
        if len == 0 {
            return Some(AtomBoundary::Cluster);
        }
        self.atom.push(character);
        let boundary = is_cluster_boundary(&self.atom, len).then_some(AtomBoundary::Cluster);
        self.atom.truncate(len);
        boundary
    }
}

fn is_cluster_boundary(text: &str, offset: usize) -> bool {
    let bytes = text.as_bytes();
    if let (Some(&previous), Some(&next)) = (bytes.get(offset.wrapping_sub(1)), bytes.get(offset))
        && previous.is_ascii()
        && next.is_ascii()
    {
        return !(previous == b'\r' && next == b'\n');
    }
    GraphemeCursor::new(offset, text.len(), true)
        .is_boundary(text, 0)
        .unwrap_or(false)
}

pub(super) fn clip_row_columns(
    snapshot: &TabSnapshot,
    row: u32,
    columns: Range<u32>,
) -> Range<u32> {
    let len = snapshot.line_len(row);
    let clip = |column: u32, bias: Bias, fallback: u32| {
        let column = column.min(len);
        let (_, _, to_next_tab_stop) =
            snapshot.tab_point_to_fold_point(TabPoint::new(row, column), Bias::Left);
        if to_next_tab_stop > 0 {
            column
        } else {
            clip_column(snapshot, row, column, bias).unwrap_or(fallback)
        }
    };
    let start = clip(columns.start, Bias::Left, 0);
    let end = clip(columns.end, Bias::Right, len);
    start..end.max(start)
}

fn row_chunks(snapshot: &TabSnapshot, row: u32, columns: Range<u32>) -> (u32, TabChunks<'_>) {
    let columns = clip_row_columns(snapshot, row, columns);
    let chunks = snapshot.chunks(
        TabPoint::new(row, columns.start)..TabPoint::new(row, columns.end),
        LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: false,
        },
        Highlights::default(),
    );
    (columns.start, chunks)
}

fn clip_column(snapshot: &TabSnapshot, row: u32, column: u32, bias: Bias) -> Option<u32> {
    let fold_point = snapshot
        .tab_point_to_fold_point(TabPoint::new(row, column), bias)
        .0;
    let clipped = snapshot.fold_point_to_tab_point(
        snapshot
            .fold_snapshot
            .clip_point_to_char_boundary(fold_point, bias),
    );
    (clipped.row() == row).then_some(clipped.column())
}

fn glyph_partitions(layout: &LineLayout) -> impl Iterator<Item = (usize, usize)> {
    let indices = layout
        .runs
        .iter()
        .flat_map(|run| &run.glyphs)
        .map(|glyph| glyph.index);
    let mut suffix_min = if indices.clone().is_sorted() {
        Vec::new()
    } else {
        indices.clone().collect::<Vec<_>>()
    };
    let mut minimum = usize::MAX;
    for index in suffix_min.iter_mut().rev() {
        minimum = minimum.min(*index);
        *index = minimum;
    }
    let mut suffix_min = suffix_min.into_iter();
    let mut prefix_max = None::<usize>;
    indices.enumerate().filter_map(move |(position, index)| {
        let minimum = suffix_min.next().unwrap_or(index);
        let partition = prefix_max
            .is_none_or(|max| max < minimum)
            .then_some((minimum, position));
        prefix_max = Some(prefix_max.map_or(index, |max| max.max(index)));
        partition
    })
}

fn slice_font_runs(
    runs: impl IntoIterator<Item = FontRun>,
    start: u32,
    columns: Range<u32>,
) -> SmallVec<[FontRun; 1]> {
    let mut sliced = SmallVec::new();
    let mut run_start = start;
    for run in runs {
        let run_end = run_start + run.len as u32;
        let sliced_start = run_start.max(columns.start);
        let sliced_end = run_end.min(columns.end);
        if sliced_start < sliced_end {
            sliced.push(FontRun {
                len: (sliced_end - sliced_start) as usize,
                font_id: run.font_id,
            });
        }
        run_start = run_end;
        if run_start >= columns.end {
            break;
        }
    }
    sliced
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Editor, EditorMode, EditorStyle,
        display_map::{
            ChunkReplacement, Crease, DisplayMap, DisplayRow, DisplaySnapshot, FoldPlaceholder,
            HighlightKey, RendererWidthsKey, SyntaxStyling, fold_map::FoldMap, inlay_map::InlayMap,
            tab_map::TabMap,
        },
        element::EditorElement,
        inlays::Inlay,
        test::{ShapingTextSystem, test_font},
    };
    use gpui::{
        AppContext as _, BorrowAppContext as _, Bounds, DevicePixels, Entity, Font, FontMetrics,
        FontWeight, GlyphId, HighlightStyle, PlatformTextSystem, RenderGlyphParams, Size,
        StrikethroughStyle, TestAppContext, TextRenderingMode, UnderlineStyle, rgba, size,
    };
    use language::{Buffer, Diagnostic, DiagnosticEntry, DiagnosticSet, Point};
    use lsp::LanguageServerId;
    use multi_buffer::{MultiBuffer, MultiBufferOffset};
    use project::{InlayId, project_settings::DiagnosticSeverity};
    use rand::prelude::*;
    use settings::SettingsStore;
    use std::{borrow::Cow, num::NonZeroU32, ops::RangeInclusive};
    use text::PointUtf16;
    use theme::LoadThemes;

    const FONT_SIZE: gpui::Pixels = px(14.);

    #[gpui::test(iterations = 10)]
    async fn test_random_long_row_edits(cx: &mut TestAppContext, mut rng: StdRng) {
        cx.background_executor.set_block_on_ticks(0..=50);
        let text = (0..4)
            .map(|_| random_text(&mut rng, 200..1_200))
            .collect::<Vec<_>>()
            .join("\n");
        let (buffer, map) = build_display_map(&text, 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));

        for _ in 0..10 {
            for _ in 0..rng.random_range(1..=3) {
                randomly_mutate(&buffer, &map, &mut rng, cx);
                let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
                assert_round_trips(&snapshot);
            }
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_matches_char_widths(&snapshot, &mut wrapper);
        }
    }

    #[gpui::test]
    async fn test_long_inlays_at_buffer_edges(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map("x", 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        let inlays = [
            Inlay::mock_hint(
                0,
                buffer_snapshot.anchor_before(MultiBufferOffset(0)),
                "漢".repeat(200),
            ),
            Inlay::mock_hint(
                1,
                buffer_snapshot.anchor_after(MultiBufferOffset(1)),
                "é".repeat(1_000),
            ),
        ];
        for inlay in inlays {
            map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_round_trips(&snapshot);
            assert_matches_char_widths(&snapshot, &mut wrapper);
        }
    }

    #[gpui::test]
    async fn test_tab_expanded_long_rows(cx: &mut TestAppContext) {
        let text = format!("{}{}", "\t".repeat(129), "é".repeat(50));
        let (_, map) = build_display_map(&text, 8, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_eq!(snapshot.line_len(DisplayRow(0)), 129 * 8 + 50 * 2);
        assert_matches_char_widths(&snapshot, &mut wrapper);
    }

    #[gpui::test]
    async fn test_long_rows_keep_grapheme_clusters_across_splices(cx: &mut TestAppContext) {
        let joined = (0..150)
            .map(|index| {
                let letter = char::from(b'a' + (index % 26) as u8);
                format!("{letter}\u{200d}\u{1f600}b\u{1f1fa}\u{1f1f8}e\u{301}")
            })
            .collect::<String>();
        let flags =
            |pairs: usize| format!("{}{}", "\u{1f1e8}\u{1f1e6}".repeat(pairs), "ab".repeat(600));
        for (text, joins_seam) in [(joined, true), (flags(40), false), (flags(300), false)] {
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            assert_cluster_atoms(&snapshot);

            let (edited, inserted) = if joins_seam {
                let seam = text.find('\u{1f600}').expect("an emoji");
                let joiner_start = seam - "a\u{200d}".len();
                (joiner_start..joiner_start + 1, "\u{1f600}")
            } else {
                (0..0, "\u{1f1e6}")
            };
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(
                        MultiBufferOffset(edited.start)..MultiBufferOffset(edited.end),
                        inserted,
                    )],
                    None,
                    cx,
                )
            });
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            assert_eq!(
                long_row.len() as usize,
                text.len() - edited.len() + inserted.len()
            );
            assert_cluster_atoms(&snapshot);
            cx.run_until_parked();
            assert_cluster_atoms(&map.update(cx, |map, cx| map.snapshot(cx)));
        }
    }

    #[gpui::test]
    async fn test_long_rows_use_measured_inlay_widths(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map(&"x".repeat(2_000), 4, cx);
        let position = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(10))
        });
        let inlay = Inlay::color(0, position, rgba(0xff0000ff));
        map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
        cx.run_until_parked();
        let inlay_width = |map: &Entity<DisplayMap>, cx: &mut TestAppContext| {
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            long_row.x_for_column(10 + "\u{25fc}".len() as u32) - long_row.x_for_column(10)
        };
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        assert_eq!(
            inlay_width(&map, cx),
            f64::from(wrapper.width_for_char('\u{25fc}'))
        );

        let key = |font_size: f32| RendererWidthsKey {
            rem_size: px(16.),
            font: test_font(),
            font_size: px(font_size),
            buffer_font: test_font(),
            editor_width: px(500.),
            line_height: px(20.),
        };
        let renderer_widths = [(ChunkRendererId::Inlay(InlayId::Color(0)), px(37.))];
        assert!(map.update(cx, |map, cx| {
            map.update_renderer_widths(key(14.), renderer_widths, cx)
        }));
        assert_eq!(inlay_width(&map, cx), 37.);
        assert!(!map.update(cx, |map, cx| {
            map.update_renderer_widths(key(14.), renderer_widths, cx)
        }));

        assert!(map.update(cx, |map, cx| {
            map.update_renderer_widths(key(28.), [], cx)
        }));
        assert_eq!(
            inlay_width(&map, cx),
            f64::from(wrapper.width_for_char('\u{25fc}'))
        );
    }

    #[gpui::test]
    async fn test_soft_wrap_counts_renderer_width_once(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map("a", 4, cx);
        let position = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(1))
        });
        let inlay = Inlay::mock_repl_result(0, position, &"r".repeat(300));
        map.update(cx, |map, cx| {
            map.splice_inlays(&[], vec![inlay], cx);
            map.set_wrap_width(Some(px(150.)), cx);
        });
        cx.run_until_parked();
        map.update(cx, |map, cx| {
            map.update_renderer_widths(
                RendererWidthsKey {
                    rem_size: px(16.),
                    font: test_font(),
                    font_size: FONT_SIZE,
                    buffer_font: test_font(),
                    editor_width: px(500.),
                    line_height: px(20.),
                },
                [(ChunkRendererId::Inlay(InlayId::ReplResult(0)), px(100.))],
                cx,
            )
        });
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_eq!(snapshot.max_point().row(), DisplayRow(0));
    }

    #[gpui::test]
    async fn test_long_rows_replace_only_standalone_invisibles(cx: &mut TestAppContext) {
        let mut scanner = StandaloneInvisibles {
            atom: "\u{d4e}".to_string(),
            last: AtomBoundary::Interior,
        };
        for (text, atom) in [
            ("", "\u{d4e}"),
            ("a", "\u{d4e}a"),
            ("\u{301}", "\u{d4e}a\u{301}"),
            ("\u{d4e}", "\u{d4e}"),
            (" ~", "~"),
            ("\u{301}", "~\u{301}"),
        ] {
            assert!(scanner.find(text, Some('\u{301}'), false).is_empty());
            assert_eq!(scanner.atom, atom);
        }
        assert!(scanner.find("\u{200b}", None, true).is_empty());
        scanner.last = AtomBoundary::Cluster;
        assert_eq!(scanner.find("\u{200b}", None, true).as_slice(), &[0..3]);
        assert_eq!(
            scanner.find("\x1f\x7f", None, true).as_slice(),
            &[0..1, 1..2]
        );

        let text = format!(
            "a{}\u{200c}b\u{200b}{}ab\u{200c}c a\u{200b}b",
            "\u{301}".repeat(15),
            "x".repeat(1_100)
        );
        let (buffer, map) = build_display_map(&text, 4, cx);
        let capped_joiner = text.find('\u{200c}').unwrap_or_default() as u32;
        let attached_joiner = text.rfind('\u{200c}').unwrap_or_default();
        let highlighted = text.rfind("ab").unwrap_or_default()..attached_joiner;
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        map.update(cx, |map, cx| {
            map.highlight_text(
                HighlightKey::Editor,
                vec![
                    buffer_snapshot.anchor_before(MultiBufferOffset(highlighted.start))
                        ..buffer_snapshot.anchor_after(MultiBufferOffset(highlighted.end)),
                ],
                HighlightStyle::color(gpui::red()),
                false,
                cx,
            )
        });
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let style = EditorStyle::default();
        let replaced = snapshot
            .highlighted_chunks(
                DisplayRow(0)..DisplayRow(1),
                LanguageAwareStyling {
                    tree_sitter: false,
                    diagnostics: false,
                },
                &style,
            )
            .filter(|chunk| matches!(chunk.replacement, Some(ChunkReplacement::Str(_))))
            .map(|chunk| chunk.text.to_string())
            .collect::<Vec<_>>();
        assert_eq!(replaced, ["\u{200b}", "\u{200b}"]);

        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let attached_joiner = attached_joiner as u32;
        assert_eq!(
            long_row.atom_range(attached_joiner).start,
            attached_joiner - 1
        );
        assert_eq!(long_row.atom_range(capped_joiner), 0..capped_joiner + 3);
        for (column, expected) in [
            (1, "a"),
            (capped_joiner, &text[..capped_joiner as usize]),
            (capped_joiner + 3, ""),
            (attached_joiner, "b"),
        ] {
            assert_eq!(cluster_prefix(snapshot.tab_snapshot(), 0, column), expected);
        }
        let language_aware = LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: false,
        };
        for columns in [0..long_row.len(), capped_joiner..long_row.len()] {
            let replaced = snapshot
                .highlighted_chunks_in_row(DisplayRow(0), columns.clone(), language_aware, &style)
                .filter(|chunk| matches!(chunk.replacement, Some(ChunkReplacement::Str(_))))
                .map(|chunk| chunk.text.to_string())
                .collect::<Vec<_>>();
            assert_eq!(replaced, ["\u{200b}", "\u{200b}"], "{columns:?}");
        }
    }

    #[gpui::test]
    fn test_cluster_prefix_recovers_unbounded_context_without_shaping(cx: &mut TestAppContext) {
        for clusters in [
            vec![format!("a{}", "\u{301}".repeat(256))],
            vec!["🇺🇸".to_string(); 34],
            vec![format!("{}🇺🇸", "\u{d4e}".repeat(256)), "🇺🇸".to_string()],
            vec![format!("👩{}\u{200d}💻", "\u{301}".repeat(256))],
            vec![format!("{}क", "क्".repeat(128))],
        ] {
            let text = format!(
                "{}{}{}",
                "x".repeat(128),
                clusters.concat(),
                "z".repeat(MAX_LINE_LEN)
            );
            let row = unmeasured_long_row(&text, cx);
            assert!(row.geometry.value.get().is_none());
            let mut start = 128;
            for cluster in clusters {
                for offset in [
                    0,
                    1,
                    4,
                    63,
                    64,
                    65,
                    cluster.len() / 2,
                    cluster.len() - 1,
                    cluster.len(),
                ] {
                    let offset = offset.min(cluster.len());
                    let expected = if offset == cluster.len() {
                        ""
                    } else {
                        &cluster[..cluster.floor_char_boundary(offset)]
                    };
                    assert_eq!(
                        cluster_prefix(&row.snapshot, 0, (start + offset) as u32),
                        expected,
                        "cluster {cluster:?}, offset {offset}"
                    );
                }
                start += cluster.len();
            }
            for column in [0, 63, 64, 65, start, text.len() - 1, text.len()] {
                assert_eq!(cluster_prefix(&row.snapshot, 0, column as u32), "");
            }
            assert!(row.geometry.value.get().is_none());
        }
    }

    #[test]
    fn test_measured_dense_unicode_paints_visible_glyphs() {
        let mut cx = TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(IgnoredMarksTextSystem(ShapingTextSystem(
                gpui::NoopTextSystem,
            ))),
        );
        crate::editor_tests::init_test(&mut cx, |_| {});
        for (character, marks, count) in [('a', 524_288, 1), ('!', 8_192, 65)] {
            let text = format!(
                "{}END",
                format!("{character}{}", "\u{34f}".repeat(marks)).repeat(count)
            );
            let expected = (0..count)
                .map(|index| (index * (2 * marks + 1), character))
                .chain(
                    "END"
                        .char_indices()
                        .map(|(index, character)| (text.len() - 3 + index, character)),
                )
                .collect::<Vec<_>>();
            let cx = cx.add_empty_window();
            let editor = cx.update(|window, cx| {
                let buffer = MultiBuffer::build_simple(&text, cx);
                cx.new(|cx| Editor::new(EditorMode::full(), buffer, None, window, cx))
            });
            editor.update(cx, |editor, cx| {
                editor.set_soft_wrap_mode(language::language_settings::SoftWrap::None, cx);
            });
            let mut style = editor.update(cx, |editor, cx| editor.style(cx).clone());
            style.text.font_size = gpui::AbsoluteLength::from(px(10.));
            crate::element::tests::assert_long_row_paints(&editor, &style, &expected, cx);
        }
    }

    #[gpui::test]
    async fn test_long_rows_keep_flag_pairs_after_long_prepend_clusters(cx: &mut TestAppContext) {
        let flags = "\u{1f1fa}\u{1f1f8}".repeat(4);
        for prepend_count in [339, 340, 341] {
            let text = format!("{}{flags}{}", "x".repeat(1_100), "z".repeat(200));
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();

            let prepend = "\u{d4e}".repeat(prepend_count);
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(MultiBufferOffset(1_100)..MultiBufferOffset(1_100), prepend)],
                    None,
                    cx,
                )
            });
            cx.run_until_parked();
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
            let flags_start = 1_100 + prepend_count as u32 * "\u{d4e}".len() as u32;
            let atom_starts = long_row
                .geometry()
                .atoms
                .columns
                .iter()
                .copied()
                .filter(|start| flags_start <= *start && *start <= flags_start + flags.len() as u32)
                .map(|start| start - flags_start)
                .collect::<Vec<_>>();
            assert_eq!(atom_starts, [8, 16, 24, 32], "{prepend_count} prepends");
        }
    }

    #[gpui::test]
    async fn test_long_rows_after_whole_buffer_edits(cx: &mut TestAppContext) {
        let text = format!(
            "{}{}\n{}",
            "\ta\n".repeat(50),
            "\tx".repeat(800),
            "\tb\n".repeat(50)
        );
        let (_, map) = build_display_map(&text, 4, cx);
        let mut wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        cx.run_until_parked();
        assert_matches_char_widths(&map.update(cx, |map, cx| map.snapshot(cx)), &mut wrapper);

        set_tab_size(8, cx);
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert_round_trips(&snapshot);
        cx.run_until_parked();
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        assert!(snapshot.long_row(DisplayRow(50)).is_some());
        assert_matches_char_widths(&snapshot, &mut wrapper);
    }

    #[gpui::test]
    async fn test_folded_buffers_do_not_widen_long_rows(cx: &mut TestAppContext) {
        init_settings(4, cx);
        let long_text = "x".repeat(2_000);
        let buffer = cx.update(|cx| {
            MultiBuffer::build_multi(
                [
                    (
                        long_text.as_str(),
                        vec![Point::new(0, 0)..Point::new(0, 2_000)],
                    ),
                    ("short", vec![Point::new(0, 0)..Point::new(0, 5)]),
                ],
                cx,
            )
        });
        let map = new_display_map(&buffer, cx);
        cx.run_until_parked();
        assert!(
            map.update(cx, |map, cx| map.snapshot(cx))
                .max_long_row_width()
                > 0.
        );

        let long_buffer_id = buffer.read_with(cx, |buffer, cx| {
            buffer
                .snapshot(cx)
                .excerpts()
                .next()
                .map(|excerpt| excerpt.context.start.buffer_id)
                .expect("an excerpt")
        });
        map.update(cx, |map, cx| map.fold_buffers([long_buffer_id], cx));
        assert_eq!(
            map.update(cx, |map, cx| map.snapshot(cx))
                .max_long_row_width(),
            0.
        );
    }

    #[test]
    fn test_max_width_index_stays_bounded_and_invalidates() {
        for widths in [Vec::new(), vec![(5, 0.)]] {
            let rows = LongRows::default();
            let index = rows.finish_width_index(0, widths);
            for range in [0..0, 0..u32::MAX, 5..6, 6..u32::MAX] {
                assert_eq!(index.max_width_in(range), 0.);
            }
        }
        let cx = &mut shaping_context();
        cx.background_executor.set_block_on_ticks(0..=0);
        let short_rows = format!("{}\n", "x".repeat(257)).repeat(1_024);
        let text = format!("{short_rows}{}\n{}", "x".repeat(2_048), "x".repeat(1_536));
        let (buffer, map) = build_display_map(&text, 4, cx);
        set_test_shaping_style(&map, cx);
        let original = map.update(cx, |map, cx| map.snapshot(cx));
        let long_rows = &original.wrap_snapshot().long_rows;
        let widths = long_rows.widths.as_ref().expect("shaping metrics");
        assert_eq!(widths.shaped_rows.load(atomic::Ordering::Relaxed), 0);
        assert!(Arc::ptr_eq(
            &long_rows.rows,
            &long_rows.interpolate(original.tab_snapshot(), &[]).rows,
        ));
        assert_eq!(
            long_rows.max_width_in(original.tab_snapshot(), 0..1_024),
            0.
        );
        let index = long_rows.width_index.get().expect("complete index").clone();
        let expected_rows = [(1_024, 2_048. * 7.), (1_025, 1_536. * 7.)];
        assert_eq!(index.rows.as_ref(), expected_rows);
        for start in [0, 1, 511, 1_023, 1_024, 1_025, 1_026, u32::MAX] {
            for end in [0, 1, 511, 1_023, 1_024, 1_025, 1_026, u32::MAX] {
                let range = start..end;
                let expected = expected_rows
                    .iter()
                    .filter(|(row, _)| range.contains(row))
                    .fold(0_f64, |maximum, (_, width)| maximum.max(*width));
                assert_eq!(
                    long_rows
                        .clone()
                        .max_width_in(original.tab_snapshot(), range),
                    expected,
                    "{start}..{end}",
                );
                assert!(Arc::ptr_eq(
                    long_rows.width_index.get().expect("cached index"),
                    &index
                ));
            }
        }
        let duplicate = long_rows.finish_width_index(index.version, expected_rows.to_vec());
        assert!(Arc::ptr_eq(&index, &duplicate));
        assert_eq!(widths.shaped_rows.load(atomic::Ordering::Relaxed), 2);

        let mut tab_snapshot = original.tab_snapshot().clone();
        tab_snapshot.version += 1;
        let unchanged = long_rows.interpolate(&tab_snapshot, &[]);
        tab_snapshot.version += 1;
        let unchanged = unchanged.interpolate(&tab_snapshot, &[]);
        assert_eq!(
            unchanged.max_width_in(&tab_snapshot, 0..u32::MAX),
            2_048. * 7.
        );
        assert_eq!(widths.shaped_rows.load(atomic::Ordering::Relaxed), 2);
        for row in [1_024, 1_025] {
            let previous = long_rows
                .row(original.tab_snapshot(), row)
                .expect("original row");
            let current = unchanged.row(&tab_snapshot, row).expect("unchanged row");
            assert!(Arc::ptr_eq(previous.geometry(), current.geometry()));
            assert!(current.geometry.previous.lock().is_none());
        }

        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(
                    MultiBufferOffset(short_rows.len())
                        ..MultiBufferOffset(short_rows.len() + 2_048),
                    "x".repeat(1_025),
                )],
                None,
                cx,
            );
        });
        let edited = map.update(cx, |map, cx| map.snapshot(cx));
        assert_eq!(edited.max_long_row_width(), 1_536. * 7.);
        assert_eq!(original.max_long_row_width(), 2_048. * 7.);
        assert_eq!(widths.shaped_rows.load(atomic::Ordering::Relaxed), 3);

        let snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        for color in [None, Some(gpui::blue())] {
            map.update(cx, |map, cx| {
                map.highlight_text(
                    HighlightKey::Editor,
                    vec![
                        snapshot.anchor_after(Point::new(1_025, 0))
                            ..snapshot.anchor_before(Point::new(1_025, 1_536)),
                    ],
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        color,
                        ..HighlightStyle::default()
                    },
                    false,
                    cx,
                );
            });
            let restyled = map.update(cx, |map, cx| map.snapshot(cx));
            assert_eq!(
                restyled.tab_snapshot().version,
                edited.tab_snapshot().version
            );
            assert_eq!(restyled.max_long_row_width(), 1_536. * 8.);
            assert_eq!(edited.max_long_row_width(), 1_536. * 7.);
            assert_eq!(original.max_long_row_width(), 2_048. * 7.);
            assert_eq!(widths.shaped_rows.load(atomic::Ordering::Relaxed), 4);
            assert!(
                restyled
                    .long_row(DisplayRow(1_025))
                    .expect("restyled row")
                    .geometry
                    .previous
                    .lock()
                    .is_none()
            );
        }
    }

    #[gpui::test]
    async fn test_long_rows_track_trailing_whitespace(cx: &mut TestAppContext) {
        let (buffer, map) = build_display_map(&format!("x{}", " ".repeat(10_000)), 4, cx);
        let trailing_whitespace_start = |map: &Entity<DisplayMap>, cx: &mut TestAppContext| {
            cx.run_until_parked();
            map.update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .and_then(|long_row| long_row.trailing_whitespace_start())
        };
        assert_eq!(trailing_whitespace_start(&map, cx), Some(1));

        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [(MultiBufferOffset(5_000)..MultiBufferOffset(5_000), "y")],
                None,
                cx,
            )
        });
        assert_eq!(trailing_whitespace_start(&map, cx), Some(5_001));

        let row_end = buffer.read_with(cx, |buffer, cx| {
            buffer.snapshot(cx).anchor_after(MultiBufferOffset(10_002))
        });
        let hint = Inlay::mock_hint(0, row_end, "hint ".to_string());
        map.update(cx, |map, cx| map.splice_inlays(&[], vec![hint], cx));
        assert_eq!(trailing_whitespace_start(&map, cx), Some(10_007));
    }

    #[gpui::test]
    async fn test_row_windows_keep_styles_from_before_their_start(cx: &mut TestAppContext) {
        init_settings(128, cx);
        let text = format!("abc\n{}\t{}", "x".repeat(79), "y".repeat(3_000));
        let buffer = cx.new(|cx| Buffer::local(text, cx));
        buffer.update(cx, |buffer, cx| {
            let diagnostics = DiagnosticSet::new(
                [DiagnosticEntry::new(
                    PointUtf16::new(0, 0)..PointUtf16::new(0, 3),
                    Diagnostic {
                        severity: lsp::DiagnosticSeverity::ERROR,
                        group_id: 1,
                        message: "error".into(),
                        ..Diagnostic::default()
                    },
                )],
                buffer,
            );
            buffer.update_diagnostics(LanguageServerId(0), diagnostics, cx)
        });
        let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
        let buffer_snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
        let map = new_display_map(&buffer, cx);
        let hint = Inlay::mock_hint(
            0,
            buffer_snapshot.anchor_after(MultiBufferOffset(1)),
            "h".repeat(4_096),
        );
        map.update(cx, |map, cx| {
            map.splice_inlays(&[], vec![hint], cx);
            map.highlight_text(
                HighlightKey::Editor,
                vec![
                    buffer_snapshot.anchor_before(Point::new(1, 79))
                        ..buffer_snapshot.anchor_after(Point::new(1, 80)),
                ],
                HighlightStyle::color(gpui::red()),
                false,
                cx,
            );
        });
        cx.run_until_parked();

        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let style = EditorStyle::default();
        let language_aware = LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: true,
        };
        let inlay_window = merge_chunks(
            snapshot
                .highlighted_chunks_in_row(DisplayRow(0), 100..350, language_aware, &style)
                .map(|chunk| (chunk.text, chunk.diagnostic_underline_severity)),
        );
        assert_eq!(
            inlay_window,
            [("h".repeat(250), Some(lsp::DiagnosticSeverity::ERROR))]
        );
        let tab_window = merge_chunks(
            snapshot
                .highlighted_chunks_in_row(DisplayRow(1), 90..110, language_aware, &style)
                .map(|chunk| (chunk.text, chunk.style.and_then(|style| style.color))),
        );
        assert_eq!(tab_window, [(" ".repeat(20), Some(gpui::red()))]);
    }

    #[test]
    fn test_long_row_positions_match_whole_line_shaping_after_edits() {
        let cx = &mut shaping_context();
        let mut state = 7_u64;
        let random_text = (0..1_500)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                ['A', 'V', 'f', 'i', 'x'][(state >> 33) as usize % 5]
            })
            .collect::<String>();
        let ligature_runs = (1..30)
            .map(|run| format!("x{}", "w".repeat(run * 7 % 101 + 20)))
            .collect::<String>();
        let forced_runs = [0, 1].map(|lead| format!("{}{}", "x".repeat(lead), "w".repeat(1_501)));
        let reordered = format!("a\u{915}{}b", ShapingTextSystem::PRE_BASE_VOWEL).repeat(150);
        for text in [random_text, ligature_runs, reordered]
            .into_iter()
            .chain(forced_runs)
        {
            let (buffer, map) = build_display_map(&text, 4, cx);
            cx.run_until_parked();
            assert_matches_whole_line_shaping(&map, None, cx);
            for (offset, inserted) in [(0, "w"), (700, "w"), (3, "\u{301}w")] {
                buffer.update(cx, |buffer, cx| {
                    let offset = buffer.snapshot(cx).text().floor_char_boundary(offset);
                    buffer.edit(
                        [(
                            MultiBufferOffset(offset)..MultiBufferOffset(offset),
                            inserted,
                        )],
                        None,
                        cx,
                    )
                });
                cx.run_until_parked();
                assert_matches_whole_line_shaping(&map, None, cx);
            }
        }
    }

    #[test]
    fn test_visible_window_keeps_paintable_glyphs_and_shares_geometry() {
        let cx = &mut shaping_context();
        for (text, shared) in [
            ("x".repeat(2_048), true),
            (
                (b' '..=b'~').map(char::from).collect::<String>().repeat(16),
                true,
            ),
            ("AV".repeat(1_024), true),
            ("漢".repeat(1_024), true),
            ("e\u{301}".repeat(512), false),
            ("a\u{93f}".repeat(512), false),
            ("fi".repeat(1_024), false),
        ] {
            let row = unmeasured_long_row(&text, cx);
            let expected = row.widths.wrapper.layout_line(
                &text,
                &[FontRun {
                    len: text.len(),
                    font_id: row.widths.base_font,
                }],
            );
            let expected_glyphs = positioned_glyphs(&expected, 0, 0.);
            let geometry = row.geometry();
            let piece = &geometry.pieces[0];
            if shared {
                assert!(Arc::ptr_eq(
                    &geometry.atoms.columns,
                    &piece.index.partition_columns
                ));
                assert!(Arc::ptr_eq(
                    &geometry.atoms.positions,
                    &piece.index.positions
                ));
                assert!(piece.index.partition_glyphs.is_none());
            } else {
                assert!(!Arc::ptr_eq(
                    &geometry.atoms.positions,
                    &piece.index.positions
                ));
            }
            for alignment in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
                for scroll in [
                    0.,
                    8.,
                    f64::from(expected.width) / 2.,
                    f64::from(expected.width) - 1.,
                ] {
                    let viewport = 47.5;
                    let free = viewport - f64::from(expected.width);
                    let left = scroll
                        - match alignment {
                            TextAlign::Left => 0.,
                            TextAlign::Center => free / 2.,
                            TextAlign::Right => free,
                        };
                    let visible = |glyph: &(FontId, GlyphId, usize, f64, Pixels)| {
                        let width = f64::from(
                            row.widths
                                .text_system
                                .bounding_box(glyph.0, expected.font_size)
                                .size
                                .width,
                        );
                        glyph.3 < left + viewport && glyph.3 + width > left
                    };
                    let window = row.visible_window(scroll, alignment, viewport);
                    let actual = window
                        .segments()
                        .iter()
                        .flat_map(|columns| {
                            row.window_layout(columns.clone(), left)
                                .into_iter()
                                .flat_map(|piece| {
                                    positioned_glyphs(
                                        &piece.layout,
                                        piece.columns.start as usize,
                                        piece.x,
                                    )
                                })
                        })
                        .filter(&visible)
                        .collect::<Vec<_>>();
                    let expected = expected_glyphs
                        .iter()
                        .filter(|glyph| visible(glyph))
                        .cloned()
                        .collect::<Vec<_>>();
                    if text.is_ascii()
                        && text.starts_with('x')
                        && alignment == TextAlign::Left
                        && scroll == 8.
                    {
                        assert_eq!(expected.first().map(|glyph| glyph.2), Some(0));
                    }
                    assert_eq!(actual, expected, "{alignment:?}, scroll {scroll}");
                }
            }
        }
    }

    #[test]
    fn test_atom_cursor_preserves_reordered_indices_without_rewinding() {
        for columns in [&[][..], &[2][..], &[0, 4, 8, 12][..], &[2, 4, 8, 12][..]] {
            let mut cursor = AtomCursor::new(columns);
            let mut furthest = 0;
            for column in [1, 0, 9, 4, 12, 8, 12, 0, u32::MAX as usize, 0, 12] {
                let expected = columns.iter().rposition(|start| *start as usize <= column);
                assert_eq!(cursor.at(column), expected, "{columns:?}, {column}");
                furthest = furthest.max(expected.map_or(0, |index| index + 1));
                assert_eq!(cursor.end, furthest);
            }
            assert_eq!(cursor.end, columns.len());
        }
    }

    #[test]
    fn test_retained_slice_preserves_precise_positions_and_glyph_dependencies() {
        for origin in [16_777_216., 1_099_511_627_776.] {
            let positions = [
                0.125,
                origin + 0.125,
                origin + 0.5,
                origin - 0.125,
                origin + 0.875,
                origin + 1.375,
            ];
            let width = origin + 1.625;

            let piece_for_indices = |text: &str, indices: [usize; 6]| {
                let fonts = [(FontId(0), 3), (FontId(1), text.len() - 3)]
                    .map(|(font_id, len)| FontRun { font_id, len });
                retained_piece(
                    text,
                    width,
                    &fonts,
                    indices
                        .into_iter()
                        .zip(positions)
                        .enumerate()
                        .map(|(id, (index, x))| {
                            (FontId(id / 3), GlyphId(id as u32), index, x, px(id as f32))
                        }),
                )
            };
            let mut piece = piece_for_indices("éabçdefg", [0, 4, 2, 4, 6, 8]);
            piece.columns = 4_096..4_106;
            piece.x = 123.5;
            let index = piece.index.as_ref();
            assert_eq!(
                index
                    .partition_columns
                    .iter()
                    .map(|column| *column as usize)
                    .chain([index.len])
                    .collect::<Vec<_>>(),
                [0, 2, 6, 8, 10]
            );
            assert_eq!(
                atom_offsets_x(&piece.layout, &[0, 2, 6, 7, 8, 9], index),
                [
                    0.125,
                    origin - 0.125,
                    origin + 0.875,
                    origin + 1.125,
                    origin + 1.375,
                    origin + 1.5,
                ]
            );

            let sliced = piece
                .window(&(4_099..4_101), 123.5 + origin + 0.5)
                .expect("visible dependency");
            assert_eq!(sliced.columns, 4_098..4_102);
            assert_eq!(sliced.text.as_str(), "abç");
            assert_eq!(sliced.x, 123.5 + origin + 0.5);
            assert_eq!(sliced.layout.width, px(0.375));
            assert_eq!(sliced.layout.len, 4);
            assert!(sliced.is_shaped_with(
                [(FontId(0), 1), (FontId(1), 3)].map(|(font_id, len)| FontRun { font_id, len })
            ));
            assert_eq!(
                positioned_glyphs(&sliced.layout, 0, 0.),
                [
                    (FontId(0), GlyphId(1), 2, -0.375, px(1.)),
                    (FontId(0), GlyphId(2), 0, 0., px(2.)),
                    (FontId(1), GlyphId(3), 2, -0.625, px(3.)),
                ]
            );
            let end = piece.window(&(4_104..4_105), 0.).expect("last glyph");
            assert_eq!(
                (end.x, end.layout.width),
                (123.5 + origin + 1.375, px(0.25))
            );
            assert_eq!(end.columns, 4_104..4_106);
            assert!(piece.window(&(4_106..4_107), 0.).is_none());
            let rebased = piece
                .window(&piece.columns, 123.5 + origin + 0.5)
                .expect("rebased run");
            let nested = rebased.window(&(4_104..4_105), 0.).expect("retained slice");
            assert_eq!(nested.columns, end.columns);
            assert_eq!(nested.x, end.x);
            assert_eq!(
                positioned_glyphs(&nested.layout, 0, 0.),
                positioned_glyphs(&end.layout, 0, 0.)
            );
            assert_eq!(nested.layout.width, end.layout.width);
            let text = "éabçdefg".repeat(16);
            let piece = piece_for_indices(&text, [0; 6]);
            let sliced = piece.window(&(3..5), 0.).expect("indivisible dependency");
            assert_eq!(sliced.columns, 0..text.len() as u32);
            assert!(Arc::ptr_eq(&piece.layout, &sliced.layout));
            let rebased = piece
                .window(&(3..5), origin + 0.125)
                .expect("visible glyphs");
            assert_eq!(rebased.columns, 0..text.len() as u32);
            assert_eq!(rebased.text.as_str().as_ptr(), piece.text.as_str().as_ptr());
            assert_eq!(rebased.x, origin + 0.125);
            assert_eq!(rebased.layout.width, px(1.5));
            assert_eq!(
                positioned_glyphs(&rebased.layout, 0, 0.)
                    .into_iter()
                    .skip(1)
                    .map(|glyph| (glyph.2, glyph.3))
                    .collect::<Vec<_>>(),
                [(0, 0.), (0, 0.375), (0, -0.25), (0, 0.75), (0, 1.25)]
            );
            let end = piece
                .window(&(3..5), width + 10.)
                .expect("offscreen dependency");
            assert_eq!((end.x, end.layout.width), (width, Pixels::ZERO));
        }
    }

    #[test]
    fn test_long_rows_keep_same_font_run_boundaries() {
        let padding = "x".repeat(10_000);
        let text = format!("{padding}fi{padding}");
        let split = padding.len() + 1;
        for highlight in [
            HighlightStyle::color(gpui::red()),
            HighlightStyle {
                underline: Some(UnderlineStyle {
                    thickness: px(1.),
                    color: None,
                    wavy: false,
                }),
                ..HighlightStyle::default()
            },
            HighlightStyle {
                strikethrough: Some(StrikethroughStyle {
                    thickness: px(1.),
                    color: None,
                }),
                ..HighlightStyle::default()
            },
        ] {
            let cx = &mut shaping_context();
            let (buffer, map) = build_display_map(&text, 4, cx);
            set_test_shaping_style(&map, cx);
            let snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
            map.update(cx, |map, cx| {
                map.highlight_text(
                    HighlightKey::Editor,
                    vec![
                        snapshot.anchor_after(MultiBufferOffset(0))
                            ..snapshot.anchor_before(MultiBufferOffset(split)),
                    ],
                    highlight,
                    false,
                    cx,
                );
            });
            assert_matches_whole_line_shaping(
                &map,
                Some(&[0..split, split..text.len()].map(|columns| FontRun {
                    len: columns.len(),
                    font_id: ShapingTextSystem::REGULAR,
                })),
                cx,
            );
        }
    }

    #[test]
    fn test_font_run_cursor_preserves_boundaries_across_skipped_spans() {
        let runs = [FontId(0), FontId(0), FontId(1)].map(|font_id| FontRun { len: 4, font_id });
        let mut fonts = FontRunsCursor::new(&runs);
        for (columns, expected) in [
            (0..0, Vec::new()),
            (0..3, vec![(FontId(0), 3)]),
            (3..5, vec![(FontId(0), 1), (FontId(0), 1)]),
            (9..12, vec![(FontId(1), 3)]),
            (12..12, Vec::new()),
        ] {
            assert_eq!(
                fonts
                    .slice(columns)
                    .into_iter()
                    .map(|run| (run.font_id, run.len))
                    .collect::<Vec<_>>(),
                expected,
            );
        }
    }

    #[test]
    fn test_chunk_seams_preserve_contextual_substitutions_beyond_lookahead() {
        for (seam, context) in [(64, 9), (8_192, 65)] {
            let cx = &mut TestAppContext::build_with_text_system(
                gpui::TestDispatcher::new(0),
                None,
                Arc::new(IgnoredMarksTextSystem(ShapingTextSystem(
                    gpui::NoopTextSystem,
                ))),
            );
            let text = format!(
                "{}Q{}Z{}",
                "x".repeat(seam - 1),
                "x".repeat(context),
                "x".repeat(MAX_LINE_LEN)
            );
            let long_row = unmeasured_long_row(&text, cx);
            let wrapper = &long_row.widths.wrapper;
            let advance = f64::from(FONT_SIZE * ShapingTextSystem::ADVANCE);
            let expected_width = advance * (text.len() + 1) as f64;
            let expected = wrapper.layout_line(
                &text,
                &[FontRun {
                    len: text.len(),
                    font_id: wrapper.font_id(),
                }],
            );
            assert_eq!(f64::from(expected.width), expected_width);
            assert_eq!(
                positioned_glyphs(&expected, 0, 0.),
                (0..text.len())
                    .map(|column| (
                        wrapper.font_id(),
                        GlyphId(if column == seam - 1 { 9 } else { 1 }),
                        column,
                        advance * (column + usize::from(column >= seam)) as f64,
                        Pixels::ZERO,
                    ))
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                (long_row.width(), long_row.x_for_column(seam as u32)),
                (expected_width, advance * (seam + 1) as f64),
            );
            assert_window_matches_layout(&long_row, 0..long_row.len(), 0., &expected);
            let window =
                long_row.chunk_aligned(&RowWindow::from_iter([seam as u32 - 1..seam as u32 + 2]));
            for columns in window.segments() {
                assert_window_matches_layout(
                    &long_row,
                    columns.clone(),
                    advance * seam as f64,
                    &expected,
                );
            }
        }
    }

    #[test]
    fn test_ignored_marks_keep_complete_run_shaping() {
        let cx = &mut TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(IgnoredMarksTextSystem(ShapingTextSystem(
                gpui::NoopTextSystem,
            ))),
        );
        for marks in [16, 512, 16_385] {
            let text = format!(
                "{}f{}i{}",
                "x".repeat(63),
                "\u{34f}".repeat(marks),
                "x".repeat(1_024)
            );
            let row = unmeasured_long_row(&text, cx);
            let wrapper = &row.widths.wrapper;
            let expected = wrapper.layout_line(
                &text,
                &[FontRun {
                    len: text.len(),
                    font_id: wrapper.font_id(),
                }],
            );
            assert_eq!(
                expected.width,
                FONT_SIZE * ShapingTextSystem::ADVANCE * (63. + 1.5 + 1_024.)
            );
            assert_eq!(
                expected
                    .runs
                    .iter()
                    .flat_map(|run| &run.glyphs)
                    .filter(|glyph| glyph.index == 63)
                    .map(|glyph| glyph.id)
                    .collect::<Vec<_>>(),
                [GlyphId(2)],
            );
            assert_eq!(row.width(), f64::from(expected.width));
            assert_window_matches_layout(&row, 0..row.len(), 0., &expected);
        }
        let text = format!("\u{34f}{}", "fi".repeat(1_024));
        let row = unmeasured_long_row(&text, cx);
        assert_long_row_matches_whole_line_shaping(&row, &text, None, cx);
        assert_eq!(
            row.window_layout(0..row.len(), 0.)
                .iter()
                .map(|piece| (piece.columns.clone(), piece.text.as_str()))
                .collect::<Vec<_>>(),
            [(0..row.len(), text.as_str())],
        );
        let row = unmeasured_long_row(&"\u{34f}".repeat(2_048), cx);
        assert_eq!(row.width(), 0.);
        assert_eq!(row.x_for_column(row.len() / 2), 0.);
        assert_eq!(row.column_at_x(0.), 0);
        assert_eq!(row.column_at_x(1.), row.len());
        let window = row.visible_window(0., TextAlign::Left, 100.);
        assert_eq!(window.segments(), &[0..row.len()]);
        let layout = row.window_layout(0..row.len(), 0.);
        assert_eq!(
            layout
                .iter()
                .flat_map(|piece| piece.layout.runs.iter())
                .count(),
            0
        );
    }

    #[test]
    fn test_hit_testing_before_first_glyph() {
        let cx = &mut shaping_context();
        let mut row = unmeasured_long_row(&"x".repeat(2 * MAX_LINE_LEN), cx);
        row.geometry = Arc::new(RowGeometryState {
            value: OnceLock::from(Arc::new(RowGeometry {
                len: row.len(),
                width: 16.,
                content_end: row.len(),
                atoms: AtomIndex {
                    columns: Arc::from([0, 1]),
                    positions: Arc::from([1., 8.]),
                },
                pieces: Vec::new(),
                paint_overhang: 0.0..0.0,
                font_runs: row.resolved_font_runs(),
                metrics: row.widths.metrics,
            })),
            ..RowGeometryState::default()
        });
        for (x, expected) in [
            (-1., 0),
            (0., 0),
            (0.5, 0),
            (1., 0),
            (8., 1),
            (16., row.len()),
            (17., row.len()),
        ] {
            assert_eq!(row.column_at_x(x), expected);
            assert_eq!(row.closest_column_for_x(x), expected);
        }
        let font_id = row.widths.base_font;
        let mut piece = retained_piece(
            "xx",
            16.,
            &[FontRun { len: 2, font_id }],
            [(0, 1.), (1, 8.)].map(|(index, x)| (font_id, GlyphId(1), index, x, Pixels::ZERO)),
        );
        piece.columns = 1..3;
        piece.x = 4.;
        row.geometry = Arc::new(RowGeometryState {
            value: OnceLock::from(Arc::new(RowGeometry {
                len: row.len(),
                width: 20.,
                content_end: row.len(),
                atoms: AtomIndex {
                    columns: Arc::from([0, 1, 2]),
                    positions: Arc::from([0., 5., 12.]),
                },
                pieces: vec![piece],
                paint_overhang: 0.0..0.0,
                font_runs: row.resolved_font_runs(),
                metrics: row.widths.metrics,
            })),
            ..RowGeometryState::default()
        });
        for x in [4., 4.5, 5.] {
            assert_eq!(row.column_at_x(x), 1);
            assert_eq!(row.closest_column_for_x(x), 1);
        }
    }

    #[test]
    fn test_restyles_keep_whole_line_shaping() {
        for (character, len, bold) in [("w", 20_001, 1..20_001), ("x", 20_000, 10_000..10_100)] {
            let cx = &mut shaping_context();
            cx.background_executor.set_block_on_ticks(0..=0);
            let text = character.repeat(len);
            let (buffer, map) = build_display_map(&text, 4, cx);
            set_test_shaping_style(&map, cx);
            let old = test_long_row(&map, cx);
            let old_width = old.width();
            let runs = [
                (0..bold.start, ShapingTextSystem::REGULAR),
                (bold.clone(), ShapingTextSystem::BOLD),
                (bold.end..len as u32, ShapingTextSystem::REGULAR),
            ]
            .map(|(columns, font_id)| FontRun {
                len: columns.len(),
                font_id,
            });
            let snapshot = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx));
            map.update(cx, |map, cx| {
                map.highlight_text(
                    HighlightKey::Editor,
                    vec![
                        snapshot.anchor_after(MultiBufferOffset(bold.start as usize))
                            ..snapshot.anchor_before(MultiBufferOffset(bold.end as usize)),
                    ],
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        ..HighlightStyle::default()
                    },
                    false,
                    cx,
                );
            });
            assert_matches_whole_line_shaping(&map, Some(&runs), cx);
            cx.run_until_parked();
            assert_matches_whole_line_shaping(&map, Some(&runs), cx);
            assert_eq!(old.width(), old_width);
        }
    }

    #[test]
    fn test_contextual_edits_invalidate_complete_runs() {
        let cx = &mut TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(IgnoredMarksTextSystem(ShapingTextSystem(
                gpui::NoopTextSystem,
            ))),
        );
        cx.background_executor.set_block_on_ticks(0..=0);
        let text = format!(
            "{}Q{}Y{}",
            "x".repeat(8_191),
            "x".repeat(65),
            "x".repeat(1_024)
        );
        let (buffer, map) = build_display_map(&text, 4, cx);
        let original = test_long_row(&map, cx);
        let advance = f64::from(FONT_SIZE * ShapingTextSystem::ADVANCE);
        assert_eq!(original.width(), text.len() as f64 * advance);
        for (suffix, extra) in [("Z", 1), ("Y", 0)] {
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(MultiBufferOffset(8_257)..MultiBufferOffset(8_258), suffix)],
                    None,
                    cx,
                )
            });
            let row = test_long_row(&map, cx);
            assert_eq!(row.width(), (text.len() + extra) as f64 * advance);
            assert_eq!(row.x_for_column(8_192), (8_192 + extra) as f64 * advance);
            assert_matches_whole_line_shaping(&map, None, cx);
            assert_eq!(original.width(), text.len() as f64 * advance);
        }
    }

    #[test]
    fn test_long_row_shaping_source_updates_offscreen() {
        for syntax_source in [true, false] {
            let mut cx = shaping_context();
            crate::editor_tests::init_test(&mut cx, |_| {});
            let text = format!("\"{}\"", "i".repeat(20_000));
            let language = languages::rust_lang();
            let buffer = cx.new(|cx| {
                Buffer::local(format!("{text}{}{text}", "\n".repeat(100)), cx)
                    .with_language(language.clone(), cx)
            });
            let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
            let cx = cx.add_empty_window();
            let editor = cx.update(|window, cx| {
                cx.new(|cx| Editor::new(EditorMode::full(), buffer.clone(), None, window, cx))
            });
            let map = editor.read_with(cx, |editor, _| editor.display_map.clone());
            let mut style = editor.update(cx, |editor, cx| editor.style(cx).clone());
            style.text.font_size = gpui::AbsoluteLength::from(px(16.));
            style.text.font_weight = FontWeight::NORMAL;
            editor.update(cx, |editor, cx| {
                editor.set_soft_wrap_mode(language::language_settings::SoftWrap::None, cx)
            });
            cx.run_until_parked();
            cx.background_executor.set_block_on_ticks(0..=0);
            for weight in [None, Some(FontWeight::BOLD), None] {
                let highlight = HighlightStyle {
                    font_weight: weight,
                    color: Some(gpui::blue()),
                    ..HighlightStyle::default()
                };
                style.syntax = Arc::new(theme::SyntaxTheme::new_test_styles([(
                    "string",
                    if syntax_source {
                        highlight
                    } else {
                        HighlightStyle::default()
                    },
                )]));
                language.set_theme(&style.syntax);
                if !syntax_source {
                    editor.update(cx, |editor, cx| {
                        editor.clear_highlights(HighlightKey::Editor, cx);
                        if weight.is_some() {
                            let snapshot = editor.buffer().read(cx).snapshot(cx);
                            editor.highlight_text(
                                HighlightKey::Editor,
                                [0, 100]
                                    .map(|row| {
                                        snapshot.anchor_after(Point::new(row, 0))
                                            ..snapshot
                                                .anchor_before(Point::new(row, text.len() as u32))
                                    })
                                    .to_vec(),
                                highlight,
                                cx,
                            );
                        }
                    });
                }
                cx.draw(point(px(0.), px(0.)), size(px(500.), px(500.)), |_, _| {
                    EditorElement::new(&editor, style.clone())
                });
                let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
                for row in [DisplayRow(0), DisplayRow(100)] {
                    let row = snapshot.long_row(row).expect("long row");
                    assert_eq!(
                        row.width(),
                        text.len() as f64 * (8. + if weight.is_some() { 1. } else { 0. })
                    );
                }
            }
        }
    }

    #[test]
    fn test_long_row_shaping_source_local_updates() {
        let mut cx = shaping_context();
        crate::editor_tests::init_test(&mut cx, |_| {});
        let cx = cx.add_empty_window();
        let editor = cx.update(|window, cx| {
            let buffer = MultiBuffer::build_simple(&"w".repeat(20_000), cx);
            cx.new(|cx| Editor::new(EditorMode::full(), buffer, None, window, cx))
        });
        let map = editor.read_with(cx, |editor, _| editor.display_map.clone());
        let mut style = editor.update(cx, |editor, cx| editor.style(cx).clone());
        style.text.font_size = gpui::AbsoluteLength::from(px(16.));
        style.text.font_weight = FontWeight::NORMAL;
        style.show_underlines = true;
        editor.update(cx, |editor, cx| {
            editor.set_soft_wrap_mode(language::language_settings::SoftWrap::None, cx)
        });
        cx.draw(point(px(0.), px(0.)), size(px(500.), px(500.)), |_, _| {
            EditorElement::new(&editor, style.clone())
        });
        cx.background_executor.set_block_on_ticks(0..=0);
        let initial = map
            .update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("long row");
        assert_eq!(initial.width(), 120_000.);
        for underlined in [true, false] {
            editor.update(cx, |editor, cx| {
                editor.clear_highlights(HighlightKey::Editor, cx);
                if underlined {
                    let snapshot = editor.buffer().read(cx).snapshot(cx);
                    editor.highlight_text(
                        HighlightKey::Editor,
                        vec![
                            snapshot.anchor_after(MultiBufferOffset(10_001))
                                ..snapshot.anchor_before(MultiBufferOffset(10_002)),
                        ],
                        HighlightStyle {
                            underline: Some(UnderlineStyle {
                                thickness: px(1.),
                                color: None,
                                wavy: false,
                            }),
                            ..HighlightStyle::default()
                        },
                        cx,
                    );
                }
            });
            let row = map
                .update(cx, |map, cx| map.snapshot(cx))
                .long_row(DisplayRow(0))
                .expect("long row");
            assert_eq!(row.width(), if underlined { 120_004. } else { 120_000. });
        }
        let buffer = editor.read_with(cx, |editor, _| editor.buffer().clone());
        for (range, inserted, expected) in [
            (10_000..10_000, "i", vec![120_008.]),
            (10_000..10_000, "\n", vec![60_000., 60_008.]),
            (10_000..10_001, "", vec![120_008.]),
        ] {
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(
                        MultiBufferOffset(range.start)..MultiBufferOffset(range.end),
                        inserted,
                    )],
                    None,
                    cx,
                )
            });
            let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
            for (row, width) in expected.into_iter().enumerate() {
                assert_eq!(
                    snapshot
                        .long_row(DisplayRow(row as u32))
                        .expect("edited row")
                        .width(),
                    width
                );
            }
            assert_eq!(initial.width(), 120_000.);
        }
    }

    struct IgnoredMarksTextSystem(ShapingTextSystem);

    impl PlatformTextSystem for IgnoredMarksTextSystem {
        fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> anyhow::Result<()> {
            self.0.add_fonts(fonts)
        }

        fn all_font_names(&self) -> Vec<String> {
            self.0.all_font_names()
        }

        fn font_id(&self, descriptor: &Font) -> anyhow::Result<FontId> {
            self.0.font_id(descriptor)
        }

        fn font_metrics(&self, font_id: FontId) -> FontMetrics {
            self.0.font_metrics(font_id)
        }

        fn typographic_bounds(
            &self,
            font_id: FontId,
            glyph_id: GlyphId,
        ) -> anyhow::Result<Bounds<f32>> {
            self.0.typographic_bounds(font_id, glyph_id)
        }

        fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> anyhow::Result<Size<f32>> {
            self.0.advance(font_id, glyph_id)
        }

        fn glyph_for_char(&self, font_id: FontId, character: char) -> Option<GlyphId> {
            self.0.glyph_for_char(font_id, character)
        }

        fn glyph_raster_bounds(
            &self,
            params: &RenderGlyphParams,
        ) -> anyhow::Result<Bounds<DevicePixels>> {
            self.0.glyph_raster_bounds(params)
        }

        fn rasterize_glyph(
            &self,
            params: &RenderGlyphParams,
            bounds: Bounds<DevicePixels>,
        ) -> anyhow::Result<(Size<DevicePixels>, Vec<u8>)> {
            self.0.rasterize_glyph(params, bounds)
        }

        fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
            let mut filtered = String::new();
            let mut offsets = Vec::new();
            let mut start = 0;
            let fonts = runs
                .iter()
                .map(|run| {
                    let filtered_start = filtered.len();
                    for (offset, character) in text[start..start + run.len].char_indices() {
                        if character != '\u{34f}' {
                            filtered.push(character);
                            offsets.extend(start + offset..start + offset + character.len_utf8());
                        }
                    }
                    start += run.len;
                    FontRun {
                        len: filtered.len() - filtered_start,
                        font_id: run.font_id,
                    }
                })
                .collect::<Vec<_>>();
            let mut layout = self.0.layout_line(&filtered, font_size, &fonts);
            let patterns = [9, 65].map(|count| format!("Q{}Z", "x".repeat(count)));
            let mut extra = Pixels::ZERO;
            for glyph in layout.runs.iter_mut().flat_map(|run| &mut run.glyphs) {
                glyph.position.x += extra;
                if filtered
                    .get(glyph.index..)
                    .is_some_and(|text| patterns.iter().any(|pattern| text.starts_with(pattern)))
                {
                    glyph.id = GlyphId(9);
                    extra += font_size * ShapingTextSystem::ADVANCE;
                }
                glyph.index = *offsets.get(glyph.index).expect("unfiltered glyph index");
            }
            layout.width += extra;
            layout.len = text.len();
            layout
        }

        fn recommended_rendering_mode(
            &self,
            font_id: FontId,
            font_size: Pixels,
        ) -> TextRenderingMode {
            self.0.recommended_rendering_mode(font_id, font_size)
        }
    }

    fn merge_chunks<'a, T: PartialEq>(
        chunks: impl Iterator<Item = (&'a str, T)>,
    ) -> Vec<(String, T)> {
        let mut merged = Vec::<(String, T)>::new();
        for (text, value) in chunks {
            match merged.last_mut() {
                Some((last_text, last_value)) if *last_value == value => last_text.push_str(text),
                _ => merged.push((text.to_string(), value)),
            }
        }
        merged
    }

    fn test_long_row(map: &Entity<DisplayMap>, cx: &mut TestAppContext) -> LongRow {
        map.update(cx, |map, cx| map.snapshot(cx))
            .long_row(DisplayRow(0))
            .expect("a long row")
    }

    fn set_test_shaping_style(map: &Entity<DisplayMap>, cx: &mut TestAppContext) {
        let font = test_font();
        let style = EditorStyle {
            text: gpui::TextStyle {
                font_family: font.family,
                font_features: font.features,
                font_fallbacks: font.fallbacks,
                font_size: gpui::AbsoluteLength::from(FONT_SIZE),
                font_weight: font.weight,
                font_style: font.style,
                ..gpui::TextStyle::default()
            },
            ..EditorStyle::default()
        };
        map.update(cx, |map, cx| {
            map.set_shaping_style(&style, SyntaxStyling::default(), cx);
        });
    }

    fn shaping_context() -> TestAppContext {
        TestAppContext::build_with_text_system(
            gpui::TestDispatcher::new(0),
            None,
            Arc::new(ShapingTextSystem(gpui::NoopTextSystem)),
        )
    }

    fn assert_matches_whole_line_shaping(
        map: &Entity<DisplayMap>,
        runs: Option<&[FontRun]>,
        cx: &mut TestAppContext,
    ) {
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let long_row = snapshot.long_row(DisplayRow(0)).expect("a long row");
        let text = snapshot.buffer_snapshot().text();
        assert_long_row_matches_whole_line_shaping(&long_row, &text, runs, cx);
    }

    fn assert_long_row_matches_whole_line_shaping(
        long_row: &LongRow,
        text: &str,
        runs: Option<&[FontRun]>,
        cx: &mut TestAppContext,
    ) {
        let wrapper = cx.update(|cx| cx.text_system().line_wrapper(test_font(), FONT_SIZE));
        let base_runs = [FontRun {
            len: text.len(),
            font_id: wrapper.font_id(),
        }];
        let runs = runs.unwrap_or(&base_runs);
        let layout = wrapper.layout_line(text, runs);
        assert_window_matches_layout(long_row, 0..long_row.len(), 0., &layout);
        let columns = text
            .grapheme_indices(true)
            .map(|(start, _)| start)
            .chain([text.len()])
            .collect::<Vec<_>>();
        for (column, expected) in columns
            .iter()
            .copied()
            .zip(cluster_interpolated_x(&layout, &columns))
        {
            let actual = long_row.x_for_column(column as u32);
            assert!(
                (actual - expected).abs() < 0.01,
                "column {column} of {} bytes: {actual} != {expected}",
                text.len()
            );
        }
    }

    fn cluster_interpolated_x(layout: &LineLayout, columns: &[usize]) -> Vec<ScrollPixelOffset> {
        let mut glyph_starts = std::collections::BTreeMap::<usize, f64>::new();
        for glyph in layout.runs.iter().flat_map(|run| &run.glyphs) {
            let x = f64::from(glyph.position.x);
            glyph_starts
                .entry(glyph.index)
                .and_modify(|start| *start = start.min(x))
                .or_insert(x);
        }
        let width = f64::from(layout.width);
        columns
            .iter()
            .enumerate()
            .map(|(index, &column)| {
                if column >= layout.len {
                    return width;
                }
                let next_column = columns.get(index + 1).copied().unwrap_or(layout.len);
                if glyph_starts.range(column..next_column).next().is_some() {
                    return f64::from(layout.x_for_index(column));
                }
                let Some((&cluster_start, &cluster_x)) = glyph_starts.range(..column).next_back()
                else {
                    return glyph_starts.values().next().copied().unwrap_or(width);
                };
                let (cluster_end, cluster_end_x) = glyph_starts
                    .range(column..)
                    .next()
                    .map_or((layout.len, width), |(index, x)| (*index, *x));
                let interior = columns
                    .iter()
                    .filter(|other| cluster_start < **other && **other < cluster_end)
                    .collect::<Vec<_>>();
                let rank = interior
                    .iter()
                    .position(|other| **other == column)
                    .unwrap_or(0)
                    + 1;
                cluster_x + (cluster_end_x - cluster_x) * rank as f64 / (interior.len() + 1) as f64
            })
            .collect()
    }

    fn assert_window_matches_layout(
        long_row: &LongRow,
        columns: Range<u32>,
        origin_x: ScrollPixelOffset,
        expected: &LineLayout,
    ) {
        let window = long_row.window_layout(columns.clone(), origin_x);

        let actual = window
            .iter()
            .flat_map(|piece| {
                assert!(columns.start <= piece.columns.start && piece.columns.end <= columns.end);
                positioned_glyphs(&piece.layout, piece.columns.start as usize, piece.x)
            })
            .collect::<Vec<_>>();
        let expected = positioned_glyphs(expected, 0, 0.)
            .into_iter()
            .filter(|glyph| columns.contains(&(glyph.2 as u32)))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{columns:?}");
    }

    fn retained_piece(
        text: &str,
        width: f64,
        fonts: &[FontRun],
        glyphs: impl IntoIterator<Item = (FontId, GlyphId, usize, f64, Pixels)>,
    ) -> TextPiece {
        let mut runs = Vec::<ShapedRun>::new();
        let mut positions = Vec::new();
        for (font_id, id, index, x, y) in glyphs {
            if runs.last().is_none_or(|run| run.font_id != font_id) {
                runs.push(ShapedRun {
                    font_id,
                    glyphs: Vec::new(),
                });
            }
            let run = runs.last_mut().expect("glyph run");
            run.glyphs.push(ShapedGlyph {
                id,
                position: point(px(x as f32), y),
                index,
                is_emoji: false,
            });
            positions.push(x);
        }
        let layout = Arc::new(LineLayout {
            len: text.len(),
            width: px(width as f32),
            runs,
            ..LineLayout::default()
        });
        TextPiece {
            columns: 0..text.len() as u32,
            x: 0.,
            text: SharedString::new(text),
            index: Arc::new(LayoutIndex::new(&layout, fonts, positions, width)),
            layout,
            runs: SmallVec::from_slice(fonts),
        }
    }

    fn positioned_glyphs(
        layout: &LineLayout,
        column: usize,
        x: ScrollPixelOffset,
    ) -> Vec<(FontId, GlyphId, usize, ScrollPixelOffset, Pixels)> {
        layout
            .runs
            .iter()
            .flat_map(|run| {
                run.glyphs.iter().map(move |glyph| {
                    (
                        run.font_id,
                        glyph.id,
                        column + glyph.index,
                        x + f64::from(glyph.position.x),
                        glyph.position.y,
                    )
                })
            })
            .collect()
    }

    fn unmeasured_long_row(text: &str, cx: &mut TestAppContext) -> LongRow {
        init_settings(4, cx);
        cx.update(|cx| {
            let buffer = MultiBuffer::build_simple(text, cx);
            let (_, inlay_snapshot) = InlayMap::new(buffer.read(cx).snapshot(cx));
            let (_, fold_snapshot) = FoldMap::new(inlay_snapshot);
            let (_, snapshot) = TabMap::new(fold_snapshot, NonZeroU32::new(4).expect("tab size"));
            LongRows::new(
                cx.text_system().line_wrapper(test_font(), FONT_SIZE),
                cx.text_system().clone(),
                snapshot.version,
            )
            .row(&snapshot, 0)
            .expect("a long row")
        })
    }

    fn build_display_map(
        text: &str,
        tab_size: u32,
        cx: &mut TestAppContext,
    ) -> (Entity<MultiBuffer>, Entity<DisplayMap>) {
        init_settings(tab_size, cx);
        let buffer = cx.update(|cx| MultiBuffer::build_simple(text, cx));
        let map = new_display_map(&buffer, cx);
        (buffer, map)
    }

    fn init_settings(tab_size: u32, cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            crate::init(cx);
            theme_settings::init(LoadThemes::JustBase, cx);
        });
        set_tab_size(tab_size, cx);
    }

    fn set_tab_size(tab_size: u32, cx: &mut TestAppContext) {
        cx.update(|cx| {
            cx.update_global::<SettingsStore, _>(|store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings.project.all_languages.defaults.tab_size = NonZeroU32::new(tab_size);
                });
            });
        });
    }

    fn new_display_map(
        buffer: &Entity<MultiBuffer>,
        cx: &mut TestAppContext,
    ) -> Entity<DisplayMap> {
        cx.new(|cx| {
            DisplayMap::new(
                buffer.clone(),
                test_font(),
                FONT_SIZE,
                None,
                1,
                1,
                FoldPlaceholder::test(),
                DiagnosticSeverity::Warning,
                cx,
            )
        })
    }

    fn randomly_mutate(
        buffer: &Entity<MultiBuffer>,
        map: &Entity<DisplayMap>,
        rng: &mut StdRng,
        cx: &mut TestAppContext,
    ) {
        let text = buffer.read_with(cx, |buffer, cx| buffer.snapshot(cx).text());
        let offset = |rng: &mut StdRng, range: RangeInclusive<usize>| {
            let mut offset = rng.random_range(range);
            while !text.is_char_boundary(offset) {
                offset -= 1;
            }
            offset
        };
        let start = offset(rng, 0..=text.len());
        let end = offset(rng, start..=(start + 60).min(text.len()));
        match rng.random_range(0..8) {
            0 if start < end => {
                let creases = vec![Crease::simple(
                    MultiBufferOffset(start)..MultiBufferOffset(end),
                    FoldPlaceholder::test(),
                )];
                map.update(cx, |map, cx| map.fold(creases, cx));
            }
            1 => {
                let position = buffer.read_with(cx, |buffer, cx| {
                    buffer.snapshot(cx).anchor_after(MultiBufferOffset(start))
                });
                let inlay = Inlay::mock_hint(
                    rng.random_range(0..usize::MAX),
                    position,
                    random_text(rng, 1..200).as_str(),
                );
                map.update(cx, |map, cx| map.splice_inlays(&[], vec![inlay], cx));
            }
            _ => {
                let mut edits = Vec::new();
                let mut edit_start = start;
                for _ in 0..rng.random_range(1..=3) {
                    if edit_start >= text.len() {
                        break;
                    }
                    let edit_end = offset(rng, edit_start..=(edit_start + 40).min(text.len()));
                    let inserted = if rng.random_ratio(1, 6) {
                        "\n".to_string()
                    } else {
                        random_text(rng, 0..200)
                    };
                    edits.push((
                        MultiBufferOffset(edit_start)..MultiBufferOffset(edit_end),
                        inserted,
                    ));
                    edit_start = offset(rng, edit_end..=(edit_end + 80).min(text.len()));
                }
                buffer.update(cx, |buffer, cx| buffer.edit(edits, None, cx));
            }
        }
    }

    fn random_text(rng: &mut StdRng, chars: Range<usize>) -> String {
        const ALPHABET: &[char] = &['a', 'W', ' ', '\t', 'é', '漢', '😀', '\u{0}', '\u{7f}'];
        let len = rng.random_range(chars);
        (0..len)
            .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())])
            .collect()
    }

    fn assert_round_trips(snapshot: &DisplaySnapshot) {
        for row in 0..=snapshot.max_point().row().0 {
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                continue;
            };
            let line = snapshot.line(DisplayRow(row));
            assert_eq!(long_row.len() as usize, line.len());
            let mut previous_x = -1.;
            for (column, _) in line.char_indices().step_by(7) {
                let x = long_row.x_for_column(column as u32);
                assert!(
                    x > previous_x,
                    "row {row} column {column}: {x} <= {previous_x}"
                );
                assert_eq!(long_row.closest_column_for_x(x) as usize, column);
                previous_x = x;
            }
        }
    }

    fn assert_cluster_atoms(snapshot: &DisplaySnapshot) {
        for row in 0..=snapshot.max_point().row().0 {
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                continue;
            };

            let line = snapshot.line(DisplayRow(row));
            let expected_atoms = line
                .grapheme_indices(true)
                .map(|(start, grapheme)| start..start + grapheme.len())
                .collect::<Vec<_>>();
            let atom_starts = long_row
                .geometry()
                .atoms
                .columns
                .iter()
                .map(|start| *start as usize)
                .collect::<Vec<_>>();
            assert_eq!(
                atom_starts,
                expected_atoms
                    .iter()
                    .map(|atom| atom.start)
                    .collect::<Vec<_>>(),
                "row {row}"
            );
            for atom in expected_atoms {
                let x = long_row.x_for_column(atom.start as u32);
                let inner =
                    atom.start + line[atom.clone()].chars().next().map_or(0, char::len_utf8);
                if inner < atom.end {
                    assert_eq!(
                        long_row.x_for_column(inner as u32),
                        long_row.x_for_column(atom.end as u32),
                        "row {row}"
                    );
                }
                assert_eq!(
                    long_row.closest_column_for_x(x) as usize,
                    atom.start,
                    "row {row}"
                );
            }
        }
    }

    fn assert_matches_char_widths(snapshot: &DisplaySnapshot, wrapper: &mut LineWrapper) {
        for row in 0..=snapshot.max_point().row().0 {
            let line = snapshot.line(DisplayRow(row));
            let Some(long_row) = snapshot.long_row(DisplayRow(row)) else {
                assert!(
                    line.len() <= MAX_LINE_LEN,
                    "row {row} has no long row geometry"
                );
                continue;
            };
            let standalone_invisibles = line
                .grapheme_indices(true)
                .filter(|(_, grapheme)| single_invisible(grapheme).is_some())
                .map(|(start, grapheme)| start..start + grapheme.len())
                .collect::<Vec<_>>();
            let mut expected_x = 0.;
            for (index, (column, character)) in line.char_indices().enumerate() {
                if index % 5 == 0 {
                    let x = long_row.x_for_column(column as u32);
                    assert!(
                        (x - expected_x).abs() < 1e-2,
                        "row {row} column {column}: {x} != {expected_x}"
                    );
                }
                let measured = if standalone_invisibles
                    .iter()
                    .any(|range| range.start == column)
                {
                    replacement(character).unwrap_or(character)
                } else {
                    character
                };
                expected_x += f64::from(wrapper.width_for_char(measured));
            }
            assert!((long_row.width() - expected_x).abs() < 1e-2);
        }
    }
}
