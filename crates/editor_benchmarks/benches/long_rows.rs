use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use editor::{
    Editor, EditorMode, MultiBuffer, SelectionEffects,
    display_map::{DisplayPoint, DisplayRow, DisplaySnapshot},
};
use gpui::{
    AnyWindowHandle, AppContext as _, AsyncApp, BenchAppContext, Context, Entity, Focusable as _,
    FontRun, IntoElement, ParentElement as _, Render, Styled as _, Task, Window, div, point,
};
use language::{Buffer, LineEnding, Point, Rope, language_settings::SoftWrap};
use zed_actions::editor::MoveDown;

#[gpui::bench(
    inputs = long_row_characters(),
    group = "Scroll an unwrapped long row",
    input_name = "character",
    sample_size = 10
)]
fn editor_scroll_long_row(character: &char, cx: &mut BenchAppContext) {
    init_context(cx);

    let text = Rope::from(character.to_string().repeat(LONG_ROW_CHARACTERS).as_str());
    let oracle_text = character.to_string().repeat(256);
    let mut session_window = None::<AnyWindowHandle>;
    let mut oracle_reported = false;
    cx.bench_renderer_session(
        LONG_ROW_SESSION_TIMEOUT,
        move |cx| {
            let window =
                *session_window.get_or_insert_with(|| cx.add_empty_window().window_handle());
            let text = text.clone();
            let editor_slot = Rc::new(RefCell::new(None));
            let renders = Rc::new(Cell::new(0));
            let (columns_per_character, character_advance, leading_characters) = cx
                .update_window(window, |_, window, cx| {
                    let editor = cx.new(|cx| unwrapped_editor(text, window, cx));
                    editor_slot.replace(Some(editor.clone()));
                    window.replace_root(cx, |_, _| LongRowRoot {
                        editor: editor_slot.clone(),
                        renders: renders.clone(),
                    });
                    window.focus(&editor.focus_handle(cx), cx);
                    editor.update(cx, |editor, cx| {
                        let style = editor.style(cx);
                        let font_size = style.text.font_size.to_pixels(window.rem_size());
                        let wrapper = cx.text_system().line_wrapper(style.text.font(), font_size);
                        let (layout, positions, width) = wrapper.layout_line_with_precise_positions(
                            &oracle_text,
                            &[FontRun {
                                len: oracle_text.len(),
                                font_id: wrapper.font_id(),
                            }],
                        );
                        let advance = width / 256.;
                        let em_width = f64::from(
                            window.text_system().em_layout_width(wrapper.font_id(), font_size),
                        );
                        let columns_per_character = advance / em_width;
                        assert!(columns_per_character.is_finite() && columns_per_character > 0.);
                        let paint_width = layout
                            .runs
                            .iter()
                            .flat_map(|run| {
                                let width = f64::from(
                                    cx.text_system().bounding_box(run.font_id, font_size).size.width,
                                );
                                run.glyphs.iter().map(move |glyph| (glyph.index, width))
                            })
                            .zip(positions)
                            .map(|((index, width), x)| {
                                x + width - advance * (index / character.len_utf8()) as f64
                            })
                            .fold(0., f64::max);
                        let leading_characters = (paint_width / advance).ceil() + 1.;
                        if !oracle_reported {
                            eprintln!(
                                "native scroll oracle {character:?}: advance={advance}px, \
                                 paint_width={paint_width}px, maximum_leading_characters={leading_characters}"
                            );
                            oracle_reported = true;
                        }
                        (columns_per_character, advance, leading_characters)
                    })
                })
                .expect("the benchmark window must remain open");
            let stopped = Arc::new(AtomicBool::new(false));
            let session = LongRowSession {
                editor: editor_slot,
                renders,
                stopped: stopped.clone(),
                bytes_per_character: character.len_utf8() as u32,
                columns_per_character,
                character_advance,
                leading_characters,
                scroll_x: LONG_ROW_SCROLL_START,
                requested_scroll: None,
                visible_start_character: 0,
            };
            (session, window, stopped)
        },
        |session, turn, window, cx| {
            let Some(editor) = session.editor.borrow().clone() else {
                return;
            };
            editor.update(cx, |editor, cx| {
                let display_snapshot = editor.display_snapshot(cx);
                assert_eq!(
                    display_snapshot.max_point(),
                    DisplayPoint::new(
                        DisplayRow(0),
                        LONG_ROW_CHARACTERS as u32 * session.bytes_per_character,
                    ),
                    "the complete line must remain unwrapped at turn {turn}"
                );
                if turn > 0 {
                    let requested_scroll = session
                        .requested_scroll
                        .take()
                        .expect("each scroll turn must follow a scroll request");
                    session.validate_scroll(
                        requested_scroll, turn, editor, &display_snapshot, window, cx,
                    );
                }
                if turn == LONG_ROW_SCROLL_TURNS {
                    session.stopped.store(true, Ordering::Release);
                    return;
                }
                session.scroll_x += LONG_ROW_SCROLL_STEP;
                let position = point(session.scroll_x, 0.);
                editor.set_scroll_position(position, window, cx);
                assert_eq!(
                    editor.scroll_position(cx),
                    position,
                    "scroll turn {turn} did not accept the scroll request"
                );
                session.requested_scroll = Some(RequestedScroll {
                    expected_start_character: position.x / session.columns_per_character,
                    renders: session.renders.get(),
                });
            });
        },
    );
}

#[gpui::bench(
    inputs = long_row_byte_counts(),
    group = "Cold open, edit, move and type in unwrapped rows",
    input_name = "bytes_per_row",
    sample_size = 10
)]
fn editor_cold_long_rows(bytes_per_row: &usize, cx: &mut BenchAppContext) {
    init_context(cx);
    let row = "x".repeat(*bytes_per_row);
    let text = Rope::from(format!("{row}\n{row}").as_str());
    let expected_text = Arc::<str>::from(format!("{row}x\n{}!x!x", "x".repeat(bytes_per_row - 2)));
    drop(row);
    let row_length = *bytes_per_row as u32;
    let mut session_window = None::<AnyWindowHandle>;
    cx.bench_renderer_session(
        Duration::from_secs(240),
        |cx| {
            let window =
                *session_window.get_or_insert_with(|| cx.add_empty_window().window_handle());
            let editor = Rc::new(RefCell::new(None));
            let renders = Rc::new(Cell::new(0));
            cx.update_window(window, |_, window, cx| {
                window.replace_root(cx, |_, _| LongRowRoot {
                    editor: editor.clone(),
                    renders: renders.clone(),
                });
            })
            .expect("the benchmark window must remain open");
            let stopped = Arc::new(AtomicBool::new(false));
            let session = ColdLongRowSession {
                editor,
                renders,
                ready: Rc::new(Cell::new(true)),
                stopped: stopped.clone(),
                task: None,
                app: cx.update(|cx| cx.to_async()),
                expected_text: expected_text.clone(),
                row_length,
            };
            (session, window, stopped)
        },
        |session, _, window, cx| {
            if !session.ready.replace(false) {
                return;
            }
            let editor = session.editor.borrow().clone();
            if editor.is_some() && session.renders.get() == 0 {
                session.ready.set(true);
                return;
            }
            if editor.is_none() {
                session.renders.set(0);
            }
            let editor_slot = session.editor.clone();
            let renders = session.renders.clone();
            let ready = session.ready.clone();
            let stopped = session.stopped.clone();
            let text = text.clone();
            let window = window.window_handle();
            session.task = Some(cx.spawn(async move |cx| {
                cx.update_window(window, |_, window, cx| {
                    if let Some(editor) = editor {
                        renders.set(0);
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                let end = Point::new(0, row_length);
                                buffer.edit([(end..end, "x")], None, cx);
                            });
                            editor.move_down(&MoveDown, window, cx);
                            let snapshot = editor.display_snapshot(cx);
                            assert_eq!(
                                editor
                                    .selections
                                    .all::<Point>(&snapshot)
                                    .into_iter()
                                    .map(|selection| selection.range())
                                    .collect::<Vec<_>>(),
                                [row_length - 2, row_length - 1].map(|column| {
                                    let point = Point::new(1, column);
                                    point..point
                                })
                            );
                            editor.handle_input("!", window, cx);
                        });
                        stopped.store(true, Ordering::Release);
                    } else {
                        let editor = cx.new(|cx| unwrapped_editor(text, window, cx));
                        editor.update(cx, |editor, cx| {
                            editor.change_selections(
                                SelectionEffects::no_scroll(),
                                window,
                                cx,
                                |selections| {
                                    selections.select_ranges([row_length - 2, row_length - 1].map(
                                        |column| {
                                            let point = Point::new(0, column);
                                            point..point
                                        },
                                    ));
                                },
                            );
                        });
                        window.focus(&editor.focus_handle(cx), cx);
                        editor_slot.replace(Some(editor));
                    }
                    window.refresh();
                })
                .expect("the benchmark window must remain open");
                ready.set(true);
            }));
        },
    );
}

struct ColdLongRowSession {
    editor: Rc<RefCell<Option<Entity<Editor>>>>,
    renders: Rc<Cell<u64>>,
    ready: Rc<Cell<bool>>,
    stopped: Arc<AtomicBool>,
    task: Option<Task<()>>,
    app: AsyncApp,
    expected_text: Arc<str>,
    row_length: u32,
}

impl Drop for ColdLongRowSession {
    fn drop(&mut self) {
        let editor = self.editor.borrow_mut().take();
        if std::thread::panicking() {
            return;
        }
        assert!(self.stopped.load(Ordering::Acquire));
        assert!(self.renders.get() > 0, "typing must render");
        self.app.update(|cx| {
            editor
                .expect("the cold-open workload must create an editor")
                .update(cx, |editor, cx| {
                    let snapshot = editor.display_snapshot(cx);
                    assert_eq!(
                        snapshot.max_point(),
                        DisplayPoint::new(DisplayRow(1), self.row_length + 2)
                    );
                    assert_eq!(snapshot.line_len(DisplayRow(0)), self.row_length + 1);
                    assert_eq!(editor.selections.count(), 2);
                    assert_eq!(
                        editor
                            .selections
                            .all::<Point>(&snapshot)
                            .into_iter()
                            .map(|selection| selection.range())
                            .collect::<Vec<_>>(),
                        [self.row_length - 1, self.row_length + 1].map(|column| {
                            let point = Point::new(1, column);
                            point..point
                        })
                    );
                    assert_eq!(editor.text(cx).as_str(), self.expected_text.as_ref());
                });
        });
    }
}

struct LongRowSession {
    editor: Rc<RefCell<Option<Entity<Editor>>>>,
    renders: Rc<Cell<u64>>,
    stopped: Arc<AtomicBool>,
    bytes_per_character: u32,
    columns_per_character: f64,
    character_advance: f64,
    leading_characters: f64,
    scroll_x: f64,
    requested_scroll: Option<RequestedScroll>,
    visible_start_character: u32,
}

impl LongRowSession {
    fn validate_scroll(
        &mut self,
        requested_scroll: RequestedScroll,
        turn: u64,
        editor: &Editor,
        display_snapshot: &DisplaySnapshot,
        window: &mut Window,
        cx: &mut Context<Editor>,
    ) {
        assert!(
            self.renders.get() > requested_scroll.renders,
            "no render followed the scroll request preceding turn {turn}"
        );
        let expected = requested_scroll.expected_start_character;
        let pixel_tolerance = 0.5 / f64::from(window.scale_factor());
        let actual_scroll = editor.scroll_position(cx).x / self.columns_per_character;
        assert!(
            (actual_scroll - expected).abs() * self.character_advance <= pixel_tolerance,
            "scroll drift at turn {turn}: {actual_scroll} != {expected} characters"
        );
        let details = editor.text_layout_details(window, cx);
        for character in [expected.ceil() as u32, LONG_ROW_CHARACTERS as u32 - 1] {
            let column = character * self.bytes_per_character;
            let expected_x = f64::from(character) * self.character_advance;
            let actual_x = display_snapshot
                .x_for_display_point(DisplayPoint::new(DisplayRow(0), column), &details);
            assert!(
                (actual_x - expected_x).abs() <= pixel_tolerance,
                "native position drift at character {character}, turn {turn}: \
                 {actual_x}px != {expected_x}px, bound={pixel_tolerance}px"
            );
            for fraction in [-0.25, 0.25] {
                assert_eq!(
                    display_snapshot.display_column_for_x(
                        DisplayRow(0),
                        (f64::from(character) + fraction) * self.character_advance,
                        &details,
                    ),
                    column,
                    "native hit test at character {character} + {fraction}, turn {turn}"
                );
            }
        }
        let visible_ranges = editor.visible_text_ranges(display_snapshot, cx);
        let [visible_range] = visible_ranges.as_slice() else {
            panic!("expected one visible range at turn {turn}, got {visible_ranges:?}");
        };
        assert!(
            visible_range.start.row == 0
                && visible_range.end.row == 0
                && visible_range.start.column < visible_range.end.column
                && visible_range.start.column % self.bytes_per_character == 0
                && visible_range.end.column % self.bytes_per_character == 0,
            "invalid visible range {visible_range:?} at turn {turn}"
        );
        let visible_start_character = visible_range.start.column / self.bytes_per_character;
        assert!(
            visible_start_character > self.visible_start_character
                && f64::from(visible_start_character) >= expected - self.leading_characters
                && f64::from(visible_start_character) <= expected + LONG_ROW_CHARACTER_TOLERANCE,
            "visible range at turn {turn} starts at character {visible_start_character}, \
             expected after character {} and within {}..={} from native font bounds",
            self.visible_start_character,
            expected - self.leading_characters,
            expected + LONG_ROW_CHARACTER_TOLERANCE,
        );
        self.visible_start_character = visible_start_character;
    }
}

struct RequestedScroll {
    expected_start_character: f64,
    renders: u64,
}

impl Drop for LongRowSession {
    fn drop(&mut self) {
        self.editor.borrow_mut().take();
    }
}

struct LongRowRoot {
    editor: Rc<RefCell<Option<Entity<Editor>>>>,
    renders: Rc<Cell<u64>>,
}

impl Render for LongRowRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().size_full().children(self.editor.borrow().clone())
    }
}

const LONG_ROW_CHARACTERS: usize = 20_000_000;
const LONG_ROW_SCROLL_TURNS: u64 = 8;
const LONG_ROW_SCROLL_START: f64 = 1_000_000.;
const LONG_ROW_SCROLL_STEP: f64 = 200.;
const LONG_ROW_CHARACTER_TOLERANCE: f64 = 1.;
const LONG_ROW_SESSION_TIMEOUT: Duration = Duration::from_secs(30);

fn long_row_characters() -> Vec<char> {
    vec!['x', 'é', '漢']
}

fn long_row_byte_counts() -> Vec<usize> {
    vec![20 * 1024, 2 * 1024 * 1024, 20 * 1024 * 1024]
}

fn unwrapped_editor(text: Rope, window: &mut Window, cx: &mut Context<Editor>) -> Editor {
    let buffer = cx.new(|cx| Buffer::local_normalized(text, LineEnding::Unix, cx));
    let buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
    let mut editor = Editor::new(EditorMode::full(), buffer, None, window, cx);
    editor.set_soft_wrap_mode(SoftWrap::None, cx);
    editor
}

fn init_context(cx: &mut BenchAppContext) {
    cx.update(|cx| {
        settings::init(cx);
        assets::Assets.load_test_fonts(cx);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
        editor::init(cx);
    });
}

gpui::bench_group!(benches, editor_scroll_long_row, editor_cold_long_rows);
gpui::bench_main!(benches);
