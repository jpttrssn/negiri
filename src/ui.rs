// SPDX-License-Identifier: GPL-3.0-or-later

//! The widget tree: library page, frame grid, detail view, editing panel, and
//! the help overlay — pure functions of `&AppModel` returning `Element`s. The
//! message flow and state live in `crate::app`; this module only reads.

use std::path::Path;

use crate::app::{
    AppModel, Message, Roll, RollDateField, THUMB_SIZE, TILE_ASPECT, Tile, Thumb,
    detail_zoom_delta, highlight_lift, highlight_power_for_lift, shadow_lift, shadow_power_for_lift,
};
use crate::detail_area::DetailArea;
use crate::error::FrameError;
use crate::exif_writer;
use crate::film::{BaseMode, FILM_CHOICES, FilmPreset};
use crate::fl;
use crate::i18n::fl_dyn;
use crate::library::{
    LibraryCell, format_roll_card_dates, library_cell_index, library_cells,
};
use cosmic::iced::alignment::{Horizontal, Vertical};
use cosmic::iced::widget::{Grid, MouseArea, Stack, grid};
use cosmic::iced::{ContentFit, Length};
use cosmic::prelude::*;
use cosmic::widget::{self, icon};

/// Basic EXIF readout for the frame-info drawer, parsed lazily from the RAW
/// file on demand and cached on its [`Tile`]. Every field is an already
/// formatted display string (`None` = absent in the file).
#[derive(Debug, Clone, Default)]
pub(crate) struct FrameMeta {
    width: Option<String>,
    height: Option<String>,
    make: Option<String>,
    model: Option<String>,
    iso: Option<String>,
    exposure: Option<String>,
    aperture: Option<String>,
    focal: Option<String>,
    lens: Option<String>,
    date: Option<String>,
}

/// Renders the library page: a responsive grid of selectable cells. The Add
/// Roll tile is always the first cell; when no rolls (or no matches) remain, a
/// centered hint overlays the empty space beside the still-present add tile.
pub(crate) fn library_view(app: &AppModel) -> Element<'_, Message> {
    let space_s = cosmic::theme::spacing().space_s;

    let cells = library_cells(
        &app.rolls,
        app.search_filter.as_deref().unwrap_or(""),
        &app.month_aliases,
    );
    let selected_index = library_cell_index(app.library_selection.as_ref(), &cells);
    // The hint overlays only when a search filters every real roll away (the
    // Add Roll tile is hidden while searching, so `cells` can be empty).
    let empty = !cells
        .iter()
        .any(|cell| matches!(cell, LibraryCell::Roll(_)));

    let grid = Grid::with_children(cells.iter().enumerate().map(|(index, cell)| {
        let selected = selected_index == Some(index);
        match cell {
            LibraryCell::AddRoll => add_roll_tile(selected),
            LibraryCell::Roll(roll) => roll_tile(roll, selected),
        }
    }))
    .fluid(THUMB_SIZE)
    .height(grid::Sizing::AspectRatio(TILE_ASPECT))
    .spacing(space_s);

    let body = widget::scrollable(widget::container(grid).width(Length::Fill).padding(space_s))
        // Report the viewport so `Nav` can scroll the highlighted tile into
        // view when it moves beyond the visible area.
        .id(scrollable_id("rolls-grid"))
        .on_scroll(Message::GridViewport)
        .height(Length::Fill);

    // A truly empty library shows no hint (the Add Roll tile alone is
    // self-explanatory), and a grid with matching rolls shows none either —
    // only the search-mismatch case (rolls exist but none match) overlays.
    if app.rolls.is_empty() || !empty {
        return body.into();
    }

    // Nothing matches: keep the hint over the space the grid leaves empty.
    let hint = widget::container(widget::text(fl!("no-rolls-found")))
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Horizontal::Center)
        .align_y(Vertical::Center);

    let mut page = Stack::with_capacity(2);
    page = page.push(body);
    page = page.push(hint);
    page.width(Length::Fill).height(Length::Fill).into()
}

/// The stable widget [`Id`] names the mounted grid's [`Scrollable`], the target
/// for keyboard scroll-into-view effects. Only one page's grid is mounted at a
/// time, so the ids never collide.
pub(crate) fn scrollable_id(name: &'static str) -> cosmic::iced::widget::Id {
    cosmic::iced::widget::Id::new(name)
}

/// The stable widget [`Id`] of the header's search input, so activating search
/// can focus it. It is only mounted while search is active.
pub(crate) fn search_input_id() -> cosmic::iced::widget::Id {
    cosmic::iced::widget::Id::new("search-input")
}

/// Renders an open roll's frame grid, with the detail view overlaid on an
/// opaque surface when a frame is selected.
///
/// The grid stays mounted (scroll position persists) under the detail surface
/// that captures input, so the detail view cannot leak wheel/clicks to it.
pub(crate) fn frames_view(app: &AppModel) -> Element<'_, Message> {
    let space_s = cosmic::theme::spacing().space_s;

    let tiles: Element<'_, Message> = if app.tiles.is_empty() {
        widget::container(widget::text(fl!("no-files")))
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into()
    } else {
        let grid = Grid::with_children(app.tiles.iter().map(|tile| {
            tile_view(
                tile,
                app.selected_frames.contains(&tile.name),
                is_calibration_frame(
                    app.roll.base_mode(),
                    app.roll.calibration_frame(),
                    &tile.name,
                ),
            )
        }))
        .fluid(THUMB_SIZE)
        .height(grid::Sizing::AspectRatio(TILE_ASPECT))
        .spacing(space_s);

        widget::scrollable(widget::container(grid).width(Length::Fill).padding(space_s))
            .id(scrollable_id("frames-grid"))
            .on_scroll(Message::GridViewport)
            .height(Length::Fill)
            .into()
    };

    let mut page = Stack::with_capacity(1);
    page = page.push(tiles);

    if let Some(detail) = detail_view(app) {
        page = page.push(
            widget::container(
                MouseArea::new(detail)
                    .on_press(Message::Ignore)
                    .on_double_press(Message::Ignore)
                    .on_double_click(Message::Ignore)
                    .on_release(Message::Ignore)
                    .on_right_press(Message::Ignore)
                    .on_right_release(Message::Ignore)
                    .on_middle_press(Message::Ignore)
                    .on_middle_release(Message::Ignore)
                    .on_scroll(|_delta| Message::Ignore),
            )
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|theme| cosmic::iced::widget::container::Style {
                // The shader renders the letterbox/pillarbox bars transparent,
                // so this container color shows through around the image. In
                // crop mode the surround goes white — the working backdrop for
                // judging the crop — and the dim overlay marks the excluded
                // area.
                background: Some(cosmic::iced::Background::Color(if app.crop_mode {
                    cosmic::iced::Color::WHITE
                } else {
                    theme.cosmic().background(false).base.into()
                })),
                ..Default::default()
            }),
        );
    }

    page.width(Length::Fill).height(Length::Fill).into()
}

/// Renders the editing panel for the context drawer.
///
/// Shown whenever a selection is active; the body is empty while the GPU
/// shader is still loading, since `detail_view` already shows the cached
/// thumbnail during that brief decode gap. The drawer pane supplies the
/// width and padding, so the panel fills the available space.
pub(crate) fn editing_panel(app: &AppModel) -> Element<'_, Message> {
    let space_s = cosmic::theme::spacing().space_s;

    if app.detail_shader.is_none() {
        return widget::column::with_capacity(1)
            .push("Loading...")
            .spacing(space_s)
            .width(Length::Fill)
            .into();
    }

    let label = widget::text(fl!("exposure-label"));
    // The `0.05 EV` step matches `edit_manifest::EV_TICK`, the keyboard nudge,
    // and the fixed-point storage grid.
    let slider = widget::slider(-3.0..=4.0, app.exposure_ev, Message::ExposureChanged)
        .step(0.05_f32)
        // A finished drag is an edit flush point.
        .on_release(Message::EditSave);

    // Tone-editing controls: three pivoted powers re-shape the GPU texture
    // via a uniform-only remap — contrast pivots at the image's measured
    // mid-gray, shadows at the measured white point, highlights at the
    // measured shadow anchor. Grid thumbnails are unaffected; every detail
    // open starts from the stored edits. The Highlights and Shadows sliders
    // expose their power as a stop-based "lift value" (`highlight_lift`/
    // `shadow_lift`) centered on the identity, so dragging right brightens the
    // region instead of crushing it — the direction other photo apps use, with
    // the sweet spot in the middle of the track. The two arms use opposite
    // power↔lift maps because their pivots sit at opposite ends. Contrast keeps
    // its conventional sense (up = more contrast).
    let contrast_label = widget::text(fl!("contrast-label"));
    let contrast_slider = widget::slider(
        0.25..=4.0,
        app.curve_contrast,
        // When any slider moves, the other values travel along so the
        // remap always composes the full curve, not a half-updated one.
        move |contrast| Message::CurveChanged(contrast, app.curve_highlights, app.curve_shadows),
    )
    .step(0.05_f32)
    // A finished drag is an edit flush point, like exposure.
    .on_release(Message::EditSave);
    let highlights_label = widget::text(fl!("highlights-label"));
    let highlights_slider = widget::slider(
        -2.0..=2.0,
        highlight_lift(app.curve_highlights),
        move |lift| {
            Message::CurveChanged(
                app.curve_contrast,
                highlight_power_for_lift(lift),
                app.curve_shadows,
            )
        },
    )
    .step(0.05_f32)
    .on_release(Message::EditSave);
    let shadows_label = widget::text(fl!("shadows-label"));
    let shadows_slider = widget::slider(
        -2.0..=2.0,
        shadow_lift(app.curve_shadows),
        move |lift| {
            Message::CurveChanged(
                app.curve_contrast,
                app.curve_highlights,
                shadow_power_for_lift(lift),
            )
        },
    )
    .step(0.05_f32)
    .on_release(Message::EditSave);
    // Keyboard crop readout + arm hint. The four values are the live margins
    // removed from each edge in source pixels (top/right/bottom/left).
    let crop_readout = widget::text(fl!(
        "crop-readout",
        top = app.crop.top,
        right = app.crop.right,
        bottom = app.crop.bottom,
        left = app.crop.left
    ));
    // Display rotation readout: a cumulative counter-clockwise quarter-turn
    // value (0° / 90° / 180° / 270°) on top of the EXIF orientation. The
    // keyboard (`r`) routes the step through the edit-key machinery.
    let rotation_degrees = u32::from(app.rotation) * 90_u32;
    let rotation_readout = widget::text(fl!("rotation-readout", degrees = rotation_degrees));
    let reset_all = widget::button::standard(fl!("reset-all")).on_press(Message::ResetAll);
    let reset_crop = widget::button::standard(fl!("reset-crop")).on_press(Message::ResetCrop);

    widget::column::with_capacity(20)
        .push(label)
        .push(slider)
        .push(contrast_label)
        .push(contrast_slider)
        .push(highlights_label)
        .push(highlights_slider)
        .push(shadows_label)
        .push(shadows_slider)
        .push(reset_all)
        .push(widget::divider::horizontal::default())
        .push(rotation_readout)
        .push(crop_readout)
        .push(reset_crop)
        .spacing(space_s)
        .width(Length::Fill)
        .into()
}

/// The fixed width (logical points) reserved for every help row's key chip, so
/// the chip and its description align in a column like a table. Wide enough for
/// the longest key (`Ctrl+C / Ctrl+V`).
pub(crate) const HELP_KEY_WIDTH: f32 = 140.0;

/// The help-overlay shortcut list: one entry per section, each a list of
/// `(key, fluent-label)` rows. Section and label IDs are looked up at render
/// time via [`crate::i18n::fl_dyn`] (the `fl!` macro needs literals).
pub(crate) const HELP_SECTIONS: &[(&str, &[(&str, &str)])] = &[
    (
        "help-section-general",
        &[
            ("?", "help-key-help"),
            ("Esc", "help-key-escape"),
            ("Space", "help-key-space"),
            ("c", "help-key-crop-mode"),
            ("e", "help-key-export"),
            ("Ctrl+A", "help-key-select-all"),
            ("Ctrl+C / Ctrl+V", "help-key-copy-paste"),
        ],
    ),
    (
        "help-section-navigation",
        &[
            ("h / j / k / l", "help-nav-move"),
            ("← ↑ ↓ →", "help-nav-move"),
            ("Enter", "help-nav-open"),
            ("Ctrl+F", "help-nav-search"),
        ],
    ),
    (
        "help-section-editing",
        &[
            ("- / =", "help-edit-exposure"),
            ("[ / ]", "help-edit-contrast"),
            ("; / '", "help-edit-highlights"),
            (", / .", "help-edit-shadows"),
            ("r", "help-edit-rotate"),
        ],
    ),
    (
        "help-section-crop-mode",
        &[
            ("h / j / k / l", "help-crop-move"),
            ("← ↑ ↓ →", "help-crop-move"),
            ("- / =", "help-crop-resize"),
            ("Shift", "help-crop-nudge"),
            ("c / Esc", "help-crop-exit"),
        ],
    ),
];

/// One help row: an accent key chip + its description. The chip sits in a
/// fixed-width column (`HELP_KEY_WIDTH`) left-aligned, so every row's key
/// values and descriptions line up vertically like a table.
pub(crate) fn help_row<'a>(key: &'a str, label: &'a str, space_s: u16, space_m: u16) -> Element<'a, Message> {
    widget::row::with_capacity(2)
        .push(
            widget::container(
                widget::container(widget::text::body(key))
                    .padding([2, space_s])
                    .class(cosmic::theme::Container::custom(|theme| {
                        cosmic::iced::widget::container::Style {
                            text_color: Some(theme.cosmic().accent_text_color().into()),
                            ..Default::default()
                        }
                    })),
            )
            .width(Length::Fixed(HELP_KEY_WIDTH))
            .align_x(Horizontal::Left),
        )
        .push(widget::text::body(fl_dyn(label)))
        .spacing(space_m)
        .align_y(Vertical::Center)
        .into()
}

/// One help section as a grid cell: its section title followed by its rows.
pub(crate) fn help_section<'a>(
    section: &'a str,
    rows: &'a [(&'a str, &'a str)],
    space_s: u16,
    space_m: u16,
) -> Element<'a, Message> {
    widget::column::with_capacity(rows.len() + 1)
        .push(widget::text::title2(fl_dyn(section)))
        .extend(
            rows.iter()
                .map(|(key, label)| help_row(key, label, space_s, space_m)),
        )
        .spacing(space_s)
        .into()
}

/// The global keyboard-help overlay (toggled by `?`): a full-window scrim that
/// swallows all pointer input (`Message::Ignore`, like the detail surface) with
/// a card that fills the window (minus a scrim margin) listing every keyboard
/// shortcut. Sections are laid out as grid cells that flow into as many columns
/// as the width allows (`Grid::fluid`) and stretch to fill the card's height
/// (`EvenlyDistribute`), so the content fills the available space instead of
/// scrolling. While it is shown, `update()` gates every other message so the
/// covered UI stays inert.
pub(crate) fn help_overlay(_app: &AppModel) -> Element<'_, Message> {
    let space_m = cosmic::theme::spacing().space_m;
    let space_s = cosmic::theme::spacing().space_s;

    let grid = Grid::with_children(
        HELP_SECTIONS
            .iter()
            .map(|(section, rows)| help_section(section, rows, space_s, space_m)),
    )
    .fluid(280.0)
    .height(grid::Sizing::EvenlyDistribute(Length::Fill))
    .spacing(space_m);

    let card = widget::container(
        widget::column::with_capacity(HELP_SECTIONS.len() + 2)
            .push(widget::text::title1(fl!("help-title")))
            .push(widget::text::caption(fl!("help-dismiss")))
            .push(grid)
            .spacing(space_m),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(space_m)
    .class(cosmic::theme::Container::Primary);

    widget::container(
        MouseArea::new(card)
            .on_press(Message::Ignore)
            .on_double_click(Message::Ignore)
            .on_double_press(Message::Ignore)
            .on_right_press(Message::Ignore)
            .on_right_release(Message::Ignore)
            .on_middle_press(Message::Ignore)
            .on_middle_release(Message::Ignore)
            .on_scroll(|_| Message::Ignore),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .padding(space_m)
    .class(cosmic::theme::Container::custom(|theme| {
        cosmic::iced::widget::container::Style {
            background: Some(cosmic::iced::Background::Color(
                cosmic::iced::Color::from(theme.cosmic().background(false).base).scale_alpha(0.6),
            )),
            ..Default::default()
        }
    }))
    .into()
}

/// Parses the basic EXIF readout (dimensions, camera, ISO, shutter, aperture,
/// focal length, lens, capture date) from a RAW file. Runs on the blocking
/// pool when the frame-info drawer opens and is cached on the tile; `Err` for
/// files the EXIF reader cannot parse (non-TIFF-based formats, corrupt files).
///
/// `display_value()` already formats the fields nicely (`1/250`, `2.8`,
/// `50.0`, `2024-05-01 …`), so the panel renders them as strings directly.
pub(crate) fn load_frame_meta(dir: &Path, name: &str) -> Result<FrameMeta, FrameError> {
    let file = std::fs::File::open(dir.join(name)).map_err(|_| FrameError::Meta)?;
    let exif = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .map_err(|_| FrameError::Meta)?;

    let field = |tag: exif::Tag| -> Option<String> {
        exif.get_field(tag, exif::In::PRIMARY)
            .map(|value| value.display_value().to_string())
    };
    // CR2/DNG may carry the recorded dimensions on the pixel tags rather than
    // the plain IFD width/height, so prefer ImageWidth/ImageLength and fall
    // back to PixelXDimension/PixelYDimension.
    let dimension = |primary: exif::Tag, pixel: exif::Tag| field(primary).or_else(|| field(pixel));

    Ok(FrameMeta {
        width: dimension(exif::Tag::ImageWidth, exif::Tag::PixelXDimension),
        height: dimension(exif::Tag::ImageLength, exif::Tag::PixelYDimension),
        make: field(exif::Tag::Make),
        model: field(exif::Tag::Model),
        iso: field(exif::Tag::PhotographicSensitivity),
        exposure: field(exif::Tag::ExposureTime),
        aperture: field(exif::Tag::FNumber),
        focal: field(exif::Tag::FocalLength),
        lens: field(exif::Tag::LensModel),
        date: field(exif::Tag::DateTimeOriginal),
    })
}

/// The frame-info drawer body for the highlighted frame: its dimensions and
/// basic EXIF readout (the file name is the drawer title, set by the caller).
/// When the roll has a start date, a roll-derived "Original date" row leads the
/// panel with the frame's exact exported `DateTimeOriginal` (the roll's start
/// date plus the frame's full-roll offset in seconds, the same value the export
/// stamps). That row depends only on the roll, so it shows even while the lazy
/// EXIF parse is in flight or failed. Shows a loading placeholder while the
/// lazy parse is in flight and only rows for fields the file actually carries;
/// a failed parse renders a quiet hint.
pub(crate) fn frame_info_panel<'a>(app: &'a AppModel, name: &str) -> Element<'a, Message> {
    let space_s = cosmic::theme::spacing().space_s;

    // The synthetic capture timestamp this frame will be stamped with on
    // export — the roll's start date plus the frame's full-roll position in
    // seconds (the tile order is the grid order, which is also the export
    // order). Only present when the roll carries a start date; independent of
    // the file's own EXIF.
    let dated = app.roll.start_date().and_then(|start| {
        app.tiles
            .iter()
            .position(|tile| tile.name == name)
            .and_then(|index| exif_writer::shifted_datetime(start, index))
    });

    let tile = app.tiles.iter().find(|tile| tile.name == name);
    let meta_rows: Vec<Element<'_, Message>> =
        if let Some(meta) = tile.and_then(|t| t.meta.as_ref()) {
            let mut rows = Vec::with_capacity(8);
            if let (Some(width), Some(height)) = (&meta.width, &meta.height) {
                rows.push(
                    widget::text::body(fl!(
                        "frame-dimensions",
                        dimensions = format!("{width} × {height}")
                    ))
                    .into(),
                );
            }
            let camera = match (&meta.make, &meta.model) {
                (Some(make), Some(model)) => format!("{make} {model}"),
                (Some(make), None) => make.clone(),
                (None, Some(model)) => model.clone(),
                _ => String::new(),
            };
            if !camera.is_empty() {
                rows.push(widget::text::body(fl!("frame-camera", camera = camera)).into());
            }
            if let Some(value) = meta.iso.clone() {
                rows.push(widget::text::body(fl!("frame-iso", iso = value)).into());
            }
            if let Some(value) = meta.exposure.clone() {
                rows.push(widget::text::body(fl!("frame-exposure", exposure = value)).into());
            }
            if let Some(value) = meta.aperture.clone() {
                rows.push(widget::text::body(fl!("frame-aperture", aperture = value)).into());
            }
            if let Some(value) = meta.focal.clone() {
                rows.push(widget::text::body(fl!("frame-focal", focal = value)).into());
            }
            if let Some(value) = meta.lens.clone() {
                rows.push(widget::text::body(fl!("frame-lens", lens = value)).into());
            }
            if let Some(value) = meta.date.clone() {
                rows.push(widget::text::body(fl!("frame-date", date = value)).into());
            }
            rows
        } else if tile.is_some_and(|t| t.meta_failed) {
            vec![widget::text(fl!("frame-info-unavailable")).into()]
        } else {
            vec![widget::text(fl!("frame-info-loading")).into()]
        };

    // The roll-derived stamp leads the panel, above whatever the file itself
    // carries, so a frame with no (or unreadable) EXIF still shows the date it
    // would be exported with.
    let mut rows = Vec::with_capacity(meta_rows.len() + usize::from(dated.is_some()));
    if let Some(date) = dated {
        rows.push(widget::text::body(fl!("frame-original-date", date = date)).into());
    }
    rows.extend(meta_rows);

    widget::column::with_capacity(rows.len())
        .extend(rows)
        .spacing(space_s)
        .width(Length::Fill)
        .into()
}

/// A labeled ISO-date text field for the roll-info drawer, committed on Enter
/// via [`Message::RollDateDraftSubmit`]. Seeded from the live draft so an
/// in-progress edit survives view re-renders.
fn roll_date_field(label: String, value: &str, field: RollDateField) -> Element<'_, Message> {
    widget::column::with_capacity(2)
        .push(widget::text(label))
        .push(
            widget::text_input(fl!("roll-date-placeholder"), value)
                .width(Length::Fill)
                .on_input(move |v| Message::RollDateDraftChange(field, v))
                .on_submit(move |_| Message::RollDateDraftSubmit(field)),
        )
        .spacing(cosmic::theme::spacing().space_xs)
        .into()
}

/// Renders the roll metadata drawer: the editable roll name heading, its full
/// path, frame count, and cover file, then the roll dates (start + optional
/// end, committed on Enter), the film preset, and the removal action. The
/// drawer pane supplies the width/padding.
pub(crate) fn roll_info_panel<'a>(
    roll: &'a Roll,
    name_draft: &'a str,
    start_draft: &'a str,
    end_draft: &'a str,
) -> Element<'a, Message> {
    let space_xs = cosmic::theme::spacing().space_xs;

    let remove = widget::button::destructive(fl!("remove-roll"))
        .on_press(Message::RemoveRoll(roll.dir.clone()));

    // The roll's display name is the drawer heading, editable in place: typing
    // feeds the draft, Enter commits it (empty reverts to the directory leaf).
    let name = widget::container(
        widget::text_input(fl!("roll-name-placeholder"), name_draft)
            .width(Length::Fill)
            .on_input(Message::RollNameDraftChange)
            .on_submit(|_| Message::RollNameDraftSubmit),
    )
    .width(Length::Fill);

    // The film preset selector: None (no inversion), Generic, then the named
    // stocks in `FILM_CHOICES` order. Index order MUST match `FilmPreset::index()`.
    let mut preset_options = Vec::with_capacity(FILM_CHOICES.len());
    for preset in FILM_CHOICES {
        match preset.stock() {
            Some(stock) => preset_options.push(stock.name.to_owned()),
            None => preset_options.push(fl!("preset-none")),
        }
    }
    let roll_dir = roll.dir.clone();
    let preset = widget::dropdown::dropdown(
        preset_options,
        Some(roll.preset.index()),
        move |index| Message::RollPresetChanged(roll_dir.clone(), FilmPreset::from_index(index)),
    )
    .width(Length::Fill);

    // The base strategy selector: how the roll's black point is resolved. Only
    // meaningful while a film (inversion) is chosen, so the row is hidden for a
    // non-inverted roll. Index order MUST match `BaseMode::index()`.
    let base = if roll.preset.stock().is_some() {
        let base_dir = roll.dir.clone();
        let base = widget::dropdown::dropdown(
            vec![
                fl!("base-preset"),
                fl!("base-auto-per-frame"),
                fl!("base-auto-selected-frame"),
            ],
            Some(roll.base_mode.index()),
            move |index| Message::RollBaseModeChanged(base_dir.clone(), BaseMode::from_index(index)),
        )
        .width(Length::Fill);
        Some(base)
    } else {
        None
    };

    widget::column::with_capacity(14)
        .push(name)
        .push(widget::divider::horizontal::default())
        .push(widget::text::body(fl!(
            "roll-path",
            path = roll.dir.display().to_string()
        )))
        .push(widget::text::body(fl!(
            "roll-frames-line",
            frames = roll.frame_count.to_string()
        )))
        .push(widget::text::body(fl!(
            "roll-cover",
            cover = roll.cover.clone().unwrap_or_else(|| fl!("roll-no-cover"))
        )))
        .push(widget::divider::horizontal::default())
        .push(widget::text(fl!("roll-dates-label")))
        .push(roll_date_field(
            fl!("roll-date-start-label"),
            start_draft,
            RollDateField::Start,
        ))
        .push(roll_date_field(
            fl!("roll-date-end-label"),
            end_draft,
            RollDateField::End,
        ))
        .push(widget::divider::horizontal::default())
        .push(widget::text(fl!("preset-label")))
        .push(preset)
        .push_maybe(base.map(|base| {
            let row: Element<'_, Message> = widget::column::with_capacity(2)
                .push(widget::text(fl!("base-label")))
                .push(base)
                .spacing(space_xs)
                .width(Length::Fill)
                .into();
            row
        }))
        .push(widget::divider::horizontal::default())
        .push(remove)
        .spacing(space_xs)
        .width(Length::Fill)
        .into()
}

/// Renders the library grid's first tile: an "Add Roll" card that opens the
/// folder picker on double-click. Mirrors a roll card's surface, radius, and
/// selection ring so it reads and behaves as another tile — single-clicking
/// selects it (highlight), double-clicking (or Enter) opens the picker — with
/// a large plus icon.
fn add_roll_tile(selected: bool) -> Element<'static, Message> {
    let content = widget::container(
        icon::from_name("list-add-symbolic")
            .size(130)
            .icon()
            .opacity(0.50),
    )
    .align_x(Horizontal::Center)
    .align_y(Vertical::Center)
    .height(Length::Fill)
    .width(Length::Fill);

    // Single-click selects the tile (highlight only); double-click opens the
    // folder picker.
    let card: Element<'_, Message> = MouseArea::new(content)
        .on_press(Message::AddRollSelected)
        .on_double_click(Message::AddRoll)
        .into();

    selectable_tile(card, selected)
}

/// Renders a library page roll card, filling the square cell the grid assigns
/// it. Single-clicking anywhere on the card selects the roll; double-clicking
/// (or Enter) drills into its frame grid. The selection highlight is an accent
/// ring drawn OVER the card (see the tail of this function), so the full-bleed
/// cover never hides it.
fn roll_tile(roll: &Roll, selected: bool) -> Element<'_, Message> {
    let space_xs = cosmic::theme::spacing().space_xs;
    // The tile's corner radius follows the COSMIC system Roundness setting
    // (rounded / slightly-rounded / square) exactly as `Container::Primary`
    // resolves it, so the full-bleed cover's top corners clip to the tile's
    // own curves.
    let radius = cosmic::theme::active().cosmic().corner_radii.radius_s[0];

    // The cover runs full-bleed: flush against the card's top edge with no
    // padding, and clipped to the tile's corners — the wgpu image renderer
    // discards fragments outside the rounded box, so the curves genuinely cut
    // into the photo. Only the TOP corners are rounded: iced-wgpu's image
    // shader applies the radius vec rotated 180° from the `border::Radius`
    // struct order (the first two entries render on the bottom), so the top
    // clip is expressed with `bottom()`. The cover's bottom edge is interior,
    // so its bottom corners stay square. Crop-to-fill (`ContentFit::Cover`)
    // makes any aspect fill the area: landscape crops almost nothing, portrait
    // center-crops the top/bottom. Placeholders stay centered in the same
    // sheet.
    let content: Element<'_, Message> = match &roll.thumb {
        Thumb::Ready(handle) => widget::image(handle.clone())
            .width(Length::Fill)
            .height(Length::Fill)
            .content_fit(ContentFit::Cover)
            .border_radius(cosmic::iced::border::Radius::default().bottom(radius))
            .into(),
        Thumb::Loading => widget::container(icon::from_name("image-loading-symbolic").icon())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into(),
        Thumb::Failed => widget::container(icon::from_name("image-missing-symbolic").icon())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into(),
    };

    // Title (flush-left) plus a single caption row: the frame count in its own
    // left column and, when a start date is set, the date (or start – end
    // range) in a right column — a display-only mirror of the roll-info
    // drawer's dates. Both sit in a block padded 12px (`space_xs`) away from
    // the card's edges; its top padding is also the gap below the full-bleed
    // cover. The two `Length::Fill` row cells split the row into equal columns;
    // an undated roll simply leaves the right cell out.
    let mut meta = widget::row::with_capacity(2);
    meta = meta.push(
        widget::container(widget::text::caption(fl!(
            "roll-frames",
            count = roll.frame_count
        )))
        .width(Length::Fill)
        .align_x(Horizontal::Left),
    );
    if let Some(start) = &roll.start_date {
        let month_names: [String; 12] = [
            fl!("month-01"),
            fl!("month-02"),
            fl!("month-03"),
            fl!("month-04"),
            fl!("month-05"),
            fl!("month-06"),
            fl!("month-07"),
            fl!("month-08"),
            fl!("month-09"),
            fl!("month-10"),
            fl!("month-11"),
            fl!("month-12"),
        ];
        let months: [&str; 12] = std::array::from_fn(|i| month_names[i].as_str());
        let dates = widget::text::caption(
            match format_roll_card_dates(&months, start, roll.end_date.as_deref()) {
                // Malformed dates (hand-edited manifests) keep today's raw ISO
                // display instead of inventing a format.
                None => match &roll.end_date {
                    Some(end) => fl!("roll-card-dates", start = start.clone(), end = end.clone()),
                    None => fl!("roll-card-date", start = start.clone()),
                },
                Some(formatted) => formatted,
            },
        );
        meta = meta.push(
            widget::container(dates)
                .width(Length::Fill)
                .align_x(Horizontal::Right),
        );
    }
    let info: Element<'_, Message> = widget::container(
        widget::column::with_capacity(2)
            .push(widget::text(&roll.name))
            .push(meta)
            .spacing(space_xs)
            .align_x(Horizontal::Left),
    )
    .width(Length::Fill)
    .padding(space_xs)
    .into();

    // The whole card is one interactive surface — selectable even while its
    // cover is still decoding, openable by double-click anywhere, not just the
    // image.
    let card = widget::column::with_capacity(2)
        .push(content)
        .push(info)
        .spacing(0);

    let card: Element<'_, Message> = MouseArea::new(card)
        .on_press(Message::RollSelected(roll.dir.clone()))
        .on_double_click(Message::RollActivated(roll.dir.clone()))
        .into();

    selectable_tile(card, selected)
}

/// Wraps a library tile's interactive content in the shared selection chrome: a
/// `Container::Primary` surface (background only, its radius follows the theme)
/// with an accent ring drawn ON TOP when selected. The selection is NOT a card
/// style here: iced paints a container's border behind its children, which a
/// full-bleed cover would cover up, so the ring is a separate overlay layer.
fn selectable_tile(content: Element<'_, Message>, selected: bool) -> Element<'_, Message> {
    let surface = widget::container(content)
        .width(Length::Fill)
        .height(Length::Fill)
        .class(cosmic::theme::Container::Primary);

    let mut stack = Stack::with_capacity(1);
    stack = stack.push(surface);

    // Selection: an accent ring drawn ON TOP of this card.
    if selected {
        stack = stack.push(selection_ring());
    }

    stack.width(Length::Fill).height(Length::Fill).into()
}

/// The selection highlight shared by every selectable tile (library rolls and
/// open-roll frames): a transparent overlay ring drawn ON TOP of the card,
/// rounded to the theme radius, so the accent border is visible around a
/// full-bleed preview on every side. Decorative only — no `MouseArea` — so it
/// never eats the clicks the card below expects.
fn selection_ring() -> Element<'static, Message> {
    widget::container(
        widget::Space::new()
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .class(cosmic::theme::Container::custom(|theme| {
        cosmic::iced::widget::container::Style {
            border: cosmic::iced::border::Border {
                color: theme.cosmic().accent.base.into(),
                width: 2.0,
                radius: theme.cosmic().corner_radii.radius_s.into(),
            },
            ..Default::default()
        }
    }))
    .into()
}

/// Renders a single frame tile of the open roll, filling the square cell the
/// grid assigns it. Single-clicking selects (accent ring highlight); a
/// double-click (or Enter on the highlighted tile) opens the detail view. Like
/// the library cards, the tile is a surface + Stack with the ring overlay; the
/// image itself stays `Contain` so a film frame's full framing is never cropped.
///
/// `is_calibration_frame` overlays a small accent dot in the tile's bottom-right
/// corner (see [`calibration_frame_dot`]) to mark the frame whose clear-film
/// plateau drives the roll's black point under the auto-selected-frame preset.
fn tile_view(tile: &Tile, selected: bool, is_calibration_frame: bool) -> Element<'_, Message> {
    // The tile is the image alone in its square cell with ContentFit::Contain
    // (no cropping); placeholders stay centered in the same sheet. The frame
    // name now lives in the frame-info context drawer (its title), not on the
    // tile.
    let content: Element<'_, Message> = match &tile.thumb {
        Thumb::Ready(handle) => widget::image(handle.clone())
            .width(Length::Fill)
            .height(Length::Fill)
            .content_fit(ContentFit::Contain)
            .into(),
        Thumb::Loading => widget::container(icon::from_name("image-loading-symbolic").icon())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into(),
        Thumb::Failed => widget::container(icon::from_name("image-missing-symbolic").icon())
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(Horizontal::Center)
            .align_y(Vertical::Center)
            .into(),
    };

    let card: Element<'_, Message> = MouseArea::new(content)
        .on_press(Message::FrameSelected(tile.name.clone()))
        .on_double_click(Message::ThumbnailActivated(tile.name.clone()))
        .into();

    let mut stack = Stack::with_capacity(1);
    stack = stack.push(
        widget::container(card)
            .width(Length::Fill)
            .height(Length::Fill)
            .class(cosmic::theme::Container::Primary),
    );

    if selected {
        stack = stack.push(selection_ring());
    }

    if is_calibration_frame {
        stack = stack.push(calibration_frame_dot());
    }

    stack.width(Length::Fill).height(Length::Fill).into()
}

/// Whether `name` carries the auto-calibration dot: it is the roll's designated
/// calibration frame AND the roll's base mode is [`BaseMode::AutoSelectedFrame`]
/// (the only mode that reads a designated frame).
#[must_use]
fn is_calibration_frame(base_mode: BaseMode, calibration_frame: Option<&str>, name: &str) -> bool {
    base_mode == BaseMode::AutoSelectedFrame && calibration_frame == Some(name)
}

/// The auto-calibration indicator: a small theme-accent circle pinned to the
/// frame tile's bottom-right corner. A pure `Stack` overlay, so it never
/// disturbs the tile's layout (the surface and any selection ring keep their
/// exact sizes); the fixed-size dot sits in a Fill container aligned to the
/// corner via the `Stack`'s fill distribution.
fn calibration_frame_dot() -> Element<'static, Message> {
    const DOT_SIZE: f32 = 12.0;
    const DOT_PAD: f32 = 8.0;

    let dot = widget::container(widget::Space::new())
        .width(Length::Fixed(DOT_SIZE))
        .height(Length::Fixed(DOT_SIZE))
        .class(cosmic::theme::Container::custom(|theme| {
            cosmic::iced::widget::container::Style {
                background: Some(theme.cosmic().accent.base.into()),
                border: cosmic::iced::border::Border {
                    // A thin on-accent ring keeps the dot visible over both
                    // light and dark tile content.
                    color: theme.cosmic().accent.on.into(),
                    width: 1.5,
                    radius: (DOT_SIZE / 2.0).into(),
                },
                ..Default::default()
            }
        }));

    widget::container(dot)
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Horizontal::Right)
        .align_y(Vertical::Bottom)
        .padding(DOT_PAD)
        .into()
}

/// Renders the detail view for the selected file, if any, over the still-mounted
/// thumbnail grid (which the caller paints an opaque, theme-colored surface on).
///
/// Shows the cached thumbnail while the GPU shader is loading, then
/// crossfades: the thumbnail sits on top of the shader in a Stack and
/// fades out via `.opacity()`.
fn detail_view(app: &AppModel) -> Option<Element<'_, Message>> {
    let space_s = cosmic::theme::spacing().space_s;
    let name = app.selected.as_ref()?;
    let tile = app.tiles.iter().find(|tile| &tile.name == name)?;

    let preview: Element<'_, Message> = match (
        app.detail_shader.as_ref(),
        app.detail_thumb.as_ref(),
        app.detail_thumb_opacity > 0.0,
    ) {
        (Some(shader), Some(thumb), true) => {
            // Crossfade in progress — thumbnail fading out over shader.
            Stack::with_children([
                shader.view().into(),
                widget::image(thumb.clone())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .content_fit(ContentFit::Contain)
                    .opacity(app.detail_thumb_opacity)
                    .into(),
            ])
            .into()
        }
        (Some(shader), _, _) => shader.view().into(),
        (None, _, _) => {
            // Shader not ready — show the cached thumbnail or a status icon.
            match &tile.thumb {
                Thumb::Ready(handle) => widget::image(handle.clone())
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .content_fit(ContentFit::Contain)
                    .into(),
                Thumb::Loading => icon::from_name("image-loading-symbolic").icon().into(),
                Thumb::Failed => icon::from_name("image-missing-symbolic").icon().into(),
            }
        }
    };

    // The shader widget is `Length::Fill × Length::Fill`; bounding it inside
    // a Fill × Fill container caps its widget bounds at the parent column's
    // allocation. The WGSL's `contained_uv` does `ContentFit::Contain` math so
    // the image letterbox/pillarbox-fits within those bounds with
    // transparent bars (alpha=0 + `BlendState::ALPHA_BLENDING`) — no overflow
    // into the header or any other widget.
    Some(
        widget::column::with_capacity(2)
            .push(
                // Interactive surface for the preview: wheel zooms, press+drag
                // pans. The outer full-surface `MouseArea` in `view` still
                // swallows events over the caption/bars and blocks the grid
                // behind; iced delivers to the inner widget first, so its
                // `capture_event()` wins and the outer handlers are skipped.
                // `DetailArea.on_move` reports cursor positions relative to
                // the widget center, matching the center-relative pan offset
                // the zoom-anchor math compares against.
                DetailArea::new(
                    widget::container(preview)
                        .width(Length::Fill)
                        .height(Length::Fill),
                )
                .on_scroll(|delta| Message::DetailZoom(detail_zoom_delta(delta)))
                .on_resize(Message::DetailAreaResized)
                .on_press(Message::DetailPanPress)
                .on_move(Message::DetailPanMove)
                .on_release(Message::DetailPanRelease)
                // Grab/grasping hand once the view is pannable (zoomed past
                // contain fit) and while a drag is in progress, so the preview
                // advertises that press+drag pans.
                .interaction(app.detail_cursor_interaction()),
            )
            .spacing(space_s)
            .align_x(Horizontal::Center)
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn is_calibration_frame_gates_on_the_base_mode_and_designated_name() {
        // The dot only appears for the designated frame under the
        // AutoSelectedFrame base mode.
        let frame = "IMG_0007.DNG";
        assert!(is_calibration_frame(
            BaseMode::AutoSelectedFrame,
            Some(frame),
            frame
        ));
        // Same base mode, wrong frame → no dot.
        assert!(!is_calibration_frame(
            BaseMode::AutoSelectedFrame,
            Some(frame),
            "IMG_0008.DNG"
        ));
        // No designated frame yet → no dot.
        assert!(!is_calibration_frame(
            BaseMode::AutoSelectedFrame,
            None,
            frame
        ));
        // Every other base mode ignores the designated frame.
        for mode in [BaseMode::Preset, BaseMode::AutoPerFrame] {
            assert!(!is_calibration_frame(mode, Some(frame), frame));
        }
    }
}
