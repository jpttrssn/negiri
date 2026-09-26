// SPDX-License-Identifier: GPL-3.0-or-later

use crate::config::Config;
use crate::edit_manifest::{self, RollManifest};
use crate::error::FrameError;
use crate::export::{
    ExportOptions, export_format_choice, export_fraction, export_frames, export_overwrite_choice,
    is_export_artifact, options_for_choice,
};
use crate::film::{
    BaseConfig, BaseMode, FILM_CHOICES, FilmPreset, MIN_PLAUSIBLE_BASE, MonoStock, measure_base,
};
use crate::fl;
use crate::library::{
    LibraryCell, MonthAliases, library_cell_index, library_cells,
    nav_target, paginate, record_roll_base_mode, record_roll_dates, record_roll_name,
    record_roll_preset, roll_dates_valid, valid_iso_date,
};
use crate::pipeline::{DetailDecode, convert_thumbnail, decode_raw_detail, resized_dims};
use crate::shader;
use crate::ui::{
    FrameMeta, editing_panel, frame_info_panel, frames_view, help_overlay, library_view,
    load_frame_meta, roll_info_panel, scrollable_id, search_input_id,
};
use cosmic::Application;
use cosmic::app::context_drawer;
use cosmic::cosmic_config::{self, CosmicConfigEntry};
use cosmic::iced::futures::SinkExt;
use cosmic::iced::keyboard;
use cosmic::iced::widget::scrollable::Viewport;
use cosmic::iced::widget::Stack;
use cosmic::iced::{Length, Point, Size, Subscription};
use cosmic::prelude::*;
use cosmic::widget::{self, about::About, icon, image::Handle, menu, toaster};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");
const APP_ICON: &[u8] =
    include_bytes!("../resources/icons/hicolor/scalable/apps/io.github.jpttrssn.curvectrl.svg");

/// Maximum dimension of decoded RAW thumbnails, also the maximum Page 1 tile width.
pub(crate) const THUMB_SIZE: f32 = 384.0;
/// Maximum dimension of the hi-res decode displayed in the detail view.
///
/// Sharp at typical window sizes while keeping the decoded buffer a fraction
/// of a full sensor frame.
const HI_RES_SIZE: u32 = 2048;
/// Opacity units per second for the hi-res crossfade (~0.2 s full ramp).
const FADE_SPEED: f32 = 5.0;
/// Aspect ratio (width / height) of the image area of a Page 1 tile.
pub(crate) const TILE_ASPECT: f32 = 1.0;

/// Maximum number of thumbnail decodes in flight at once.
///
/// Decodes run on the blocking thread pool, so several proceed in parallel on
/// multicore machines. The bound keeps the transient footprint (each in-flight
/// decode holds one full-resolution RAW buffer until it finishes downscaling)
/// to a few dozen hundred MB, not the whole roll.
const MAX_CONCURRENT_THUMBS: usize = 4;

/// Number of detail-view overview (2048px) mono buffers to cache most-recently
/// used. Sized to cover a typical film roll (~40 frames). Each entry is the
/// fixed 2048 overview decode (~12–17 MB), so the worst case is bounded at
/// ~600 MB; native zoom level-up decodes are never cached.
const DETAIL_CACHE_CAPACITY: usize = 40;

/// Maximum number of neighbor detail preload decodes in flight at once.
/// Preloads run on their own bounded channel, separate from the single
/// critical detail slot, so they never starve the currently-viewed frame.
const MAX_CONCURRENT_PRELOADS: usize = 2;

/// How many frames on each side of the currently-viewed frame to preload into
/// the overview LRU cache (so Left/Right paging to a neighbor is instant).
const DETAIL_PRELOAD_DISTANCE: usize = 1;

/// Keyboard shortcut step for exposure (EV) when an edit control is adjusted
/// with a bare key.
const EDIT_STEP_EV: f32 = 0.50;
/// Keyboard shortcut nudge step for exposure (EV) with the Shift modifier.
/// Equal to `edit_manifest::EV_TICK` (and the exposure slider's step), so every
/// exposure edit lands on the fixed-point storage grid.
const EDIT_NUDGE_EV: f32 = 0.05;
/// Keyboard shortcut step for a tone-curve power (contrast/highlights/shadows)
/// with a bare key. Highlights and Shadows apply it to their user-facing lift
/// value in stops, so a step UP lifts the region.
const EDIT_STEP_CURVE: f32 = 0.20;
/// Keyboard shortcut nudge step for a tone-curve power with the Shift modifier.
const EDIT_NUDGE_CURVE: f32 = 0.05;
/// Keyboard shortcut step for a crop-window move: a bare move key (`h`/`j`/`k`/`l`
/// or an arrow) translates the crop window by this many source pixels.
const CROP_STEP_PX: i32 = 5;
/// Keyboard shortcut nudge step for a crop move with the Shift modifier (a 1px
/// fine adjustment).
const CROP_NUDGE_PX: i32 = 1;
/// Keyboard shortcut step for a crop-window resize: a bare `-`/`=` grows or
/// shrinks the crop window (about its center, aspect-locked) so its long edge
/// changes by this many source pixels. Shift switches to the 1px `CROP_NUDGE_PX`.
const CROP_RESIZE_STEP_PX: i32 = 10;
/// The smallest crop-window long edge (source pixels) a resize is allowed to
/// shrink to, so a resize can never collapse the crop to a degenerate sliver.
const MIN_CROP_WINDOW: u32 = 4;

/// Minimum padding (logical points) kept around the image on every side while
/// the detail view is in crop mode: the shader's contain-fit base shrinks by
/// this amount so a white margin is always visible, even when the image aspect
/// exactly matches the widget. Zooming in grows the image into the padding.
const CROP_MODE_PADDING: f32 = 100.0;

/// Maximum detail-view zoom in `log2` units: 1.0 = contain fit, each +1
/// doubles the rendered scale, so this caps at 2^6 = 64× the fit scale.
const MAX_DETAIL_ZOOM: f32 = 7.0;

/// Detail-view zoom (log2 units) at which the native hi-res decode is
/// prefetched on a preview whose 1:1 cap sits above it. 2.0 = 2× contain:
/// safely past the 2048 overview's own 1:1 on typical widgets, before the
/// view has upscaled 2048 pixels far enough to look soft. On larger previews
/// the 1:1 cap can sit BELOW this — see [`AppModel::native_level_up_zoom`],
/// which fires the level-up at the earlier of this threshold and the cap so
/// the trigger is always reachable.
const NATIVE_ZOOM_THRESHOLD: f32 = 2.0;

/// wgpu's typical `max_texture_dimension_2d` ceiling. The native level-up
/// caps its target at this so a >8K sensor never asks for an oversized upload.
const MAX_TEXTURE_EDGE: u32 = 8192;

/// Debounce window (ms) between the last keystroke in the search input and the
/// library re-filter. The input text still updates instantly; only the
/// applied filter (`search_filter`) waits for this quiet period.
const SEARCH_DEBOUNCE_MS: u64 = 300;

/// The application model stores app-specific state used to describe its interface and
/// drive its logic.
// Keep the handful of independent state flags as plain bools: they gate
// mutually-unrelated behavior (multi-select, detail decode levels, crop mask,
// export, async decode failures, overwrite dialog), so a shared bitmask or
/// Detail-view zoom in `log2` units: `1.0` = contain fit (the whole frame),
/// each `+1` doubles the rendered scale. Newtyped so a zoom (a `log2` of a
/// scale factor) is never confused with a pixel or length value; `.0` is the
/// raw `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Zoom(pub(crate) f32);

impl Zoom {
    /// Contain fit: the whole frame visible.
    const CONTAIN: Self = Self(1.0);
}

// The many feature flags on the model are mutually exclusive by message
// routing; grouping them would only obscure each flag's meaning.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct AppModel {
    /// Application state which is managed by the COSMIC runtime.
    core: cosmic::Core,
    /// Display a context drawer with the designated page if defined.
    context_page: ContextPage,
    /// Per-view context-drawer open/closed memory (see [`DrawerMemory`]).
    drawer_memory: DrawerMemory,
    /// The about page for this app.
    about: About,
    /// Key bindings for the application's menu bar, consumed by
    /// `cosmic::widget::menu::items` (values are unit [`MenuAction`]s, hence the
    /// allow: the map shape is imposed by the menu API, not by choice).
    #[allow(clippy::zero_sized_map_values)]
    key_binds: HashMap<menu::KeyBind, MenuAction>,
    /// Configuration data that persists between application runs.
    config: Config,
    /// Film rolls listed on the library page, one directory of negatives each.
    pub(crate) rolls: Vec<Roll>,
    /// Directory of the roll currently drilled into (its frame grid and the
    /// detail view). `None` shows the library page of rolls.
    active: Option<PathBuf>,
    /// The library cell single-clicked (or arrow-key selected) on the library
    /// page: either the always-first Add Roll tile or a real roll. `None`
    /// (initial state) until the user selects a cell; drives the selected-tile
    /// highlight, the `Space` metadata drawer, and Enter/arrow navigation.
    pub(crate) library_selection: Option<LibrarySelection>,
    /// Frame-grid highlight: the tile single-clicked (or last opened / paged
    /// to) inside an open roll. Distinct from `selected` (the opened detail
    /// frame) so the highlight survives closing the detail view and drives the
    /// accent ring, Enter, and arrow-key navigation. Also the "primary" of any
    /// multi-selection: it is always a member of `selected_frames`.
    frame_selected: Option<String>,
    /// Multi-selected frames on the open roll's grid. A plain click (or the
    /// primary `frame_selected`) always lands here; Ctrl+click toggles a frame
    /// in/out, Shift+click selects the range from the anchor through the
    /// clicked frame, and Ctrl+A selects every frame in the roll. Used as the
    /// batch target for copy/paste edits.
    pub(crate) selected_frames: HashSet<String>,
    /// Last frame used as the Shift+click range anchor within the grid.
    selection_anchor: Option<String>,
    /// The modifier keys currently held, delivered by the global keyboard
    /// subscription's `ModifiersChanged` event. A frame click reads this to
    /// distinguish a plain click (clear + select) from a Ctrl+click (toggle a
    /// frame in/out of the multi-selection) or a Shift+click (range), since
    /// iced's `MouseArea` delivers no modifier info.
    modifiers: keyboard::Modifiers,
    /// Whether an editing shortcut key (an `AdjustEdit` character) is currently
    /// held. Keyboard steps mutate the live preview (RAM + shader) on every
    /// press/repeat like a slider drag, and commit once when the key is
    /// released — mirroring the slider's drag/release cadence instead of
    /// persisting + re-baking the thumbnail per auto-repeated press.
    editing_key_held: bool,
    /// Number of columns the library grid last laid out (tracked from window
    /// resizes, matching iced's `Grid::fluid` math), so Up/Down keyboard
    /// navigation can jump by exactly one row.
    grid_cols: usize,
    /// The most recent viewport of whichever grid is mounted (library or frame
    /// page — they are mutually exclusive). Drives keyboard scroll-into-view:
    /// knowing the visible height, content height, and current translation lets
    /// `Nav` reveal the highlighted tile precisely. Cleared when the page
    /// changes; re-captured by each grid's `on_scroll`.
    grid_viewport: Option<Viewport>,
    /// The window height from the last resize, used only to estimate the grid
    /// viewport height before the first real scroll has been observed.
    window_height: f32,
    /// Names handed to the bounded in-flight roll-cover decodes, so re-baked
    /// roll tiles never double-spawn against the startup chain (memory bound).
    cover_inflight: Vec<PathBuf>,
    /// Search is library-only: while a roll is open the frame grid is never
    /// searched and the header control is hidden. Input state: `None` hides
    /// the search input (the header shows only the search icon); `Some(term)`
    /// shows the input with `term` as the live text, updated on every
    /// keystroke (echoing cosmic-files, the mere presence of the input is the
    /// toggle, not the text). The *applied* filter lives in
    /// [`Self::search_filter`].
    search: Option<String>,
    /// The applied search filter, committed only after `SEARCH_DEBOUNCE_MS`
    /// passes with no further keystrokes. The library page reads this instead
    /// of [`Self::search`], so typing stays instant while re-filtering lags
    /// behind it.
    pub(crate) search_filter: Option<String>,
    /// Monotonic serial for debounce commits: bumped on every keystroke, clear,
    /// activate and escape; a `SearchCommitted` message whose serial does not
    /// match is stale and is discarded.
    search_version: u64,
    /// Month-name aliases for the library search, built once at startup from
    /// the localized `month-01`…`month-12` short names plus the English full
    /// and 3-letter names. The fluent locale is fixed for the session, so this
    /// never needs rebuilding.
    pub(crate) month_aliases: MonthAliases,
    /// File entries from the open roll, displayed as tiles on its frame grid.
    pub(crate) tiles: Vec<Tile>,
    /// File shown enlarged in the detail view in place of the grid, if any.
    pub(crate) selected: Option<String>,
    /// Names handed to the bounded in-flight thumbnail decodes, so re-baked
    /// tiles never double-spawn against the startup chain (memory bound).
    thumb_inflight: Vec<String>,
    /// The frame whose EXIF parse is currently in flight for the frame-info
    /// drawer, so a highlight change to the same frame does not double-spawn.
    frame_meta_inflight: Option<String>,
    /// Persisted per-file edits for the film roll, loaded at startup and
    /// reconciled against the files on disk on each scan. Writes to the
    /// manifest happen only on explicit flush messages, never per frame.
    pub(crate) roll: RollManifest,
    /// GPU shader program for the detail view, rendering mono data with
    /// live exposure adjustment.  `None` while the decode is in flight.
    pub(crate) detail_shader: Option<shader::DetailProgram>,
    /// File handed to the one permitted in-flight hi-res detail decode.
    detail_inflight: Option<String>,
    /// True once the native-resolution decode has been requested or found
    /// unnecessary (sensor already ≤ the overview size); guards the zoom
    /// trigger against re-spawning a level-up on every wheel event.
    detail_native_queued: bool,
    /// Name of the frame whose detail decode failure was already logged for
    /// the current selection, so wheel-driven re-decode retries of the same
    /// (doomed) frame log once instead of once per notch.
    detail_logged_failure: Option<String>,
    /// Current opacity of the thumbnail layer during crossfade (1.0→0.0).
    pub(crate) detail_thumb_opacity: f32,
    /// Timestamp of the last animation frame for framerate-independent fading.
    detail_last_frame: Option<Instant>,
    /// Cached thumbnail kept visible over the shader during crossfade.
    pub(crate) detail_thumb: Option<Handle>,
    /// Detail-view zoom in `log2` units: 1.0 = contain fit (whole frame),
    /// each +1 doubles the rendered scale.
    detail_zoom: Zoom,
    /// The detail preview's laid-out logical size, reported by `DetailArea`
    /// via `DetailAreaResized`; drives the 1:1 ("100%") zoom cap.
    ///
    /// This is view geometry, so it is NOT cleared by `clear_detail`: frame
    /// paging keeps `DetailArea` mounted with unchanged bounds and therefore
    /// never re-publishes, and dropping the measurement would silently disable
    /// the cap (the wheel would fall back to `MAX_DETAIL_ZOOM`) until a drawer
    /// toggle forced a resize.
    detail_area_size: Option<Size>,
    /// Pan offset of the image center from the widget center (logical points).
    pub(crate) detail_pan: Point,
    /// True while the user is pressing/dragging the detail preview (grab-pan).
    detail_panning: bool,
    /// Most recent cursor position over the detail preview, widget-relative
    /// logical points; anchors wheel-zoom at the cursor.
    detail_cursor: Option<Point>,
    /// Contrast power previewed in the detail view, pivoting the live tone
    /// curve at the image's measured mid-gray. Non-persisted: resets to
    /// `1.0` (identity) on every detail open. GPU-uniform only — the grid
    /// thumbnails always render with the default curve.
    pub(crate) curve_contrast: f32,
    /// Highlights power previewed in the detail view, pivoting the live tone
    /// curve at the image's measured 10th-percentile shadow anchor. Same
    /// lifecycle and rules as [`Self::curve_contrast`].
    pub(crate) curve_highlights: f32,
    /// Shadows power previewed in the detail view, pivoting the live tone
    /// curve at the image's measured white point. Same lifecycle and rules as
    /// [`Self::curve_contrast`].
    pub(crate) curve_shadows: f32,
    /// Exposure compensation in EV (−3.00 to +3.00).
    pub(crate) exposure_ev: f32,
    /// Live keyboard crop margins for the detail view; loaded from the stored
    /// manifest per open and applied to the GPU shader as a uniform UV-remap.
    pub(crate) crop: edit_manifest::CropMargins,
    /// Live display rotation for the detail view: cumulative counter-clockwise
    /// 90° quarter-turns (`0`…`3`) on top of the EXIF orientation. Loaded from
    /// the stored manifest per open and applied to the GPU shader as a uniform.
    pub(crate) rotation: u8,
    /// Draft text for the two roll date fields in the roll-info drawer, seeded
    /// from the selected roll's committed dates on selection change and
    /// committed back on submit.
    roll_date_drafts: RollDateDrafts,
    /// Draft text for the roll-info drawer's editable name heading, seeded from
    /// the selected roll's current name on selection change and committed on
    /// submit.
    roll_name_draft: String,
    /// Whether the detail view is in crop mode — a focused, VIM-like modal
    /// state toggled by the bare `c` key or the View → Crop mode menu item.
    /// Only meaningfully on with a detail view open. While active the surround
    /// renders white and the shader dims everything outside the crop (the old
    /// "view dimmed crop area" overlay), frame paging is disabled, and the
    /// keyboard accepts only the h/j/k/l crop trims. Never persisted.
    pub(crate) crop_mode: bool,
    /// Whether the global keyboard-help overlay is shown (toggled by `?`).
    /// While on, the overlay covers the body and the rest of `update()` is
    /// gated so navigation/editing keys do not act on the covered UI. Never
    /// persisted.
    help_visible: bool,
    /// Edit values as they were when the detail panel was opened — the stored
    /// manifest values for the selected file. `ResetAll` restores these, not
    /// the identity, so reset reverts the panel to its opened state.
    reset_exposure_ev: f32,
    reset_curve_contrast: f32,
    reset_curve_highlights: f32,
    reset_curve_shadows: f32,
    /// Crop as it was when the detail panel was opened; `ResetAll` restores it.
    reset_crop: edit_manifest::CropMargins,
    /// Rotation as it was when the detail panel was opened; `ResetAll`
    /// restores it. Tracked separately from [`Self::rotation`]'s zeroing in
    /// [`Self::clear_detail`] like the other reset snapshots.
    reset_rotation: u8,
    /// The last edit copied for paste (Ctrl+C), as a full [`ToneEdit`]. `None`
    /// until the user copies — a paste with nothing copied is a no-op.
    clipboard: Option<edit_manifest::ToneEdit>,
    /// Monotonic counter incremented each time a new detail decode finishes;
    /// stamped into [`DetailProgram::image_id`] so the GPU pipeline
    /// recognises a new image and rebuilds its texture.
    next_image_id: u64,
    /// LRU of decoded detail overviews keyed by (roll dir, file name), so
    /// returning to a recently-viewed frame doesn't re-decode the RAW. Survives
    /// roll switches and detail close; eviction is global (see
    /// [`DETAIL_CACHE_CAPACITY`]). Only overview (2048px) buffers are stored.
    detail_cache: LruCache<(PathBuf, FilmPreset, String), DetailMono>,
    /// (roll dir, file name) handed to the bounded neighbor preload decodes,
    /// so a frame already being preloaded (or already cached) is never spawned
    /// twice. Independent of the single critical detail slot.
    detail_preload_inflight: Vec<(PathBuf, String)>,
    /// Completion toasts shown over the window (e.g. the export summary).
    /// Auto-dismissing: each toast is removed when its [`Message::ToastClose`]
    /// fires after the toast's duration.
    toasts: toaster::Toasts<Message>,
    /// Whether the native folder dialog is currently open. The portal dialog is
    /// modal and blocks this window's input, but a second `ExportRequested`
    /// could still fire during the async gap before the dialog appears, so a
    /// flag prevents stacking two dialogs.
    export_pending: bool,
    /// Live export progress as `(done, total)` frames, or `None` while no batch
    /// is running. Drives the header's circular progress ring; the stream task
    /// clears it via [`Message::ExportDone`] when the batch finishes.
    export_progress: Option<(usize, usize)>,
}

/// Which roll date field a draft/commit message targets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RollDateField {
    Start,
    End,
}

/// Draft ISO date text for the two roll-info drawer date fields (start and
/// optional end). Seeded from the selected roll's committed dates; updated by
/// typing; committed back to the roll and its manifest on submit. Holds plain
/// `String`s so an in-progress edit stays stable while the read-only view
/// re-renders. `key` is the roll directory the drafts were seeded from, so a
/// changed library selection (or a fresh drawer) reseeds on the next sync.
#[derive(Debug, Clone)]
struct RollDateDrafts {
    /// The roll directory the drafts were seeded from (`None` = never seeded).
    key: Option<PathBuf>,
    start: String,
    end: String,
}

impl RollDateDrafts {
    /// Fresh drafts from a roll's committed dates.
    fn from_dates(key: PathBuf, start: Option<&str>, end: Option<&str>) -> Self {
        Self {
            key: Some(key),
            start: start.unwrap_or("").to_owned(),
            end: end.unwrap_or("").to_owned(),
        }
    }

    /// The draft for a given field, updated in place (used for typing).
    fn set(&mut self, field: RollDateField, value: String) {
        match field {
            RollDateField::Start => self.start = value,
            RollDateField::End => self.end = value,
        }
    }

    /// The draft for a given field.
    fn get(&self, field: RollDateField) -> &str {
        match field {
            RollDateField::Start => &self.start,
            RollDateField::End => &self.end,
        }
    }
}

/// The selectable cell on the library page: either the leading Add Roll tile
/// (shown only while no search is active) or a real roll. Modeling both with a
/// single type makes the selection, highlight, keyboard navigation, and the
/// open action uniform across the grid — the add tile is selected and Entered
/// exactly like a roll card.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LibrarySelection {
    /// The Add Roll tile (grid cell 0 while no search is active). Enter /
    /// double-click opens the folder picker; it has no directory or metadata
    /// of its own.
    AddRoll,
    /// A real roll card, by its directory.
    Roll(PathBuf),
}

/// A film roll: a user-chosen directory of negatives, listed as a cover tile
/// on the library page. Double-clicking (or Enter on the selected roll) drills
/// into its frame grid.
#[derive(Debug, Clone)]
pub(crate) struct Roll {
    /// Absolute directory holding this roll's negatives.
    pub dir: PathBuf,
    /// Display name: the manifest's human-readable label when set, otherwise
    /// the directory's final component.
    pub name: String,
    /// The roll's cover file name (first sorted non-dot file), if any.
    pub cover: Option<String>,
    /// Number of regular non-dot files in the roll directory, surfaced in the
    /// roll-info metadata drawer.
    pub frame_count: usize,
    /// The film-inversion preset this roll's frames render with. The single
    /// in-memory source of truth for decodes; persisted to the roll's edit
    /// manifest (see [`edit_manifest::RollManifest::preset`]).
    pub preset: FilmPreset,
    /// How this roll's black point is resolved (preset base / auto per frame /
    /// auto selected frame). Orthogonal to the film choice; persisted to the
    /// roll's edit manifest (see [`edit_manifest::RollManifest::base_mode`]).
    pub base_mode: BaseMode,
    /// The roll's start date (ISO `YYYY-MM-DD`), if already set. Mirrors the
    /// edit manifest; display-only (no decode impact).
    pub start_date: Option<String>,
    /// The roll's optional end date (ISO `YYYY-MM-DD`), if already set.
    pub end_date: Option<String>,
    /// Decoded cover thumbnail state.
    pub thumb: Thumb,
}

/// A file entry displayed as a tile on the open roll's frame grid.
pub(crate) struct Tile {
    /// File name.
    pub(crate) name: String,
    /// Decoded thumbnail state.
    pub(crate) thumb: Thumb,
    /// Lazy EXIF/dimension readout for the frame-info drawer (parsed once per
    /// session); `None` until the first drawer request.
    pub(crate) meta: Option<FrameMeta>,
    /// Set when the file's metadata could not be parsed, so a failed read is
    /// not re-attempted on every drawer open.
    pub(crate) meta_failed: bool,
}

/// A decoded detail-view overview: the linear pre-sRGB mono buffer plus its
/// geometry, as delivered by [`decode_raw_detail`]. Exactly what an
/// [`shader::DetailProgram`] needs to (re)build without re-decoding
/// the RAW. Cached by the detail LRU keyed on (roll dir, film preset, file
/// name), so a preset change never serves a stale inversion.
#[derive(Debug, Clone)]
struct DetailMono {
    mono: Vec<f32>,
    width: u32,
    height: u32,
    /// The sensor's true long edge AFTER cropping but BEFORE the downscale, so
    /// a served cache entry can decide whether the overview was already native.
    src_long_edge: u32,
    /// Film-negative inversion carried through the cache: `Some((stock, base))`
    /// when the frame is a negative the shader density-inverts per fragment
    /// (`base` is the per-frame clear-film transmission measured from the
    /// sensor mono), `None` for an already-positive scan. Rebuilds an identical
    /// shader from a cache hit without re-decoding the RAW.
    inversion: Option<(MonoStock, f32)>,
}

/// A fixed-capacity least-recently-used map keyed by (roll dir, film preset,
/// file name).
///
/// Backed by a `HashMap` for O(1) lookup plus a `VecDeque` of keys as the
/// recency index: `get` moves the key to the back, `insert` pops the front
/// (least-recent) key once the capacity is exceeded and hands back the evicted
/// value so the caller can release (drop) its memory. Dependency-free and pure,
/// so the eviction order is unit-tested.
struct LruCache<K, V> {
    map: std::collections::HashMap<K, V>,
    order: std::collections::VecDeque<K>,
    capacity: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    fn new(capacity: usize) -> Self {
        LruCache {
            map: std::collections::HashMap::with_capacity(capacity),
            order: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the cached value for `key`, marking it most-recently used.
    fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.bump(key.clone());
        }
        self.map.get(key)
    }

    /// Inserts `value` under `key`, marking it most-recently used. Once the
    /// capacity is exceeded the least-recently-used entry is evicted and its
    /// value returned (so the caller can drop it); `None` when nothing was
    /// evicted. With a zero capacity every insert is immediately evicted and
    /// returned, so the cache stays empty.
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return Some(value);
        }
        let evicted = if !self.map.contains_key(&key) && self.map.len() >= self.capacity {
            self.order
                .pop_front()
                .and_then(|oldest| self.map.remove(&oldest))
        } else {
            None
        };
        self.map.insert(key.clone(), value);
        self.bump(key);
        evicted
    }

    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Drops every entry whose key no longer passes `keep`, returning the
    /// removed values so the caller can release (drop) them eagerly. The
    /// recency order of the surviving entries is preserved. Used to clear one
    /// roll's cached overviews when its film preset changes.
    fn retain(&mut self, mut keep: impl FnMut(&K) -> bool) -> Vec<V> {
        let mut dropped = Vec::new();
        self.order.retain(|key| {
            if keep(key) {
                true
            } else {
                if let Some(value) = self.map.remove(key) {
                    dropped.push(value);
                }
                false
            }
        });
        dropped
    }

    #[cfg(test)]
    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Moves `key` to the back of the recency order (most-recently used). If it
    /// was not present, keeps the order consistent by removing any duplicate.
    fn bump(&mut self, key: K) {
        if let Some(position) = self.order.iter().position(|k| k == &key) {
            self.order.remove(position);
        }
        self.order.push_back(key);
    }
}

/// Thumbnail loading state of a [`Tile`] or roll cover.
#[derive(Debug, Clone)]
pub(crate) enum Thumb {
    /// The file has not been decoded yet.
    Loading,
    /// A decoded RGBA thumbnail.
    Ready(Handle),
    /// The file could not be decoded.
    Failed,
}

/// Messages emitted by the application and its widgets.
#[derive(Debug, Clone)]
pub(crate) enum Message {
    /// Close the detail view, returning to the grid.
    DetailClosed,
    /// A hi-res decode for the detail view finished, returning the true
    /// sensor-linear mono buffer for the GPU shader. Carries the preset the
    /// decode ran under so the LRU is keyed consistently with the buffer
    /// contents.
    DetailReady(String, FilmPreset, Result<DetailDecode, FrameError>),
    /// A neighbor preload decode finished. Unlike [`Message::DetailReady`] this
    /// only lands into the detail LRU cache; it never becomes the active shader.
    DetailPreloaded(PathBuf, String, FilmPreset, Result<DetailDecode, FrameError>),
    /// The startup roll scan finished.
    RollsLoaded(Vec<Roll>),
    /// A single roll was scanned after being added; push it into the library.
    RollInfoLoaded(Roll),
    /// A roll cover decode finished.
    CoverReady(PathBuf, Result<Handle, FrameError>),
    /// The folder picker returned a roll directory to add, along with the
    /// film preset chosen for it in the dialog. The base strategy is not part
    /// of the dialog (the open dialog carries a single choice), so a new roll
    /// starts on the preset base and the user sets auto in the roll-info drawer.
    RollAdded(PathBuf, FilmPreset),
    /// The selected roll's film preset (which film inverts it) was changed from
    /// the roll-info drawer.
    RollPresetChanged(PathBuf, FilmPreset),
    /// The selected roll's base strategy (preset base / auto per frame / auto
    /// selected frame) was changed from the roll-info drawer.
    RollBaseModeChanged(PathBuf, BaseMode),
    /// The user asked to make the currently viewed frame the roll's
    /// auto-calibration frame: its clear-film plateau is measured once and
    /// recorded in the roll manifest as the black point every frame inverts
    /// against. Only meaningful while the roll's base mode is
    /// [`BaseMode::AutoSelectedFrame`].
    CalibrateBaseFromFrame,
    /// A background measurement of the roll's auto-calibration frame landed:
    /// `name` is the designated frame and `Option<f32>` its measured (and
    /// plausibility-filtered) clear-film transmission. Applied only while the
    /// roll still uses the `AutoSelectedFrame` base mode with that same frame.
    CalibrationBaseMeasured(PathBuf, String, Option<f32>),
    /// The user pressed the Add roll button.
    AddRoll,
    /// A roll tile was single-clicked — select it (highlight + metadata
    /// drawer target). Does not drill in.
    RollSelected(PathBuf),
    /// Remove a roll from the library (the library page owns the roll list).
    /// Non-destructive: drops the directory from the persisted config so it no
    /// longer shows as a roll card — the files and their edit manifest on disk
    /// are left untouched and the roll can be re-added later.
    RemoveRoll(PathBuf),
    /// Remove the roll currently selected in the library (menu-driven variant
    /// of [`Message::RemoveRoll`]); no-ops when no roll is selected.
    RemoveSelectedRoll,
    /// The Add Roll tile was single-clicked — select it (highlight). Does not
    /// open the folder picker; that still needs a double click or Enter.
    AddRollSelected,
    /// Open the library cell currently selected by a single click
    /// (`library_selection`), from Enter or a double click — a roll drills into
    /// its frame grid, the Add Roll tile opens the folder picker; on the frame
    /// page it opens the frame currently highlighted by `frame_selected`.
    OpenSelected,
    /// A roll tile was double-clicked — drill into its frame grid.
    RollActivated(PathBuf),
    /// Arrow-key navigation. On the library page it moves the roll selection;
    /// inside an open roll it moves the frame highlight, or pages the detail
    /// view left/right when one is open.
    Nav(MoveDir),
    /// A frame tile was single-clicked — select it (accent ring) without
    /// opening the detail view. Honors the current modifier state: a plain
    /// click clears and selects, Ctrl+click toggles membership, Shift+click
    /// selects the range from the anchor through this frame.
    FrameSelected(String),
    /// Select every frame in the open roll (Ctrl+A).
    SelectAllFrames,
    /// The held modifier keys changed (delivered as `ModifiersChanged` by the
    /// global keyboard subscription). Stored so frame clicks can distinguish
    /// plain/Ctrl/Shift selection (iced's `MouseArea` carries no modifier
    /// info).
    ModifiersChanged(keyboard::Modifiers),
    /// A grid scrollable reported its geometry (bounds, content height, current
    /// translation). Cached so keyboard navigation can reveal the highlighted
    /// tile by scrolling the grid when it moves out of the visible viewport.
    GridViewport(Viewport),
    /// The frame scan for an opened roll finished.
    RollOpened(PathBuf, Vec<String>),
    /// The frame-info drawer's lazy EXIF parse for a frame finished; carries
    /// the parsed metadata (or a failure, which is cached so it is not retried).
    FrameInfoReady(String, Result<FrameMeta, FrameError>),
    /// Activate the search field: reveal the header input (and focus it),
    /// mirroring cosmic-files' search icon toggle. No-op if already active.
    SearchActivate,
    /// Deactivate the search field via the input's clear button: hide the
    /// input and drop any term, returning the header to the search icon.
    SearchClear,
    /// The active search term changed (typed into the header input).
    SearchInput(String),
    /// A debounced search term is ready to apply (fired `SEARCH_DEBOUNCE_MS`
    /// after the last keystroke). The inner serial must match `search_version`
    /// or the commit is stale and is discarded.
    SearchCommitted(String, u64),
    LaunchUrl(String),
    /// A surface action from a menu popup (Wayland): forwarded to the cosmic
    /// runtime, which creates/destroys the popup surface backing the menus.
    Surface(cosmic::surface::Action),
    ThumbReady(String, Result<Handle, FrameError>),
    /// A thumbnail was double-clicked, opening it in the detail view.
    ThumbnailActivated(String),
    /// Animation tick driving the hi-res crossfade.
    DetailFadeTick,
    /// The user moved the exposure slider.
    ExposureChanged(f32),
    /// A keyboard shortcut adjusted one of the editing controls by a signed
    /// delta (positive = increase). Routes through the same RAM + live-shader
    /// path as the matching slider; no-op unless a detail view is open.
    AdjustEdit(EditAdjust),
    /// An editing-shortcut key was released: commit any keyboard adjustments
    /// made while it was held (persist + re-bake), mirroring the slider's
    /// release commit instead of committing per auto-repeated press.
    EditKeyReleased,
    /// Wheel-scroll zoom in the detail view; payload is the change in zoom
    /// units (log2 of the scale ratio), positive = zoom in, negative = out.
    DetailZoom(f32),
    /// The detail preview was (re)laid out; carries its new logical size. Used
    /// to recompute the 1:1 ("100%") zoom cap, which depends on the preview
    /// area's size (window resize, drawer open/close, …).
    DetailAreaResized(Size),
    /// The user pressed the mouse on the detail preview — grab-pan begins.
    DetailPanPress,
    /// The cursor moved over the detail preview; while panning this shifts
    /// the image. Carries the widget-relative cursor position (logical points).
    DetailPanMove(Point),
    /// The mouse was released or left the preview — grab-pan ends.
    DetailPanRelease,
    /// The live tone curve changed: new contrast, highlights, and shadows
    /// powers. Applies to the shader as a uniform-only remap.
    CurveChanged(f32, f32, f32),
    /// Reset every first-class edit (exposure + tone curve) to their
    /// identities in one action, and persist the reset like any other edit.
    ResetAll,
    /// Reset only the crop margins to zero (show the full frame), leaving the
    /// exposure and tone edits untouched, and persist the reset.
    ResetCrop,
    /// The user typed into a roll date field in the roll-info drawer. Carries
    /// the affected field and the new draft text (kept in RAM so typing
    /// doesn't fight a read-only view); nothing is committed until submit.
    RollDateDraftChange(RollDateField, String),
    /// A roll date field was submitted (Enter/return): validate the ISO
    /// `YYYY-MM-DD` draft (empty clears the date), persist it to the roll's
    /// manifest, and update the in-memory roll and card.
    RollDateDraftSubmit(RollDateField),
    /// The user typed into the roll-info drawer's editable name heading.
    /// Carries the new draft text (kept in RAM so typing doesn't fight a
    /// read-only view); nothing is committed until submit.
    RollNameDraftChange(String),
    /// The editable name heading was submitted (Enter/return): persist the
    /// trimmed draft (empty reverts to the directory leaf) to the roll's
    /// manifest and update the in-memory roll and card.
    RollNameDraftSubmit,
    /// Toggle the detail view's crop mode: a focused modal state (bare `c` or
    /// View → Crop mode) in which the surround renders white and the shader
    /// dims everything outside the crop, frame paging is disabled, and only
    /// the h/j/k/l crop trims act. A no-op outside a detail view.
    ToggleCropMode,
    /// Toggle the global keyboard-help overlay (bare `?`). While on, the rest
    /// of `update()` is gated so the covered UI ignores navigation/editing.
    ToggleHelp,
    /// Flush the in-memory roll edits to the manifest file on disk.
    EditSave,
    /// Copy the focused frame's full edit (exposure + tone curve) to the
    /// clipboard as a [`ToneEdit`] (Ctrl+C).
    CopyEdits,
    /// Paste the copied edit onto every multi-selected frame, or the focused
    /// frame when nothing else is selected (Ctrl+V).
    PasteEdits,
    /// Consume an input event without acting on it, blocking the grid
    /// beneath the detail view's input surface.
    Ignore,
    /// The user asked to export: open the native folder dialog (File → Export…
    /// or the bare `e` key).
    ExportRequested,
    /// One export frame finished: `done` of `total` frames are written (or
    /// skipped), driving the header's circular progress ring. Ticks may drop
    /// under channel backpressure without harm — the batch still completes.
    ExportProgress {
        done: usize,
        total: usize,
    },
    /// The folder dialog finished: `Some((dest, options))` exports the batch into
    /// `dest` with `options`, `None` means the dialog was cancelled.
    ExportChosen(Option<(PathBuf, ExportOptions)>),
    /// The export batch finished, carrying how many frames succeeded, how many
    /// were skipped (already present when overwrite was off), how many failed,
    /// the destination folder for the completion toast, and — when the roll had
    /// a start date — that date, so the toast can say the exports are stamped
    /// and dated.
    ExportDone {
        ok: usize,
        skipped: usize,
        failed: usize,
        dest: PathBuf,
        start_date: Option<String>,
    },
    /// A toast's duration elapsed (or its close button was pressed): dismiss
    /// it from the toaster.
    ToastClose(toaster::ToastId),
    /// Quit the application, persisting any open edits first.
    Quit,
    ToggleContextPage(ContextPage),
    /// Toggle the current page's context drawer (bare Space). Since the
    /// keyboard subscription's filter closure cannot capture app state, this
    /// defers the page choice to the update handler: editing while a detail
    /// view is open, roll info otherwise.
    ToggleContext,
    UpdateConfig(Config),
}

/// A direction for arrow-key navigation of a grid selection (library rolls or
/// open-roll frames), and for paging the detail view left/right.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MoveDir {
    Left,
    Right,
    Up,
    Down,
}

/// A single editing-control adjustment from a keyboard shortcut: which control
/// and the signed delta to apply (positive = increase). The step (coarse vs
/// nudge) is resolved at construction by `edit_adjust_for`, so the handler
/// only clamps against the control's range and applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EditAdjust {
    Exposure(f32),
    Contrast(f32),
    Highlights(f32),
    Shadows(f32),
    /// Rotate the display one quarter-turn counter-clockwise (a discrete step,
    /// no delta payload; unlike the numeric adjusts it doesn't hold-repeat a
    /// magnitude, but the edit-key machinery still commits on release).
    RotateCcw,
}

/// Create a COSMIC application from the app model
impl cosmic::Application for AppModel {
    /// The async executor that will be used to run your application's commands.
    type Executor = cosmic::executor::Default;

    /// Data that your application receives to its init method.
    type Flags = ();

    /// Messages which the application and its widgets will emit.
    type Message = Message;

    /// Unique identifier in RDNN (reverse domain name notation) format.
    const APP_ID: &'static str = "io.github.jpttrssn.curvectrl";

    fn core(&self) -> &cosmic::Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut cosmic::Core {
        &mut self.core
    }

    /// Initializes the application with any given flags and startup commands.
    #[allow(clippy::too_many_lines)] // Big struct literal; one field per line.
    fn init(
        mut core: cosmic::Core,
        _flags: Self::Flags,
    ) -> (Self, Task<cosmic::Action<Self::Message>>) {
        // Create the about widget
        let about = About::default()
            .name(fl!("app-title"))
            .icon(widget::icon::from_svg_bytes(APP_ICON))
            .version(env!("CARGO_PKG_VERSION"))
            .links([(fl!("repository"), REPOSITORY)])
            .license(env!("CARGO_PKG_LICENSE"));

        // The editing panel drawer sits beside the detail view as a side pane
        // that resizes the content, rather than overlaying it.
        core.window.context_is_overlay = false;

        // Construct the app model with the runtime's core.
        let mut app = AppModel {
            core,
            context_page: ContextPage::default(),
            drawer_memory: DrawerMemory::default(),
            about,
            key_binds: HashMap::from([
                (
                    menu::KeyBind {
                        modifiers: vec![],
                        key: keyboard::Key::Character(" ".into()),
                    },
                    MenuAction::Details,
                ),
                (
                    menu::KeyBind {
                        modifiers: vec![menu::key_bind::Modifier::Ctrl],
                        key: keyboard::Key::Character("a".into()),
                    },
                    MenuAction::SelectAll,
                ),
                (
                    menu::KeyBind {
                        modifiers: vec![menu::key_bind::Modifier::Ctrl],
                        key: keyboard::Key::Character("c".into()),
                    },
                    MenuAction::CopyEdits,
                ),
                (
                    menu::KeyBind {
                        modifiers: vec![menu::key_bind::Modifier::Ctrl],
                        key: keyboard::Key::Character("v".into()),
                    },
                    MenuAction::PasteEdits,
                ),
                // Bare `e` exports the current selection; the key_binds map has
                // hint text for the menu bar, while the true trigger lives in
                // the global keyboard subscription (NoModifier KeyBinds are
                // label-only in this cosmic menu API).
                (
                    menu::KeyBind {
                        modifiers: vec![],
                        key: keyboard::Key::Character("e".into()),
                    },
                    MenuAction::Export,
                ),
                // Bare `c` toggles crop mode; same label-only convention as
                // `e` above.
                (
                    menu::KeyBind {
                        modifiers: vec![],
                        key: keyboard::Key::Character("c".into()),
                    },
                    MenuAction::ToggleCropMode,
                ),
            ]),
            // Optional configuration file for an application.
            config: cosmic_config::Config::new(Self::APP_ID, Config::VERSION)
                .map(|context| match Config::get_entry(&context) {
                    Ok(config) => config,
                    Err((_errors, config)) => {
                        // for why in errors {
                        //     tracing::error!(%why, "error loading app config");
                        // }

                        config
                    }
                })
                .unwrap_or_default(),
            rolls: Vec::new(),
            active: None,
            library_selection: None,
            frame_selected: None,
            selected_frames: HashSet::new(),
            selection_anchor: None,
            modifiers: keyboard::Modifiers::default(),
            editing_key_held: false,
            // A 3-wide grid is a safe initial guess until the first resize.
            grid_cols: 3,
            // No viewport is known until the grid lays out and scrolls; the
            // resize handler estimates the height before that.
            grid_viewport: None,
            window_height: 600.0,
            cover_inflight: Vec::new(),
            // The localized short month names (month order 1–12) seed the
            // search aliases; English full/3-letter names fill the rest.
            month_aliases: MonthAliases::new(&[
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
            ]),
            search: None,
            search_filter: None,
            search_version: 0,
            tiles: Vec::new(),
            selected: None,
            thumb_inflight: Vec::new(),
            frame_meta_inflight: None,
            roll: RollManifest::default(),
            detail_shader: None,
            detail_inflight: None,
            detail_native_queued: false,
            detail_logged_failure: None,
            detail_thumb_opacity: 1.0,
            detail_last_frame: None,
            detail_thumb: None,
            detail_zoom: Zoom::CONTAIN,
            detail_area_size: None,
            detail_pan: Point::default(),
            detail_panning: false,
            detail_cursor: None,
            curve_contrast: 1.0,
            curve_highlights: 1.0,
            curve_shadows: 1.0,
            exposure_ev: edit_manifest::DEFAULT_EXPOSURE_EV,
            crop: edit_manifest::CropMargins::default(),
            roll_date_drafts: RollDateDrafts {
                key: None,
                start: String::new(),
                end: String::new(),
            },
            roll_name_draft: String::new(),
            rotation: 0,
            crop_mode: false,
            help_visible: false,
            reset_exposure_ev: edit_manifest::DEFAULT_EXPOSURE_EV,
            reset_curve_contrast: 1.0,
            reset_curve_highlights: 1.0,
            reset_curve_shadows: 1.0,
            reset_crop: edit_manifest::CropMargins::default(),
            reset_rotation: 0,
            clipboard: None,
            next_image_id: 0,
            detail_cache: LruCache::new(DETAIL_CACHE_CAPACITY),
            detail_preload_inflight: Vec::new(),
            toasts: toaster::Toasts::new(Message::ToastClose),
            export_pending: false,
            export_progress: None,
        };

        // Set the window title and scan the configured roll directories.
        let rolls = app.config.rolls.clone();
        let command = Task::batch([
            app.update_title(),
            cosmic::task::future(async { Message::RollsLoaded(load_rolls(rolls).await) }),
        ]);

        (app, command)
    }

    /// Track the content width so arrow-key navigation of the library grid can
    /// mirror iced's `Grid::fluid` column count exactly
    /// (`ceil((width + spacing) / (max_width + spacing))`), and remember the
    /// window height for the pre-scroll viewport estimate.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn on_window_resize(&mut self, _id: cosmic::iced::window::Id, width: f32, height: f32) {
        let spacing = f32::from(cosmic::theme::spacing().space_s);
        self.grid_cols = grid_num_cols(width, spacing).max(1);
        self.window_height = height;
    }

    /// Escape pressed outside any widget that captured it (routed here by
    /// libcosmic's `keyboard_nav` subscription). Dismisses the help overlay
    /// first (VIM-like "leave the modal"), then exits crop mode, then closes
    /// the roll / detail view (also clearing an active library search) via the
    /// shared `DetailClosed` path.
    fn on_escape(&mut self) -> Task<cosmic::Action<Self::Message>> {
        if self.help_visible {
            cosmic::task::message(Message::ToggleHelp)
        } else if self.crop_mode {
            cosmic::task::message(Message::ToggleCropMode)
        } else {
            cosmic::task::message(Message::DetailClosed)
        }
    }

    /// Ctrl+F pressed outside any widget that captured it (routed here by
    /// libcosmic's `keyboard_nav` subscription). Reveals and focuses the header
    /// search input, exactly like clicking the search icon.
    fn on_search(&mut self) -> Task<cosmic::Action<Self::Message>> {
        cosmic::task::message(Message::SearchActivate)
    }

    /// Elements to pack at the start of the header bar.
    #[allow(clippy::too_many_lines)] // the menu bar legitimately builds every item inline
    fn header_start(&self) -> Vec<Element<'_, Self::Message>> {
        // Remove roll is only actionable on the rolls view when a roll is
        // selected; inside a roll (or with the add tile / nothing selected) it
        // shows disabled on the menu.
        let roll_selected = self.active.is_none()
            && matches!(&self.library_selection, Some(LibrarySelection::Roll(_)));
        let remove_roll = if roll_selected {
            menu::Item::Button(fl!("menu-remove-roll"), None, MenuAction::RemoveRoll)
        } else {
            menu::Item::ButtonDisabled(fl!("menu-remove-roll"), None, MenuAction::RemoveRoll)
        };

        // Export is only actionable when there is at least one frame to write
        // inside the open roll; while a batch runs it is disabled (a second
        // export cannot overlap the running one).
        let exporting = self.export_progress.is_some();
        let export = if exporting || !self.has_export_targets() {
            menu::Item::ButtonDisabled(fl!("menu-export"), None, MenuAction::Export)
        } else {
            menu::Item::Button(fl!("menu-export"), None, MenuAction::Export)
        };

        let file_menu = menu::Tree::with_children(
            menu::root(fl!("menu-file")).apply(Element::from),
            menu::items(
                &self.key_binds,
                vec![
                    menu::Item::Button(fl!("menu-add-roll"), None, MenuAction::AddRoll),
                    remove_roll,
                    menu::Item::Divider,
                    export,
                    menu::Item::Divider,
                    menu::Item::Button(fl!("menu-quit"), None, MenuAction::Quit),
                ],
            ),
        );

        // Copy/paste edits are only actionable while a frame selection exists
        // (a focused/highlighted frame or an open detail view); on the library
        // page no frame is ever selected, even when a roll is. Paste
        // additionally needs a non-empty edit clipboard to apply.
        let frame_context = self.frame_selected.is_some() || self.selected.is_some();
        let copy_edits = if frame_context {
            menu::Item::Button(fl!("menu-copy-edits"), None, MenuAction::CopyEdits)
        } else {
            menu::Item::ButtonDisabled(fl!("menu-copy-edits"), None, MenuAction::CopyEdits)
        };
        let paste_edits = if frame_context && self.clipboard.is_some() {
            menu::Item::Button(fl!("menu-paste-edits"), None, MenuAction::PasteEdits)
        } else {
            menu::Item::ButtonDisabled(fl!("menu-paste-edits"), None, MenuAction::PasteEdits)
        };

// "Calibrate from this frame" designates the current frame as the
        // roll's auto-calibration frame — only meaningful under the
        // Auto-selected-frame base mode of an open roll. The handler no-ops
        // without an eligible frame, so the menu only gates on the base mode.
        let calibrate_enabled = self.active.is_some()
            && self.roll.base_mode() == BaseMode::AutoSelectedFrame;
        let calibrate_items = if calibrate_enabled {
            vec![
                menu::Item::Divider,
                menu::Item::Button(fl!("menu-calibrate-base"), None, MenuAction::CalibrateBase),
            ]
        } else {
            vec![
                menu::Item::Divider,
                menu::Item::ButtonDisabled(
                    fl!("menu-calibrate-base"),
                    None,
                    MenuAction::CalibrateBase,
                ),
            ]
        };

        let edit_menu = menu::Tree::with_children(
            menu::root(fl!("menu-edit")).apply(Element::from),
            menu::items(
                &self.key_binds,
                [
                    vec![
                        menu::Item::Button(fl!("menu-select-all"), None, MenuAction::SelectAll),
                        menu::Item::Divider,
                        copy_edits,
                        paste_edits,
                    ],
                    calibrate_items,
                ]
                .concat(),
            ),
        );

        // The generic Details item opens the context drawer: the editing panel
        // while a detail view is open, the frame-info drawer on the frames grid
        // with a highlighted frame, or the roll-info drawer on the library
        // page. It stays enabled exactly where the drawer can open — a detail
        // view, a highlighted frame inside a roll, or a roll selected on the
        // library page (the page guards make it a no-op everywhere else).
        let details_enabled = self.selected.is_some()
            || (self.active.is_some() && self.frame_selected.is_some())
            || (self.active.is_none()
                && matches!(&self.library_selection, Some(LibrarySelection::Roll(_))));
        let details = if details_enabled {
            menu::Item::Button(fl!("menu-details"), None, MenuAction::Details)
        } else {
            menu::Item::ButtonDisabled(fl!("menu-details"), None, MenuAction::Details)
        };

        let crop_mode = if self.selected.is_some() {
            menu::Item::Button(fl!("menu-crop-mode"), None, MenuAction::ToggleCropMode)
        } else {
            menu::Item::ButtonDisabled(fl!("menu-crop-mode"), None, MenuAction::ToggleCropMode)
        };

        let view_menu = menu::Tree::with_children(
            menu::root(fl!("menu-view")).apply(Element::from),
            menu::items(
                &self.key_binds,
                vec![
                    menu::Item::Button(fl!("about"), None, MenuAction::About),
                    details,
                    menu::Item::Divider,
                    crop_mode,
                ],
            ),
        );

        // Menu popups must be backed by real Wayland surfaces (and know which
        // window to anchor to), so the bar forwards surface actions to the
        // cosmic runtime — matching how cosmic-files wires its menu bar.
        let menu_bar = menu::bar(vec![file_menu, edit_menu, view_menu])
            .window_id_maybe(self.core().main_window_id())
            .on_surface_action(Message::Surface)
            .item_width(menu::ItemWidth::Uniform(250));

        vec![menu_bar.into()]
    }

    /// Elements to pack at the end of the header bar.
    fn header_end(&self) -> Vec<Element<'_, Self::Message>> {
        // The editing drawer is no longer toggled from a header button — it
        // opens automatically with the detail view (see `open_frame`) and is
        // hidden/revealed by the Space context-drawer toggle — so the header
        // end packs only the search control, the detail loading spinner, and
        // (while a batch runs) the export progress ring.

        // Search is library-only: it filters roll names and dates on the
        // library page and is hidden entirely while a roll is open (the frame
        // grid is never searched). Mirroring cosmic-files, the input is only
        // shown once search is active: an inactive state packs a search icon
        // that reveals (and focuses) the input, which then replaces the icon
        // until it is cleared.
        let mut end = Vec::new();
        if self.active.is_none() {
            let search: Element<'_, Message> = if let Some(term) = &self.search {
                cosmic::widget::text_input::search_input(fl!("search-rolls"), term)
                    .width(Length::Fixed(240.0))
                    .id(search_input_id())
                    .on_clear(Message::SearchClear)
                    .on_input(Message::SearchInput)
                    .into()
            } else {
                widget::button::icon(icon::from_name("system-search-symbolic"))
                    .tooltip(fl!("search-toggle"))
                    .on_press(Message::SearchActivate)
                    .padding(8)
                    .into()
            };
            end.push(search);
        }

        // The detail-view loading spinner sits at the far right of the header:
        // an indeterminate circular throbber that turns on while any hi-res
        // decode runs for the open detail view — the 2048 overview on open and
        // the native level-up on zoom — and off the moment that decode lands
        // or supersedes. Zero screen space when idle, matching the export ring
        // beside it. `selected` also gates it: `detail_inflight` is NOT cleared
        // when Escape closes the view (only roll-close does), so this keeps a
        // decode still in flight from spinning forever on the grid below.
        if self.selected.is_some() && self.detail_inflight.is_some() {
            let spinner = cosmic::widget::indeterminate_circular().size(18.0);
            let spinner = cosmic::widget::tooltip(
                spinner,
                widget::text(fl!("detail-loading")),
                cosmic::widget::tooltip::Position::Bottom,
            );
            end.push(spinner.into());
        }

        // The COSMIC-Files-style export indicator sits at the far right of the
        // header: a small circular determinate progress ring that exists only
        // while a batch is running — zero screen space when idle. Hovering
        // shows which frame the batch is on.
        if let Some((done, total)) = self.export_progress {
            let ring =
                cosmic::widget::determinate_circular(export_fraction(done, total)).size(18.0);
            let ring = cosmic::widget::tooltip(
                ring,
                widget::text(fl!("export-progress", done = done, total = total)),
                cosmic::widget::tooltip::Position::Bottom,
            );
            end.push(ring.into());
        }
        end
    }

    /// Display a context drawer if the context page is requested.
    fn context_drawer(&self) -> Option<context_drawer::ContextDrawer<'_, Self::Message>> {
        if !self.core.window.show_context {
            return None;
        }

        match self.context_page {
            ContextPage::About => Some(context_drawer::about(
                &self.about,
                |url| Message::LaunchUrl(url.to_string()),
                Message::ToggleContextPage(ContextPage::About),
            )),
            ContextPage::Editing => {
                // Without a selection there is nothing to edit.
                let name = self.selected.as_ref()?;

                Some(
                    context_drawer::context_drawer(
                        editing_panel(self),
                        Message::ToggleContextPage(ContextPage::Editing),
                    )
                    .title(name),
                )
            }
            ContextPage::RollInfo => {
                // Only meaningful on the library page with a roll selection:
                // the drawer shows metadata for that roll, so an active roll, a
                // stale, or a missing (or Add-Roll-tile) selection closes the
                // drawer rather than rendering an empty panel.
                if self.active.is_some() {
                    return None;
                }
                let LibrarySelection::Roll(dir) = self.library_selection.as_ref()? else {
                    return None;
                };
                let roll = self.rolls.iter().find(|roll| &roll.dir == dir)?;

                Some(
                    context_drawer::context_drawer(
                        roll_info_panel(
                            roll,
                            &self.roll_name_draft,
                            &self.roll_date_drafts.start,
                            &self.roll_date_drafts.end,
                        ),
                        Message::ToggleContextPage(ContextPage::RollInfo),
                    )
                    .title(fl!("roll-info-title")),
                )
            }
            ContextPage::FrameInfo => {
                // Only meaningful on the frames grid with a highlighted frame:
                // the drawer shows that frame's name/dimensions/EXIF, so an
                // open detail view, an inactive roll, or no highlight closes it.
                if self.selected.is_some() || self.active.is_none() {
                    return None;
                }
                let name = self.frame_selected.as_ref()?;
                if !self.tiles.iter().any(|tile| &tile.name == name) {
                    return None;
                }

                Some(
                    context_drawer::context_drawer(
                        frame_info_panel(self, name),
                        Message::ToggleContextPage(ContextPage::FrameInfo),
                    )
                    .title(name),
                )
            }
        }
    }

    /// Describes the interface based on the current state of the application model.
    ///
    /// Application events will be processed through the view. Any messages emitted by
    /// events received by widgets will be passed to the update method.
    fn view(&self) -> Element<'_, Self::Message> {
        // The page body holds only the content: there is no in-roll toolbar
        // (the back-to-rolls button was removed — Esc is the way out; search
        // lives in the header, and Add Roll is the first library tile). The
        // column wrapper also applies the Fill sizing the views otherwise
        // shrink to.
        let content: Element<_> = match self.active.as_deref() {
            Some(_) => frames_view(self),
            None => library_view(self),
        };

        let content: Element<_> = widget::column::with_capacity(1)
            .push(content)
            .spacing(cosmic::theme::spacing().space_s)
            .height(Length::Fill)
            .width(Length::Fill)
            .into();

        // Overlay the toaster (completion toasts) on top of the whole window,
        // then the global help overlay above everything when it is open.
        let content = toaster::toaster(&self.toasts, content);
        if self.help_visible {
            Stack::with_children([content, help_overlay(self)]).into()
        } else {
            content
        }
    }

    /// Register subscriptions for this application.
    ///
    /// Subscriptions are long-running async tasks running in the background which
    /// emit messages to the application through a channel. They can be dynamically
    /// stopped and started conditionally based on application state, or persist
    /// indefinitely.
    #[allow(clippy::too_many_lines)] // each subscription arm is a compact match guard
    fn subscription(&self) -> Subscription<Self::Message> {
        // Add subscriptions which are always active.
        let mut subscriptions = vec![
            // Escape, Ctrl+F, Tab, and F11 are handled by libcosmic's built-in
            // `keyboard_nav` subscription (routed to `on_escape`/`on_search`),
            // so this subscription only maps the app-specific keys below.
            //
            // `Subscription::filter_map` requires a NON-CAPTURING (zero-sized)
            // closure, so the crop-mode routing below cannot read `self` — it
            // maps key events unconditionally and the message handlers do the
            // crop-mode gating (`Message::Nav` and `Message::AdjustEdit`).
            keyboard::listen().filter_map(|event| match event {
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Named(named),
                    ..
                } => match named {
                    keyboard::key::Named::Enter => Some(Message::OpenSelected),
                    keyboard::key::Named::ArrowLeft => Some(Message::Nav(MoveDir::Left)),
                    keyboard::key::Named::ArrowRight => Some(Message::Nav(MoveDir::Right)),
                    keyboard::key::Named::ArrowUp => Some(Message::Nav(MoveDir::Up)),
                    keyboard::key::Named::ArrowDown => Some(Message::Nav(MoveDir::Down)),
                    _ => None,
                },
                // Track the held modifier state (Ctrl/Shift/Alt) delivered by
                // the platform so frame clicks can decide their selection
                // behavior — iced's `MouseArea` carries no modifier info.
                keyboard::Event::ModifiersChanged(modifiers) => {
                    Some(Message::ModifiersChanged(modifiers))
                }
                // The spacebar carries no Named variant in this iced fork, so it arrives
                // as a character — matched by payload. A bare space (no
                // modifiers) toggles the active page's context drawer, which
                // replaced the old Ctrl+Space binding (the full-screen
                // preview feature that used bare Space was removed).
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == " " => Some(Message::ToggleContext),
                // Ctrl+A selects every frame in the open roll.
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if modifiers.control() && character == "a" => Some(Message::SelectAllFrames),
                // Ctrl+C / Ctrl+V copy and paste edits. The handlers no-op
                // when there is nothing focused to copy (or nothing copied to
                // paste), and a focused text input captures these keys first
                // (so search-field copy/paste is unaffected).
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if modifiers.control() && character == "c" => Some(Message::CopyEdits),
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if modifiers.control() && character == "v" => Some(Message::PasteEdits),
                // Bare `e` (no modifiers) opens the export destination picker
                // for the current selection. Reached only when no focused text
                // input swallowed the key first (search/crop fields capture
                // bare characters), and placed before the editing-key wildcard
                // so `e` never routes into an edit adjust.
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == "e" => Some(Message::ExportRequested),
                // Bare `c` toggles crop mode — the focused, VIM-like modal
                // state for trimming the crop with h/j/k/l. Placed before the
                // editing-key wildcard and the arrow-nav arms so `c` never
                // routes into an edit adjust; the handler no-ops outside a
                // detail view. Auto-repeat is ignored so a held `c` toggles
                // once rather than flapping the mode.
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    repeat,
                    ..
                } if !modifiers.control() && !repeat && character == "c" => {
                    Some(Message::ToggleCropMode)
                }
                // `?` toggles the global keyboard-help overlay. This iced fork
                // delivers `key_without_modifiers`, so Shift+`/` arrives as `/`
                // with Shift set. Gated on `!repeat` so a held `?` toggles once.
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    repeat,
                    ..
                } if !modifiers.control() && !repeat && modifiers.shift() && character == "/" => {
                    Some(Message::ToggleHelp)
                }
                // VIM-style navigation: bare `h`/`j`/`k`/`l` mirror the arrow keys via the
                // same `Nav` message — navigating outside crop mode, moving the
                // crop window inside it (Shift handled by `Nav`; repeats like
                // arrows).
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == "h" => Some(Message::Nav(MoveDir::Left)),
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == "j" => Some(Message::Nav(MoveDir::Down)),
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == "k" => Some(Message::Nav(MoveDir::Up)),
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() && character == "l" => Some(Message::Nav(MoveDir::Right)),
                // Editing shortcuts: bare (no Ctrl) keys that map to one of the
                // editing controls; holding Shift switches to the fine nudge
                // step. Mapped unconditionally — the `AdjustEdit` handler gates
                // on crop mode (only h/j/k/l crop trims pass through while it
                // is on, and they are inert outside it) and no-ops without a
                // detail view.
                keyboard::Event::KeyPressed {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control() => {
                    edit_adjust_for(character.as_str(), modifiers.alt(), modifiers.shift())
                        .map(Message::AdjustEdit)
                }
                // Releasing an editing-shortcut key ends the hold: commit the
                // adjustments made while it was down (one persist + re-bake per
                // hold, not per auto-repeated press). Only fires for keys that
                // actually map to an editing control, so releasing a modifier
                // or a non-edit character stays a no-op. The update handler
                // ignores the release if no editing key is recorded as held.
                keyboard::Event::KeyReleased {
                    key: keyboard::Key::Character(character),
                    modifiers,
                    ..
                } if !modifiers.control()
                    && edit_adjust_for(character.as_str(), modifiers.alt(), modifiers.shift())
                        .is_some() =>
                {
                    Some(Message::EditKeyReleased)
                }
                _ => None,
            }),
            // Watch for application configuration changes.
            self.core()
                .watch_config::<Config>(Self::APP_ID)
                .map(|update| {
                    // for why in update.errors {
                    //     tracing::error!(?why, "app config error");
                    // }

                    Message::UpdateConfig(update.config)
                }),
        ];

        // Drive the thumbnail crossfade while opacity is ramping down.
        if self.selected.is_some()
            && self.detail_shader.is_some()
            && self.detail_thumb.is_some()
            && self.detail_thumb_opacity > 0.0
        {
            subscriptions.push(
                cosmic::iced::time::every(std::time::Duration::from_millis(16))
                    .map(|_| Message::DetailFadeTick),
            );
        }

        Subscription::batch(subscriptions)
    }

    /// Handles messages emitted by the application and its widgets.
    ///
    /// Tasks may be returned for asynchronous execution of code in the background
    /// on the application's async runtime.
    #[allow(clippy::too_many_lines)] // Message dispatch; arms stay inline for readability.
    fn update(&mut self, message: Self::Message) -> Task<cosmic::Action<Self::Message>> {
        // Reseed the roll-info drawer's date drafts whenever the library roll
        // selection changes (a different roll always starts from its own
        // committed dates; stale uncommitted text is dropped). Runs before
        // dispatch so every path — navigation, Space, direct selection — is
        // covered by one guard.
        self.sync_roll_date_drafts();

        // While the global help overlay is shown, it is modal: only `?`/Escape
        // (ToggleHelp), the plumbing that must keep flowing underneath
        // (modifier/viewport/resize tracking, decode + frame-info + cover
        // landings, the fade tick, toasts, config, export progress), and the
        // inert `Ignore` are dispatched; every other input (navigation, open,
        // Space, editing, crop, export) is swallowed so it cannot act on the
        // covered UI.
        if self.help_visible
            && !matches!(
                message,
                Message::ToggleHelp
                    | Message::Ignore
                    | Message::ModifiersChanged(_)
                    | Message::GridViewport(_)
                    | Message::DetailAreaResized(_)
                    | Message::ThumbReady(_, _)
                    | Message::RollOpened(_, _)
                    | Message::CoverReady(_, _)
                    | Message::DetailReady(_, _, _)
                    | Message::DetailPreloaded(_, _, _, _)
                    | Message::FrameInfoReady(_, _)
                    | Message::RollInfoLoaded(_)
                    | Message::RollsLoaded(_)
                    | Message::DetailFadeTick
                    | Message::ToastClose(_)
                    | Message::UpdateConfig(_)
                    | Message::ExportProgress { .. }
                    | Message::ExportDone { .. }
            )
        {
            return Task::none();
        }

        match message {
            Message::DetailClosed => {
                self.persist_roll();
                // Escape first deactivates an active search (hiding the header
                // input back to the search icon) before any other close
                // behavior, mirroring cosmic-files. Search is library-only, so
                // this applies only on the library page — inside a roll the
                // term is left intact for the return to the library.
                if self.active.is_none() && self.search.is_some() {
                    self.search = None;
                    self.search_filter = None;
                    self.search_version = self.search_version.wrapping_add(1);
                    return Task::none();
                }
                // Escape never closes a context drawer (only Space toggles it).
                // On the library page it is a no-op beyond the search handling
                // above.
                if self.active.is_none() {
                    return Task::none();
                }
                if self.selected.is_some() {
                    // Close the detail view immediately — the frame highlight
                    // survives so the grid still shows where you were. The
                    // editing drawer is not closed; the grid's own remembered
                    // drawer state takes over.
                    self.selected = None;
                    self.clear_detail();
                    self.restore_drawer_for(DrawerView::Grid);
                    return self.ensure_frame_info_loaded().unwrap_or_else(Task::none);
                }
                // On the bare grid Escape backs out of the roll entirely,
                // resetting every detail- and roll-page field so nothing from
                // the closed roll leaks into the library (the frame highlight
                // is dropped as part of the reset; edits were already flushed
                // by `persist_roll` at the top of this arm).
                self.active = None;
                self.selected = None;
                self.frame_selected = None;
                self.selected_frames.clear();
                self.selection_anchor = None;
                self.grid_viewport = None;
                self.tiles = Vec::new();
                self.frame_meta_inflight = None;
                self.thumb_inflight.clear();
                self.detail_inflight = None;
                self.detail_preload_inflight.clear();
                // The RAM manifest is dropped with the roll; re-opened rolls
                // re-load it (see `RollOpened`). The overview LRU is
                // deliberately kept: it survives roll switches by design.
                self.roll = edit_manifest::RollManifest::default();
                self.clear_detail();
                self.restore_drawer_for(DrawerView::Library);
                Task::none()
            }

            Message::DetailReady(name, preset, result) => {
                self.handle_detail_ready(&name, preset, result)
            }

            Message::DetailPreloaded(dir, name, preset, result) => {
                self.handle_detail_preloaded(&dir, &name, preset, result)
            }

            Message::ThumbnailActivated(name) => self.open_frame(&name),

            Message::DetailFadeTick => {
                let dt = self
                    .detail_last_frame
                    .map_or(0.0, |t| t.elapsed().as_secs_f32().min(0.1));
                self.detail_last_frame = Some(Instant::now());
                self.detail_thumb_opacity = (self.detail_thumb_opacity - dt * FADE_SPEED).max(0.0);
                // Crossfade complete — drop the thumbnail cache.
                if self.detail_thumb_opacity <= 0.0 {
                    self.detail_thumb = None;
                }
                Task::none()
            }

            Message::ExposureChanged(ev) => {
                // RAM-only until an edit flush point; the shader stays live.
                // Shared with the keyboard `AdjustEdit` path via `set_exposure`.
                self.set_exposure(ev);
                Task::none()
            }

            Message::AdjustEdit(adjust) => {
                // Keyboard shortcuts only make sense while a detail view (and
                // its editing controls) are on screen.
                //
                // Crop mode is a focused modal state: the h/j/k/l keys MOVE the
                // crop window, and `-`/`=` (which normally adjust exposure)
                // RESIZE it about its center. Every OTHER edit key is inert
                // inside it. Blocked adjusts never set `editing_key_held`, so a
                // later key release stays a no-op instead of committing a stale
                // hold.
                if self.crop_mode {
                    let step = if self.modifiers.shift() {
                        CROP_NUDGE_PX
                    } else {
                        CROP_RESIZE_STEP_PX
                    };
                    // `-`/`=` arrive as `Exposure(∓ev)`; the sign picks the
                    // resize direction (negative shrinks, positive grows) and
                    // the step/nudge comes from the Shift state. (Crop-window
                    // MOVE is not an edit adjust — `h`/`j`/`k`/`l` route to
                    // `Message::Nav`, like the arrows.)
                    if let EditAdjust::Exposure(delta) = adjust {
                        self.editing_key_held = true;
                        let delta = if delta < 0.0 { -step } else { step };
                        self.apply_crop_resize(delta);
                    }
                    return Task::none();
                }
                // Every other adjust acts as usual.
                self.editing_key_held = true;
                self.apply_edit_adjust(adjust);
                Task::none()
            }

            Message::EditKeyReleased => {
                // The held editing key was released: commit the adjustments
                // made while it was down exactly once (persist + re-bake tile
                // + cover), mirroring the slider's `EditSave` on release. If
                // nothing was held (stray release, or a release after the
                // detail view closed) this is a no-op.
                if self.editing_key_held {
                    self.editing_key_held = false;
                    self.commit_edit()
                } else {
                    Task::none()
                }
            }

            Message::DetailZoom(delta) => {
                let (new_zoom, new_pan) = apply_detail_zoom(
                    self.detail_zoom.0,
                    self.detail_pan,
                    self.detail_cursor,
                    delta,
                    self.max_detail_zoom_for_wheel(),
                );
                self.detail_zoom = Zoom(new_zoom);
                self.detail_pan = new_pan;
                if let Some(shader) = &mut self.detail_shader {
                    shader.set_view(new_zoom, new_pan);
                }
                // Level up to native once the view crosses the trigger zoom;
                // the trigger adapts to the preview size so it always stays
                // reachable (see `native_level_up_zoom`). The inflight gate
                // keeps wheels that keep zooming through a running decode
                // (up to the projected native cap) from re-entering the busy
                // slot — the landing re-pump covers a zoom taken mid-load.
                if self.detail_inflight.is_none()
                    && self.detail_zoom.0 >= self.native_level_up_zoom()
                    && !self.detail_native_queued
                {
                    return self.decode_detail_next();
                }
                Task::none()
            }

            Message::DetailAreaResized(size) => {
                self.detail_area_size = Some(size);
                self.reclamp_detail_zoom();
                Task::none()
            }

            Message::DetailPanPress => {
                self.detail_panning = true;
                Task::none()
            }

            Message::DetailPanMove(pt) => {
                let prev = self.detail_cursor;
                self.detail_cursor = Some(pt);
                if self.detail_panning
                    && let Some(prev) = prev
                {
                    // Grab-pan: the image follows the cursor 1:1 in logical
                    // points, independent of zoom.
                    self.detail_pan.x += pt.x - prev.x;
                    self.detail_pan.y += pt.y - prev.y;
                    if let Some(shader) = &mut self.detail_shader {
                        shader.set_view(self.detail_zoom.0, self.detail_pan);
                    }
                }
                Task::none()
            }

            Message::DetailPanRelease => {
                self.detail_panning = false;
                Task::none()
            }

            Message::CurveChanged(contrast, highlights, shadows) => {
                // RAM-only until an edit flush point (slider `on_release`,
                // `DetailClosed`, window close) — same lifecycle as exposure.
                // Shared with the keyboard `AdjustEdit` path via `set_curve`.
                self.set_curve(contrast, highlights, shadows);
                Task::none()
            }

            Message::ResetAll => {
                // Reset every persisted edit back to the state when the edit
                // panel was opened (the stored manifest values for the
                // selected file), in RAM, push the live shader back to that
                // state, and let the normal flush points persist it (the
                // button is one unconscious click away from a slider, so
                // mirror the sliders' choice of only-mutate-RAM: the same
                // `EditSave` points persist it).
                self.exposure_ev = self.reset_exposure_ev;
                self.curve_contrast = self.reset_curve_contrast;
                self.curve_highlights = self.reset_curve_highlights;
                self.curve_shadows = self.reset_curve_shadows;
                self.crop = self.reset_crop;
                self.rotation = self.reset_rotation;
                if let Some(selected) = &self.selected {
                    self.roll.set_exposure(selected, self.reset_exposure_ev);
                    self.roll.set_curve(
                        selected,
                        self.reset_curve_contrast,
                        self.reset_curve_highlights,
                        self.reset_curve_shadows,
                    );
                    self.roll.set_crop(selected, self.reset_crop);
                    self.roll.set_rotation(selected, self.reset_rotation);
                }
                if let Some(shader) = &mut self.detail_shader {
                    shader.set_exposure(self.reset_exposure_ev);
                    shader.set_curve(
                        self.reset_curve_contrast,
                        self.reset_curve_highlights,
                        self.reset_curve_shadows,
                    );
                    shader.set_crop(self.reset_crop);
                    shader.set_rotation(self.reset_rotation);
                }
                self.reclamp_detail_zoom();
                Task::none()
            }

            // Reset ONLY the crop to zero (full frame), leaving exposure/tone
            // untouched, and persist the reset immediately.
            Message::ResetCrop => {
                self.crop = edit_manifest::CropMargins::default();
                if let Some(selected) = &self.selected {
                    self.roll.set_crop(selected, self.crop);
                }
                if let Some(shader) = &mut self.detail_shader {
                    shader.set_crop(self.crop);
                }
                self.reclamp_detail_zoom();
                self.commit_edit()
            }

            // A roll date field in the roll-info drawer gained a keystroke:
            // keep the draft text in RAM (never touching the committed date)
            // so the read-only view can re-render it. Commit happens only on
            // submit.
            Message::RollDateDraftChange(field, value) => {
                self.roll_date_drafts.set(field, value);
                Task::none()
            }

            // A roll date field was submitted (Enter/return): an empty draft
            // clears the date, a valid ISO `YYYY-MM-DD` draft commits it
            // (normalized), an invalid one is dropped by re-seeding the drafts
            // from the committed dates. The in-memory roll (and thus the
            // library card) is updated and the manifest is flushed.
            Message::RollDateDraftSubmit(field) => {
                let draft = self.roll_date_drafts.get(field).trim().to_owned();
                if !draft.is_empty() && !valid_iso_date(&draft) {
                    self.sync_roll_date_drafts();
                    return Task::none();
                }
                let Some(LibrarySelection::Roll(dir)) = self.library_selection.clone() else {
                    self.sync_roll_date_drafts();
                    return Task::none();
                };
                let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) else {
                    self.sync_roll_date_drafts();
                    return Task::none();
                };
                let next = if draft.is_empty() { None } else { Some(draft) };
                // A committed date must not contradict the other one: the roll
                // may not end before it starts (ending the export-relevant
                // start/end coherence). A conflicting submit is dropped by
                // re-seeding the drafts, like a malformed date.
                let coherent = match field {
                    RollDateField::Start => {
                        roll_dates_valid(next.as_deref(), roll.end_date.as_deref())
                    }
                    RollDateField::End => {
                        roll_dates_valid(roll.start_date.as_deref(), next.as_deref())
                    }
                };
                if !coherent {
                    self.sync_roll_date_drafts();
                    return Task::none();
                }
                match field {
                    RollDateField::Start => roll.start_date = next,
                    RollDateField::End => roll.end_date = next,
                }
                record_roll_dates(&roll.dir, roll.start_date.clone(), roll.end_date.clone());
                self.roll_date_drafts = RollDateDrafts::from_dates(
                    dir,
                    roll.start_date.as_deref(),
                    roll.end_date.as_deref(),
                );
                Task::none()
            }

            // The roll-info drawer's editable name heading gained a keystroke:
            // keep the draft text in RAM (never touching the committed name) so
            // the read-only view can re-render it. Commit happens only on submit.
            Message::RollNameDraftChange(value) => {
                self.roll_name_draft = value;
                Task::none()
            }

            // The editable heading was submitted (Enter/return): the trimmed
            // draft becomes the roll's display name — an empty draft clears
            // the label so the roll falls back to its directory leaf. The
            // in-memory roll (and thus the library card, search, and sort) is
            // updated and the manifest is flushed.
            Message::RollNameDraftSubmit => {
                let draft = self.roll_name_draft.trim().to_owned();
                let Some(LibrarySelection::Roll(dir)) = self.library_selection.clone() else {
                    self.sync_roll_date_drafts();
                    return Task::none();
                };
                let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) else {
                    self.sync_roll_date_drafts();
                    return Task::none();
                };
                roll.name = if draft.is_empty() {
                    // Clear the label: the directory leaf is derived at load.
                    record_roll_name(&roll.dir, None);
                    dir.file_name()
                        .and_then(|name| name.to_str())
                        .map_or_else(|| roll.name.clone(), str::to_owned)
                } else {
                    record_roll_name(&roll.dir, Some(draft.clone()));
                    draft
                };
                self.roll_name_draft = roll.name.clone();
                Task::none()
            }

            Message::ToggleCropMode => {
                // Crop mode only means anything over the detail view; without
                // one the toggle is a no-op (the bare `c` key reaches it from
                // any page).
                if let Some(shader) = &mut self.detail_shader {
                    self.crop_mode = !self.crop_mode;
                    shader.set_show_mask(self.crop_mode);
                    shader.set_pad(if self.crop_mode {
                        CROP_MODE_PADDING
                    } else {
                        0.0
                    });
                }
                // The minimum padding shrinks the contain-fit base, which
                // changes the 1:1 zoom cap; pull the current zoom back into
                // range if the cap dropped.
                self.reclamp_detail_zoom();
                // Leaving crop mode commits the crop session (persist + re-bake
                // the tile/cover), so any trims made while the mode was on land
                // on disk exactly when it closes — via `c`, Escape (routed here
                // by `on_escape`), or the View menu checkbox.
                if self.crop_mode {
                    Task::none()
                } else {
                    self.commit_edit()
                }
            }

            Message::ToggleHelp => {
                self.help_visible = !self.help_visible;
                Task::none()
            }

            Message::RollsLoaded(rolls) => {
                self.rolls = rolls;
                // A refresh supersedes any earlier cover chain.
                self.cover_inflight.clear();
                // A roll selection pointing at a roll that left the config is
                // stale; drop it and close a metadata drawer showing it. An Add
                // Roll selection always stays valid (the tile is always present).
                if matches!(
                    &self.library_selection,
                    Some(LibrarySelection::Roll(dir)) if !self.rolls.iter().any(|roll| &roll.dir == dir)
                ) {
                    self.library_selection = None;
                    if self.context_page == ContextPage::RollInfo {
                        self.core_mut().set_show_context(false);
                        self.drawer_memory.set(DrawerView::Library, false);
                    }
                }
                self.select_first_visible_roll();
                self.decode_covers()
            }

            Message::RollInfoLoaded(roll) => {
                if self.rolls.iter().any(|existing| existing.dir == roll.dir) {
                    return Task::none();
                }
                self.rolls.push(roll);
                self.rolls.sort_by(|a, b| a.name.cmp(&b.name));
                // The selection already points at the new roll (`RollAdded`),
                // so the card just lands alongside the others.
                self.decode_covers()
            }

            Message::CoverReady(dir, result) => {
                if let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) {
                    roll.thumb = match result {
                        Ok(handle) => Thumb::Ready(handle),
                        Err(_) => Thumb::Failed,
                    };
                }
                self.cover_inflight.retain(|pending| pending != &dir);
                self.decode_covers()
            }

            Message::AddRoll => open_roll_picker(),

            Message::RollAdded(dir, preset) => {
                // The library list is app-wide: persist the new roll no matter
                // where the folder picker was invoked from.
                if !self.rolls.iter().any(|roll| roll.dir == dir) {
                    self.config.rolls.push(dir.to_string_lossy().into_owned());
                    self.persist_config();
                }
                // Record the chosen preset. The manifest is the persistence
                // layer and the in-memory roll (on a re-add) the decode source:
                // writing first means the just-spawned scan reads it back. The
                // base strategy starts at the default (Preset), settable from
                // the roll-info drawer.
                record_roll_preset(&dir, preset);
                if let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) {
                    roll.preset = preset;
                    roll.thumb = Thumb::Loading;
                }
                // Mark the new roll selected (the selection survives roll exit,
                // so backing out lands on it). The cover scan for the card
                // continues in parallel.
                self.set_library_selection(LibrarySelection::Roll(dir.clone()));
                let scan_dir = dir.clone();
                let card = cosmic::task::future(async move {
                    Message::RollInfoLoaded(load_roll(scan_dir).await)
                });
                // Drill straight into the new roll's frame grid, from wherever
                // the app was when the roller was chosen.
                let open = self.open_roll(dir);
                Task::batch([card, open, self.decode_covers()])
            }

            Message::RollPresetChanged(dir, preset) => {
                record_roll_preset(&dir, preset);
                if let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) {
                    roll.preset = preset;
                    roll.thumb = Thumb::Loading;
                }
                let mut tasks = vec![self.decode_covers()];
                // The drawer is library-only today, so the active branch is a
                // defensive re-bake (kept so a film change can never render a
                // stale inversion for an open roll). A film change leaves the
                // base strategy (and any calibration) untouched.
                if self.active.as_deref() == Some(dir.as_path()) {
                    self.roll.set_preset(preset);
                    tasks.extend(self.reflow_roll_after_change(&dir));
                }
                Task::batch(tasks)
            }

            Message::RollBaseModeChanged(dir, mode) => {
                record_roll_base_mode(&dir, mode);
                if let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) {
                    roll.base_mode = mode;
                    roll.thumb = Thumb::Loading;
                }
                let mut tasks = vec![self.decode_covers()];
                if mode == BaseMode::AutoSelectedFrame {
                    // The auto-selected-frame base mode designates ONE frame
                    // whose clear-film plateau becomes the roll's black point.
                    // Default it to the roll's first sorted frame (mirroring
                    // the cover selection) and measure its base so the mode
                    // works out of the box; `apply_calibration_measurement`
                    // stores the value.
                    if let Some(first) = self.roll_first_frame(&dir) {
                        let mut manifest = edit_manifest::load_roll_manifest(&dir);
                        if manifest.calibration_frame().is_none() {
                            manifest.set_calibration_frame(&first);
                            if let Err(err) = edit_manifest::save_roll_manifest(&dir, &manifest) {
                                log::error!(
                                    "failed to write roll manifest {}: {err}",
                                    edit_manifest::manifest_path(&dir).display()
                                );
                            }
                        }
                        if self.active.as_deref() == Some(dir.as_path())
                            && self.roll.calibration_frame().is_none()
                        {
                            self.roll.set_calibration_frame(&first);
                        }
                        tasks.push(Self::measure_calibration_frame(dir.clone()));
                    }
                } else {
                    // Leaving the auto-selected-frame base mode drops the
                    // calibration reference: a stale frame/base must not
                    // linger for the other modes.
                    let mut manifest = edit_manifest::load_roll_manifest(&dir);
                    if manifest.calibration_frame().is_some()
                        || manifest.calibrated_base().is_some()
                    {
                        manifest.clear_calibration();
                        if let Err(err) = edit_manifest::save_roll_manifest(&dir, &manifest) {
                            log::error!(
                                "failed to write roll manifest {}: {err}",
                                edit_manifest::manifest_path(&dir).display()
                            );
                        }
                    }
                    if self.active.as_deref() == Some(dir.as_path()) {
                        self.roll.clear_calibration();
                    }
                }
                if self.active.as_deref() == Some(dir.as_path()) {
                    self.roll.set_base_mode(mode);
                    tasks.extend(self.reflow_roll_after_change(&dir));
                }
                Task::batch(tasks)
            }

            Message::CalibrateBaseFromFrame => self.calibrate_base_from_frame(),

            Message::CalibrationBaseMeasured(dir, name, base) => {
                self.apply_calibration_measurement(&dir, &name, base)
            }

            Message::RollSelected(dir) => {
                self.set_library_selection(LibrarySelection::Roll(dir));
                Task::none()
            }

            Message::AddRollSelected => {
                self.set_library_selection(LibrarySelection::AddRoll);
                Task::none()
            }

            Message::RemoveRoll(dir) => self.remove_roll(&dir),

            Message::RemoveSelectedRoll => match self.library_selection.clone() {
                Some(LibrarySelection::Roll(dir)) => self.remove_roll(&dir),
                _ => Task::none(),
            },

            Message::OpenSelected => {
                if self.active.is_some() {
                    // Frame page: open the highlighted frame in the detail view.
                    if let Some(name) = self.frame_selected.clone() {
                        self.open_frame(&name)
                    } else {
                        Task::none()
                    }
                } else {
                    match self.library_selection.clone() {
                        // A roll drills in; the Add Roll tile opens the folder
                        // picker, mirroring how Enter opens a selected roll.
                        Some(LibrarySelection::Roll(dir)) => self.open_roll(dir),
                        Some(LibrarySelection::AddRoll) => open_roll_picker(),
                        None => Task::none(),
                    }
                }
            }

            Message::Nav(dir) => {
                // Crop mode is a focused, VIM-like modal state: arrow keys MOVE
                // the crop window (Left/Right/Up/Down) instead of paging frames
                // or moving a grid highlight, which stay disabled while it is
                // on. Outside crop mode the arrows page the detail view or move
                // the grid highlight as usual.
                if self.crop_mode {
                    // The move direction is expressed visually: pressing ← moves
                    // the crop window left on screen even when the frame is
                    // rotated (`crop_move_direction` compensates for the CCW
                    // display rotation).
                    let direction = crop_move_direction(dir, self.rotation);
                    let delta = if self.modifiers.shift() {
                        CROP_NUDGE_PX
                    } else {
                        CROP_STEP_PX
                    };
                    self.editing_key_held = true;
                    self.apply_crop_move(direction, delta);
                    return Task::none();
                }
                if self.active.is_some() {
                    // Inside a roll. With the detail view open, Left/Right page
                    // through the frames; on the bare grid they move the
                    // highlight.
                    if self.selected.is_some() {
                        let current = self
                            .selected
                            .as_ref()
                            .and_then(|name| self.tiles.iter().position(|tile| tile.name == *name));
                        if let Some(target) =
                            current.and_then(|idx| paginate(idx, self.tiles.len(), dir))
                        {
                            let name = self.tiles[target].name.clone();
                            return self.open_frame(&name);
                        }
                        return Task::none();
                    }

                    // Bare frame grid: move the highlight, then reveal it if it
                    // stepped out of the viewport.
                    let selected = self
                        .frame_selected
                        .as_ref()
                        .and_then(|name| self.tiles.iter().position(|tile| tile.name == *name));
                    let len = self.tiles.len();
                    let cols = self.nav_cols();
                    if let Some(target) = nav_target(selected, len, cols, dir) {
                        let name = self.tiles[target].name.clone();
                        self.frame_selected = Some(name.clone());
                        // Shift+arrow extends the multi-selection (keeping it
                        // additive); a plain arrow collapses to the new primary.
                        if self.modifiers.shift() {
                            self.selected_frames.insert(name.clone());
                        } else {
                            self.selected_frames.clear();
                            self.selected_frames.insert(name.clone());
                        }
                        self.selection_anchor = Some(name.clone());
                        return Task::batch([
                            self.scroll_selection_into_view("frames-grid", target, len, cols),
                            self.ensure_frame_info_loaded().unwrap_or_else(Task::none),
                        ]);
                    }
                    return Task::none();
                }

                // Library grid: move the selection over every visible cell —
                // the leading Add Roll tile (only when no search is active),
                // then the filtered rolls — and reveal it out of the viewport.
                // While a search is active the Add Roll tile is hidden, so an
                // empty match set yields no destination at all.
                let cells = library_cells(
                    &self.rolls,
                    self.search_filter.as_deref().unwrap_or(""),
                    &self.month_aliases,
                );
                let selected = library_cell_index(self.library_selection.as_ref(), &cells);
                let len = cells.len();
                let cols = self.nav_cols();
                if let Some(target) = nav_target(selected, len, cols, dir) {
                    self.set_library_selection(cells[target].selection());
                    return self.scroll_selection_into_view("rolls-grid", target, len, cols);
                }
                Task::none()
            }

            Message::FrameSelected(name) => {
                // The clicked tile is always the keyboard focus / primary, even
                // when a multi-select toggle removes it from the selection set.
                self.frame_selected = Some(name.clone());
                let (updated, anchor) = apply_frame_click(
                    std::mem::take(&mut self.selected_frames),
                    &name,
                    self.modifiers.control(),
                    self.modifiers.shift(),
                    self.selection_anchor.as_deref(),
                    &self.tiles,
                );
                self.selected_frames = updated;
                self.selection_anchor = anchor;
                self.ensure_frame_info_loaded().unwrap_or_else(Task::none)
            }

            Message::SelectAllFrames => {
                if self.active.is_some() {
                    self.selected_frames =
                        self.tiles.iter().map(|tile| tile.name.clone()).collect();
                }
                Task::none()
            }

            Message::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers;
                Task::none()
            }

            Message::GridViewport(viewport) => {
                self.grid_viewport = Some(viewport);
                Task::none()
            }

            Message::RollActivated(dir) => self.open_roll(dir),

            Message::RollOpened(dir, files) => {
                // A stale scan from a roll closed mid-scan must not land.
                if self.active.as_deref() != Some(dir.as_path()) {
                    return Task::none();
                }
                // Load the roll manifest and reconcile it against what is on
                // disk, so edits for removed files never reattach to a name
                // that later returns.
                self.roll = edit_manifest::load_roll_manifest(&dir);
                edit_manifest::reconcile(&mut self.roll, &files);

                // The frame grid remounts; any cached viewport is stale until
                // it scrolls again.
                self.grid_viewport = None;
                self.thumb_inflight.clear();

                self.tiles = files
                    .into_iter()
                    .map(|name| Tile {
                        name,
                        thumb: Thumb::Loading,
                        meta: None,
                        meta_failed: false,
                    })
                    .collect();

                // Pre-select the first frame so the grid always has a
                // highlight (the detail view stays closed; Enter opens it).
                if let Some(first) = self.tiles.first() {
                    self.frame_selected = Some(first.name.clone());
                    self.selected_frames.clear();
                    self.selected_frames.insert(first.name.clone());
                    self.selection_anchor = Some(first.name.clone());
                }

                // The grid's remembered drawer state can only be applied once a
                // frame is highlighted (the pre-select above), so re-apply it
                // here — navigation never closes the drawer.
                self.restore_drawer_for(DrawerView::Grid);
                // A roll opened under the auto-selected-frame base mode with no
                // designated frame yet (e.g. a freshly added roll whose base
                // mode was picked in the Add Roll dialog) defaults it to the
                // first frame and measures its base.
                let mut tasks = vec![];
                if self.roll.base_mode() == BaseMode::AutoSelectedFrame
                    && self.roll.calibration_frame().is_none()
                {
                    let first = self.tiles.first().map(|tile| tile.name.clone());
                    if let Some(first) = first {
                        self.roll.set_calibration_frame(&first);
                        self.persist_roll();
                        tasks.push(Self::measure_calibration_frame(dir.clone()));
                    }
                }
                tasks.extend([
                    self.decode_next(),
                    // With the grid's drawer open, parse the freshly
                    // pre-selected frame's EXIF so the panel never shows a
                    // stuck "Loading…" after re-entering a roll.
                    self.ensure_frame_info_loaded().unwrap_or_else(Task::none),
                ]);
                Task::batch(tasks)
            }

            Message::SearchActivate => {
                // Search is library-only; a roll's frame grid is never
                // searched, so activation is a no-op while a roll is open.
                if self.active.is_some() {
                    return Task::none();
                }
                if self.search.is_none() {
                    self.search = Some(String::new());
                    self.search_filter = None;
                    self.search_version = self.search_version.wrapping_add(1);
                }
                // Focus the (now visible) input.
                cosmic::widget::text_input::focus(search_input_id())
            }

            Message::SearchClear => {
                self.search = None;
                self.search_filter = None;
                self.search_version = self.search_version.wrapping_add(1);
                Task::none()
            }

            Message::SearchInput(term) => {
                // The input box shows the keystrokes immediately; the applied
                // filter waits for `SEARCH_DEBOUNCE_MS` of quiet typing.
                self.search = Some(term.clone());
                self.search_version = self.search_version.wrapping_add(1);
                let version = self.search_version;
                cosmic::task::future(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(SEARCH_DEBOUNCE_MS)).await;
                    Message::SearchCommitted(term, version)
                })
            }

            Message::SearchCommitted(term, version) => {
                // Drop stale commits: any keystroke, clear or escape since this
                // task spawned bumped `search_version`.
                if version == self.search_version {
                    self.search_filter = Some(term);
                }
                Task::none()
            }

            Message::ThumbReady(name, result) => {
                if let Some(tile) = self.tiles.iter_mut().find(|tile| tile.name == name) {
                    match &result {
                        Ok(_) => rebake_trace(format_args!("ThumbReady: {name} -> Ready")),
                        Err(_) => rebake_trace(format_args!("ThumbReady: {name} -> Failed")),
                    }
                    tile.thumb = match result {
                        Ok(handle) => Thumb::Ready(handle),
                        Err(_) => Thumb::Failed,
                    };
                } else {
                    rebake_trace(format_args!("ThumbReady: {name} NOT FOUND in tiles"));
                }

                self.thumb_inflight.retain(|pending| pending != &name);
                self.decode_next()
            }

            Message::FrameInfoReady(name, result) => {
                if let Some(tile) = self.tiles.iter_mut().find(|tile| tile.name == name) {
                    match result {
                        Ok(meta) => tile.meta = Some(meta),
                        Err(_) => tile.meta_failed = true,
                    }
                }
                if self.frame_meta_inflight.as_deref() == Some(name.as_str()) {
                    self.frame_meta_inflight = None;
                }
                Task::none()
            }

            Message::ToggleContextPage(context_page) => {
                // The drawer's close button (X) routes here with its own page;
                // the View → About menu routes here with `About`. About is a
                // transient panel: the menu opens it, its X (or Space) dismisses
                // it by restoring the current view's remembered drawer state,
                // and the view drawers' X closes them, clearing that view's
                // remembered state.
                if context_page == ContextPage::About && self.context_page != ContextPage::About {
                    self.context_page = ContextPage::About;
                    self.core_mut().set_show_context(true);
                } else if self.context_page == ContextPage::About {
                    self.restore_drawer_for(self.current_view());
                } else if context_page == self.current_view().page() {
                    self.core_mut().set_show_context(false);
                    self.drawer_memory.set(self.current_view(), false);
                }
                Task::none()
            }

            Message::ToggleContext => {
                // Bare Space (and View → Details): toggle only the current
                // view's context drawer. About, if showing, is dismissed and
                // the view's own remembered state is restored.
                let view = self.current_view();
                // A genuinely showing About (menu-opened) is dismissed by
                // restoring the view's remembered drawer state — see #18.
                if self.context_page == ContextPage::About && self.core.window.show_context {
                    self.restore_drawer_for(view);
                    return Task::none();
                }
                // A "phantom" About is only `ContextPage`'s launch default
                // (nothing is showing): clear it and fall through so the very
                // first toggle opens the drawer instead of silently restoring
                // the remembered-closed state.
                if self.context_page == ContextPage::About {
                    self.context_page = view.page();
                }
                if self.core.window.show_context {
                    self.drawer_memory.set(view, false);
                    self.core_mut().set_show_context(false);
                    return Task::none();
                }
                if !self.view_drawer_valid(view) {
                    return Task::none();
                }
                self.drawer_memory.set(view, true);
                self.context_page = view.page();
                self.core_mut().set_show_context(true);
                if view == DrawerView::Grid {
                    return self.ensure_frame_info_loaded().unwrap_or_else(Task::none);
                }
                Task::none()
            }

            Message::UpdateConfig(config) => {
                self.config = config;
                Task::none()
            }

            Message::Ignore => Task::none(),

            Message::ExportRequested => {
                // Never stack a second dialog onto a pending one: the portal
                // dialog is modal once visible, but a stray keyboard or menu
                // double-fire can still land in the async gap before it shows.
                if self.export_pending {
                    return Task::none();
                }
                // A batch already running means a new export would queue behind
                // it; refuse the dialog outright rather than open one that
                // resolves into a no-op (the menu item is disabled to match).
                if self.export_progress.is_some() {
                    return Task::none();
                }
                // The keyboard `e` trigger fires on any page; without a selected
                // frame inside an open roll there is nothing to export (the File
                // menu item is disabled to match).
                if !self.has_export_targets() {
                    return Task::none();
                }
                self.export_pending = true;

                let dialog = cosmic::dialog::file_chooser::open::Dialog::new()
                    .title(fl!("export-title"))
                    .accept_label(fl!("export-pick-folder"))
                    .choice(export_format_choice())
                    .choice(export_overwrite_choice())
                    .open_folder();

                cosmic::task::future(async move {
                    match dialog.await {
                        Ok(response) => {
                            // The chosen folder is the destination; every frame
                            // keeps its own `<stem>.<ext>`.
                            let Ok(dest) = response.url().to_file_path() else {
                                return Message::ExportChosen(None);
                            };
                            // The response lists the selected (id, value) pairs:
                            // the "format" choice carries the preset key and the
                            // "overwrite" checkbox carries its state. A backend
                            // that dropped a choice falls back to the defaults.
                            let key = response
                                .choices()
                                .iter()
                                .find(|(id, _)| id == "format")
                                .map_or("jpeg-90", |(_, value)| value.as_str());
                            let options = options_for_choice(key).with_overwrite(
                                response
                                    .choices()
                                    .iter()
                                    .find(|(id, _)| id == "overwrite")
                                    .is_some_and(|(_, value)| value == "true"),
                            );
                            Message::ExportChosen(Some((dest, options)))
                        }
                        // Cancelled (or a portal failure) is a no-op.
                        Err(_) => Message::ExportChosen(None),
                    }
                })
            }

            Message::ExportChosen(result) => {
                self.export_pending = false;
                match result {
                    Some((dest, options)) => self.begin_export(dest, options),
                    None => Task::none(),
                }
            }

            Message::ExportProgress { done, total } => {
                self.export_progress = Some((done, total));
                Task::none()
            }

            Message::ExportDone {
                ok,
                skipped,
                failed,
                dest,
                start_date,
            } => {
                // The batch is over: hide the header ring before the summary
                // toast lands.
                self.export_progress = None;
                let toast = if failed == 0 && skipped == 0 {
                    toaster::Toast::new(match start_date {
                        Some(date) => fl!(
                            "export-done-dated",
                            count = ok,
                            dir = dest.display().to_string(),
                            date = date
                        ),
                        None => fl!("export-done", count = ok, dir = dest.display().to_string()),
                    })
                } else if failed == 0 {
                    toaster::Toast::new(fl!(
                        "export-done-skipped",
                        count = ok,
                        dir = dest.display().to_string(),
                        skipped = skipped
                    ))
                } else {
                    let total = ok + failed;
                    toaster::Toast::new(fl!("export-failed", count = total, failed = failed))
                };
                self.toasts.push(toast).map(cosmic::Action::App)
            }

            Message::ToastClose(id) => {
                self.toasts.remove(id);
                Task::none()
            }

            Message::EditSave => {
                // Slider release (and the close/flush points) commit the
                // dragged edit: persist the manifest, then re-bake the active
                // tile + roll cover through the shared commit path used by the
                // keyboard-release commit too.
                self.commit_edit()
            }

            Message::CopyEdits => {
                self.copy_edits();
                Task::none()
            }

            Message::PasteEdits => self.paste_edits(),

            Message::Quit => {
                // Flush any in-progress edit to disk, then ask the window to
                // close (the cosmic runtime runs our on_close_requested hook,
                // which persists a final time before exiting).
                self.persist_roll();
                self.persist_config();
                Task::done(cosmic::Action::Cosmic(cosmic::app::Action::Close))
            }

            // Forward menu-bar popup surface actions to the cosmic runtime,
            // which creates/destroys the actual popup surfaces on Wayland.
            Message::Surface(action) => {
                cosmic::task::message(cosmic::Action::Cosmic(cosmic::app::Action::Surface(action)))
            }

            Message::LaunchUrl(url) => match open::that_detached(&url) {
                Ok(()) => Task::none(),
                Err(err) => {
                    log::error!("failed to open {url:?}: {err}");
                    Task::none()
                }
            },
        }
    }

    /// Called when the user requests an app window to be closed; flush any
    /// unsaved edits before the window goes away.
    fn on_close_requested(&self, _id: cosmic::iced::window::Id) -> Option<Self::Message> {
        Some(Message::EditSave)
    }
}

impl AppModel {
    /// Updates the header and window titles.
    pub fn update_title(&mut self) -> Task<cosmic::Action<Message>> {
        if let Some(id) = self.core.main_window_id() {
            self.set_window_title(fl!("app-title"), id)
        } else {
            Task::none()
        }
    }

    /// Ensures the library page always has a selection once rolls exist:
    /// whenever nothing is selected, highlight the first visible roll
    /// (respecting an active search). No-op while a roll is open, while no roll
    /// is visible, or when a selection already exists.
    fn select_first_visible_roll(&mut self) {
        if self.active.is_some() || self.library_selection.is_some() {
            return;
        }
        let cells = library_cells(
            &self.rolls,
            self.search_filter.as_deref().unwrap_or(""),
            &self.month_aliases,
        );
        if let Some(LibraryCell::Roll(roll)) = cells
            .iter()
            .find(|cell| matches!(cell, LibraryCell::Roll(_)))
        {
            self.set_library_selection(LibrarySelection::Roll(roll.dir.clone()));
        }
    }

    /// Selects a library cell and reseeds the roll-info drafts in the same pass
    /// the selection changes, so the drawer never shows a previous roll's edits.
    fn set_library_selection(&mut self, selection: LibrarySelection) {
        self.library_selection = Some(selection);
        self.sync_roll_date_drafts();
    }

    /// Drills into a roll's frame grid (double click, or Enter on the selected
    /// roll). The outgoing roll keeps its unsaved edits; the detail/editing
    /// state is reset for the fresh roll, and a library metadata drawer closes
    /// since it can never show while a roll is open.
    fn open_roll(&mut self, dir: PathBuf) -> Task<cosmic::Action<Message>> {
        if self.active.as_deref() == Some(dir.as_path()) {
            return Task::none();
        }
        self.persist_roll();
        self.active = Some(dir.clone());
        self.selected = None;
        self.frame_selected = None;
        self.selected_frames.clear();
        self.selection_anchor = None;
        self.grid_viewport = None;
        self.frame_meta_inflight = None;
        self.clear_detail();
        // Entering the grid: the drawer (if the grid's memory says open) shows
        // the frame-info panel; navigation never closes it.
        self.restore_drawer_for(DrawerView::Grid);
        cosmic::task::future(async move {
            let files = load_files_in(dir.clone()).await;
            Message::RollOpened(dir, files)
        })
    }

    /// Opens a frame in the detail view (double-click, Enter on the highlighted
    /// tile, or Left/Right paging while a detail view is open). Persists unsaved
    /// tweaks to the outgoing file, loads the stored edits, updates both the
    /// detail selection and the grid highlight, and starts the decode chain.
    fn open_frame(&mut self, name: &str) -> Task<cosmic::Action<Message>> {
        if self.selected.as_deref() != Some(name) {
            // Persist unsaved tweaks to the outgoing file first.
            self.persist_roll();
            // Read the stored edits BEFORE the decode builds the shader, which
            // consumes `self.exposure_ev` and the tone (via `set_curve` in
            // `handle_detail_ready`).
            let stored_tone = self.roll.tone(name);
            self.selected = Some(name.to_owned());
            self.frame_selected = Some(name.to_owned());
            // The opened frame becomes the primary of the multi-selection (and
            // the Shift+click anchor), so copy/paste batches stay consistent
            // with what is on screen.  Shift+open extends; a plain open
            // collapses the set to the single opened frame.
            if self.modifiers.shift() {
                self.selected_frames.insert(name.to_owned());
            } else {
                self.selected_frames.clear();
                self.selected_frames.insert(name.to_owned());
            }
            self.selection_anchor = Some(name.to_owned());
            self.clear_detail();
            self.exposure_ev = stored_tone.exposure_ev;
            self.curve_contrast = stored_tone.curve_contrast;
            self.curve_highlights = stored_tone.curve_highlights;
            self.curve_shadows = stored_tone.curve_shadows;
            self.crop = self.roll.crop(name);
            self.rotation = self.roll.rotation(name) & 3;
            // Entering the detail view: the drawer (if the detail's memory says
            // open) shows the editing panel; navigation never closes it.
            self.restore_drawer_for(DrawerView::Detail);
            // Anchor the reset snapshot to the opened state (the stored
            // manifest values), so Reset reverts here rather than to identity.
            self.reset_exposure_ev = stored_tone.exposure_ev;
            self.reset_curve_contrast = stored_tone.curve_contrast;
            self.reset_curve_highlights = stored_tone.curve_highlights;
            self.reset_curve_shadows = stored_tone.curve_shadows;
            self.reset_crop = self.crop;
            self.reset_rotation = self.rotation;
        }

        // The editing drawer stays hidden on a fresh open; the user brings it
        // up with Space or the View → Details… menu when they
        // want the controls. A same-session paging to a neighbour file likewise
        // leaves whatever context state is current untouched.

        Task::batch([
            self.decode_detail_next(),
            self.preload_detail_neighbors(name),
        ])
    }

    /// The frames the next export should target: every multi-selected frame
    /// when the user has built a multi-selection, otherwise just the focused/
    /// highlighted frame (which also mirrors the open detail frame). Empty when
    /// no roll is open or nothing is focused.
    fn export_targets(&self) -> Vec<String> {
        if self.active.is_none() {
            return Vec::new();
        }
        if self.selected_frames.is_empty() {
            return self.frame_selected.iter().cloned().collect();
        }
        // Deterministic order for the export so a re-run writes files in the
        // same sequence (a HashSet has no stable order).
        let mut names: Vec<String> = self.selected_frames.iter().cloned().collect();
        names.sort();
        names
    }

    /// Whether [`Self::export_targets`] would yield at least one frame.
    fn has_export_targets(&self) -> bool {
        self.active.is_some() && (self.frame_selected.is_some() || !self.selected_frames.is_empty())
    }

    /// Exports every target frame into `dest`, one file per frame named
    /// `<stem>.<ext>` per the given options (JPEG or lossless PNG, native or
    /// downscaled), decoding and baking each on the blocking worker pool.
    /// Streams [`Message::ExportProgress`] per finished frame (driving the
    /// header ring) plus a final [`Message::ExportDone`] summary, which clears
    /// the ring. Edits are read from the in-memory manifest — the same source
    /// the grid and detail view render from.
    fn begin_export(
        &mut self,
        dest: PathBuf,
        options: ExportOptions,
    ) -> Task<cosmic::Action<Message>> {
        let Some(dir) = self.active.clone() else {
            return Task::none();
        };
        let names = self.export_targets();
        if names.is_empty() {
            return Task::none();
        }
        // A batch is already running: refuse a second one (the disabled menu
        // item and the ExportRequested guard also block this path).
        if self.export_progress.is_some() {
            return Task::none();
        }
        // Snapshot each frame's stored edits up front (the manifest is shared and
        // could change while the batch runs, but a snapshot keeps the export of
        // one batch internally consistent). This reads the same in-memory
        // manifest the grid and detail view render from, so an export always
        // matches what is on screen — even for an edit that has been applied
        // but not yet flushed to the on-disk manifest. Each frame carries its
        // full-roll position (the sorted order the grid shows), the offset used
        // for a stable synthetic capture timestamp, and the roll's start date
        // (when set) travels along to stamp DateTimeOriginal on the output.
        let start_date = self.roll.start_date().map(str::to_owned);
        let frames: Vec<(
            String,
            edit_manifest::ToneEdit,
            edit_manifest::CropMargins,
            u8,
            usize,
        )> = names
            .into_iter()
            .map(|name| {
                let tone = self.roll.tone(&name);
                let crop = self.roll.crop(&name);
                let rotation = self.roll.rotation(&name) & 3;
                let index = self
                    .tiles
                    .iter()
                    .position(|tile| tile.name == name)
                    .unwrap_or(0);
                (name, tone, crop, rotation, index)
            })
            .collect();
        let total = frames.len();
        self.export_progress = Some((0, total));
        let preset = self.roll.preset();
        let base_config = self.roll.base_config();

        // Stream from an async channel so the UI sees per-frame ticks. After
        // each frame the sender pushes an `ExportProgress` message (dropping
        // harmlessly if the UI is behind on backpressure), and the final
        // `ExportDone` is awaited *through* the channel so it can never be
        // lost — the ring is guaranteed a matching completion message.
        cosmic::task::stream(cosmic::iced::stream::channel(1, async move |mut sender| {
            let mut tick = |done: usize, _total: usize| {
                let _ = sender.try_send(Message::ExportProgress { done, total });
            };
            let (ok, skipped, failed) = export_frames(
                dir,
                dest.clone(),
                frames,
                options,
                preset,
                base_config,
                start_date.clone(),
                &mut tick,
            )
            .await;
            let _ = sender
                .send(Message::ExportDone {
                    ok,
                    skipped,
                    failed,
                    dest,
                    start_date,
                })
                .await;
        }))
    }

    /// The column count arrow-key navigation should use: the exact grid count
    /// derived from the cached viewport width when one is known (matching iced's
    /// fluid math), falling back to the window-resize track when the grid has
    /// not scrolled yet.
    fn nav_cols(&self) -> usize {
        self.grid_viewport.as_ref().map_or_else(
            || self.grid_cols.max(1),
            |viewport| {
                let spacing = f32::from(cosmic::theme::spacing().space_s);
                grid_num_cols(viewport.bounds().width - 2.0 * spacing, spacing).max(1)
            },
        )
    }

    /// Scrolls the mounted grid so the tile at `index` (within the matched set
    /// of `len`, laid out `cols`-wide) is fully visible, when it has moved
    /// beyond the viewport. Uses the cached [`Viewport`] for precise reveal;
    /// before the first real scroll the geometry is estimated from the window
    /// height and row count. Returns the scroll effect, or `Task::none()` when
    /// the tile is already visible.
    #[allow(clippy::cast_precision_loss)] // row/cell counts are far below f32's exact range
    fn scroll_selection_into_view(
        &self,
        name: &'static str,
        index: usize,
        len: usize,
        cols: usize,
    ) -> Task<cosmic::Action<Message>> {
        let spacing = f32::from(cosmic::theme::spacing().space_s);
        let padding = spacing;

        let (cell_width, viewport_height, viewport_offset_y, content_height) =
            if let Some(viewport) = &self.grid_viewport {
                let available = viewport.bounds().width - 2.0 * padding;
                let cols = grid_num_cols(available, spacing).max(1);
                let cell = (available - spacing * (cols as f32 - 1.0)) / cols as f32;
                (
                    cell,
                    viewport.bounds().height,
                    viewport.absolute_offset().y,
                    viewport.content_bounds().height,
                )
            } else {
                // No viewport yet: the grid sits at the top, its height is the
                // window (≈80 px of header sizing leaves the content space),
                // and the content is sized from the row count at THUMB cells.
                let rows = len.div_ceil(cols.max(1));
                let content_height = 2.0 * padding
                    + rows as f32 * (THUMB_SIZE + spacing)
                    + spacing * (rows.saturating_sub(1)) as f32;
                (
                    THUMB_SIZE,
                    (self.window_height - 80.0).max(1.0),
                    0.0,
                    content_height,
                )
            };

        match reveal_target_y(
            cols,
            index,
            spacing,
            padding,
            cell_width,
            viewport_height,
            viewport_offset_y,
            content_height,
        ) {
            Some(y) => cosmic::iced::widget::scrollable::scroll_to::<cosmic::Action<Message>>(
                scrollable_id(name),
                cosmic::iced::widget::scrollable::AbsoluteOffset {
                    x: None,
                    y: Some(y),
                },
            ),
            None => Task::none(),
        }
    }

    /// Spawns decoding of up to [`MAX_CONCURRENT_THUMBS`] pending thumbnails.
    ///
    /// Decoding is bounded rather than strictly sequential so several
    /// `spawn_blocking` RAW decodes overlap on multicore machines; the count
    /// keeps the transient footprint to a few full-resolution buffers. A name
    /// already in flight is never spawned again, so a re-bake requested
    /// mid-chain waits for the running decode instead of racing it. The
    /// exposure read here is the value at spawn time, so every tile (initial
    /// chain or re-bake) bakes in the version of the edit that is current when
    /// it actually decodes.
    fn decode_next(&mut self) -> Task<cosmic::Action<Message>> {
        let Some(dir) = self.active.clone() else {
            return Task::none();
        };
        let capacity = MAX_CONCURRENT_THUMBS.saturating_sub(self.thumb_inflight.len());
        if capacity == 0 {
            rebake_trace(format_args!(
                "decode_next: capacity 0 (inflight={})",
                self.thumb_inflight.len()
            ));
            return Task::none();
        }

        let pending: Vec<(
            String,
            edit_manifest::ToneEdit,
            edit_manifest::CropMargins,
            u8,
        )> = self
            .tiles
            .iter()
            .filter(|tile| matches!(tile.thumb, Thumb::Loading))
            .filter(|tile| !self.thumb_inflight.iter().any(|name| name == &tile.name))
            .take(capacity)
            .map(|tile| {
                let tone = self.roll.tone(&tile.name);
                let crop = self.roll.crop(&tile.name);
                let rotation = self.roll.rotation(&tile.name) & 3;
                (tile.name.clone(), tone, crop, rotation)
            })
            .collect();

        if pending.is_empty() {
            rebake_trace(format_args!(
                "decode_next: no loading tiles (tiles={}, inflight={})",
                self.tiles.len(),
                self.thumb_inflight.len()
            ));
            return Task::none();
        }
        rebake_trace(format_args!(
            "decode_next: spawning {} [{:?}]",
            pending.len(),
            pending.iter().map(|(n, ..)| n.as_str()).collect::<Vec<_>>()
        ));

        // The in-memory manifest is loaded synchronously in `RollOpened`
        // before the tile pump runs, so it carries the preset even while the
        // library card scan (`RollInfoLoaded`) is still in flight for a
        // freshly-added roll — `self.rolls` can be a step behind on that path.
        let preset = self.roll.preset();
        let base_config = self.roll.base_config();

        self.thumb_inflight
            .extend(pending.iter().map(|(name, ..)| name.clone()));

        Task::batch(
            pending
                .into_iter()
                .map(move |(name, tone, crop, rotation)| {
                    cosmic::task::future(decode_thumbnail(
                        dir.clone(),
                        name,
                        tone,
                        crop,
                        rotation,
                        preset,
                        base_config,
                    ))
                }),
        )
    }

    /// Spawns decoding of up to [`MAX_CONCURRENT_THUMBS`] roll-cover
    /// thumbnails, mirroring the frame chain's bounds and de-duplication.
    fn decode_covers(&mut self) -> Task<cosmic::Action<Message>> {
        let capacity = MAX_CONCURRENT_THUMBS.saturating_sub(self.cover_inflight.len());
        if capacity == 0 {
            return Task::none();
        }

        let pending: Vec<(PathBuf, String, FilmPreset)> = self
            .rolls
            .iter()
            .filter(|roll| matches!(roll.thumb, Thumb::Loading))
            .filter(|roll| !self.cover_inflight.iter().any(|dir| dir == &roll.dir))
            .filter_map(|roll| {
                roll.cover
                    .clone()
                    .map(|name| (roll.dir.clone(), name, roll.preset))
            })
            .take(capacity)
            .collect();

        if pending.is_empty() {
            return Task::none();
        }

        self.cover_inflight
            .extend(pending.iter().map(|(dir, _, _)| dir.clone()));

        Task::batch(
            pending
                .into_iter()
                .map(|(dir, name, preset)| cosmic::task::future(decode_cover(dir, name, preset))),
        )
    }

    /// Re-bake the active frame's grid thumbnail and, if that frame is also
    /// this roll's cover, the library roll card — the shared re-decode step of
    /// every commit (slider release, keyboard-shortcut release, crop commits)
    /// so the grid and roll preview reflect the latest edit. The bake reads the
    /// crop/tone from the in-memory roll at decode time, so it reflects a live
    /// edit without a prior `persist_roll()` (`commit_edit` persists first so
    /// the cover's on-disk manifest read is fresh too).
    fn re_bake_edit(&mut self) -> Task<cosmic::Action<Message>> {
        let mut tasks = Vec::with_capacity(2);
        let Some(name) = self.selected.clone() else {
            return Task::batch(tasks);
        };
        rebake_trace(format_args!(
            "re_bake_edit: selected={name} active={}",
            self.active.is_some()
        ));
        if let Some(tile) = self.tiles.iter_mut().find(|tile| tile.name == name) {
            tile.thumb = Thumb::Loading;
            rebake_trace(format_args!("re_bake_edit: tile {name} -> Loading"));
        }
        tasks.push(self.decode_next());
        if let Some(active) = self.active.as_ref() {
            let Some(roll) = self
                .rolls
                .iter_mut()
                .find(|roll| roll.dir == *active && roll.cover.as_deref() == Some(name.as_str()))
            else {
                return Task::batch(tasks);
            };
            roll.thumb = Thumb::Loading;
            rebake_trace(format_args!("re_bake_edit: roll cover {name} -> Loading"));
            tasks.push(self.decode_covers());
        }
        Task::batch(tasks)
    }

    /// Commits the open frame's edits: persists the RAM manifest to disk, then
    /// re-bakes the affected grid tile + roll cover. This is the single commit
    /// point shared by slider release (`EditSave`), keyboard-shortcut release
    /// (`EditKeyReleased`), and the discrete crop commits (typed margin submit,
    /// reset crop). Editing steps themselves mutate live (RAM + shader) only;
    /// the commit happens once per interaction, like a slider drag/release.
    fn commit_edit(&mut self) -> Task<cosmic::Action<Message>> {
        self.persist_roll();
        self.re_bake_edit()
    }

    /// Measures the currently viewed frame's clear-film plateau and records it
    /// as the roll's calibrated black point, then re-renders everything under
    /// the new base. The frame must yield a plausible measurement
    /// (`>= MIN_PLAUSIBLE_BASE`); otherwise the calibration is left untouched —
    /// a frame with no clear film (e.g. one shot against a grey card) must not
    /// poison the roll's rendering.
    fn calibrate_base_from_frame(&mut self) -> Task<cosmic::Action<Message>> {
        let Some(dir) = self.active.clone() else {
            return Task::none();
        };
        let Some(name) = self
            .selected
            .clone()
            .or_else(|| self.frame_selected.clone())
        else {
            return Task::none();
        };
        let preset = self.roll.preset();
        // Only the auto-selected-frame base mode designates a calibration
        // frame; elsewhere the base strategy is not a per-frame designation.
        if self.roll.base_mode() != BaseMode::AutoSelectedFrame {
            return Task::none();
        }
        // The viewed frame's overview is usually cached; measure straight from
        // it (the exact mono the user is looking at). A cold frame falls back
        // to a one-shot decode that lands in `CalibrationBaseMeasured`.
        let Some(cached) = self.detail_cache.get(&(dir.clone(), preset, name.clone())) else {
            // Designate the frame in RAM AND on disk before the decode returns,
            // since `apply_calibration_measurement` reads the manifest back
            // from disk and would otherwise drop the result as stale.
            self.roll.set_calibration_frame(&name);
            self.persist_roll();
            return cosmic::task::future(measure_frame_base(dir, name, preset));
        };
        let base = measure_base(&cached.mono).filter(|base| *base >= MIN_PLAUSIBLE_BASE);
        self.roll.set_calibration_frame(&name);
        match base {
            Some(base) => self.roll.set_calibrated_base(base),
            None => self.roll.base = None,
        }
        self.reflow_base()
    }

    /// Spawns a one-shot measurement of the roll's designated auto-calibration
    /// frame, posting its measured clear-film transmission back as
    /// [`Message::CalibrationBaseMeasured`].
    fn measure_calibration_frame(dir: PathBuf) -> Task<cosmic::Action<Message>> {
        // The designated frame (and the film preset the decode runs under) live
        // on disk — the drawer is library-only and the RAM manifest may be for
        // another roll or stale, so read both fresh.
        let manifest = edit_manifest::load_roll_manifest(&dir);
        let Some(name) = manifest.calibration_frame().map(str::to_owned) else {
            return Task::none();
        };
        cosmic::task::future(measure_frame_base(dir, name, manifest.preset()))
    }

    /// Applies a landed calibration-frame measurement to the roll's manifest:
    /// records the designated frame (already set on the defaulting path) and
    /// its measured base, then re-renders every rendering under the new black
    /// point. Guarded so a measurement racing a base-mode/frame change is
    /// dropped.
    fn apply_calibration_measurement(
        &mut self,
        dir: &Path,
        name: &str,
        base: Option<f32>,
    ) -> Task<cosmic::Action<Message>> {
        let mut manifest = edit_manifest::load_roll_manifest(dir);
        let stale = manifest.base_mode() != BaseMode::AutoSelectedFrame
            || manifest.calibration_frame() != Some(name);
        if stale {
            return Task::none();
        }
        manifest.set_calibration_frame(name);
        match base {
            Some(base) => manifest.set_calibrated_base(base),
            None => manifest.base = None,
        }
        if let Err(err) = edit_manifest::save_roll_manifest(dir, &manifest) {
            log::error!(
                "failed to write roll manifest {}: {err}",
                edit_manifest::manifest_path(dir).display()
            );
        }
        if self.active.as_deref() == Some(dir) {
            self.roll.set_calibration_frame(name);
            match base {
                Some(base) => self.roll.set_calibrated_base(base),
                None => self.roll.base = None,
            }
            self.reflow_base()
        } else {
            // Library page: only the roll card's cover needs re-baking under
            // the now-resolved base (frames decode from the disk manifest when
            // the roll is opened).
            if let Some(roll) = self.rolls.iter_mut().find(|roll| roll.dir == dir) {
                roll.thumb = Thumb::Loading;
            }
            self.decode_covers()
        }
    }

    /// The roll's first sorted frame name: the `AutoSelectedFrame` default
    /// calibration frame. Mirrors the cover selection (the first sorted
    /// non-dot file) so the defaulted calibration frame matches what the user
    /// sees first in the grid.
    fn roll_first_frame(&self, dir: &Path) -> Option<String> {
        if self.active.as_deref() == Some(dir) {
            self.tiles.first().map(|tile| tile.name.clone())
        } else {
            self.rolls
                .iter()
                .find(|roll| roll.dir == dir)
                .and_then(|roll| roll.cover.clone())
        }
    }

    /// Re-bakes every rendering of the OPEN roll after a film-preset or
    /// base-mode change: re-bakes the grid tiles and drops every detail buffer
    /// decoded under the old profile (the live shader, the roll's LRU entries,
    /// and any in-flight native level-up), then re-pumps so the next frames
    /// come up under the new profile. Exposure/curve/crop edits survive (they
    /// are applied in-shader). The caller has already recorded the new choice
    /// on `self.roll` and pushed `decode_covers` into its task batch.
    fn reflow_roll_after_change(&mut self, dir: &Path) -> Vec<Task<cosmic::Action<Message>>> {
        for tile in &mut self.tiles {
            tile.thumb = Thumb::Loading;
        }
        self.thumb_inflight.clear();
        self.detail_shader = None;
        self.detail_native_queued = false;
        self.detail_logged_failure = None;
        self.detail_inflight = None;
        self.detail_preload_inflight.clear();
        self.detail_thumb = None;
        self.detail_last_frame = None;
        let _ = self.detail_cache.retain(|(cached_dir, _, _)| cached_dir != dir);
        let current = self.selected.clone().unwrap_or_default();
        vec![
            self.decode_next(),
            self.decode_detail_next(),
            self.preload_detail_neighbors(&current),
        ]
    }

    /// Rebuilds every rendering after a base-calibration change: the live
    /// detail decode, the grid tiles, and the roll covers all carry a black
    /// point resolved at decode time, so the cached overviews and decodes are
    /// dropped and the roll is re-decoded under the new base mode. The on-disk
    /// manifest is written first since the cover path reads it from disk.
    fn reflow_base(&mut self) -> Task<cosmic::Action<Message>> {
        self.persist_roll();
        // A cached overview's `inversion` (or an in-flight decode) was resolved
        // under the old base mode and must not be served: drop the live shader,
        // the LRU, and the in-flight bookkeeping.
        self.detail_shader = None;
        self.detail_native_queued = false;
        self.detail_logged_failure = None;
        self.detail_inflight = None;
        self.detail_preload_inflight.clear();
        self.detail_thumb = None;
        self.detail_last_frame = None;
        let _ = self.detail_cache.retain(|_| false);
        // Re-bake every grid tile and the open roll's cover under the new base.
        for tile in &mut self.tiles {
            tile.thumb = Thumb::Loading;
        }
        self.thumb_inflight.clear();
        if let Some(active) = self.active.as_ref()
            && let Some(roll) = self.rolls.iter_mut().find(|roll| &roll.dir == active)
        {
            roll.thumb = Thumb::Loading;
        }
        let current = self.selected.clone().unwrap_or_default();
        Task::batch([
            self.decode_next(),
            self.decode_covers(),
            self.decode_detail_next(),
            self.preload_detail_neighbors(&current),
        ])
    }

    /// Spawns the hi-res decode behind the detail view when one is due.
    ///
    /// Two progressive levels share the single in-flight slot: the 2048
    /// overview while the shader is absent, then a native-resolution re-decode
    /// once the zoom reaches the level-up trigger ([`Self::native_level_up_zoom`],
    /// the earlier of the current texture's 1:1 cap and the fixed
    /// [`NATIVE_ZOOM_THRESHOLD`]) — the flag makes the level-up one-shot, and
    /// the inflight slot makes extra wheels no-ops.
    fn decode_detail_next(&mut self) -> Task<cosmic::Action<Message>> {
        if self.detail_inflight.is_some() || self.selected.is_none() {
            if self.detail_inflight.is_some() {
                detail_trace(format_args!(
                    "trigger swallowed: inflight slot busy (zoom={:.3})",
                    self.detail_zoom.0
                ));
            }
            return Task::none();
        }

        // The active roll's in-memory manifest is the decode source of truth
        // (loaded synchronously in `RollOpened`, kept in sync on preset
        // changes), so this is correct even while the library card scan for a
        // freshly added roll is still in flight.
        let preset = self.roll.preset();

        // Level 1 is native unless the sensor's long edge exceeds the wgpu
        // texture ceiling; `resize_area` treats a cap ≥ the source edge as
        // identity, so ≤8K scans decode at true full resolution.
        let cap: u32 = if self.detail_shader.is_none() {
            HI_RES_SIZE
        } else if !self.detail_native_queued {
            MAX_TEXTURE_EDGE
        } else {
            detail_trace(format_args!(
                "trigger swallowed: native already queued (zoom={:.3})",
                self.detail_zoom.0
            ));
            return Task::none();
        };

        let Some(name) = self.selected.clone() else {
            return Task::none();
        };

        // An overview request that's already in the LRU cache is served without
        // re-decoding the RAW (the whole point of the cache): build the shader
        // from the cached mono and return — the preview is ready immediately
        // with no thumbnail crossfade. Native level-up requests are never
        // cached/served; they always re-decode on demand. (The cached mono is
        // cloned out so the cache keeps serving future hits; the clone is a
        // fraction of the cost of a full RAW decode.)
        if cap == HI_RES_SIZE
            && let Some(dir) = self.active.clone()
            && let Some(cached) = self
                .detail_cache
                .get(&(dir, preset, name.clone()))
                .map(|c| {
                    (
                        c.mono.clone(),
                        c.width,
                        c.height,
                        c.src_long_edge,
                        c.inversion,
                    )
                })
        {
            let (mono, width, height, src_long_edge, inversion) = cached;
            detail_trace(format_args!(
                "cache hit: {name} {width}x{height} (src {src_long_edge})"
            ));
            self.install_detail_shader(mono, width, height, src_long_edge, inversion);
            return Task::none();
        }

        detail_trace(format_args!(
            "trigger fire: cap={cap}, shader={}, zoom={:.3}",
            self.detail_shader.is_some(),
            self.detail_zoom.0
        ));

        self.detail_inflight = Some(name.clone());

        let Some(dir) = self.active.clone() else {
            return Task::none();
        };

        let base_config = self.roll.base_config();

        cosmic::task::future(decode_detail(dir, name, cap, preset, base_config))
    }

    /// Builds and installs the detail shader from a decoded mono buffer,
    /// applying the current exposure, zoom/pan and tone curve, and marking the
    /// native level-up done when the source was already at (or below) the
    /// overview cap. Used when serving an LRU cache hit so the preview appears
    /// immediately without re-decoding the RAW.
    fn install_detail_shader(
        &mut self,
        mono: Vec<f32>,
        width: u32,
        height: u32,
        src_long_edge: u32,
        inversion: Option<(MonoStock, f32)>,
    ) {
        let image_id = self.next_image_id;
        self.next_image_id = self.next_image_id.wrapping_add(1);
        self.detail_shader = Some(shader::DetailProgram::new(
            mono,
            width,
            height,
            self.exposure_ev,
            self.crop,
            self.rotation,
            image_id,
            src_long_edge,
            inversion,
        ));
        if let Some(shader) = &mut self.detail_shader {
            shader.set_view(self.detail_zoom.0, self.detail_pan);
            shader.set_curve(self.curve_contrast, self.curve_highlights, self.curve_shadows);
            shader.set_crop(self.crop);
            shader.set_rotation(self.rotation);
            // A fresh program starts with the mask off; re-assert the current
            // crop-mode state so the dim overlay survives re-installs.
            shader.set_show_mask(self.crop_mode);
            shader.set_pad(if self.crop_mode {
                CROP_MODE_PADDING
            } else {
                0.0
            });
        }
        // A new texture (first decode or the native level-up) changes the 1:1
        // cap, so re-derive it and clamp the current zoom.
        self.reclamp_detail_zoom();
        // A cached overview that was already at native resolution needs no
        // level-up re-decode.
        if src_long_edge <= HI_RES_SIZE {
            self.detail_native_queued = true;
        }
    }

    /// Handles a finished neighbor preload decode: lands the overview into the
    /// LRU cache (dropping any evicted buffer) and frees its preload slot.
    /// A preload never touches `detail_inflight` or the active shader, and a
    /// stale landing (roll switched away mid-decode) is still a valid cached
    /// overview, so it is inserted regardless.
    fn handle_detail_preloaded(
        &mut self,
        dir: &PathBuf,
        name: &str,
        preset: FilmPreset,
        result: Result<DetailDecode, FrameError>,
    ) -> Task<cosmic::Action<Message>> {
        self.detail_preload_inflight
            .retain(|(pending_dir, pending_name)| pending_dir != dir || pending_name != name);

        if let Ok(DetailDecode {
            mono,
            width,
            height,
            src_long_edge,
            inversion,
        }) = result
        {
            let _evicted = self.detail_cache.insert(
                (dir.clone(), preset, name.to_string()),
                DetailMono {
                    mono,
                    width,
                    height,
                    src_long_edge,
                    inversion,
                },
            );
        }
        Task::none()
    }

    /// Preloads the frames `DETAIL_PRELOAD_DISTANCE` either side of `name` in
    /// the roll into the LRU cache, so Left/Right paging to a neighbor is
    /// instant. Runs on its own bounded channel, separate from the single
    /// critical detail slot. Already-cached and already-in-flight frames are
    /// skipped.
    fn preload_detail_neighbors(&mut self, name: &str) -> Task<cosmic::Action<Message>> {
        let Some(dir) = self.active.clone() else {
            return Task::none();
        };

        let Some(current) = self.tiles.iter().position(|tile| tile.name == name) else {
            return Task::none();
        };

        // Build the neighbor names in priority order (closest first) so the
        // bounded slots fill with the most useful frames first.
        let mut neighbors = Vec::new();
        for step in 1..=DETAIL_PRELOAD_DISTANCE {
            if let Some(prev) = current.checked_sub(step) {
                neighbors.push(self.tiles[prev].name.clone());
            }
            if let Some(next) = current.checked_add(step).filter(|&i| i < self.tiles.len()) {
                neighbors.push(self.tiles[next].name.clone());
            }
        }

        let capacity = MAX_CONCURRENT_PRELOADS.saturating_sub(self.detail_preload_inflight.len());
        if capacity == 0 {
            return Task::none();
        }

        let preset = self.roll.preset();

        let pending: Vec<String> = neighbors
            .into_iter()
            .filter(|n| {
                let key = (dir.clone(), preset, n.clone());
                !self.detail_cache.contains(&key)
                    && !self
                        .detail_preload_inflight
                        .iter()
                        .any(|(pd, pn)| pd == &dir && pn == n)
            })
            .take(capacity)
            .collect();

        if pending.is_empty() {
            return Task::none();
        }

        self.detail_preload_inflight
            .extend(pending.iter().map(|n| (dir.clone(), n.clone())));

        let base_config = self.roll.base_config();

        Task::batch(
            pending
                .into_iter()
                .map(|n| cosmic::task::future(preload_detail(dir.clone(), n, preset, base_config))),
        )
    }

    /// Reset all detail-view buffers, crossfade state, the view transform, and
    /// the non-persisted tone-curve preview (each open starts at the identity).
    fn clear_detail(&mut self) {
        self.detail_shader = None;
        self.detail_thumb_opacity = 1.0;
        self.detail_last_frame = None;
        self.detail_thumb = None;
        self.detail_zoom = Zoom::CONTAIN;
        // `detail_area_size` deliberately survives: it is the preview's laid-out
        // geometry, not per-frame detail state. Frame paging keeps `DetailArea`
        // mounted with unchanged bounds, so it never re-publishes a size; that
        // left the field `None` and the 1:1 zoom cap disabled (falling back to
        // `MAX_DETAIL_ZOOM`) until the editing drawer first changed the bounds —
        // letting the wheel zoom past native 100% (see the measured-size note on
        // the field).
        self.detail_native_queued = false;
        self.detail_logged_failure = None;
        self.detail_pan = Point::default();
        self.detail_panning = false;
        self.detail_cursor = None;
        // Any held editing key is dead once the detail view (and its edit
        // context) goes away; a late `EditKeyReleased` will find it clear and
        // no-op rather than committing a stale selection.
        self.editing_key_held = false;
        self.curve_contrast = 1.0;
        self.curve_highlights = 1.0;
        self.curve_shadows = 1.0;
        self.exposure_ev = edit_manifest::DEFAULT_EXPOSURE_EV;
        self.crop = edit_manifest::CropMargins::default();
        self.rotation = 0;
        self.crop_mode = false;
        // The reset snapshot mirrors the live edit values' lifecycle: reset
        // to identity on close; the next `ThumbnailActivated` re-syncs it.
        self.reset_exposure_ev = edit_manifest::DEFAULT_EXPOSURE_EV;
        self.reset_curve_contrast = 1.0;
        self.reset_curve_highlights = 1.0;
        self.reset_curve_shadows = 1.0;
        self.reset_crop = edit_manifest::CropMargins::default();
        self.reset_rotation = 0;
    }

    /// The effective maximum zoom for the current detail view: the 1:1 ("100%")
    /// point (one image pixel per physical screen pixel) for the live shader's
    /// texture at the preview's current size, capped by the absolute
    /// [`MAX_DETAIL_ZOOM`] guard. Falls back to [`MAX_DETAIL_ZOOM`] until both
    /// a shader and a reported preview size exist.
    fn max_detail_zoom(&self) -> f32 {
        let Some(shader) = &self.detail_shader else {
            return MAX_DETAIL_ZOOM;
        };
        let Some(size) = self.detail_area_size else {
            return MAX_DETAIL_ZOOM;
        };
        let sf = self.core().scale_factor();
        shader
            .zoom_100(size.width, size.height, sf)
            .min(MAX_DETAIL_ZOOM)
    }

    /// The clamp applied to the detail-view wheel and view re-clamps while a
    /// hi-res decode is running: the live texture's 1:1 (see
    /// [`Self::max_detail_zoom`]) would freeze the wheel at the coarse
    /// texture's 100% point — exactly when the native level-up is in flight.
    /// While a decode runs the cap rises to the **projected** native 1:1: the
    /// source's own 1:1 computed for the texture the in-flight decode will
    /// produce (`resized_dims` with the wgpu ceiling, aspect preserved), so
    /// zooming continues through the load but never passes the point the
    /// upcoming texture's pixels cover — and the landing re-clamps to that
    /// exact same value, so there is no yank-back. Idle, it is precisely
    /// [`Self::max_detail_zoom`].
    fn max_detail_zoom_for_wheel(&self) -> f32 {
        let Some(shader) = &self.detail_shader else {
            return MAX_DETAIL_ZOOM;
        };
        let Some(size) = self.detail_area_size else {
            return self.max_detail_zoom();
        };
        if self.detail_inflight.is_none() {
            return self.max_detail_zoom();
        }
        let (src_w, src_h) = shader.source_dimensions();
        let (dw, dh) = resized_dims(src_w, src_h, MAX_TEXTURE_EDGE);
        let sf = self.core().scale_factor();
        shader
            .zoom_100_for(dw, dh, size.width, size.height, sf)
            .min(MAX_DETAIL_ZOOM)
    }

    /// The zoom (log2 units) at which the native hi-res level-up should fire:
    /// the earlier of the current texture's 1:1 ("100%") cap and the fixed
    /// [`NATIVE_ZOOM_THRESHOLD`].
    ///
    /// The cap term is what makes the trigger reachable on every layout: the
    /// wheel clamps at `max_detail_zoom()`, so on a wide full-screen preview
    /// (context drawer closed) the overview's 1:1 point can sit below 2.0 and
    /// the fixed threshold alone could never be crossed. Firing at the cap is
    /// exactly right there — the overview is already at 100% and the native
    /// texture is what unlocks any deeper zooming.
    fn native_level_up_zoom(&self) -> f32 {
        Self::native_level_up_zoom_for(self.max_detail_zoom())
    }

    /// Pure fallback for [`Self::native_level_up_zoom`], testable without an
    /// [`AppModel`]: the level-up fires at the earlier of the current 1:1 cap
    /// (`cap`) and the fixed prefetch threshold. `cap`'s fallback path (no
    /// shader or size yet) surfaces as [`MAX_DETAIL_ZOOM`], so this degrades
    /// to exactly the old fixed-threshold trigger before the preview installs.
    #[must_use]
    fn native_level_up_zoom_for(cap: f32) -> f32 {
        cap.min(NATIVE_ZOOM_THRESHOLD)
    }

    /// The cursor to show over the detail preview: a grab hand once the view is
    /// zoomed past the contain fit, and a grasping hand while a grab-pan drag is
    /// active. At fit (or before any zoom) it stays the default arrow.
    ///
    /// Any `zoom > 1.0` already overflows the preview in at least the
    /// constraining axis (`scale = contain_s · 2^(zoom−1) > contain_s`), so
    /// "zoomed" and "able to pan" coincide at that threshold.
    pub(crate) fn detail_cursor_interaction(&self) -> cosmic::iced::mouse::Interaction {
        detail_cursor_for(self.detail_zoom.0, self.detail_panning)
    }

    /// Recompute the zoom cap and pull the current zoom down to it if it shrank
    /// (texture level-up installs a larger texture and RAISES the cap; a
    /// crop/rotation/resize can LOWER it). Uses [`Self::max_detail_zoom_for_wheel`],
    /// so during a native decode the bound is the projected native 1:1 rather
    /// than the coarse texture's — a window/drawer resize mid-load must not
    /// snatch the zoom back to the overview's 100%, and the landing re-clamp
    /// lands on the exact projected value. Pushes the (possibly clamped) view
    /// to the shader so the render and state agree.
    fn reclamp_detail_zoom(&mut self) {
        let max = self.max_detail_zoom_for_wheel();
        if self.detail_zoom.0 > max {
            self.detail_zoom = Zoom(max);
            if let Some(shader) = &mut self.detail_shader {
                shader.set_view(self.detail_zoom.0, self.detail_pan);
            }
        }
    }

    /// The navigational view currently shown (which owns a context drawer).
    fn current_view(&self) -> DrawerView {
        if self.selected.is_some() {
            DrawerView::Detail
        } else if self.active.is_some() {
            DrawerView::Grid
        } else {
            DrawerView::Library
        }
    }

    /// Whether `view`'s drawer has a valid panel (so a toggle or restore never
    /// opens an empty drawer).
    fn view_drawer_valid(&self, view: DrawerView) -> bool {
        match view {
            DrawerView::Library => {
                self.active.is_none()
                    && matches!(&self.library_selection, Some(LibrarySelection::Roll(_)))
            }
            DrawerView::Grid => self.active.is_some() && self.frame_selected.is_some(),
            DrawerView::Detail => self.selected.is_some(),
        }
    }

    /// Point the context drawer at `view`'s panel with that view's remembered
    /// open/closed state, clamped to validity. Called on every view transition
    /// so the drawer follows the current view without ever closing on
    /// navigation.
    fn restore_drawer_for(&mut self, view: DrawerView) {
        self.context_page = view.page();
        let open = self.drawer_memory.get(view) && self.view_drawer_valid(view);
        self.core_mut().set_show_context(open);
    }

    /// Reseeds the roll-info drawer's date drafts whenever the library roll
    /// selection no longer matches the key the drafts were seeded from (a
    /// different roll always starts from its own committed dates; a stale
    /// selection such as the Add Roll tile clears the key so the next roll
    /// reseeds). Cheap: a path comparison — it runs at the top of `update` for
    /// every message.
    fn sync_roll_date_drafts(&mut self) {
        let Some(LibrarySelection::Roll(dir)) = self.library_selection.clone() else {
            self.roll_date_drafts.key = None;
            return;
        };
        if self.roll_date_drafts.key.as_deref() == Some(dir.as_path()) {
            return;
        }
        let (start, end, name) =
            self.rolls
                .iter()
                .find(|roll| roll.dir == dir)
                .map_or((None, None, None), |roll| {
                    (
                        roll.start_date.as_deref(),
                        roll.end_date.as_deref(),
                        Some(roll.name.as_str()),
                    )
                });
        self.roll_date_drafts = RollDateDrafts::from_dates(dir, start, end);
        self.roll_name_draft = name.unwrap_or("").to_owned();
    }

    /// Kicks off the lazy EXIF parse for the highlighted frame if the frame-info
    /// drawer needs it and it hasn't been parsed (or failed) already. Called on
    /// opening the drawer and whenever the highlight changes while it is open;
    /// returns the spawned task for the caller to run.
    fn ensure_frame_info_loaded(&mut self) -> Option<Task<cosmic::Action<Message>>> {
        if self.context_page != ContextPage::FrameInfo || !self.core.window.show_context {
            return None;
        }
        let name = self.frame_selected.clone()?;
        let dir = self.active.clone()?;
        let needs_parse = self
            .tiles
            .iter()
            .find(|tile| tile.name == name)
            .is_some_and(|tile| tile.meta.is_none() && !tile.meta_failed);
        if !needs_parse || self.frame_meta_inflight.as_deref() == Some(name.as_str()) {
            return None;
        }
        self.frame_meta_inflight = Some(name.clone());

        Some(cosmic::task::future(async move {
            let parse_name = name.clone();
            let result =
                tokio::task::spawn_blocking(move || load_frame_meta(&dir, &parse_name)).await;
            Message::FrameInfoReady(name, result.unwrap_or(Err(FrameError::Meta)))
        }))
    }

    /// Writes the in-memory roll edits to the open roll's manifest file on disk.
    fn persist_roll(&self) {
        let Some(dir) = &self.active else {
            return;
        };
        if let Err(err) = edit_manifest::save_roll_manifest(dir, &self.roll) {
            log::error!("failed to save edits: {err}");
        }
    }

    /// Persists the user-controlled `rolls` list to the app config, so added
    /// rolls survive restarts. The existing config file is written in place
    /// (a fresh `Config` context, matching the one used at load).
    fn persist_config(&self) {
        let Ok(context) = cosmic_config::Config::new(Self::APP_ID, Config::VERSION) else {
            return;
        };
        if let Err(err) = self.config.write_entry(&context) {
            log::error!("failed to save config: {err}");
        }
    }

    /// Writes the open frame's exposure live: the RAM roll edit and the GPU
    /// shader uniform. The slider (`ExposureChanged`) and a keyboard step
    /// (`EditAdjust::Exposure`) both route here; committing (persist + re-bake)
    /// stays separate so the sliders can stream drags without re-decoding
    /// thumbnails on every move.
    fn set_exposure(&mut self, ev: f32) {
        self.exposure_ev = ev;
        if let Some(selected) = &self.selected {
            self.roll.set_exposure(selected, ev);
        }
        if let Some(shader) = &mut self.detail_shader {
            shader.set_exposure(ev);
        }
    }

    /// Writes the open frame's tone-curve controls live: the RAM roll edit and
    /// the GPU shader uniforms. The slider (`CurveChanged`) and a keyboard step
    /// (`EditAdjust::Contrast`/`Highlights`/`Shadows`) both route here;
    /// committing (persist + re-bake) stays separate like
    /// [`Self::set_exposure`].
    ///
    /// The three values are stored as powers (the positive-path curve domain),
    /// but the film path re-reads them as STOP LIFTS via `log2`/`-log2` and
    /// applies them in the density domain (Contrast scales the density window;
    /// Highlights/Shadows are the toe/shoulder masks). See
    /// `docs/tone-model-density.md`.
    fn set_curve(&mut self, contrast: f32, highlights: f32, shadows: f32) {
        self.curve_contrast = contrast;
        self.curve_highlights = highlights;
        self.curve_shadows = shadows;
        if let Some(selected) = &self.selected {
            self.roll.set_curve(selected, contrast, highlights, shadows);
        }
        if let Some(shader) = &mut self.detail_shader {
            shader.set_curve(contrast, highlights, shadows);
        }
    }

    /// Applies a keyboard-shortcut edit adjustment, mirroring the slider
    /// messages: mutate the RAM roll edit and the live shader only — the
    /// commit (persist + re-bake) happens on key release via
    /// [`Self::commit_edit`], the same cadence as a slider drag/release.
    /// No-ops unless a detail view is open (there is nothing to edit
    /// otherwise).
    fn apply_edit_adjust(&mut self, adjust: EditAdjust) {
        // Only adjust while a detail view is up, so the bindings never mutate
        // edits for an unseen frame.
        if self.selected.is_none() {
            return;
        }

        match adjust {
            EditAdjust::Exposure(delta) => {
                let ev = clamp_ev(self.exposure_ev + delta);
                self.set_exposure(ev);
            }
            // Each tone arm only changes one power; the others keep their
            // current values so the composed curve stays fully defined.
            // Contrast steps its stop lift (like the other two), so the
            // "increase" keys `]` raise the mid-pivoted power.
            EditAdjust::Contrast(delta) => {
                let contrast =
                    contrast_power_for_lift(contrast_lift(self.curve_contrast) + delta);
                self.set_curve(contrast, self.curve_highlights, self.curve_shadows);
            }
            // Highlights and Shadows step their user-facing LIFT value in
            // stops, so the "increase" keys `'`/`.` lift/brighten the region
            // exactly like dragging their slider right — keeping keyboard and
            // slider directions in lockstep. The two arms use opposite
            // power↔lift maps because their pivots sit at opposite ends: a
            // highlight lift raises the shadow-pivoted power, a shadow lift
            // lowers the white-pivoted power.
            EditAdjust::Highlights(delta) => {
                let highlights =
                    highlight_power_for_lift(highlight_lift(self.curve_highlights) + delta);
                self.set_curve(self.curve_contrast, highlights, self.curve_shadows);
            }
            EditAdjust::Shadows(delta) => {
                let shadows =
                    shadow_power_for_lift(shadow_lift(self.curve_shadows) + delta);
                self.set_curve(self.curve_contrast, self.curve_highlights, shadows);
            }
            // A display rotation steps one quarter-turn counter-clockwise
            // (authoring the composite of the crop + the EXIF-upright frame;
            // the crop margins themselves are untouched). Live-only until the
            // trim key's release commits, mirroring the numeric adjusts.
            EditAdjust::RotateCcw => {
                self.apply_rotate_ccw();
            }
        }
    }

    /// Translates the crop window by `delta` source pixels in `direction`,
    /// writing the RAM roll edit and the live GPU shader uniform. Bounded by the
    /// FULL-RESOLUTION display-oriented source dims the persisted crop is
    /// authored against (`DetailProgram::source_dimensions`); with no ready
    /// detail (decode still in flight) it stays a no-op. Live-only until the
    /// edit key's release (or crop-mode close) commits.
    fn apply_crop_move(&mut self, direction: edit_manifest::CropDirection, delta: i32) {
        let Some((w, h)) = self
            .detail_shader
            .as_ref()
            .map(shader::DetailProgram::source_dimensions)
        else {
            return;
        };
        let next = move_crop_box(self.crop, direction, delta, w, h);
        self.crop = next;
        if let Some(selected) = &self.selected {
            self.roll.set_crop(selected, next);
        }
        if let Some(shader) = &mut self.detail_shader {
            shader.set_crop(next);
        }
        self.reclamp_detail_zoom();
    }

    /// Resizes the crop window about its center by `delta` source pixels
    /// (positive grows, negative shrinks; aspect preserved), writing the RAM
    /// roll edit and the live GPU shader uniform. Bounded by the
    /// FULL-RESOLUTION display-oriented source dims (see
    /// [`Self::apply_crop_move`]); with no ready detail it stays a no-op.
    /// Live-only until the edit key's release (or crop-mode close) commits.
    fn apply_crop_resize(&mut self, delta: i32) {
        let Some((w, h)) = self
            .detail_shader
            .as_ref()
            .map(shader::DetailProgram::source_dimensions)
        else {
            return;
        };
        let next = resize_crop_box(self.crop, delta, w, h);
        self.crop = next;
        if let Some(selected) = &self.selected {
            self.roll.set_crop(selected, next);
        }
        if let Some(shader) = &mut self.detail_shader {
            shader.set_crop(next);
        }
        self.reclamp_detail_zoom();
    }

    /// Rotates the open frame's display one quarter-turn counter-clockwise by
    /// writing the RAM roll edit and the live GPU shader uniform. Shared by the
    /// keyboard shortcut (via [`Self::apply_edit_adjust`]) and the editing
    /// drawer button (which additionally persists via [`Self::commit_edit`]).
    fn apply_rotate_ccw(&mut self) {
        self.rotation = (self.rotation + 1) & 3;
        if let Some(selected) = &self.selected {
            self.roll.set_rotation(selected, self.rotation);
        }
        if let Some(shader) = &mut self.detail_shader {
            shader.set_rotation(self.rotation);
        }
        self.reclamp_detail_zoom();
    }

    /// Copies the focused frame's full edit (exposure + tone curve) to the
    /// edit clipboard for a later [`Self::paste_edits`]. The focus is the
    /// primary grid frame (`frame_selected`), falling back to the open detail
    /// frame `selected`. No-op when there is nothing focused/selected.
    fn copy_edits(&mut self) {
        let source = self.frame_selected.as_deref().or(self.selected.as_deref());
        if let Some(source) = source {
            self.clipboard = Some(self.roll.tone(source));
        }
    }

    /// Pastes the copied edit onto every multi-selected frame (or the focused
    /// frame when nothing else is selected): writes each target's full edit to
    /// the RAM manifest, updates the live shader to the pasted values when the
    /// open detail frame is a target, persists, and re-bakes the affected grid
    /// thumbnails (plus the roll cover if it is one of them).
    fn paste_edits(&mut self) -> Task<cosmic::Action<Message>> {
        let Some(tone) = self.clipboard else {
            return Task::none();
        };

        // Target frames: the multi-selection, else the focused (or open) frame.
        let mut targets: Vec<String> = self
            .selected_frames
            .iter()
            .cloned()
            .chain(self.frame_selected.clone())
            .chain(self.selected.clone())
            .collect();
        targets.sort();
        targets.dedup();
        if targets.is_empty() {
            return Task::none();
        }

        // Apply to the RAM manifest for every target.
        for name in &targets {
            self.roll.set_tone(name, tone);
        }

        // If the live detail frame is a target, sync the on-screen preview
        // state and the GPU shader to the pasted values.
        if let Some(open) = self.selected.as_deref()
            && targets.iter().any(|name| name == open)
        {
            self.exposure_ev = tone.exposure_ev;
            self.curve_contrast = tone.curve_contrast;
            self.curve_highlights = tone.curve_highlights;
            self.curve_shadows = tone.curve_shadows;
            if let Some(shader) = &mut self.detail_shader {
                shader.set_exposure(tone.exposure_ev);
                shader.set_curve(tone.curve_contrast, tone.curve_highlights, tone.curve_shadows);
            }
        }

        self.persist_roll();

        // Re-bake the affected tiles so the grid reflects the new edits.
        let mut tasks = Vec::with_capacity(2);
        for name in &targets {
            if let Some(tile) = self.tiles.iter_mut().find(|tile| &tile.name == name) {
                tile.thumb = Thumb::Loading;
            }
        }
        tasks.push(self.decode_next());
        // If the open roll's cover is among the targets, its library card also
        // re-bakes to mirror the edit.
        if let Some(active) = self.active.as_ref()
            && let Some(roll) = self.rolls.iter_mut().find(|roll| {
                roll.dir == *active
                    && roll
                        .cover
                        .as_deref()
                        .is_some_and(|cover| targets.iter().any(|t| t == cover))
            })
        {
            roll.thumb = Thumb::Loading;
            tasks.push(self.decode_covers());
        }
        Task::batch(tasks)
    }

    /// Removes a roll from the library, dropping its directory from the
    /// persisted config so it stops appearing as a roll card.
    ///
    /// Non-destructive and reversible: the on-disk files and the roll's edit
    /// manifest are left untouched — only the library listing changes, so the
    /// roll can be re-added later without losing edits. No-ops when the roll
    /// is not in the library (e.g. a stale reference) or while it is open.
    fn remove_roll(&mut self, dir: &Path) -> Task<cosmic::Action<Message>> {
        // Only the library page owns the roll list; an open roll cannot be
        // removed from under its frame grid.
        if self.active.is_some() {
            return Task::none();
        }
        let removed = self.rolls.iter().any(|roll| roll.dir == dir);
        if !removed {
            return Task::none();
        }

        self.config
            .rolls
            .retain(|candidate| Path::new(candidate) != dir);
        self.persist_config();
        self.rolls.retain(|roll| roll.dir != dir);

        // If the removed roll was selected (driving the RollInfo drawer and
        // the keyboard highlight), clear the selection so no stale target
        // remains and close that drawer.
        if matches!(self.library_selection.as_ref(), Some(LibrarySelection::Roll(sel)) if *sel == dir)
        {
            self.library_selection = None;
            if self.context_page == ContextPage::RollInfo {
                self.core_mut().set_show_context(false);
                self.drawer_memory.set(DrawerView::Library, false);
            }
        }
        // Keep a selection alive: fall back to the first remaining roll when
        // more exist.
        self.select_first_visible_roll();
        Task::none()
    }

    /// Handle the completion of a hi-res detail decode, applying the result
    /// only if it matches the current selection and re-pumping if superseded.
    #[allow(clippy::too_many_lines)] // progressive-level landing branches are verbose but linear
    fn handle_detail_ready(
        &mut self,
        name: &str,
        preset: FilmPreset,
        result: Result<DetailDecode, FrameError>,
    ) -> Task<cosmic::Action<Message>> {
        if detail_result_is_current(
            self.selected.as_deref(),
            self.detail_inflight.as_deref(),
            name,
        ) {
            // Only the first load of an image restarts the thumbnail
            // crossfade; a native level-up swap must not re-flash the
            // (normally faded) thumb over the fresh hi-res texture.
            let fresh_open = self.detail_shader.is_none();
            let mut landed = false;
            match result {
                Ok(DetailDecode {
                    mono,
                    width,
                    height,
                    src_long_edge,
                    inversion,
                }) => {
                    landed = true;
                    detail_trace(format_args!(
                        "arrived ok: {name} {}x{} (src long edge {src_long_edge}), fresh={fresh_open}, zoom={:.3}",
                        width, height, self.detail_zoom.0
                    ));
                    // Stash a fresh overview decode into the LRU so returning
                    // to this frame is instant. Only the level-1 overview is
                    // cached (never a native level-up swap). Edits are applied
                    // in-shader at open time, so the cached mono stays valid
                    // across edit changes. Any evicted entry drops here, freeing
                    // its memory eagerly.
                    if fresh_open && let Some(dir) = self.active.clone() {
                        // Bound so the evicted entry (if any) is dropped here,
                        // freeing its memory eagerly; the value is otherwise
                        // unused.
                        let _evicted = self.detail_cache.insert(
                            (dir.clone(), preset, name.to_string()),
                            DetailMono {
                                mono: mono.clone(),
                                width,
                                height,
                                src_long_edge,
                                inversion,
                            },
                        );
                    }
                    // Cache the current thumbnail so it stays visible over the
                    // shader during the crossfade (first load only — a level-up
                    // swap must not re-insert the faded thumb).
                    if fresh_open
                        && let Some(tile) =
                            self.tiles.iter().find(|t| t.name == name).and_then(|t| {
                                match &t.thumb {
                                    Thumb::Ready(h) => Some(h.clone()),
                                    _ => None,
                                }
                            })
                    {
                        self.detail_thumb = Some(tile);
                    }
                    let image_id = self.next_image_id;
                    self.next_image_id = self.next_image_id.wrapping_add(1);
                    self.detail_shader = Some(shader::DetailProgram::new(
                        mono,
                        width,
                        height,
                        self.exposure_ev,
                        self.crop,
                        self.rotation,
                        image_id,
                        src_long_edge,
                        inversion,
                    ));
                    // Carry over any zoom/pan the user applied while the decode
                    // was in flight (the program starts at contain fit).
                    if let Some(shader) = &mut self.detail_shader {
                        shader.set_view(self.detail_zoom.0, self.detail_pan);
                        shader.set_curve(
                            self.curve_contrast,
                            self.curve_highlights,
                            self.curve_shadows,
                        );
                        shader.set_crop(self.crop);
                        shader.set_rotation(self.rotation);
                        // A fresh program starts with the mask off; re-assert
                        // the current crop-mode state so the dim overlay
                        // survives the level-up re-install.
                        shader.set_show_mask(self.crop_mode);
                        shader.set_pad(if self.crop_mode {
                            CROP_MODE_PADDING
                        } else {
                            0.0
                        });
                    }
                    // The native texture widens the 1:1 cap; re-derive it.
                    self.reclamp_detail_zoom();
                    // The level-up is one-shot: a decode that lands after the
                    // first shader IS the native one, and an overview that was
                    // decoded at its native resolution (sensor long edge ≤
                    // `HI_RES_SIZE`) is already full-res — in both cases there
                    // is nothing more to decode. The sensor's true long edge
                    // (post-crop, pre-downscale) is the source of truth here,
                    // NOT the capped overview width: a large sensor downscaled
                    // just under 2048 must still level up to native.
                    if !fresh_open || src_long_edge <= HI_RES_SIZE {
                        self.detail_native_queued = true;
                    }
                }
                Err(err) => {
                    detail_trace(format_args!(
                        "arrived err: {name}, fresh={fresh_open}, zoom={:.3}",
                        self.detail_zoom.0
                    ));
                    // Wheel-driven re-decodes of the same failing frame land
                    // here repeatedly; log the reason once per selection so an
                    // undecodable file shows one clear line instead of one per
                    // notch.
                    if self.detail_logged_failure.as_deref() != Some(name) {
                        log::error!("detail decode failed for {name}: {err}");
                        self.detail_logged_failure = Some(name.to_string());
                    }
                }
            }
            if fresh_open {
                self.detail_thumb_opacity = 1.0;
                self.detail_last_frame = None;
            }
            self.detail_inflight = None;

            // Landing-time re-pump: a wheel taken while a decode was running
            // fired the trigger into an occupied slot and it was swallowed.
            // If the user is already at/past the level-up zoom once a decode
            // lands, start the next level onto the just-freed slot — so
            // zooming during the overview load still reaches native. Gated on
            // the landing having succeeded: an error must not spin the slot.
            if landed && !self.detail_native_queued {
                let trigger = self.native_level_up_zoom();
                if self.detail_zoom.0 >= trigger {
                    detail_trace(format_args!(
                        "landing re-pump: zoom {:.3} >= native level-up zoom {:.3}",
                        self.detail_zoom.0, trigger
                    ));
                    return self.decode_detail_next();
                }
            }
        } else {
            detail_trace(format_args!("arrived superseded: {name}"));

            // A superseded decode finished and freed the single slot;
            // start the current selection's queued request, if any.
            self.detail_inflight = None;

            return self.decode_detail_next();
        }

        Task::none()
    }
}

/// Logs a detail-pump trace line (`RUST_LOG`-gated trace level; the
/// `EXPOSURE_TRACE_DETAIL` knob forces it). Allocation-free via
/// [`std::fmt::Arguments`]; used to diagnose the progressive two-level detail
/// decode — trigger fire/swallow, arrival, and landing re-pump.
fn detail_trace(args: std::fmt::Arguments<'_>) {
    log::trace!("[detail] {args}");
}

/// Logs a thumbnail re-bake / frame-decode trace line (the `EXPOSURE_TRACE_REBAKE`
/// knob forces the trace level). Mirrors `detail_trace`; used to diagnose why a
/// committed edit (crop or tone) is not showing up in the grid thumbnail / roll
/// cover.
fn rebake_trace(args: std::fmt::Arguments<'_>) {
    log::trace!("[rebake] {args}");
}

/// Loads one roll's metadata: display name (manifest label, falling back to
/// the directory leaf), cover file (first sorted non-dot file), the count of
/// frame files, and the film preset recorded in the roll's edit manifest
/// (defaulting to the non-inverted `None`) — with nothing decoded yet.
async fn load_roll(dir: PathBuf) -> Roll {
    let leaf = dir
        .file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| dir.to_string_lossy().into_owned(), str::to_string);
    let (cover, frame_count) = roll_cover_and_count(&dir).await;
    let manifest = edit_manifest::load_roll_manifest(&dir);
    let name = manifest.name().unwrap_or(&leaf).to_owned();
    let start_date = manifest.start_date().map(str::to_owned);
    let end_date = manifest.end_date().map(str::to_owned);
    Roll {
        dir,
        name,
        cover,
        frame_count,
        preset: manifest.preset(),
        base_mode: manifest.base_mode(),
        start_date,
        end_date,
        thumb: Thumb::Loading,
    }
}

/// Loads roll metadata for each configured roll directory, de-duplicated and
/// sorted by display name. Name order is the stable backing order: the library
/// view derives its date-descending display order from it per render.
async fn load_rolls(rolls: Vec<String>) -> Vec<Roll> {
    let mut seen = HashSet::new();
    let mut loaded = Vec::with_capacity(rolls.len());
    for entry in rolls {
        let dir = PathBuf::from(entry);
        if !seen.insert(dir.clone()) {
            continue;
        }
        loaded.push(load_roll(dir).await);
    }
    loaded.sort_by(|a, b| a.name.cmp(&b.name));
    loaded
}


/// Scans a roll directory once: returns the first regular non-dot frame file
/// name in sorted order — the roll's cover, if it has any negatives yet —
/// alongside the count of frame files (both `None`/0 for a missing or empty
/// directory). A single pass covers the cover thumbnail and the
/// metadata-drawer frame count. Export artifacts (JPEG/PNG) are not frames.
async fn roll_cover_and_count(dir: &Path) -> (Option<String>, usize) {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return (None, 0);
    };

    let mut files = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.file_type().await.is_ok_and(|ty| ty.is_file())
            && let Some(name) = entry.file_name().into_string().ok()
            && !name.starts_with('.')
            && !is_export_artifact(&name)
        {
            files.push(name);
        }
    }

    let count = files.len();
    files.sort();
    (files.into_iter().next(), count)
}

/// Scans a roll directory for its frame files and returns their sorted names.
/// Dotfiles (including the edit manifest) and export artifacts (JPEG/PNG) are
/// never shown as tiles.
async fn load_files_in(dir: PathBuf) -> Vec<String> {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return Vec::new();
    };

    let mut files = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        if entry.file_type().await.is_ok_and(|ty| ty.is_file())
            && let Some(name) = entry.file_name().into_string().ok()
            && !name.starts_with('.')
            && !is_export_artifact(&name)
        {
            files.push(name);
        }
    }

    files.sort();
    files
}










/// The add-roll dialog's "film preset" choice: whether the chosen folder holds
/// already-positive scans (regular RAWs, the non-inverted default), the generic
/// B&W profile, or a specific film stock. Mirrors the roll-info drawer's film
/// dropdown ordering ([`FILM_CHOICES`]). The response returns the selected key,
/// which the picker resolves back into a [`FilmPreset`]. (The base strategy is
/// not part of the dialog — the cosmic open dialog carries a single choice — so
/// a new roll starts on the preset base and the user sets auto in the roll-info
/// drawer.)
#[must_use]
fn roll_preset_choice() -> cosmic::dialog::file_chooser::Choice {
    let mut choice =
        cosmic::dialog::file_chooser::Choice::new("preset", &fl!("preset-label"), "none");
    for preset in FILM_CHOICES {
        match preset.stock() {
            // `None` (no inversion) uses its fluent label; stock profiles show
            // their display name.
            Some(stock) => choice = choice.insert(preset.choice_key(), stock.name),
            None => choice = choice.insert(preset.choice_key(), &fl!("preset-none")),
        }
    }
    choice
}

/// Opens the system folder picker, and on success emits [`Message::RollAdded`]
/// for the chosen directory (a cancel or portal failure is a no-op). The dialog
/// carries the film-preset choice, which lands in the [`Message::RollAdded`]
/// payload alongside the directory. Shared by the double-click handler and
/// Enter on a selected Add Roll tile.
fn open_roll_picker() -> Task<cosmic::Action<Message>> {
    cosmic::task::future(async {
        let dialog = cosmic::dialog::file_chooser::open::Dialog::new()
            .choice(roll_preset_choice())
            .open_folder();
        match dialog.await {
            Ok(response) => {
                let Ok(dir) = response.url().to_file_path() else {
                    return Message::Ignore;
                };
                // The response lists the selected (id, value) pairs; the
                // "preset" choice carries the preset key. A backend that
                // dropped the choice falls back to the default (None).
                let preset = response
                    .choices()
                    .iter()
                    .find(|(id, _)| id == "preset")
                    .map_or(FilmPreset::default(), |(_, value)| {
                        FilmPreset::from_key(value.as_str())
                    });
                Message::RollAdded(dir, preset)
            }
            // Cancelled (or a portal failure) is a no-op.
            Err(_) => Message::Ignore,
        }
    })
}



/// Updates the multi-selection after a frame tile is clicked, given the current
/// modifier state and the visible (filtered) `order` of frames.
///
/// - Plain click: replace the selection with just the clicked frame.
/// - Ctrl+click: toggle the clicked frame's membership.
/// - Shift+click: replace the selection with the inclusive range from the
///   `anchor` through the clicked frame (in `order`); a missing anchor or a
///   clicked frame outside `order` collapses to a plain single selection.
///
/// Returns the new selection set and the new shift-anchor (the clicked frame,
/// unless it was toggled out by a Ctrl+click, in which case the anchor is
/// unchanged so a later Shift+click still sources a valid frame).
fn apply_frame_click(
    mut selected: HashSet<String>,
    clicked: &str,
    ctrl: bool,
    shift: bool,
    anchor: Option<&str>,
    order: &[Tile],
) -> (HashSet<String>, Option<String>) {
    // Index of the clicked frame in the grid order (None when not present).
    let click_idx = order.iter().position(|tile| tile.name == clicked);

    if shift {
        // Range: anchor through clicked, inclusive, in display order.
        if let Some(anchor) = anchor
            && let (Some(click_idx), Some(anchor_idx)) =
                (click_idx, order.iter().position(|tile| tile.name == anchor))
        {
            let (lo, hi) = if anchor_idx <= click_idx {
                (anchor_idx, click_idx)
            } else {
                (click_idx, anchor_idx)
            };
            selected.clear();
            for tile in &order[lo..=hi] {
                selected.insert(tile.name.clone());
            }
            return (selected, Some(clicked.to_owned()));
        }
        // No usable anchor: fall through to a plain single selection.
    }

    if ctrl {
        if selected.contains(clicked) {
            selected.remove(clicked);
            // Keep the anchor stable (the toggled-off frame is gone from the
            // set but may still be the focus); return the old anchor.
            return (selected, anchor.map(str::to_owned));
        }
        selected.insert(clicked.to_owned());
        return (selected, Some(clicked.to_owned()));
    }

    // Plain click: single selection.
    selected.clear();
    selected.insert(clicked.to_owned());
    (selected, Some(clicked.to_owned()))
}

/// Column count for a fluid grid of `THUMB_SIZE` cells at `available` width,
/// mirroring iced's `Grid::fluid`/`Constraint::MaxWidth` math exactly
/// (`ceil((available + spacing) / (max + spacing))`).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn grid_num_cols(available: f32, spacing: f32) -> usize {
    (((available + spacing) / (THUMB_SIZE + spacing)).ceil()) as usize
}

/// The absolute scroll offset (content-space y) that brings the tile at
/// `index` into the grid's visible viewport, or `None` when it is already
/// fully visible. Row geometry mirrors `Grid`: each square cell is `cell_width`
/// tall and rows advance by `cell_width + spacing`, with the grid inset by
/// `padding` inside the scrollable content.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
fn reveal_target_y(
    cols: usize,
    index: usize,
    spacing: f32,
    padding: f32,
    cell_width: f32,
    viewport_height: f32,
    viewport_offset_y: f32,
    content_height: f32,
) -> Option<f32> {
    if cols == 0 {
        return None;
    }
    let row = index / cols;
    let top = padding + row as f32 * (cell_width + spacing);
    let bottom = top + cell_width;

    let target = if top < viewport_offset_y {
        top
    } else if bottom > viewport_offset_y + viewport_height {
        bottom - viewport_height
    } else {
        return None;
    };

    // Clamp to the scroller's range so an estimate can never overshoot.
    let max = (content_height - viewport_height).max(0.0);
    Some(target.clamp(0.0, max))
}


/// Converts a wheel scroll delta into a detail-view zoom change (in `log2`
/// units, so +1 = double the rendered scale, −1 = halve it).
///
/// A single wheel notch (one line) zooms half a unit; trackpad pixel deltas
/// treat a ~400 px swipe as one full unit.
pub(crate) fn detail_zoom_delta(delta: cosmic::iced::mouse::ScrollDelta) -> f32 {
    match delta {
        cosmic::iced::mouse::ScrollDelta::Lines { y, .. } => y * 0.5,
        cosmic::iced::mouse::ScrollDelta::Pixels { y, .. } => y / 400.0,
    }
}

/// The detail-preview cursor for a given zoom (`1.0` = contain fit) and
/// grab-pan state: a grasping hand while dragging, a grab hand once the view is
/// zoomed in far enough to pan, and the default arrow otherwise. Pure so the
/// policy is unit-tested; [`AppModel::detail_cursor_interaction`] feeds it the
/// live state.
#[must_use]
fn detail_cursor_for(zoom: f32, panning: bool) -> cosmic::iced::mouse::Interaction {
    use cosmic::iced::mouse::Interaction;
    if panning {
        Interaction::Grabbing
    } else if zoom > 1.0 {
        Interaction::Grab
    } else {
        Interaction::None
    }
}

/// Applies a wheel zoom `delta` (log2 units) to the detail view, clamping to
/// `[1.0, max_zoom]` (the 1:1 "100%" cap, see [`AppModel::max_detail_zoom`]).
/// Zooming all the way back out to contain fit re-centers the image. Returns
/// the new `(zoom, pan)`.
fn apply_detail_zoom(
    zoom: f32,
    pan: Point,
    cursor: Option<Point>,
    delta: f32,
    max_zoom: f32,
) -> (f32, Point) {
    let new_zoom = (zoom + delta).clamp(1.0, max_zoom);
    // At contain fit the whole frame must be centered. `zoom + delta <= 1.0`
    // is equivalent to `new_zoom == 1.0` because of the clamp above.
    let new_pan = if zoom + delta <= 1.0 {
        Point::default()
    } else {
        cursor.map_or(pan, |cursor| zoom_about_anchor(zoom, new_zoom, pan, cursor))
    };
    (new_zoom, new_pan)
}

/// Computes the pan that keeps the image point under `cursor` fixed on
/// screen while the zoom changes from `zoom_old` to `zoom_new` (log2 units).
///
/// The rendered size is proportional to `2^(zoom-1)`, so the unknown
/// contain-fit scale cancels:
/// `off1 = off0 + (1 - 2^(z1-z0)) * (cursor - off0)`.
/// Both `pan` and `cursor` are relative to the widget center; the widget
/// center itself never enters the formula.
fn zoom_about_anchor(zoom_old: f32, zoom_new: f32, pan: Point, cursor: Point) -> Point {
    let ratio = (zoom_new - zoom_old).exp2();
    let k = 1.0 - ratio;
    Point::new(
        pan.x + k * (cursor.x - pan.x),
        pan.y + k * (cursor.y - pan.y),
    )
}

/// Clamps an exposure adjustment in EV to the slider's range (−3.0..=+4.0).
fn clamp_ev(ev: f32) -> f32 {
    ev.clamp(-3.0, 4.0)
}

/// Clamps the Contrast power (a mid-pivoted power) to the slider's range
/// `0.125..=8.0` so a keyboard shortcut and the slider agree on bounds. The
/// range is a symmetric reciprocal pair around the `1.0` identity, so the
/// stop-based lift scale spans an even ±3 stops.
fn clamp_contrast_power(power: f32) -> f32 {
    power.clamp(0.125, 8.0)
}

/// Clamps a Highlights/Shadows region power to the slider's range
/// `0.25..=4.0`. A symmetric reciprocal pair around `1.0`, spanning ±2 stops
/// like [`clamp_contrast_power`] but narrower, since the region controls act on
/// one end of the tone range.
fn clamp_tone_lift_power(power: f32) -> f32 {
    power.clamp(0.25, 4.0)
}

/// The Contrast slider's user-facing "lift value" in stops: `+log2(power)`, so
/// INCREASING it raises contrast (steeper mid slope). Identity at `0.0` (power
/// `1.0` ⇔ 0 stops), at the CENTER of the symmetric `−3..=3` track; power 2.0
/// is +1 stop, power 0.5 is −1 stop.
pub(crate) fn contrast_lift(power: f32) -> f32 {
    power.log2()
}

/// The stored contrast power for a lift value (in stops) picked by a slider or
/// keyboard step. Bound-sensitive inverse of [`contrast_lift`] across the power
/// range (round-trip `lift ↔ power` is exact within the range).
pub(crate) fn contrast_power_for_lift(value: f32) -> f32 {
    clamp_contrast_power(value.exp2())
}

/// The Shadows slider's user-facing "lift value" in stops: `-log2(power)`, so
/// INCREASING it lifts/brightens the lower tones. The shadows power pivots at
/// the white point, so a lift drives the power BELOW 1.0 (a lower-tones lift
/// with white pinned). Identity at `0.0` (power 1.0 ⇔ 0 stops), at the CENTER
/// of the symmetric `−2..=2` track; power 0.5 is +1 stop of lift, power 2.0 is
/// −1 stop (crush).
pub(crate) fn shadow_lift(power: f32) -> f32 {
    -(power.log2())
}

/// The stored shadows power for a lift value (in stops) picked by a slider or
/// keyboard step. Bound-sensitive inverse of [`shadow_lift`] across the power
/// range, so a value that exits `−2..=2` clamps to the same endpoints
/// `clamp_tone_lift_power` produces (round-trip `lift ↔ power` is exact within
/// the range).
pub(crate) fn shadow_power_for_lift(value: f32) -> f32 {
    clamp_tone_lift_power((-value).exp2())
}

/// The Highlights slider's user-facing "lift value" in stops: `+log2(power)`,
/// so INCREASING it lifts/brightens the upper tones. The highlights power
/// pivots at the shadow anchor, so a lift drives the power ABOVE 1.0 (an
/// upper-tones lift with the shadow anchor pinned) — the opposite power
/// direction from [`shadow_lift`], which is exactly what makes both sliders
/// brighten their named region on a rightward drag. Identity at `0.0`, centered
/// on the symmetric `−2..=2` track like the shadows arm.
pub(crate) fn highlight_lift(power: f32) -> f32 {
    power.log2()
}

/// The stored highlights power for a lift value (in stops) picked by a slider
/// or keyboard step. Bound-sensitive inverse of [`highlight_lift`] across the
/// power range (round-trip `lift ↔ power` is exact within the range).
pub(crate) fn highlight_power_for_lift(value: f32) -> f32 {
    clamp_tone_lift_power(value.exp2())
}

/// Translates the crop window by `delta_px` in `direction`, keeping the window
/// size (and thus aspect) untouched. The window stays clamped inside the source
/// frame `[0, w] × [0, h]`.
///
/// The margins model the window's top-left as `(left, top)` and its bottom-right
/// as `(w − right, h − bottom)`, so moving the window adjusts the parallel pair
/// by equal-and-opposite amounts: moving right grows `left` and shrinks `right`,
/// moving left the reverse, etc.
/// Maps a screen-space crop move (arrow or `h`/`j`/`k`/`l`) to the EXIF-upright
/// crop-window direction that produces the same VISUAL movement, given the
/// display's cumulative counter-clockwise `rotation` quarter-turns.
///
/// The crop window is authored in the EXIF-upright source frame, while the
/// display applies a CCW `rot` on top (the shader's `rotate_uv`). Screen
/// directions cycle Left → Up → Right → Down; each CCW display turn shifts
/// which EXIF-upright edge is "up", so the texture direction is the screen
/// direction rotated by `rotation` steps in that cycle.
#[must_use]
fn crop_move_direction(dir: MoveDir, rotation: u8) -> edit_manifest::CropDirection {
    use edit_manifest::CropDirection::{Bottom, Left, Right, Top};
    let screen = match dir {
        MoveDir::Left => 0,
        MoveDir::Up => 1,
        MoveDir::Right => 2,
        MoveDir::Down => 3,
    };
    match (screen + u32::from(rotation & 3)) % 4 {
        0 => Left,
        1 => Top,
        2 => Right,
        _ => Bottom,
    }
}

fn move_crop_box(
    crop: edit_manifest::CropMargins,
    direction: edit_manifest::CropDirection,
    delta_px: i32,
    width: u32,
    height: u32,
) -> edit_manifest::CropMargins {
    use edit_manifest::CropDirection::{Bottom, Left, Right, Top};

    let cw = crop.cropped_width(width);
    let ch = crop.cropped_height(height);
    let (left, top) = match direction {
        Left => (
            clamp_window_origin(crop.left, delta_px, false, width.saturating_sub(cw)),
            crop.top,
        ),
        Right => (
            clamp_window_origin(crop.left, delta_px, true, width.saturating_sub(cw)),
            crop.top,
        ),
        Top => (
            crop.left,
            clamp_window_origin(crop.top, delta_px, false, height.saturating_sub(ch)),
        ),
        Bottom => (
            crop.left,
            clamp_window_origin(crop.top, delta_px, true, height.saturating_sub(ch)),
        ),
    };
    edit_manifest::CropMargins {
        top,
        right: width.saturating_sub(left).saturating_sub(cw),
        bottom: height.saturating_sub(top).saturating_sub(ch),
        left,
    }
}

/// Clamps the window origin coordinate after a `delta` move in the given
/// direction (`grow` = the coordinate increases), into `[0, max_origin]`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn clamp_window_origin(origin: u32, delta: i32, grow: bool, max_origin: u32) -> u32 {
    let moved = if grow {
        i64::from(origin) + i64::from(delta)
    } else {
        i64::from(origin) - i64::from(delta)
    };
    moved.clamp(0, i64::from(max_origin)) as u32
}

/// Resizes the crop window about its center by `delta_px`: a positive `delta`
/// grows the window (the long edge changes by `delta_px`, both axes scaled
/// proportionally so the aspect is preserved exactly), a negative one shrinks
/// it. The result is clamped so the long edge stays in
/// `[MIN_CROP_WINDOW, source long edge]` — a resize never collapses the crop
/// or exceeds the source.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resize_crop_box(
    crop: edit_manifest::CropMargins,
    delta_px: i32,
    width: u32,
    height: u32,
) -> edit_manifest::CropMargins {
    let cw = crop.cropped_width(width).max(1);
    let ch = crop.cropped_height(height).max(1);
    let long = u32::max(cw, ch);
    let frame_long = u32::max(width, height);
    // Scale factor so the long edge changes by `delta_px`; both axes scale
    // together, preserving the aspect exactly.
    let delta_f = f64::from(delta_px) / f64::from(long);
    let scale = (1.0 + delta_f).clamp(
        f64::from(MIN_CROP_WINDOW) / f64::from(long),
        f64::from(frame_long) / f64::from(long),
    );
    let nw = (f64::from(cw) * scale).round().clamp(1.0, f64::from(width)) as u32;
    let nh = (f64::from(ch) * scale)
        .round()
        .clamp(1.0, f64::from(height)) as u32;
    // Keep the window's center fixed; recompute margins from it.
    let cx = f64::from(crop.left) + f64::from(cw) / 2.0;
    let cy = f64::from(crop.top) + f64::from(ch) / 2.0;
    let left = (cx - f64::from(nw) / 2.0)
        .round()
        .clamp(0.0, f64::from(width - nw)) as u32;
    let top = (cy - f64::from(nh) / 2.0)
        .round()
        .clamp(0.0, f64::from(height - nh)) as u32;
    edit_manifest::CropMargins {
        top,
        right: width.saturating_sub(left).saturating_sub(nw),
        bottom: height.saturating_sub(top).saturating_sub(nh),
        left,
    }
}

/// Maps a keyboard shortcut character to an [`EditAdjust`], or `None` when the
/// key is not bound. `alt` and `shift` are the event's modifier state.
///
/// The four control pairs are laid out on a US keyboard left-to-right to match
/// the editing panel's control order (Exposure → Contrast → Highlights →
/// Shadows): `-`/`=` exposure, `[`/`]` contrast, `;`/`'` highlights, `,`/`.`
/// shadows. A bare key uses the coarse step; holding `Shift` selects the fine
/// nudge step. The "increase" key of each pair (`=`/`]`/`'`/`.`) applies a
/// POSITIVE step — for the highlights/shadows arms this steps the user-facing
/// lift value (see [`shadow_lift`]/[`highlight_lift`]), so `'`/`.` brighten the
/// region's highlights or shadows.
/// `-`/`=` stay the exposure pair here; the crop-mode handler reinterprets them
/// as the window resize. `r` rotates the display one quarter-turn
/// counter-clockwise (cumulative; modifiers ignored). The VIM movement keys
/// `h`/`j`/`k`/`l` are NOT mapped here — they route to `Message::Nav` in the
/// subscription, mirroring the arrow keys for navigation (and, in crop mode,
/// the crop-window move).
/// The `key` payload
/// is deliberately layout-stable: iced's `keyboard::listen` delivers the
/// unmodified character (`key_without_modifiers`), and the iced fork never
/// reports the `Shift`-produced symbols from the base keys used here, so nudge
/// is driven by the event's modifier state rather than by matching
/// `_`/`+`/`{`/`}`/`:`/`"`/`<`/`>`.
fn edit_adjust_for(key: &str, _alt: bool, shift: bool) -> Option<EditAdjust> {
    let ev = if shift { EDIT_NUDGE_EV } else { EDIT_STEP_EV };
    let curve = if shift {
        EDIT_NUDGE_CURVE
    } else {
        EDIT_STEP_CURVE
    };
    match key {
        "-" => Some(EditAdjust::Exposure(-ev)),
        "=" => Some(EditAdjust::Exposure(ev)),
        "[" => Some(EditAdjust::Contrast(-curve)),
        "]" => Some(EditAdjust::Contrast(curve)),
        ";" => Some(EditAdjust::Highlights(-curve)),
        "'" => Some(EditAdjust::Highlights(curve)),
        "," => Some(EditAdjust::Shadows(-curve)),
        "." => Some(EditAdjust::Shadows(curve)),
        // Rotate the display one quarter-turn counter-clockwise. A bare `r`
        // fires on press and the edit-key release commits the persist + re-bake
        // (the release arm matches through this same function, so `r` is
        // covered); modifiers are ignored like the tone-adjacent keys.
        "r" => Some(EditAdjust::RotateCcw),
        _ => None,
    }
}

/// Decodes a RAW frame from the open roll into a thumbnail message, baking in
/// the given exposure, tone curve and display rotation so the grid tile reflects
/// the stored edits (grid == detail).
async fn decode_thumbnail(
    dir: PathBuf,
    name: String,
    tone: edit_manifest::ToneEdit,
    crop: edit_manifest::CropMargins,
    rotation: u8,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> Message {
    let result = decode_raw(dir, name.clone(), move |image| {
        convert_thumbnail(image, THUMB_SIZE, tone, crop, rotation, preset, base_config)
    })
    .await;

    Message::ThumbReady(name, result)
}

/// Decodes a roll's cover file into a thumbnail message, baking in the cover
/// file's stored exposure, tone curve and display rotation from the roll's
/// manifest so the roll tile preview parallels the edited frame (grid == detail
/// for covers too). A roll with no manifest (or an unedited cover) falls back
/// to identity. The cover's own on-disk manifest also supplies the roll's base
/// mode, so a library-page cover reflects a calibration recorded from the open
/// roll without extra plumbing.
async fn decode_cover(dir: PathBuf, name: String, preset: FilmPreset) -> Message {
    let manifest = edit_manifest::load_roll_manifest(&dir);
    let tone = manifest.tone(&name);
    let crop = manifest.crop(&name);
    let rotation = manifest.rotation(&name) & 3;
    let base_config = manifest.base_config();
    let result = decode_raw(dir.clone(), name, move |image| {
        convert_thumbnail(image, THUMB_SIZE, tone, crop, rotation, preset, base_config)
    })
    .await;

    Message::CoverReady(dir, result)
}

/// Decodes a RAW frame from the open roll into a hi-res message for
/// the detail view, returning the oriented linear mono data that the GPU
/// shader uploads and applies exposure to.  `max_edge` caps the long edge in
/// pixels; the overview level uses [`HI_RES_SIZE`], the native level-up
/// [`MAX_TEXTURE_EDGE`].
async fn decode_detail(
    dir: PathBuf,
    name: String,
    max_edge: u32,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> Message {
    let result = decode_raw_detail(dir, name.clone(), max_edge, preset, base_config).await;
    Message::DetailReady(name, preset, result)
}

/// Decodes a neighbor frame's overview for the preload cache. Unlike
/// [`decode_detail`] this only populates the LRU — it never becomes the active
/// detail shader — so it always decodes at the fixed overview cap.
async fn preload_detail(
    dir: PathBuf,
    name: String,
    preset: FilmPreset,
    base_config: BaseConfig,
) -> Message {
    let result =
        decode_raw_detail(dir.clone(), name.clone(), HI_RES_SIZE, preset, base_config).await;
    Message::DetailPreloaded(dir, name, preset, result)
}

/// Measures ONE frame's clear-film transmission at the overview scale, posting
/// it back as [`Message::CalibrationBaseMeasured`]. The `AutoSelectedFrame`
/// preset's one-shot calibration path: designate a frame, decode it once,
/// and pin the whole roll's black point to its measured plateau. Implausible
/// or missing measurements arrive as `None` (the caller keeps the stock
/// fallback rather than recording a bad base).
async fn measure_frame_base(dir: PathBuf, name: String, preset: FilmPreset) -> Message {
    let base = decode_raw_detail(
        dir.clone(),
        name.clone(),
        HI_RES_SIZE,
        preset,
        BaseConfig::default(),
    )
    .await
    .ok()
    .and_then(|decoded| measure_base(&decoded.mono).filter(|base| *base >= MIN_PLAUSIBLE_BASE));
    Message::CalibrationBaseMeasured(dir, name, base)
}

/// Runs a RAW decode plus conversion on a blocking worker thread so the UI
/// never stalls on CPU-heavy work.
async fn decode_raw<F>(dir: PathBuf, name: String, convert: F) -> Result<Handle, FrameError>
where
    F: Fn(&rawloader::RawImage) -> Result<Handle, FrameError> + Send + 'static,
{
    let path = dir.join(name);

    tokio::task::spawn_blocking(move || {
        let image = match rawloader::decode_file(&path) {
            Ok(image) => image,
            Err(err) => {
                log::error!("failed to decode {}: {err}", path.display());
                return Err(FrameError::Decode {
                    path,
                    message: err.to_string(),
                });
            }
        };
        convert(&image)
    })
    .await
    .unwrap_or(Err(FrameError::ThreadPanic))
}

/// Whether a finished hi-res decode belongs to the current selection and is
/// the one this model dispatched.
///
/// Guards [`Message::DetailReady`] against results from selections that were
/// replaced or closed while their decode was still running.
fn detail_result_is_current(
    selected: Option<&str>,
    inflight: Option<&str>,
    finished: &str,
) -> bool {
    selected == Some(finished) && inflight == Some(finished)
}

/// The context page to display in the context drawer.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum ContextPage {
    #[default]
    About,
    /// The editing panel for the active detail view.
    Editing,
    /// The metadata drawer for the library roll selection (`library_selection`).
    RollInfo,
    /// The frame-info drawer on the frames grid: name, dimensions, and basic
    /// EXIF for the highlighted frame (`frame_selected`).
    FrameInfo,
}

/// The three navigational views that own a context drawer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum DrawerView {
    /// The library page → [`ContextPage::RollInfo`].
    Library,
    /// The open roll's frame grid → [`ContextPage::FrameInfo`].
    Grid,
    /// The open frame's detail view → [`ContextPage::Editing`].
    Detail,
}

impl DrawerView {
    /// The drawer page this view shows.
    #[must_use]
    const fn page(self) -> ContextPage {
        match self {
            Self::Library => ContextPage::RollInfo,
            Self::Grid => ContextPage::FrameInfo,
            Self::Detail => ContextPage::Editing,
        }
    }
}

/// Per-view context-drawer state: whether each view's drawer is open, remembered
/// across navigation. Space toggles only the current view's slot; moving
/// between views restores the incoming view's own remembered state (Esc and
/// navigation never close a drawer).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct DrawerMemory {
    library: bool,
    grid: bool,
    detail: bool,
}

impl DrawerMemory {
    #[must_use]
    const fn get(self, view: DrawerView) -> bool {
        match view {
            DrawerView::Library => self.library,
            DrawerView::Grid => self.grid,
            DrawerView::Detail => self.detail,
        }
    }

    fn set(&mut self, view: DrawerView, open: bool) {
        match view {
            DrawerView::Library => self.library = open,
            DrawerView::Grid => self.grid = open,
            DrawerView::Detail => self.detail = open,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MenuAction {
    AddRoll,
    RemoveRoll,
    Quit,
    SelectAll,
    CopyEdits,
    PasteEdits,
    Export,
    CalibrateBase,
    About,
    Details,
    ToggleCropMode,
}

impl menu::action::MenuAction for MenuAction {
    type Message = Message;

    fn message(&self) -> Self::Message {
        match self {
            MenuAction::AddRoll => Message::AddRoll,
            MenuAction::RemoveRoll => Message::RemoveSelectedRoll,
            MenuAction::Quit => Message::Quit,
            MenuAction::SelectAll => Message::SelectAllFrames,
            MenuAction::CopyEdits => Message::CopyEdits,
            MenuAction::PasteEdits => Message::PasteEdits,
            MenuAction::Export => Message::ExportRequested,
            MenuAction::CalibrateBase => Message::CalibrateBaseFromFrame,
            MenuAction::About => Message::ToggleContextPage(ContextPage::About),
            MenuAction::Details => Message::ToggleContext,
            MenuAction::ToggleCropMode => Message::ToggleCropMode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{
        apply_exposure, bake_tone, class_gains, convert_thumbnail, crop_rgba, crop_samples,
        display_source_dims, downsample_thumbnail, flatten_bayer, luma, pivots_for, resized_dims,
        resize_area, rotate_quarters, scale_crop, srgb_encode, unsharp_mask,
    };

    fn tile(name: &str) -> Tile {
        Tile {
            name: name.to_string(),
            thumb: Thumb::Loading,
            meta: None,
            meta_failed: false,
        }
    }


    #[test]
    fn load_roll_prefers_the_manifest_label_over_the_directory_leaf() {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "exposure-app-roll-name-{}-{seq}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = edit_manifest::RollManifest::default();
        manifest.set_name(Some("Rollerskates".to_owned()));
        edit_manifest::save_roll_manifest(&dir, &manifest).unwrap();
        let leaf = dir.file_name().unwrap().to_string_lossy().into_owned();

        let labeled = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(load_roll(dir.clone()));
        std::fs::remove_dir_all(&dir).unwrap();

        assert_ne!(leaf, "Rollerskates");
        assert_eq!(labeled.name, "Rollerskates");
    }

    #[test]
    fn grid_num_cols_matches_iced_ceil_math() {
        // (1200 + 16) / (384 + 16) = 3.04 → ceil 4
        assert_eq!(grid_num_cols(1200.0, 16.0), 4);
        // (400 + 16) / (384 + 16) = 1.04 → ceil 2
        assert_eq!(grid_num_cols(400.0, 16.0), 2);
    }

    #[test]
    fn reveal_target_y_no_movement_when_visible() {
        // 3-column grid, tile 4 sits in row 1 (top = 16 + 400*1 = 416) inside a
        // 900-high viewport scrolled to 0.
        assert_eq!(
            reveal_target_y(3, 4, 16.0, 16.0, 384.0, 900.0, 0.0, 3000.0),
            None
        );
    }

    #[test]
    fn reveal_target_y_scrolls_down_below_the_fold() {
        // tile 30 in row 10: top = 16 + 400*10 = 4016, bottom = 4400. Viewport
        // is 900 tall, scrolled to 1000 → fold at 1900. Scroll to 4400 - 900.
        assert_eq!(
            reveal_target_y(3, 30, 16.0, 16.0, 384.0, 900.0, 1000.0, 30000.0),
            Some(3500.0)
        );
    }

    #[test]
    fn reveal_target_y_scrolls_up_above_the_viewport() {
        // Tile 2 in row 0 (top = 16) is above a viewport scrolled to 300 —
        // scroll back to 16.
        assert_eq!(
            reveal_target_y(3, 2, 16.0, 16.0, 384.0, 900.0, 300.0, 3000.0),
            Some(16.0)
        );
    }

    #[test]
    fn reveal_target_y_clamps_to_content() {
        // tile 30 in row 10 wants 4400 - 1000 = 3400, but content is only 2700
        // tall, so the scroll clamps to 2700 - 1000 = 1700.
        assert_eq!(
            reveal_target_y(3, 30, 16.0, 16.0, 384.0, 1000.0, 0.0, 2700.0),
            Some(1700.0)
        );
    }

    #[test]
    fn reveal_target_y_zero_columns_is_no_op() {
        assert_eq!(
            reveal_target_y(0, 0, 16.0, 16.0, 384.0, 900.0, 0.0, 3000.0),
            None
        );
    }

    #[test]
    fn detail_result_applies_to_its_own_selection() {
        assert!(detail_result_is_current(Some("a"), Some("a"), "a"));
    }

    #[test]
    fn detail_result_rejected_for_replaced_selection() {
        // "a" finished, but the user already moved on to "b".
        assert!(!detail_result_is_current(Some("b"), Some("a"), "a"));
    }

    #[test]
    fn detail_result_rejected_after_close() {
        assert!(!detail_result_is_current(None, Some("a"), "a"));
    }

    #[test]
    fn detail_result_rejected_without_inflight_request() {
        assert!(!detail_result_is_current(Some("a"), None, "a"));
    }

    #[test]
    fn srgb_encode_matches_reference_values() {
        let assert_close = |value: f32, expected: f32| {
            assert!((srgb_encode(value) - expected).abs() < 1e-6);
        };

        assert_close(0.0, 0.0);
        assert_close(1.0, 1.0);
        assert_close(0.5, 0.735_356_98);
        assert!(srgb_encode(0.25) < srgb_encode(0.5));
        assert_close(-0.5, 0.0);
        assert_close(1.5, 1.0);
    }

    #[test]
    fn flatten_bayer_scales_classes_to_a_common_base() {
        // Neutral film at transmission 0.5 seen through per-class sensor casts
        // (RGGB layout: one R, two G, one B site).
        let cfa = rawloader::CFA::new("RGGB");
        let samples = [0.60, 0.50, 0.50, 0.40]; // R, G, G, B sites

        let mono = flatten_bayer(&samples, 2, 2, &cfa);

        assert!(mono.iter().all(|value| (value - 0.5).abs() < 1e-6));
    }

    #[test]
    fn flatten_bayer_respects_cfa_phase() {
        // GBRG layout: blue sits top-left; only that site carries the cast.
        let cfa = rawloader::CFA::new("GBRG");
        let samples = [0.25, 0.20, 0.30, 0.25]; // G, B, R, G sites

        let mono = flatten_bayer(&samples, 2, 2, &cfa);

        assert!(mono.iter().all(|value| (value - 0.25).abs() < 1e-6));
    }

    #[test]
    fn flatten_bayer_handles_fourth_color_sites() {
        // Emerald-class sites get their own measured base like any other.
        let cfa = rawloader::CFA::new("RGBE");
        let samples = [0.45, 0.50, 0.55, 0.50]; // R, G, B, E sites

        let mono = flatten_bayer(&samples, 2, 2, &cfa);

        assert!(mono.iter().all(|value| (value - 0.5).abs() < 1e-6));
    }

    #[test]
    fn unsharp_mask_leaves_flat_buffers_unchanged() {
        let mut flat = vec![0.42_f32; 16];

        unsharp_mask(&mut flat, 4, 4);

        assert!(flat.iter().all(|value| (value - 0.42).abs() < 1e-6));
    }

    #[test]
    fn unsharp_mask_increases_local_contrast() {
        // Horizontal mid-gray step edge across a single row.
        let mut edge = vec![0.2_f32, 0.2, 0.2, 0.8, 0.8, 0.8];

        unsharp_mask(&mut edge, 6, 1);

        assert!(edge[2] < 0.2); // dark side dips further
        assert!(edge[3] > 0.8); // bright side overshoots
    }

    #[test]
    fn unsharp_mask_clamps_to_unit_range() {
        // A lone bright spike would overshoot past white without clamping.
        let mut spike = vec![0.0_f32; 9];
        spike[4] = 1.0;

        unsharp_mask(&mut spike, 3, 3);

        assert!(spike.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(spike[4], 1.0);
    }

    #[test]
    fn luma_preserves_neutral_levels() {
        assert!((luma(&[1.0, 1.0, 1.0])[0] - 1.0).abs() < 1e-6);
        assert!((luma(&[0.25, 0.25, 0.25])[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn luma_weights_green_over_blue() {
        let green = luma(&[0.0, 1.0, 0.0])[0];
        let blue = luma(&[0.0, 0.0, 1.0])[0];

        assert!(green > blue);
        assert!((green - 0.715_2).abs() < 1e-6);
        assert!((blue - 0.072_2).abs() < 1e-6);
    }

    #[test]
    fn apply_exposure_scales_in_linear_light() {
        let mut mono = [0.5_f32; 8];

        apply_exposure(&mut mono, 1.0);
        assert!(mono.iter().all(|value| (*value - 1.0).abs() < 1e-6));

        apply_exposure(&mut mono, -1.0);
        assert!(mono.iter().all(|value| (*value - 0.5).abs() < 1e-6));

        // EV=0 → 2^0 = 1.0, identity.
        apply_exposure(&mut mono, 0.0);
        assert!(mono.iter().all(|value| (*value - 0.5).abs() < 1e-6));

        // 1 EV doubles linear light at any level.
        let mut dim = [0.125_f32; 4];
        apply_exposure(&mut dim, 1.0);
        assert!(dim.iter().all(|value| (*value - 0.25).abs() < 1e-6));
    }

    #[test]
    fn exposure_math_ev_zero_is_identity() {
        // EV=0 → 2^0 = 1.0, so exposed = mono × 1.0 = mono.
        let mono = 0.5_f32;
        let exposed = mono * 2.0_f32.powf(0.0);
        assert!((exposed - 0.5).abs() < 1e-6);

        // sRGB(0.5) ≈ 188/255.
        let expected_level = (srgb_encode(0.5) * 255.0).round() as u8;
        assert_eq!(expected_level, 188);

        // EV +1 → 0.5 × 2 = 1.0 → sRGB(1.0) = 255.
        let bright = mono * 2.0_f32.powf(1.0);
        let bright_level = (srgb_encode(bright) * 255.0).round() as u8;
        assert_eq!(bright_level, 255);

        // EV −1 → 0.5 × 0.5 = 0.25 → sRGB(0.25) ≈ 137.
        let dark = mono * 2.0_f32.powf(-1.0);
        let dark_level = (srgb_encode(dark) * 255.0).round() as u8;
        assert_eq!(dark_level, 137);

        assert!(bright_level > dark_level);
    }

    #[test]
    fn crop_extracts_inner_region() {
        let samples: Vec<f32> = (0_u16..12).map(f32::from).collect();

        let (cropped, width, height) = crop_samples(&samples, 4, 3, [1, 1, 1, 2]).unwrap();

        assert_eq!((width, height), (1, 1));
        assert_eq!(cropped, vec![6.0]);
    }

    #[test]
    fn crop_rejects_degenerate_regions() {
        assert!(crop_samples(&[0.0; 4], 2, 2, [2, 0, 0, 0]).is_none());
        assert!(crop_samples(&[0.0; 4], 2, 2, [0, 1, 0, 2]).is_none());
    }

    #[test]
    fn resize_area_averages_source_blocks() {
        // Gray ramp 1.0..16.0 over a 4x4 buffer.
        let rgb: Vec<f32> = (1_u16..=16)
            .flat_map(|value| [f32::from(value); 3])
            .collect();

        let (out, width, height) = resize_area(&rgb, 4, 4, 2, 3);

        assert_eq!((width, height), (2, 2));
        assert_eq!(
            out,
            vec![
                3.5, 3.5, 3.5, // mean of {1, 2, 5, 6}
                5.5, 5.5, 5.5, // mean of {3, 4, 7, 8}
                11.5, 11.5, 11.5, // mean of {9, 10, 13, 14}
                13.5, 13.5, 13.5, // mean of {11, 12, 15, 16}
            ]
        );
    }

    #[test]
    fn resize_area_is_identity_without_shrink() {
        let rgb: Vec<f32> = (0_u16..4).flat_map(|value| [f32::from(value); 3]).collect();

        let (out, width, height) = resize_area(&rgb, 2, 2, 384, 3);

        assert_eq!((width, height), (2, 2));
        assert_eq!(out, rgb);
    }

    #[test]
    fn resize_area_handles_single_channel_samples() {
        let samples: Vec<f32> = (1_u16..=4).map(f32::from).collect();

        let (out, width, height) = resize_area(&samples, 2, 2, 1, 1);

        assert_eq!((width, height), (1, 1));
        assert_eq!(out, vec![2.5]); // mean of {1, 2, 3, 4}
    }

    #[test]
    fn resized_dims_caps_the_long_edge_preserving_aspect() {
        // A source already under the ceiling keeps its exact dimensions.
        assert_eq!(resized_dims(3000, 2000, 8192), (3000, 2000));
        // A portrait source under the ceiling is untouched too.
        assert_eq!(resized_dims(2000, 3000, 8192), (2000, 3000));
        // A landscape source over the wgpu ceiling is capped on the long edge,
        // the short axis truncating to the same fractional scale as the real
        // downscale (6000·(8192/9000) = 5461.3 → 5461).
        assert_eq!(resized_dims(9000, 6000, 8192), (8192, 5461));
        // Portrait mirrors it: the capped edge lands on the height axis.
        assert_eq!(resized_dims(6000, 9000, 8192), (5461, 8192));
        // Degenerate dims never divide by zero or shrink below a pixel.
        assert_eq!(resized_dims(0, 3, 8192), (1, 3));
        assert_eq!(resized_dims(1, 1, 8192), (1, 1));
    }

    #[test]
    fn resized_dims_matches_resize_area_output_dims() {
        // The projection must agree with the decoder's own size math to the
        // pixel, so a projected 1:1 cap equals the cap of the real decode.
        for (w, h, max) in [
            (9000, 6000, 8192),
            (6000, 9000, 8192),
            (2000, 1500, 2048),
            (2500, 2000, 2048),
            (800, 600, 2048),
        ] {
            let samples = vec![0.0_f32; w as usize * h as usize];
            let (_, out_w, out_h) = resize_area(&samples, w, h, max, 1);
            assert_eq!((out_w, out_h), resized_dims(w, h, max));
        }
    }

    /// Builds an Integer RAW whose per-class normalization yields the supplied
    /// per-site transmissions: raw = transmission × 1000, white = 1000,
    /// black = 0, so cast compression happens exactly at sample time.
    fn raw_with_transmissions(
        width: usize,
        height: usize,
        pattern: &str,
        transmissions: Vec<u16>, // raw codes, one per site
        crops: [usize; 4],
    ) -> rawloader::RawImage {
        rawloader::RawImage {
            make: String::new(),
            model: String::new(),
            clean_make: String::new(),
            clean_model: String::new(),
            width,
            height,
            cpp: 1,
            wb_coeffs: [1.0; 4],
            whitelevels: [1000; 4],
            blacklevels: [0; 4],
            xyz_to_cam: [[0.0; 3]; 4],
            cfa: rawloader::CFA::new(pattern),
            crops,
            blackareas: Vec::new(),
            orientation: rawloader::Orientation::Normal,
            data: rawloader::RawImageData::Integer(transmissions),
        }
    }

    #[test]
    fn downsample_thumbnail_removes_casts_at_full_scale() {
        // Neutral film at normalized transmission 0.5 with per-class captures
        // (RGGB: one R, two G, one B site): the fused downscaler must rescale
        // each class onto the green-anchored reference.
        let image = raw_with_transmissions(
            2,
            2,
            "RGGB",
            vec![600, 500, 500, 400], // 0.6, 0.5, 0.5, 0.4 after normalization
            [0, 0, 0, 0],
        );

        let (mono, width, height) = downsample_thumbnail(&image, 2).unwrap();

        assert_eq!((width, height), (2, 2));
        assert!(mono.iter().all(|value| (value - 0.5).abs() < 1e-5));
    }

    #[test]
    fn downsample_thumbnail_preserves_phase_after_averaging() {
        // Same class casts compressed into one 1x1 output pixel: the per-class
        // averages must be rescalled before combining, or the cast survives.
        let image = raw_with_transmissions(2, 2, "RGGB", vec![600, 500, 500, 400], [0, 0, 0, 0]);

        let (mono, width, height) = downsample_thumbnail(&image, 1).unwrap();

        assert_eq!((width, height), (1, 1));
        assert!((mono[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn downsample_thumbnail_handles_fourth_color_sites() {
        let image = raw_with_transmissions(
            2,
            2,
            "RGBE",
            vec![450, 500, 550, 500], // R, G, B, E sites
            [0, 0, 0, 0],
        );

        let (mono, width, height) = downsample_thumbnail(&image, 1).unwrap();

        assert_eq!((width, height), (1, 1));
        assert!((mono[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn downsample_thumbnail_respects_crops_and_phase() {
        // 4x4 RGGB with per-class captures, cropped off the left border. The
        // surviving region must hit the absolute CFA (not rephase against the
        // crop origin), so the class casts compress to uniform 0.5 as usual.
        let image = raw_with_transmissions(
            4,
            4,
            "RGGB",
            vec![
                600, 500, 600, 500, // R, G, R, G
                500, 400, 500, 400, // G, B, G, B
                600, 500, 600, 500, //
                500, 400, 500, 400, //
            ],
            [0, 0, 0, 1],
        );

        let (mono, width, height) = downsample_thumbnail(&image, 2).unwrap();

        assert_eq!((width, height), (1, 2));
        assert!(mono.iter().all(|value| (value - 0.5).abs() < 1e-5));
    }

    #[test]
    fn downsample_thumbnail_rejects_degenerate_crops() {
        let image = raw_with_transmissions(2, 2, "RGGB", vec![600, 500, 500, 400], [2, 0, 0, 0]);

        assert!(downsample_thumbnail(&image, 1).is_none());
    }

    #[test]
    fn downsample_rgb_is_luma_of_block_means() {
        // Float data is normalized by the global maximum before luma collapse;
        // the block mean of luma equals luma of the block means (linearity).
        let values: Vec<f32> = vec![
            0.5, 0.5, 0.5, //
            0.6, 0.4, 0.2, //
            0.2, 0.4, 0.6, //
            0.8, 0.1, 0.1, //
        ];
        let image = rawloader::RawImage {
            make: String::new(),
            model: String::new(),
            clean_make: String::new(),
            clean_model: String::new(),
            width: 2,
            height: 2,
            cpp: 3,
            wb_coeffs: [1.0; 4],
            whitelevels: [0; 4],
            blacklevels: [0; 4],
            xyz_to_cam: [[0.0; 3]; 4],
            cfa: rawloader::CFA::new("RGGB"),
            crops: [0, 0, 0, 0],
            blackareas: Vec::new(),
            orientation: rawloader::Orientation::Normal,
            data: rawloader::RawImageData::Float(values),
        };

        let (mono, width, height) = downsample_thumbnail(&image, 1).unwrap();

        assert_eq!((width, height), (1, 1));
        // Normalized by global max 0.8 → gain 1.25; then channel means and
        // Rec.709 luma.
        let expected = {
            let r = (0.625 + 0.75 + 0.25 + 1.0) / 4.0;
            let g = (0.625 + 0.5 + 0.5 + 0.125) / 4.0;
            let b = (0.625 + 0.25 + 0.75 + 0.125) / 4.0;
            0.212_6 * r + 0.715_2 * g + 0.072_2 * b
        };
        assert!((mono[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn class_gains_anchors_to_green_when_present() {
        let gains = class_gains([0.6, 0.5, 0.4, 0.9], [false; 4]);

        assert!((gains[0] - 0.5 / 0.6).abs() < 1e-6); // R pulled up to green
        assert!((gains[1] - 1.0).abs() < 1e-6); // green is the reference
        assert!((gains[2] - 0.5 / 0.4).abs() < 1e-6); // B pulled down to green
        assert!((gains[3] - 0.5 / 0.9).abs() < 1e-6);
    }

    #[test]
    fn class_gains_falls_back_to_dimmest_class() {
        // Green absent: anchor onto the dimmest measurable class.
        let gains = class_gains([0.6, 0.5, 0.4, 0.9], [false, true, false, true]);

        assert!((gains[2] - 1.0).abs() < 1e-6); // dimmest (B) is the reference
        assert!((gains[0] - 0.4 / 0.6).abs() < 1e-6);
        assert!((gains[3] - 0.4 / 0.9).abs() < 1e-6);
    }

    #[test]
    fn class_gains_keeps_defaults_without_measurable_class() {
        assert_eq!(class_gains([0.1; 4], [true; 4]), [1.0; 4]);
    }

    #[test]
    fn none_preset_does_not_rescale_midtones_or_clamp_highlights() {
        // Regression: the `None` preset (already-positive RAW) previously ran
        // `normalize_positive`, which divided the whole frame by its own 95th
        // percentile and clamped everything above it to pure white. That baked
        // an auto-expose into the data before the EV slider and permanently
        // destroyed highlight detail. With the auto-expose removed, a frame
        // whose midtones sit at 0.5 relative to the sensor white point must
        // render those midtones as a real mid-gray — not scaled up to white —
        // and speculars above the 95th percentile must survive at the peak.
        //
        // 10x10 RGB Integer RAW, white = 1000: 97 sites at 0.5 (raw 500) and
        // 3 speculars at 1.0 (raw 1000), so the 95th percentile ≈ 0.5. Under
        // the old `normalize_positive` that anchor scaled the 0.5 midtones to
        // 1.0 and clamped the speculars — a uniform white frame. Today the
        // midtones must stay a mid-gray (~sRGB(0.5) ≈ 188).
        let mut values = Vec::with_capacity(10 * 10 * 3);
        for y in 0..10 {
            for x in 0..10 {
                let site = if (x == 0 && y == 0) || (x == 9 && y == 0) || (x == 4 && y == 9) {
                    1000
                } else {
                    500
                };
                values.extend_from_slice(&[site, site, site]);
            }
        }
        let image = rawloader::RawImage {
            make: String::new(),
            model: String::new(),
            clean_make: String::new(),
            clean_model: String::new(),
            width: 10,
            height: 10,
            cpp: 3,
            wb_coeffs: [1.0; 4],
            whitelevels: [1000; 4],
            blacklevels: [0; 4],
            xyz_to_cam: [[0.0; 3]; 4],
            cfa: rawloader::CFA::new("RGGB"),
            crops: [0, 0, 0, 0],
            blackareas: Vec::new(),
            orientation: rawloader::Orientation::Normal,
            data: rawloader::RawImageData::Integer(values),
        };

        // Identity curve + EV 0 so the sampled mono value is preserved exactly
        // through sRGB with no user gain.
        let tone = edit_manifest::ToneEdit {
            exposure_ev: 0.0,
            ..edit_manifest::ToneEdit::default()
        };
        let handle = convert_thumbnail(
            &image,
            10.0,
            tone,
            Default::default(),
            0,
            FilmPreset::None,
            BaseConfig::default(),
        )
        .expect("decode succeeds");
        let (width, height, pixels) = match &handle {
            cosmic::widget::image::Handle::Rgba {
                width,
                height,
                pixels,
                ..
            } => (*width, *height, pixels.as_ref()),
            _ => panic!("expected an RGBA handle"),
        };
        assert_eq!((width, height), (10, 10));
        // A midtone site far from any specular must render as a genuine
        // mid-gray, NOT a scaled-to-white 255 that `normalize_positive`
        // produced. Site (5,5) sits in the 0.5 bulk.
        let midtone = pixels[(5 * 10 + 5) * 4];
        assert!(midtone < 200, "midtone crushed too bright: {midtone}");
        assert!(midtone > 100, "midtone too dark: {midtone}");
        // The specular sites stay at the sensor peak (much brighter than the
        // bulk), proving highlight detail is preserved rather than crushed.
        let specular = pixels[(0 * 10 + 0) * 4];
        assert!(specular as i32 > midtone as i32 + 40, "specular crushed");
    }

    #[test]
    fn inverted_preset_brightens_on_positive_ev() {
        // User-facing EV means "brightness" for BOTH presets: +EV brightens a
        // film negative's positive exactly as it brightens an already-positive
        // scan. Since the rework moved the exposure gain onto the TRUE sensor
        // data (multiplied by 2^-EV there) and then per-fragment density-
        // inverts it, +EV must raise the density and therefore brighten the
        // positive — NOT darken it.
        //
        // A uniform frame equal to its own measured clear-film base prints all
        // black at EV 0 (every site sits at the measured black point). Raising
        // EV to +1 halves the transmission (2^-1), lifting the density to
        // log10(2), which must brighten the frame measurably.
        let values: Vec<u16> = vec![900; 10 * 10 * 3];
        let image = rawloader::RawImage {
            make: String::new(),
            model: String::new(),
            clean_make: String::new(),
            clean_model: String::new(),
            width: 10,
            height: 10,
            cpp: 3,
            wb_coeffs: [1.0; 4],
            whitelevels: [1000; 4],
            blacklevels: [0; 4],
            xyz_to_cam: [[0.0; 3]; 4],
            cfa: rawloader::CFA::new("RGGB"),
            crops: [0, 0, 0, 0],
            blackareas: Vec::new(),
            orientation: rawloader::Orientation::Normal,
            data: rawloader::RawImageData::Integer(values),
        };

        let bake = |ev: f32| {
            let tone = edit_manifest::ToneEdit {
                exposure_ev: ev,
                ..edit_manifest::ToneEdit::default()
            };
            let handle = convert_thumbnail(
                &image,
                10.0,
                tone,
                Default::default(),
                0,
                FilmPreset::Hp5Plus,
                BaseConfig::default(),
            )
            .expect("decode succeeds");
            let (width, height, pixels) = match &handle {
                cosmic::widget::image::Handle::Rgba {
                    width,
                    height,
                    pixels,
                    ..
                } => (*width, *height, pixels.as_ref()),
                _ => panic!("expected an RGBA handle"),
            };
            assert_eq!((width, height), (10, 10));
            let count = pixels.len() / 4;
            let sum: u32 = pixels.chunks_exact(4).map(|p| u32::from(p[0])).sum::<u32>();
            sum as f32 / count as f32
        };

        let ev0 = bake(0.0);
        let ev1 = bake(1.0);
        assert!(
            ev1 > ev0 + 20.0,
            "+1EV must brighten an inverted preset: {ev0} → {ev1}"
        );
    }

    #[test]
    fn apply_detail_zoom_clamps_at_both_ends() {
        let (zoom, _) = apply_detail_zoom(1.0, Point::default(), None, -1.0, MAX_DETAIL_ZOOM);
        assert_eq!(zoom, 1.0);
        let (zoom, _) = apply_detail_zoom(MAX_DETAIL_ZOOM, Point::default(), None, 9.0, MAX_DETAIL_ZOOM);
        assert_eq!(zoom, MAX_DETAIL_ZOOM);
    }

    #[test]
    fn apply_detail_zoom_clamps_at_the_hundred_percent_cap() {
        // The 1:1 cap is the maximum: zooming past it stops there, and the cap
        // itself is respected even when below MAX_DETAIL_ZOOM.
        let cap = 4.25;
        let (zoom, _) = apply_detail_zoom(4.0, Point::default(), None, 9.0, cap);
        assert_eq!(zoom, cap);
        // A cap below the current zoom clamps back down to it.
        let (zoom, _) = apply_detail_zoom(6.0, Point::default(), None, 0.0, cap);
        assert_eq!(zoom, cap);
    }

    #[test]
    fn apply_detail_zoom_returns_unchanged_pan_without_cursor() {
        let (zoom, pan) = apply_detail_zoom(2.0, Point::new(13.0, -7.0), None, 0.5, 8.0);
        assert!((zoom - 2.5).abs() < 1e-6);
        assert_eq!(pan, Point::new(13.0, -7.0));
    }

    #[test]
    fn apply_detail_zoom_recenters_when_back_to_contain_fit() {
        // Zooming all the way out must give the centered contain view.
        let (zoom, pan) =
            apply_detail_zoom(3.0, Point::new(50.0, -30.0), Some(Point::new(0.0, 0.0)), -2.0, 8.0);
        assert_eq!(zoom, 1.0);
        assert_eq!(pan, Point::default());
    }

    #[test]
    fn native_level_up_zoom_fires_at_the_earlier_of_cap_and_threshold() {
        // A preview whose 1:1 cap sits above the fixed threshold keeps the
        // prefetch trigger: level up at 2× contain before reaching the cap.
        assert_eq!(AppModel::native_level_up_zoom_for(5.0), NATIVE_ZOOM_THRESHOLD);
        // A large full-screen preview (drawer closed) can put the overview's
        // own 1:1 below 2.0; the trigger must fall back to that cap, otherwise
        // zooming (clamped at 1:1) could never reach the fixed threshold and
        // the native level-up would never fire.
        assert_eq!(AppModel::native_level_up_zoom_for(1.4), 1.4);
        assert_eq!(AppModel::native_level_up_zoom_for(1.0), 1.0);
        // The no-shader/no-size fallback cap (MAX_DETAIL_ZOOM) degrades to the
        // fixed threshold, exactly the pre-install behavior.
        assert_eq!(
            AppModel::native_level_up_zoom_for(MAX_DETAIL_ZOOM),
            NATIVE_ZOOM_THRESHOLD
        );
        assert!(
            AppModel::native_level_up_zoom_for(1.4) < NATIVE_ZOOM_THRESHOLD,
            "below-cap previews must fire strictly before the fixed threshold"
        );
    }

    #[test]
    fn zoom_about_anchor_round_trips() {
        // Zooming in then back out at the same cursor must return the exact
        // original pan (the anchored image point is pinned in both steps).
        let pan = Point::new(5.0, 6.0);
        let cursor = Point::new(-9.0, 4.0);
        let zoomed = zoom_about_anchor(2.0, 3.0, pan, cursor);
        let back = zoom_about_anchor(3.0, 2.0, zoomed, cursor);
        assert!((back.x - pan.x).abs() < 1e-5);
        assert!((back.y - pan.y).abs() < 1e-5);
    }

    #[test]
    fn zoom_about_anchor_keeps_center_pinned_when_cursor_is_center() {
        // Zooming about the image center (cursor == pan) leaves the pan
        // unchanged: center of the frame stays center of the widget.
        let pan = Point::new(12.0, -8.0);
        let out = zoom_about_anchor(2.0, 4.0, pan, Point::new(12.0, -8.0));
        assert!((out.x - pan.x).abs() < 1e-5);
        assert!((out.y - pan.y).abs() < 1e-5);
    }

    #[test]
    fn zoom_about_anchor_is_identity_at_delta_zero() {
        let pan = Point::new(5.0, 6.0);
        let out = zoom_about_anchor(2.0, 2.0, pan, Point::new(-9.0, 4.0));
        assert!((out.x - pan.x).abs() < 1e-6);
        assert!((out.y - pan.y).abs() < 1e-6);
    }

    #[test]
    fn detail_zoom_delta_converts_scroll_units() {
        let lines = cosmic::iced::mouse::ScrollDelta::Lines { x: 0.0, y: 2.0 };
        assert!((detail_zoom_delta(lines) - 1.0).abs() < 1e-6);
        let pixels = cosmic::iced::mouse::ScrollDelta::Pixels { x: 0.0, y: 400.0 };
        assert!((detail_zoom_delta(pixels) - 1.0).abs() < 1e-6);
        let up = cosmic::iced::mouse::ScrollDelta::Lines { x: 0.0, y: -1.0 };
        assert!((detail_zoom_delta(up) + 0.5).abs() < 1e-6);
    }

    #[test]
    fn detail_cursor_is_arrow_at_fit_grab_when_zoomed_and_grabbing_on_drag() {
        use cosmic::iced::mouse::Interaction;
        // Contain fit (or zoomed out) cannot pan: default arrow.
        assert_eq!(detail_cursor_for(1.0, false), Interaction::None);
        // A live grab-pan drag always grasps, even at fit.
        assert_eq!(detail_cursor_for(1.0, true), Interaction::Grabbing);
        // Zoomed past contain: the image overflows, so panning is meaningful.
        assert_eq!(detail_cursor_for(2.5, false), Interaction::Grab);
        assert_eq!(detail_cursor_for(2.5, true), Interaction::Grabbing);
    }

    #[test]
    fn lru_cache_get_marks_most_recently_used() {
        let mut cache = LruCache::new(2);
        assert!(cache.insert("a", 1).is_none());
        assert!(cache.insert("b", 2).is_none());
        // Touching "a" makes it most-recently used, so "b" is evicted next.
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.insert("c", 3), Some(2));
        assert!(cache.contains(&"a"));
        assert!(!cache.contains(&"b"));
        assert!(cache.contains(&"c"));
    }

    #[test]
    fn lru_cache_evicts_least_recently_used_at_capacity() {
        let mut cache = LruCache::new(3);
        assert!(cache.insert("a", 1).is_none());
        assert!(cache.insert("b", 2).is_none());
        assert!(cache.insert("c", 3).is_none());
        assert_eq!(cache.len(), 3);
        // Full: inserting "d" evicts the least-recently-used "a".
        assert_eq!(cache.insert("d", 4), Some(1));
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&"a"));
        for key in ["b", "c", "d"] {
            assert!(cache.contains(&key));
        }
    }

    #[test]
    fn lru_cache_insert_same_key_updates_in_place() {
        let mut cache = LruCache::new(2);
        assert!(cache.insert("a", 1).is_none());
        // Re-inserting an existing key updates the value and never evicts.
        assert!(cache.insert("a", 10).is_none());
        assert_eq!(cache.get(&"a"), Some(&10));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lru_cache_clear_drops_everything_and_reuse() {
        let mut cache = LruCache::new(2);
        assert!(cache.insert("a", 1).is_none());
        assert!(cache.insert("b", 2).is_none());
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(&"a"));
        // The cache is usable again after clearing.
        assert!(cache.insert("c", 3).is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn lru_cache_zero_capacity_insert_evicts_every_new_entry() {
        let mut cache = LruCache::new(0);
        assert_eq!(cache.insert("a", 1), Some(1));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn lru_cache_preset_keyed_entries_do_not_collide() {
        // Two presets for the same roll+frame are distinct cache entries, so a
        // preset change can never serve a stale inversion.
        let mut cache = LruCache::new(8);
        let dir = PathBuf::from("/rolls/a");
        let key_alpha = (dir.clone(), FilmPreset::Hp5Plus, "frame.DNG".to_string());
        let key_neutral = (dir, FilmPreset::None, "frame.DNG".to_string());
        assert!(cache.insert(key_alpha.clone(), 1).is_none());
        assert!(cache.insert(key_neutral.clone(), 2).is_none());
        assert_eq!(cache.get(&key_alpha), Some(&1));
        assert_eq!(cache.get(&key_neutral), Some(&2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn lru_cache_retain_drops_only_matching_dir_and_preserves_recency() {
        let mut cache = LruCache::new(8);
        let roll_a = PathBuf::from("/rolls/a");
        let roll_b = PathBuf::from("/rolls/b");
        let key_a_none = (roll_a.clone(), FilmPreset::None, "1.DNG".to_string());
        let key_a_hp5 = (roll_a.clone(), FilmPreset::Hp5Plus, "2.DNG".to_string());
        let key_b = (roll_b.clone(), FilmPreset::None, "3.DNG".to_string());
        cache.insert(key_a_none.clone(), 1);
        cache.insert(key_a_hp5.clone(), 2);
        cache.insert(key_b.clone(), 3);
        // Touch the other roll so its recency stays intact through the retain.
        assert_eq!(cache.get(&key_b), Some(&3));

        // Dropping every entry for roll `a` (any preset) leaves roll `b` alone.
        let dropped = cache.retain(|(dir, _, _)| dir != &roll_a);
        assert_eq!(dropped.len(), 2);

        assert!(!cache.contains(&key_a_none));
        assert!(!cache.contains(&key_a_hp5));
        assert!(cache.contains(&key_b));
        assert_eq!(cache.get(&key_b), Some(&3));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn clamp_ev_bounds_to_the_slider_range() {
        assert_eq!(clamp_ev(-99.0), -3.0);
        assert_eq!(clamp_ev(99.0), 4.0);
        assert_eq!(clamp_ev(0.5), 0.5);
    }

    #[test]
    fn clamp_contrast_power_bounds_to_the_slider_range() {
        assert_eq!(clamp_contrast_power(0.0), 0.125);
        assert_eq!(clamp_contrast_power(99.0), 8.0);
        assert_eq!(clamp_contrast_power(1.0), 1.0);
    }

    #[test]
    fn clamp_tone_lift_power_bounds_to_the_slider_range() {
        assert_eq!(clamp_tone_lift_power(0.0), 0.25);
        assert_eq!(clamp_tone_lift_power(99.0), 4.0);
        assert_eq!(clamp_tone_lift_power(1.0), 1.0);
    }

    #[test]
    fn contrast_lift_value_follows_the_power_direction() {
        // Identity sits at the center 0.0; raising the power raises contrast.
        assert!((contrast_lift(1.0) - 0.0).abs() < 1e-6);
        assert!((contrast_lift(2.0) - 1.0).abs() < 1e-6);
        assert!((contrast_lift(0.5) - (-1.0)).abs() < 1e-6);
        // Monotone increasing in the power, like the highlights arm.
        let (a, b) = (contrast_lift(0.6), contrast_lift(1.4));
        assert!(a < b);
    }

    #[test]
    fn shadow_lift_value_inverts_the_power_direction() {
        // Identity sits at the center 0.0 for the stop-based lift value.
        assert!((shadow_lift(1.0) - 0.0).abs() < 1e-6);
        // The log flips direction: a shadows power below 1.0 (a lower-tones
        // lift, white-pivoted) reads as a POSITIVE lift value, so dragging the
        // slider right brightens the shadows.
        assert!((shadow_lift(0.5) - 1.0).abs() < 1e-6);
        assert!((shadow_lift(2.0) - (-1.0)).abs() < 1e-6);
        // The lift value is monotone decreasing in the power, which is the point.
        let (a, b) = (shadow_lift(0.6), shadow_lift(1.4));
        assert!(a > b);
    }

    #[test]
    fn highlight_lift_value_follows_the_power_direction() {
        // Identity sits at the center 0.0.
        assert!((highlight_lift(1.0) - 0.0).abs() < 1e-6);
        // The highlights power pivots at the shadow anchor, so a lift drives
        // the power ABOVE 1.0: a positive lift value must map to power > 1 so
        // dragging the highlights slider right brightens the upper tones.
        assert!((highlight_lift(2.0) - 1.0).abs() < 1e-6);
        assert!((highlight_lift(0.5) - (-1.0)).abs() < 1e-6);
        // Monotone increasing in the power — the opposite of the shadows arm.
        let (a, b) = (highlight_lift(0.6), highlight_lift(1.4));
        assert!(a < b);
    }

    #[test]
    fn lift_maps_round_trip_across_the_power_range() {
        for power in [0.25_f32, 0.5, 1.0, 1.7, 4.0] {
            let shadow_back = shadow_power_for_lift(shadow_lift(power));
            assert!(
                (shadow_back - power).abs() < 1e-5,
                "shadows power {power} round-tripped to {shadow_back}"
            );
            let highlight_back = highlight_power_for_lift(highlight_lift(power));
            assert!(
                (highlight_back - power).abs() < 1e-5,
                "highlights power {power} round-tripped to {highlight_back}"
            );
        }
        for power in [0.125_f32, 0.25, 1.0, 2.0, 4.0, 8.0] {
            let contrast_back = contrast_power_for_lift(contrast_lift(power));
            assert!(
                (contrast_back - power).abs() < 1e-5,
                "contrast power {power} round-tripped to {contrast_back}"
            );
        }
        // Values leaving the symmetric windows clamp to the power endpoints, so
        // an exited lift track and the clamps agree on bounds. Contrast spans
        // ±3 stops; Highlights/Shadows ±2.
        assert_eq!(shadow_power_for_lift(3.0), 0.25);
        assert_eq!(shadow_power_for_lift(-3.0), 4.0);
        assert_eq!(highlight_power_for_lift(3.0), 4.0);
        assert_eq!(highlight_power_for_lift(-3.0), 0.25);
        assert_eq!(contrast_power_for_lift(3.0), 8.0);
        assert_eq!(contrast_power_for_lift(-3.0), 0.125);
        // The identity lift value maps to the identity power on every arm.
        assert!((shadow_power_for_lift(0.0) - 1.0).abs() < 1e-6);
        assert!((highlight_power_for_lift(0.0) - 1.0).abs() < 1e-6);
        assert!((contrast_power_for_lift(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn edit_adjust_maps_bare_and_shifted_keys() {
        // Exposure pair: `-`/`=` coarse, Shift nudge.
        assert_eq!(
            edit_adjust_for("-", false, false),
            Some(EditAdjust::Exposure(-EDIT_STEP_EV))
        );
        assert_eq!(
            edit_adjust_for("=", false, false),
            Some(EditAdjust::Exposure(EDIT_STEP_EV))
        );
        assert_eq!(
            edit_adjust_for("-", false, true),
            Some(EditAdjust::Exposure(-EDIT_NUDGE_EV))
        );
        assert_eq!(
            edit_adjust_for("=", false, true),
            Some(EditAdjust::Exposure(EDIT_NUDGE_EV))
        );
        // Contrast pair: `[`/`]` coarse, Shift nudge.
        assert_eq!(
            edit_adjust_for("[", false, false),
            Some(EditAdjust::Contrast(-EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for("]", false, false),
            Some(EditAdjust::Contrast(EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for("[", false, true),
            Some(EditAdjust::Contrast(-EDIT_NUDGE_CURVE))
        );
        assert_eq!(
            edit_adjust_for("]", false, true),
            Some(EditAdjust::Contrast(EDIT_NUDGE_CURVE))
        );
        // Highlights pair: `;`/`'` coarse, Shift nudge.
        assert_eq!(
            edit_adjust_for(";", false, false),
            Some(EditAdjust::Highlights(-EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for("'", false, false),
            Some(EditAdjust::Highlights(EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for(";", false, true),
            Some(EditAdjust::Highlights(-EDIT_NUDGE_CURVE))
        );
        assert_eq!(
            edit_adjust_for("'", false, true),
            Some(EditAdjust::Highlights(EDIT_NUDGE_CURVE))
        );
        // Shadows pair: `,`/`.` coarse, Shift nudge.
        assert_eq!(
            edit_adjust_for(",", false, false),
            Some(EditAdjust::Shadows(-EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for(".", false, false),
            Some(EditAdjust::Shadows(EDIT_STEP_CURVE))
        );
        assert_eq!(
            edit_adjust_for(",", false, true),
            Some(EditAdjust::Shadows(-EDIT_NUDGE_CURVE))
        );
        assert_eq!(
            edit_adjust_for(".", false, true),
            Some(EditAdjust::Shadows(EDIT_NUDGE_CURVE))
        );
        // The VIM movement keys are NOT edit adjusts — `h`/`j`/`k`/`l` route to
        // `Message::Nav` in the subscription (mirroring the arrows), so they map
        // to `None` here.
        assert_eq!(edit_adjust_for("h", false, false), None);
        assert_eq!(edit_adjust_for("j", false, false), None);
        assert_eq!(edit_adjust_for("k", false, false), None);
        assert_eq!(edit_adjust_for("l", false, false), None);
        // Shift/Alt leave them unbound too.
        assert_eq!(edit_adjust_for("h", true, true), None);
    }

    #[test]
    fn edit_adjust_ignores_unbound_keys() {
        assert_eq!(edit_adjust_for("a", false, false), None);
        assert_eq!(edit_adjust_for(" ", false, false), None);
        assert_eq!(edit_adjust_for("p", false, true), None);
        assert_eq!(edit_adjust_for("_", false, true), None);
    }

    #[test]
    fn plain_click_selects_a_single_frame() {
        let tiles = vec![tile("a"), tile("b"), tile("c")];
        let (set, anchor) = apply_frame_click(HashSet::new(), "b", false, false, None, &tiles);
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort();
        assert_eq!(v, vec!["b".to_string()]);
        assert_eq!(anchor.as_deref(), Some("b"));
    }

    #[test]
    fn ctrl_click_toggles_membership() {
        let tiles = vec![tile("a"), tile("b"), tile("c")];
        let start: HashSet<String> = ["a", "b"].into_iter().map(str::to_owned).collect();
        // Toggle "b" off.
        let (set, anchor) = apply_frame_click(start.clone(), "b", true, false, Some("a"), &tiles);
        assert_eq!(set.len(), 1);
        assert!(set.contains("a"));
        assert!(!set.contains("b"));
        assert_eq!(anchor.as_deref(), Some("a"));
        // Toggle "c" on.
        let (set, _) = apply_frame_click(start, "c", true, false, Some("a"), &tiles);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn shift_click_selects_range_anchor_to_clicked() {
        let tiles = vec![tile("a"), tile("b"), tile("c"), tile("d")];
        // Anchor "a", click "c" → selects a..=c.
        let (set, anchor) = apply_frame_click(HashSet::new(), "c", false, true, Some("a"), &tiles);
        let mut v: Vec<_> = set.into_iter().collect();
        v.sort();
        assert_eq!(v, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert_eq!(anchor.as_deref(), Some("c"));
        // Reverse range: anchor "d", click "b" → selects b..=d.
        let (set, _) = apply_frame_click(HashSet::new(), "b", false, true, Some("d"), &tiles);
        assert_eq!(set.len(), 3);
        assert!(set.contains("b") && set.contains("c") && set.contains("d"));
    }

    #[test]
    fn shift_click_without_anchor_collapses_to_single() {
        let tiles = vec![tile("a"), tile("b"), tile("c")];
        let (set, _) = apply_frame_click(HashSet::new(), "b", false, true, None, &tiles);
        assert_eq!(set.len(), 1);
        assert!(set.contains("b"));
    }

    use edit_manifest::CropDirection as Dir;
    use edit_manifest::CropMargins as Marg;

    fn assert_aspects(m: Marg, w: u32, h: u32) {
        // (w − left − right) / (h − top − bottom) must equal w / h, non-degenerate.
        let cw = m.cropped_width(w);
        let ch = m.cropped_height(h);
        assert!(
            cw > 0 && ch > 0,
            "crop collapses to a degenerate frame: {m:?}"
        );
        let left = f64::from(m.left);
        let right = f64::from(m.right);
        let top = f64::from(m.top);
        let bottom = f64::from(m.bottom);
        let lhs = f64::from(w) - left - right;
        let rhs = f64::from(h) - top - bottom;
        let lhs_ratio = lhs / rhs;
        let src_ratio = f64::from(w) / f64::from(h);
        // Margins are integer pixels; the perpendicular pair rounds the exact
        // `ar·(parallel sum)` to the nearest pixel, so a sub-percent drift is
        // inherent and acceptable for frames that keep any real size. A genuine
        // aspect violation (or a collapse) moves the ratio far more than this.
        assert!(
            (lhs_ratio - src_ratio).abs() < 0.01,
            "aspect drifted: {lhs_ratio} != {src_ratio} for {m:?}"
        );
    }

    #[test]
    fn move_crop_translates_the_window_and_keeps_size() {
        // A 200×100 crop window in a 400×300 frame, sitting at (40, 30).
        let crop = Marg {
            top: 30,
            right: 160,
            bottom: 170,
            left: 40,
        };
        // Move right by 10: left grows, right shrinks, size unchanged.
        let right = move_crop_box(crop, Dir::Right, 10, 400, 300);
        assert_eq!(right.left, 50);
        assert_eq!(right.right, 150);
        assert_eq!(right.cropped_width(400), 200);
        assert_eq!(right.cropped_height(300), 100);
        // Move left back by 10 restores the original margins exactly.
        assert_eq!(move_crop_box(right, Dir::Left, 10, 400, 300), crop);
        // Move up: top shrinks, bottom grows; size unchanged.
        let up = move_crop_box(crop, Dir::Top, 10, 400, 300);
        assert_eq!(up.top, 20);
        assert_eq!(up.bottom, 180);
        assert_eq!(up.cropped_width(400), 200);
        assert_eq!(up.cropped_height(300), 100);
        // Move down returns to the original.
        assert_eq!(move_crop_box(up, Dir::Bottom, 10, 400, 300), crop);
    }

    #[test]
    fn move_crop_clamps_the_window_inside_the_frame() {
        // Window flush to the left edge: moving further left clamps at 0.
        let crop = Marg {
            top: 30,
            right: 160,
            bottom: 170,
            left: 0,
        };
        assert_eq!(move_crop_box(crop, Dir::Left, 50, 400, 300).left, 0);
        // Full-frame window cannot move at all (max_origin is 0).
        let full = Marg::default();
        assert_eq!(move_crop_box(full, Dir::Right, 10, 400, 300), full);
        assert_eq!(move_crop_box(full, Dir::Bottom, 10, 400, 300), full);
        // Moving far right pins the window against the right edge: the 240-wide
        // window's left origin clamps to 160 (right becomes 0).
        let far = move_crop_box(crop, Dir::Right, 10_000, 400, 300);
        assert_eq!(far.left, 160);
        assert_eq!(far.right, 0);
    }

    #[test]
    fn crop_move_direction_rotates_with_the_display() {
        // Rotation 0 is the identity: screen == EXIF-upright directions.
        assert_eq!(crop_move_direction(MoveDir::Left, 0), Dir::Left);
        assert_eq!(crop_move_direction(MoveDir::Right, 0), Dir::Right);
        assert_eq!(crop_move_direction(MoveDir::Up, 0), Dir::Top);
        assert_eq!(crop_move_direction(MoveDir::Down, 0), Dir::Bottom);
        // One CCW turn puts the texture's top on the display's left, so a
        // visual Left moves the crop toward the texture top, etc.
        assert_eq!(crop_move_direction(MoveDir::Left, 1), Dir::Top);
        assert_eq!(crop_move_direction(MoveDir::Right, 1), Dir::Bottom);
        assert_eq!(crop_move_direction(MoveDir::Up, 1), Dir::Right);
        assert_eq!(crop_move_direction(MoveDir::Down, 1), Dir::Left);
        // 180° flips both axes.
        assert_eq!(crop_move_direction(MoveDir::Left, 2), Dir::Right);
        assert_eq!(crop_move_direction(MoveDir::Right, 2), Dir::Left);
        assert_eq!(crop_move_direction(MoveDir::Up, 2), Dir::Bottom);
        assert_eq!(crop_move_direction(MoveDir::Down, 2), Dir::Top);
        // Three CCW turns (one CW): the texture bottom lands on the display
        // left.
        assert_eq!(crop_move_direction(MoveDir::Left, 3), Dir::Bottom);
        assert_eq!(crop_move_direction(MoveDir::Right, 3), Dir::Top);
        assert_eq!(crop_move_direction(MoveDir::Up, 3), Dir::Left);
        assert_eq!(crop_move_direction(MoveDir::Down, 3), Dir::Right);
        // Rotation is mod 4: 4 == 0 (identity again).
        assert_eq!(crop_move_direction(MoveDir::Left, 4), Dir::Left);
    }

    #[test]
    fn resize_crop_grows_and_shrinks_about_center_preserving_aspect() {
        // A source-aspect crop window in a 400×300 frame: 200×150 (aspect 4:3),
        // centered with left=right=100 and top=bottom=75.
        let crop = Marg {
            top: 75,
            right: 100,
            bottom: 75,
            left: 100,
        };
        // Grow by 10 (long edge 200 → 210, both axes scaled by 1.05).
        let grown = resize_crop_box(crop, 10, 400, 300);
        assert_eq!(grown.cropped_width(400), 210);
        assert_eq!(grown.cropped_height(300), 158);
        assert_aspects(grown, 400, 300);
        // Center preserved: left margin is half the leftover width.
        assert_eq!(grown.left, 95);
        assert_eq!(grown.top, 71);
        // Shrink by 10 returns (round-trip) toward the start.
        let shrunk = resize_crop_box(grown, -10, 400, 300);
        assert_aspects(shrunk, 400, 300);
        assert_eq!(shrunk.cropped_width(400), 200);
        assert_eq!(shrunk.cropped_height(300), 150);
    }

    #[test]
    fn resize_crop_clamps_at_minimum_and_full_frame() {
        let crop = Marg {
            top: 75,
            right: 100,
            bottom: 75,
            left: 100,
        };
        // Growing beyond the frame clamps to full-frame margins.
        let full = resize_crop_box(crop, 10_000, 400, 300);
        assert_eq!(full, Marg::default());
        // Shrinking past the minimum clamps the long edge at MIN_CROP_WINDOW
        // (the width, since 4:3 makes width the long edge).
        let tiny = resize_crop_box(crop, -10_000, 400, 300);
        assert_eq!(tiny.cropped_width(400), MIN_CROP_WINDOW);
        assert_aspects(tiny, 400, 300);
    }

    #[test]
    fn crop_margins_cropped_dims_are_saturating() {
        let m = Marg {
            top: 5,
            right: 5,
            bottom: 5,
            left: 5,
        };
        assert_eq!(m.cropped_width(12), 2);
        assert_eq!(m.cropped_height(12), 2);
        // Oversized margins clamp to zero rather than wrapping.
        assert_eq!(m.cropped_width(4), 0);
        assert_eq!(m.cropped_height(4), 0);
    }

    #[test]
    fn scale_crop_scales_margins_onto_the_print() {
        // 4000x2000 source printed at 400x200 scales each axis by 1/10.
        let crop = Marg {
            top: 10,
            right: 20,
            bottom: 30,
            left: 40,
        };
        assert_eq!(
            scale_crop(crop, 4000, 2000, 400, 200),
            Marg {
                top: 1,
                right: 2,
                bottom: 3,
                left: 4
            }
        );
    }

    #[test]
    fn scale_crop_scales_a_rotated_print_from_display_source_dims() {
        // A Rotate90 sensor of masked dims (2000 wide, 400 tall) displays as
        // (400 wide, 2000 tall): the caller resolves those display dims before
        // scaling, so the print's horizontal axis (400) maps to the source
        // horizontal and the vertical margins compress by 400→200 / 2000.
        let crop = Marg {
            top: 10,
            right: 20,
            bottom: 30,
            left: 40,
        };
        let (disp_w, disp_h) = display_source_dims(2000, 400, rawloader::Orientation::Rotate90);
        assert_eq!((disp_w, disp_h), (400, 2000));
        assert_eq!(
            scale_crop(crop, disp_w, disp_h, 400, 200),
            Marg {
                top: 1,
                right: 20,
                bottom: 3,
                left: 40
            }
        );
    }

    #[test]
    fn display_source_dims_swaps_axes_for_rotated_sensors() {
        use rawloader::Orientation;
        for o in [
            Orientation::Rotate90,
            Orientation::Rotate270,
            Orientation::Transpose,
            Orientation::Transverse,
        ] {
            assert_eq!(display_source_dims(6000, 4000, o), (4000, 6000));
        }
        for o in [
            Orientation::Normal,
            Orientation::Unknown,
            Orientation::HorizontalFlip,
            Orientation::VerticalFlip,
            Orientation::Rotate180,
        ] {
            assert_eq!(display_source_dims(6000, 4000, o), (6000, 4000));
        }
    }

    #[test]
    fn crop_bake_resolves_display_dims_before_scaling_to_the_thumb() {
        // The thumbnail bake must interpret the stored crop in the same
        // full-resolution display-source frame the detail view authors it in.
        // A 6000x4000 landscape sensor at THUMB_SIZE (384 long edge) prints
        // 384x256; a 100 source-px left+right trim ≈ 1.67% of the frame must
        // remove ≈ 6.4 print px per side — NOT ~1 px, which is what happens
        // when the crop is instead treated as overview-texture pixels and the
        // bake scales it against the full sensor.
        let crop = Marg {
            top: 0,
            right: 100,
            bottom: 0,
            left: 100,
        };
        let (disp_w, disp_h) = display_source_dims(6000, 4000, rawloader::Orientation::Normal);
        let scaled = scale_crop(crop, disp_w, disp_h, 384, 256);
        assert_eq!(
            (scaled.left, scaled.right, scaled.top, scaled.bottom),
            (6, 6, 0, 0)
        );
    }

    #[test]
    fn crop_bake_matches_a_portrait_rotated_frame() {
        // Rotate90: crop left/right margins in the portrait display are
        // horizontal in source vertical terms — resolved display dims handle
        // the swap, so the same 100-px side crop and a 100-px top trim land at
        // the same print fractions as the landscape case.
        let crop = Marg {
            top: 100,
            right: 100,
            bottom: 0,
            left: 0,
        };
        let (disp_w, disp_h) = display_source_dims(6000, 4000, rawloader::Orientation::Rotate90);
        assert_eq!((disp_w, disp_h), (4000, 6000));
        // Portrait print of a 6000-long sensor at THUMB_SIZE: 256x384.
        let scaled = scale_crop(crop, disp_w, disp_h, 256, 384);
        assert_eq!(
            (scaled.left, scaled.right, scaled.top, scaled.bottom),
            (0, 6, 6, 0)
        );
    }

    #[test]
    fn crop_rgba_slices_the_frame_to_the_margins() {
        // A 4x4 RGBA grid; each pixel gray = row-major pixel index.
        let mut rgba = Vec::new();
        for p in 0..16u8 {
            rgba.extend_from_slice(&[p, p, p, 255]);
        }
        let crop = Marg {
            top: 1,
            right: 1,
            bottom: 1,
            left: 1,
        };
        let (out, w, h) = crop_rgba(rgba, 4, 4, crop);
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[0..4], &[5, 5, 5, 255], "top-left = source row1 col1");
        assert_eq!(&out[4..8], &[6, 6, 6, 255]);
        assert_eq!(&out[8..12], &[9, 9, 9, 255]);
        assert_eq!(&out[12..16], &[10, 10, 10, 255]);
    }

    #[test]
    fn crop_rgba_is_identity_with_no_margins() {
        let rgba: Vec<u8> = (0..4u8).flat_map(|p| [p, p, p, 255]).collect();
        let (out, w, h) = crop_rgba(rgba.clone(), 2, 2, Marg::default());
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, rgba);
    }

    #[test]
    fn crop_rgba_rejects_an_overrunning_crop_unchanged() {
        let rgba: Vec<u8> = (0..4u8).flat_map(|p| [p, p, p, 255]).collect();
        let crop = Marg {
            top: 0,
            right: 0,
            bottom: 0,
            left: 100,
        };
        let (out, w, h) = crop_rgba(rgba.clone(), 2, 2, crop);
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, rgba);
    }

    #[test]
    fn rotate_quarters_quarter_turns_a_square_grid_ccw() {
        // A 2x2 RGBA grid; each pixel gray = its position (0..3).
        let rgba: Vec<u8> = (0..4u8).flat_map(|p| [p, p, p, 255]).collect();
        // One CCW turn keeps a square's dims but moves the original TOP edge
        // to the display's LEFT column (shader `rot == 1`: (1-v, u) puts the
        // display top-left at texture top-right, and the display left column
        // walks the original top row right→left as it goes down).
        let (out, w, h) = rotate_quarters(rgba.clone(), 2, 2, 1);
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[0..4], &[1, 1, 1, 255], "top-left = source top-right");
        assert_eq!(
            &out[8..12],
            &[0, 0, 0, 255],
            "bottom-left = source top-left"
        );
        // Two turns: pure 180° reversal, dims unchanged.
        let (out, w, h) = rotate_quarters(rgba.clone(), 2, 2, 2);
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[0..4], &[3, 3, 3, 255]);
        assert_eq!(&out[12..16], &[0, 0, 0, 255]);
        // Three turns (one CW): the original TOP edge lands on the display's
        // RIGHT column, so the display top-left samples the source BOTTOM-left.
        let (out, w, h) = rotate_quarters(rgba.clone(), 2, 2, 3);
        assert_eq!((w, h), (2, 2));
        assert_eq!(&out[0..4], &[2, 2, 2, 255], "top-left = source bottom-left");
        // Four turns wrap to the identity (mirrors `& 3` in the shader).
        let (out, w, h) = rotate_quarters(rgba.clone(), 2, 2, 4);
        assert_eq!((w, h), (2, 2));
        assert_eq!(out, rgba);
    }

    #[test]
    fn rotate_quarters_swaps_dims_on_an_odd_turn_and_composes_crop() {
        // A 3-wide × 2-tall grid; pixel value = row-major index.
        let mut rgba = Vec::new();
        for p in 0..6u8 {
            rgba.extend_from_slice(&[p, p, p, 255]);
        }
        let (out, w, h) = rotate_quarters(rgba.clone(), 3, 2, 1);
        assert_eq!((w, h), (2, 3), "odd turn swaps the print dims");
        // Output grid (2 wide × 3 tall): source 0 1 2 / 3 4 5 rotates CCW to
        // 2 5 / 1 4 / 0 3. Mapping: source (x,y) → output (ox=y, oy=w−1−x).
        assert_eq!(
            &out[0..4],
            &[2, 2, 2, 255],
            "output top-left = source top-right"
        );
        assert_eq!(
            &out[4..8],
            &[5, 5, 5, 255],
            "output top-right = source bottom-right"
        );
        assert_eq!(
            &out[8..12],
            &[1, 1, 1, 255],
            "output middle-left = source mid top"
        );
        assert_eq!(
            &out[16..20],
            &[0, 0, 0, 255],
            "output bottom-left = source top-left"
        );
        // Compose with a crop: quarter-turn the result of cropping a 3×3 frame
        // to its 1-px margins (a 1×1 center pixel), like the bake's
        // crop-then-rotate ordering.
        let mut rgba = Vec::new();
        for p in 0..9u8 {
            rgba.extend_from_slice(&[p, p, p, 255]);
        }
        let (cropped, cw, ch) = crop_rgba(
            rgba.clone(),
            3,
            3,
            Marg {
                top: 1,
                right: 1,
                bottom: 1,
                left: 1,
            },
        );
        assert_eq!((cw, ch), (1, 1));
        let (out, w, h) = rotate_quarters(cropped, cw, ch, 1);
        assert_eq!((w, h), (1, 1));
        assert_eq!(&out[0..4], &[4, 4, 4, 255], "crop center pixel survives");
    }


    /// The GPU side of the bake-parity tests: a Rust simulation of `exposure.wgsl`
    /// `shade()` where the tone remap comes from the same 2048×1 R16Float tone
    /// LUT the shader texture-samples (decoded to f32 + linear interpolation,
    /// via [`shader::sample_tone_lut_f32`]). The CPU bakes call [`tone_model`]
    /// exactly; the only approximation anywhere is the LUT itself, so this
    /// diffing bounds that approximation rather than a duplicated expression.
    #[allow(clippy::too_many_arguments)]
    fn gpu_fragment(
        mono: f32,
        exposure: f32,
        inv: bool,
        inv_base: f32,
        inv_d_max: f32,
        inv_gamma: f32,
        region_shadows: f32,
        region_highlights: f32,
        tone_lut: &[f32],
    ) -> f32 {
        let mono_linear = mono;
        let v = if inv {
            let transmission = mono_linear * exposure;
            let clamped = transmission.clamp(1e-6, inv_base);
            let density = -(clamped / inv_base).ln() / 10.0_f32.ln();
            let position = (density / inv_d_max).clamp(0.0, 1.0);
            let positive = position.powf(inv_gamma);
            // Density-domain region shape (mirrors the WGSL film branch:
            // `film::region_shape` with K = 2). The film path does NOT sample
            // the tone LUT.
            let toe = (1.0 - positive).powf(2.0);
            let shoulder = positive.powf(2.0);
            (positive + region_shadows * toe + region_highlights * shoulder)
                .clamp(0.0, 1.0)
        } else {
            let remapped = shader::sample_tone_lut_f32(tone_lut, mono_linear);
            (remapped * exposure).clamp(0.0, 1.0)
        };
        srgb_encode(v)
    }

    /// Decode a `build_tone_lut` half-float byte buffer back to f32, matching
    /// what the GPU's R16Float sampler would hand back (modulo the sampler's
    /// own linear interpolation).
    fn half_lut_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| shader::half_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    }

    /// Deterministic LCG so the differential tests are reproducible run-to-run.
    #[allow(clippy::cast_possible_truncation)]
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Top 23 mantissa bits of [1.0, 2.0), mapped to [0.0, 1.0).
            let bits = ((self.0 >> 40) as u32 & 0x007F_FFFF) | 0x3F80_0000;
            f32::from_bits(bits) - 1.0
        }
    }

    #[test]
    fn cpu_bake_matches_gpu_lut_simulation_on_random_frames() {
        // Randomize every input (frame values, EV, curve powers, preset, base
        // mode) and assert `bake_tone` (grid/export, exact `tone_model`) == the
        // GPU detail path (the same tone model delivered through a half-float
        // LUT + linear interpolation, simulated via `gpu_fragment`) — the
        // structural guarantee that the three render paths cannot drift beyond
        // the accepted LUT approximation. `pivots_for` is called on the
        // untouched frame on both sides, so they share identical pivots.
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        for case in 0..64 {
            let ev = rng.next() * 7.0 - 3.0;
            let tone = edit_manifest::ToneEdit {
                exposure_ev: ev,
                // Contrast spans its full ±3-stop power range (0.125..8.0);
                // Highlights/Shadows their ±2-stop range (0.25..4.0).
                curve_contrast: (rng.next() * 6.0 - 3.0).exp2(),
                curve_highlights: (rng.next() * 4.0 - 2.0).exp2(),
                curve_shadows: (rng.next() * 4.0 - 2.0).exp2(),
            };
            let preset = if rng.next() < 0.5 {
                FilmPreset::None
            } else {
                FilmPreset::Hp5Plus
            };
            let base_config = BaseConfig {
                calibrated: (rng.next() < 0.4).then(|| rng.next() * 0.5 + 0.3),
                auto: rng.next() < 0.3,
            };
            let mut mono: Vec<f32> = (0..256).map(|_| rng.next()).collect();
            // Stress the clamps with hard extremes.
            mono[0] = 0.0;
            mono[1] = 1.0;
            mono[64] = 1e-6;

            let mut baked = mono.clone();
            bake_tone(&mut baked, tone, preset, base_config);

            let (stock_and_base, pivots) = pivots_for(&mono, tone, preset, base_config);
            let (shadow, mid, white) = pivots;
            // The positive path delivers the curve through the LUT; the film
            // path evaluates its density-domain shape inline (mirrored by
            // `gpu_fragment`).
            let lut = half_lut_bytes_to_f32(&shader::build_tone_lut(
                tone.curve_contrast,
                tone.curve_highlights,
                tone.curve_shadows,
                shadow,
                mid,
                white,
            ));
            // Stop lifts recovered from the decoded powers, matching the maps
            // the app/pipeline use.
            let contrast_stops = tone.curve_contrast.log2();
            let shadows_stops = -tone.curve_shadows.log2();
            let highlights_stops = tone.curve_highlights.log2();
            let (exposure, inv, inv_base, inv_d_max, inv_gamma, region_shadows, region_highlights) =
                match stock_and_base {
                    Some((stock, base)) => (
                        shader::sensor_gain(tone.exposure_ev, true),
                        true,
                        base,
                        crate::film::effective_d_max(stock.d_max, contrast_stops),
                        stock.gamma,
                        crate::film::region_strength(shadows_stops),
                        crate::film::region_strength(highlights_stops),
                    ),
                    None => (
                        shader::sensor_gain(tone.exposure_ev, false),
                        false,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                    ),
                };

            for (i, &sample) in mono.iter().enumerate() {
                let reference = gpu_fragment(
                    sample,
                    exposure,
                    inv,
                    inv_base,
                    inv_d_max,
                    inv_gamma,
                    region_shadows,
                    region_highlights,
                    &lut,
                );
                let baked_value = baked[i];
                assert!(
                    (baked_value - reference).abs() <= 5e-4,
                    "case {case} pixel {i} ({sample}): bake {baked_value} vs gpu {reference}"
                );
            }
        }
    }
}
