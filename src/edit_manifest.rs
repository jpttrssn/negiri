// SPDX-License-Identifier: GPL-3.0-or-later

use crate::film::{self, Develop};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

/// File name of the per-roll edit manifest inside a roll directory.
///
/// Deliberately names the file after its function rather than the application,
/// so an application rename never orphans existing manifests.
pub const ROLL_MANIFEST_FILE: &str = ".film-roll.toml";

/// Exposure applied to an image until an edit records otherwise.
pub const DEFAULT_EXPOSURE_EV: f32 = 0.7;

/// Contrast power applied until an edit records otherwise (identity `1.0`).
pub const DEFAULT_CONTRAST: f32 = 1.0;

/// Black density anchor applied until an edit records otherwise (identity `0`).
pub const DEFAULT_BLACK: f32 = 0.0;

/// White density anchor applied until an edit records otherwise.
pub const DEFAULT_WHITE: f32 = film::DEFAULT_D_MAX;

/// Midtone pivot offset applied until an edit records otherwise (identity `0`).
pub const DEFAULT_PIVOT: f32 = 0.0;

/// The storage grid's scale: values are stored to the nearest `1 / SCALE`
/// (one millistop / milli-EV, one milli-density, one milli-fraction), in each
/// control's own domain.
///
/// The manifest stores the *real* control value rounded onto this grid, not a
/// count on a UI grid, so retuning a control's slider range or steps can never
/// reinterpret a stored value. Changing this scale only changes the precision of
/// future writes; it never changes what an existing value means. It is stated as
/// an exact integer divisor so a rounded value re-rounds to the same `f32` (a
/// clean `0.7` stays `0.7`, not `0.70000005`). The UI steps below are whole
/// multiples of the grid (pinned by `steps_are_whole_multiples_of_the_storage_grid`),
/// so a drag or keypress always persists exactly with no rounding jump.
pub const STORAGE_SCALE: f32 = 1000.0;

/// The exposure one-tick (Shift) step, in EV — a UI concern, free to retune.
pub const EV_TICK: f32 = 0.01;

/// The exposure coarse drag step, in EV — a UI concern, free to retune.
pub const EV_STEP: f32 = 0.05;

/// The contrast / black / white / pivot one-tick (Shift) step — a UI concern,
/// free to retune.
pub const TONE_TICK: f32 = 0.01;

/// The contrast / black / white / pivot coarse drag step — a UI concern, free to
/// retune.
pub const TONE_STEP: f32 = 0.05;

/// The exposure control's whole domain in EV: both the slider's range (see
/// [`exposure_range`]) and the write clamp. A UI concern, free to retune.
const EXPOSURE_EV_MIN: f32 = -3.0;
const EXPOSURE_EV_MAX: f32 = 4.0;
/// The contrast lift control's domain in stops (the centered `−3..=3` stop
/// track, power `0.125..=8.0`).
const CONTRAST_LIFT_MIN: f32 = -3.0;
const CONTRAST_LIFT_MAX: f32 = 3.0;
/// The black density anchor's domain (±2.0 density).
const BLACK_MIN: f32 = -2.0;
const BLACK_MAX: f32 = 2.0;
/// The white density anchor's domain (`0.5..5.0` density).
const WHITE_MIN: f32 = 0.5;
const WHITE_MAX: f32 = 5.0;
/// The midtone pivot offset's domain (the `−0.5..=0.5` window-fraction range).
const PIVOT_MIN: f32 = -0.5;
const PIVOT_MAX: f32 = 0.5;

/// The exposure slider's range in EV (`−3.0..=4.0`).
pub fn exposure_range() -> RangeInclusive<f32> {
    EXPOSURE_EV_MIN..=EXPOSURE_EV_MAX
}

/// The contrast slider's range in stop-lift (`−3.0..=3.0`).
pub fn contrast_lift_range() -> RangeInclusive<f32> {
    CONTRAST_LIFT_MIN..=CONTRAST_LIFT_MAX
}

/// The same contrast domain in the raw power the develop functions take
/// (`0.125..=8.0`): the lift bounds mapped through the power curve. Because the
/// lift bounds are symmetric, the result is a reciprocal pair around the `1.0`
/// identity, so the stop-lift track spans an even ±3 stops.
pub fn contrast_power_range() -> RangeInclusive<f32> {
    CONTRAST_LIFT_MIN.exp2()..=CONTRAST_LIFT_MAX.exp2()
}

/// The black anchor slider's range in density (`−2.0..=2.0`).
pub fn black_range() -> RangeInclusive<f32> {
    BLACK_MIN..=BLACK_MAX
}

/// The white anchor slider's range in density (`0.5..=5.0`).
pub fn white_range() -> RangeInclusive<f32> {
    WHITE_MIN..=WHITE_MAX
}

/// The midtone pivot slider's range in window fraction (`−0.5..=0.5`).
pub fn pivot_range() -> RangeInclusive<f32> {
    PIVOT_MIN..=PIVOT_MAX
}

/// Clamps `value` into a control's `range()`.
///
/// `RangeInclusive::clamp` needs `Ord`, which `f32` deliberately does not
/// implement (NaN has no order), so this is the float-safe equivalent.
pub fn clamp_to(value: f32, range: RangeInclusive<f32>) -> f32 {
    value.clamp(*range.start(), *range.end())
}

/// Rounds `value` onto the real-unit storage grid (the nearest step of the
/// [`STORAGE_SCALE`] grid).
///
/// This is the only quantization the manifest applies. It keeps writes clean and
/// hand-editable (`0.4`, not `0.40000001`) and stops float drift accumulating
/// across repeated steps. Because the stored number is the control's real value,
/// a later change to a range or UI step cannot reinterpret it.
fn quantize(value: f32) -> f32 {
    (value * STORAGE_SCALE).round() / STORAGE_SCALE
}

/// Serializes an `f32` control value as an `f64` rounded onto the storage grid.
///
/// `toml` widens an `f32` to `f64` before formatting (`2.4f32` becomes
/// `2.4000000953674316`), so a raw `f32` field would leak the full binary value
/// into a hand-edited file. Rounding at `f64` precision first keeps the text a
/// clean decimal (`2.4`) while the value still decodes to the same grid `f32`.
#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `serialize_with` signature mandates `&f32`
fn serialize_grid<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    let scale = f64::from(STORAGE_SCALE);
    serializer.serialize_f64((f64::from(*value) * scale).round() / scale)
}

/// The `Option<f32>` twin of [`serialize_grid`] for an optional field (the roll
/// `base`); `skip_serializing_if` already excludes `None`.
// serde's `serialize_with` signature mandates `&Option<f32>`.
#[allow(clippy::ref_option, clippy::trivially_copy_pass_by_ref)]
fn serialize_grid_opt<S: Serializer>(
    value: &Option<f32>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => serialize_grid(value, serializer),
        None => serializer.serialize_none(),
    }
}

/// Serializable per-file edits, stored as the controls' real domain values
/// (f32) on the [`STORAGE_SCALE`] grid rather than as UI ticks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditData {
    /// Exposure compensation in EV (`0.7` = `+0.70 EV`).
    #[serde(default = "default_exposure_ev", serialize_with = "serialize_grid")]
    pub exposure_ev: f32,
    /// Contrast lift in stops about the midtone pivot (`+log2` of the develop
    /// power), `0.0` = identity.
    #[serde(default = "default_contrast_lift", serialize_with = "serialize_grid")]
    pub contrast_lift: f32,
    /// Black density anchor, `0.0` = identity.
    #[serde(default = "default_black", serialize_with = "serialize_grid")]
    pub black: f32,
    /// White density anchor ([`DEFAULT_WHITE`] default).
    #[serde(default = "default_white", serialize_with = "serialize_grid")]
    pub white: f32,
    /// Midtone pivot offset in window fraction, `0.0` = identity.
    #[serde(default = "default_pivot_offset", serialize_with = "serialize_grid")]
    pub pivot_offset: f32,
    /// Source-pixel crop margins removed from each edge. Missing in older
    /// manifests stays the all-zero (no-crop) [`CropMargins::default`].
    ///
    /// A crop is a first-class edit that persists and renders everywhere, but
    /// it is deliberately NOT part of [`ToneEdit`] so copy/paste never carries
    /// a crop onto another frame.
    #[serde(default)]
    pub crop: CropMargins,
    /// User-requested display rotation on TOP of the RAW's EXIF orientation:
    /// cumulative counter-clockwise 90° quarter-turns (`0`…`3`). Only the
    /// display (and export) of the frame, so the crop margins are still
    /// authored in the EXIF-upright source frame and are NOT re-interpreted
    /// here. Missing in older manifests stays `0` (no user rotation).
    #[serde(default)]
    pub rotation: u8,
}

impl Default for EditData {
    /// A fresh, un-edited entry: default exposure, identity tone curve, no crop.
    fn default() -> Self {
        Self {
            exposure_ev: DEFAULT_EXPOSURE_EV,
            contrast_lift: 0.0,
            black: 0.0,
            white: DEFAULT_WHITE,
            pivot_offset: 0.0,
            crop: CropMargins::default(),
            rotation: 0,
        }
    }
}

/// `#[serde(default)]` target so a manifest entry without the exposure field (or
/// carrying only a legacy `_ticks` key) loads the current base exposure rather
/// than a raw `0`.
fn default_exposure_ev() -> f32 {
    DEFAULT_EXPOSURE_EV
}

/// `#[serde(default)]` target for the identity contrast lift.
fn default_contrast_lift() -> f32 {
    0.0
}

/// `#[serde(default)]` target for the identity black density anchor.
fn default_black() -> f32 {
    0.0
}

/// `#[serde(default)]` target so a manifest entry without the white field loads
/// the default white density anchor.
fn default_white() -> f32 {
    DEFAULT_WHITE
}

/// `#[serde(default)]` target for the identity midtone pivot offset.
fn default_pivot_offset() -> f32 {
    0.0
}

/// A per-file edit aggregated into one value the decode/thumbnail pipelines
/// can thread through without a tuple or per-field reads.
///
/// This is the runtime `f32` view read from [`EditData`]'s stored real values:
/// the four develop shape controls plus exposure. The
/// clear-film `base` is roll-level (the designated calibration frame), so
/// building a [`Develop`] takes it alongside this type via
/// [`ToneEdit::to_develop`]. All fields are identities at `Default`, so a
/// manifest entry without an edit paints identical to `EditData::default()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneEdit {
    pub exposure_ev: f32,
    /// Contrast power about the midtone pivot (`1.0` = identity).
    pub contrast: f32,
    /// Black density anchor (`0.0` = identity).
    pub black: f32,
    /// White density anchor ([`DEFAULT_WHITE`] = identity).
    pub white: f32,
    /// Midtone pivot offset (`0.0` = identity).
    pub pivot_offset: f32,
}

impl Default for ToneEdit {
    fn default() -> Self {
        Self {
            exposure_ev: DEFAULT_EXPOSURE_EV,
            contrast: DEFAULT_CONTRAST,
            black: DEFAULT_BLACK,
            white: DEFAULT_WHITE,
            pivot_offset: DEFAULT_PIVOT,
        }
    }
}

impl ToneEdit {
    /// The value stored for a file carrying no edits.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build the pointwise density develop for this frame, anchored on the
    /// roll's resolved clear-film transmission `base`.
    #[must_use]
    pub fn to_develop(self, base: f32) -> Develop {
        Develop {
            exposure_ev: self.exposure_ev,
            base,
            black: self.black,
            white: self.white,
            contrast: self.contrast,
            pivot_offset: self.pivot_offset,
        }
    }
}

/// Source-pixel margins removed from each edge of a frame by a keyboard-driven
/// crop, preserving the natural aspect ratio.
///
/// `top`/`right`/`bottom`/`left` are in source image pixels (raw photosites) and
/// are applied at decode time, so a 1px margin trims 1 real sensor pixel
/// regardless of display scale. The all-zero [`Default`] is identity (no crop).
///
/// Deliberately kept OUT of [`ToneEdit`]: copy/paste copies only tone, so a crop
/// can never be pasted onto another frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CropMargins {
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub left: u32,
}

impl CropMargins {
    /// The cropped width given the source width, as a non-negative [`u32`].
    #[must_use]
    pub fn cropped_width(self, source: u32) -> u32 {
        source.saturating_sub(self.left).saturating_sub(self.right)
    }

    /// The cropped height given the source height, as a non-negative [`u32`].
    #[must_use]
    pub fn cropped_height(self, source: u32) -> u32 {
        source.saturating_sub(self.top).saturating_sub(self.bottom)
    }
}

/// Which edge a keyboard crop press trims. Only the four edges are exposed; a
/// corner is built by trimming two adjoining edges sequentially. The anchor is
/// auto-selected as the opposite edge's midpoint, and the perpendicular margins
/// derive from the aspect ratio to keep the frame ratio-locked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropDirection {
    Top,
    Bottom,
    Left,
    Right,
}

/// A film roll directory's app-owned manifest: roll metadata plus per-file
/// edits, keyed by file name within the roll. The directory itself is the
/// scope, so equal file names across two rolls never collide.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RollManifest {
    /// Manifest format revision; bumped when the on-disk schema changes.
    ///
    /// Pre-release, so a storage change is NOT versioned: the fields moved from
    /// `*_ticks` counts to real values (`exposure_ev`, `contrast_lift`, `black`,
    /// `white`, `pivot_offset`) in place, and a legacy `*_ticks` key is simply
    /// unknown and loads its field default (earlier work reads back near
    /// identity).
    /// The schema has never shipped, so there is nothing to migrate; a released
    /// build must bump this instead.
    pub version: u32,
    /// Optional human-readable roll label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Edits keyed by file name within the roll.
    #[serde(default)]
    pub edits: HashMap<String, EditData>,
    /// The roll's resolved clear-film transmission: the measured plateau of
    /// [`Self::calibration_frame`]. Every frame's density develop anchors on
    /// this same base. Absent falls back to [`crate::film::DEFAULT_FILM_BASE`]
    /// until a calibration frame is measured.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_grid_opt"
    )]
    pub base: Option<f32>,
    /// The frame name whose measured clear-film plateau pins the roll's base.
    /// Defaults to the roll's first sorted frame; `None` until one is
    /// designated (and never a decode input — only the dot indicator and the
    /// calibration flow read it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_frame: Option<String>,
    /// The roll's start date (the first shot), an ISO `YYYY-MM-DD` string set
    /// from the roll-info context drawer. Absent means undated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_date: Option<String>,
    /// The roll's optional end date (the last shot), ISO `YYYY-MM-DD`. Absent
    /// means the roll is undated or a single-day roll.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_date: Option<String>,
    /// The film stock the roll was shot on (free text, e.g. `Kodak Tri-X 400`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub film: Option<String>,
    /// Where the roll was shot (free text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// The camera the roll was shot with (free text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    /// The lens(es) used for the roll (free text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens: Option<String>,
    /// Free-form developer/process notes for the roll.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_notes: Option<String>,
}

/// The roll's free-form film metadata (film stock, location, camera, lens, and
/// developer notes) as one aggregate, mirroring how [`ToneEdit`] aggregates the
/// per-frame edit scalars. Each field is an optional string; `None` means the
/// field was never recorded (and serializes to nothing, keeping untouched rolls'
/// manifests clean).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RollMeta {
    pub film: Option<String>,
    pub location: Option<String>,
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub developer_notes: Option<String>,
}

impl RollMeta {
    /// The committed form of these fields: every value trimmed, with a
    /// whitespace-only value becoming `None` (so a cleared field is recorded as
    /// absent). Applied to the drawer's drafts on each debounced commit.
    #[must_use]
    pub fn normalized(&self) -> Self {
        fn clean(value: Option<&str>) -> Option<String> {
            let trimmed = value?.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        }
        Self {
            film: clean(self.film.as_deref()),
            location: clean(self.location.as_deref()),
            camera: clean(self.camera.as_deref()),
            lens: clean(self.lens.as_deref()),
            developer_notes: clean(self.developer_notes.as_deref()),
        }
    }
}

impl Default for RollManifest {
    fn default() -> Self {
        Self {
            version: 1,
            name: None,
            edits: HashMap::new(),
            base: None,
            calibration_frame: None,
            start_date: None,
            end_date: None,
            film: None,
            location: None,
            camera: None,
            lens: None,
            developer_notes: None,
        }
    }
}

impl RollManifest {
    /// Records the exposure for `name`, updating an existing entry in place.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_exposure(&mut self, name: &str, exposure_ev: f32) {
        let value = quantize(clamp_to(exposure_ev, exposure_range()));
        if let Some(edit) = self.edits.get_mut(name) {
            edit.exposure_ev = value;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    exposure_ev: value,
                    ..Default::default()
                },
            );
        }
    }

    /// Records the develop shape for `name` (contrast, black, white, midtone
    /// pivot), updating an existing entry in place. The contrast power is
    /// converted to its stored stop-lift (`+log2`), then every value is clamped
    /// to its control range and quantized onto the [`STORAGE_SCALE`] grid; the identities
    /// are contrast `1.0`, black `0.0`, white [`DEFAULT_WHITE`], pivot `0.0`.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_develop(&mut self, name: &str, tone: ToneEdit) {
        let contrast_lift = quantize(clamp_to(tone.contrast, contrast_power_range()).log2());
        let black = quantize(clamp_to(tone.black, black_range()));
        let white = quantize(clamp_to(tone.white, white_range()));
        let pivot_offset = quantize(clamp_to(tone.pivot_offset, pivot_range()));
        if let Some(edit) = self.edits.get_mut(name) {
            edit.contrast_lift = contrast_lift;
            edit.black = black;
            edit.white = white;
            edit.pivot_offset = pivot_offset;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    contrast_lift,
                    black,
                    white,
                    pivot_offset,
                    ..EditData::default()
                },
            );
        }
    }

    /// The full edit for `name` as a single [`ToneEdit`], reading the stored
    /// real values (with the stored contrast lift converted back to a power).
    /// Files (or manifest fields) never touched fall back to their identities.
    ///
    /// Each value is clamped to its current control range in RAM only — nothing
    /// is written back. A stored value left out of range by a narrowed control
    /// therefore reads as its clamped bound, and the untouched value on disk is
    /// preserved until the next edit write (so widening the control restores it).
    ///
    /// The crop margins are intentionally NOT part of this aggregate: it is
    /// the copy/paste payload, and a crop must never be copied/pasted.
    #[must_use]
    pub fn tone(&self, name: &str) -> ToneEdit {
        self.edits
            .get(name)
            .map_or_else(ToneEdit::identity, |edit| ToneEdit {
                exposure_ev: clamp_to(edit.exposure_ev, exposure_range()),
                contrast: clamp_to(edit.contrast_lift, contrast_lift_range()).exp2(),
                black: clamp_to(edit.black, black_range()),
                white: clamp_to(edit.white, white_range()),
                pivot_offset: clamp_to(edit.pivot_offset, pivot_range()),
            })
    }

    /// Replaces the full edit for `name` with `tone` (exposure + develop shape
    /// in one step — copy/paste), updating an existing entry in place.
    ///
    /// A paste never touches the target's crop or rotation: the crop margins
    /// and the user rotation survive [`Self::set_tone`] unchanged (or stay
    /// default on a fresh file). Values are clamped to their control ranges and
    /// quantized onto the [`STORAGE_SCALE`] grid.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_tone(&mut self, name: &str, tone: ToneEdit) {
        let existing = self
            .edits
            .get(name)
            .map_or(CropMargins::default(), |edit| edit.crop);
        let rotation = self.edits.get(name).map_or(0, |edit| edit.rotation);
        self.edits.insert(
            name.to_owned(),
            EditData {
                exposure_ev: quantize(clamp_to(tone.exposure_ev, exposure_range())),
                contrast_lift: quantize(clamp_to(tone.contrast, contrast_power_range()).log2()),
                black: quantize(clamp_to(tone.black, black_range())),
                white: quantize(clamp_to(tone.white, white_range())),
                pivot_offset: quantize(clamp_to(tone.pivot_offset, pivot_range())),
                crop: existing,
                rotation,
            },
        );
    }

    /// The crop margins for `name`; files (or manifest fields) never touched
    /// fall back to the all-zero (no-crop) default.
    #[must_use]
    pub fn crop(&self, name: &str) -> CropMargins {
        self.edits
            .get(name)
            .map_or_else(CropMargins::default, |edit| edit.crop)
    }

    /// Records the crop margins for `name`, updating an existing entry in
    /// place. Replaces the whole margins set at once (the keyboard trims build
    /// it up via [`Self::set_crop_amount`]).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_crop(&mut self, name: &str, crop: CropMargins) {
        if let Some(edit) = self.edits.get_mut(name) {
            edit.crop = crop;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    crop,
                    ..EditData::default()
                },
            );
        }
    }

    /// The user rotation for `name`; files (or manifest fields) never touched
    /// fall back to `0` (no rotation).
    #[must_use]
    pub fn rotation(&self, name: &str) -> u8 {
        self.edits.get(name).map_or(0, |edit| edit.rotation)
    }

    /// Records the cumulative counter-clockwise 90° rotation for `name`,
    /// updating an existing entry in place.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_rotation(&mut self, name: &str, rotation: u8) {
        if let Some(edit) = self.edits.get_mut(name) {
            edit.rotation = rotation;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    rotation,
                    ..Default::default()
                },
            );
        }
    }

    /// The roll's human-readable label, or `None` when it uses the directory
    /// leaf as its display name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Records the roll's display label. `Some` overrides the directory leaf;
    /// `None` clears it back to the leaf (the key disappears on the next save,
    /// keeping the default manifest clean for never-renamed rolls).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    /// The roll's calibrated black point (the measured clear-film transmission
    /// of [`Self::calibration_frame`]), if the calibration frame was measured
    /// and recorded.
    #[must_use]
    #[allow(dead_code)] // kept for the manifest tests; the app reads `base_or_default`
    pub const fn calibrated_base(&self) -> Option<f32> {
        self.base
    }

    /// The roll's resolved clear-film transmission: the measured calibration
    /// base, or [`film::DEFAULT_FILM_BASE`] until a calibration frame is
    /// measured. Every frame's develop anchors its density here.
    #[must_use]
    pub fn base_or_default(&self) -> f32 {
        self.base.unwrap_or(film::DEFAULT_FILM_BASE)
    }

    /// The roll's start date (ISO `YYYY-MM-DD`), or `None` if undated.
    #[must_use]
    pub fn start_date(&self) -> Option<&str> {
        self.start_date.as_deref()
    }

    /// The roll's optional end date (ISO `YYYY-MM-DD`), or `None` if unset.
    #[must_use]
    pub fn end_date(&self) -> Option<&str> {
        self.end_date.as_deref()
    }

    /// Records the roll's start and optional end dates as ISO `YYYY-MM-DD`
    /// strings (raw user input, validated by the UI layer). Passing `None`
    /// clears the corresponding field; an end date without a start date is
    /// still accepted but meaningless until a start date is recorded.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_dates(&mut self, start: Option<String>, end: Option<String>) {
        self.start_date = start;
        self.end_date = end;
    }

    /// The roll's free-form film metadata as one [`RollMeta`] aggregate
    /// (decoded from the flat manifest fields).
    #[must_use]
    pub fn meta(&self) -> RollMeta {
        RollMeta {
            film: self.film.clone(),
            location: self.location.clone(),
            camera: self.camera.clone(),
            lens: self.lens.clone(),
            developer_notes: self.developer_notes.clone(),
        }
    }

    /// Records the roll's free-form film metadata, replacing all five fields at
    /// once. A `None` field clears it (and serializes to nothing).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_meta(&mut self, meta: RollMeta) {
        self.film = meta.film;
        self.location = meta.location;
        self.camera = meta.camera;
        self.lens = meta.lens;
        self.developer_notes = meta.developer_notes;
    }

    /// The frame name designated as the roll's calibration reference (the
    /// black-point source), if any. `None` before a frame is designated.
    #[must_use]
    pub fn calibration_frame(&self) -> Option<&str> {
        self.calibration_frame.as_deref()
    }

    /// Designates `name` as the roll's calibration frame: its measured
    /// clear-film plateau pins the roll's black point. The numeric base itself
    /// is recorded separately via [`Self::set_calibrated_base`].
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_calibration_frame(&mut self, name: &str) {
        self.calibration_frame = Some(name.to_owned());
    }

    /// Records a calibrated black point measured from the roll's calibration
    /// frame. Every frame in the roll then inverts against this same base.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_calibrated_base(&mut self, base: f32) {
        self.base = Some(quantize(base));
    }

    /// Clears the calibration reference: both the designated frame and its
    /// measured base.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    #[allow(dead_code)] // kept for the manifest tests
    pub fn clear_calibration(&mut self) {
        self.base = None;
        self.calibration_frame = None;
    }
}

/// Full path to the roll manifest inside `dir`.
#[must_use]
pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(ROLL_MANIFEST_FILE)
}

/// Parses a manifest from its TOML text. `Err` carries the TOML error; the
/// caller reports it (with the path) and falls back to a default.
fn parse_manifest(text: &str) -> Result<RollManifest, toml::de::Error> {
    toml::from_str(text)
}

/// Builds a manifest from a raw file read, applying the fallback policy: a
/// missing manifest maps to a default roll, and an unreadable, non-UTF-8, or
/// malformed one likewise logs and yields a default, so a broken file never
/// blocks scanning the library. Shared by the sync and async loaders.
fn manifest_from_read(read: std::io::Result<Vec<u8>>, path: &Path) -> RollManifest {
    match read {
        Ok(bytes) => match std::str::from_utf8(&bytes) {
            Ok(text) => match parse_manifest(text) {
                Ok(manifest) => manifest,
                Err(err) => {
                    log::error!("malformed edit manifest {}: {err}", path.display());
                    RollManifest::default()
                }
            },
            Err(err) => {
                log::error!("non-UTF-8 edit manifest {}: {err}", path.display());
                RollManifest::default()
            }
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => RollManifest::default(),
        Err(err) => {
            log::error!("failed to read edit manifest {}: {err}", path.display());
            RollManifest::default()
        }
    }
}

/// Loads the roll manifest for `dir` synchronously. Callers on the async runtime
/// should prefer [`load_roll_manifest_async`] so the read doesn't block the
/// executor.
#[must_use]
pub fn load_roll_manifest(dir: &Path) -> RollManifest {
    let path = manifest_path(dir);
    manifest_from_read(std::fs::read(&path), &path)
}

/// Loads the roll manifest for `dir` without blocking the async executor: the
/// read runs on the blocking pool via [`tokio::fs`]. Same fallback policy as
/// [`load_roll_manifest`].
pub async fn load_roll_manifest_async(dir: &Path) -> RollManifest {
    let path = manifest_path(dir);
    manifest_from_read(tokio::fs::read(&path).await, &path)
}

/// Writes the roll manifest to `dir` atomically via a temp file and rename.
///
/// Unknown fields in an existing manifest are tolerated when loading (future
/// versions never break a scan) but are not preserved on save.
pub fn save_roll_manifest(dir: &Path, manifest: &RollManifest) -> std::io::Result<()> {
    let text = toml::to_string_pretty(manifest).map_err(std::io::Error::other)?;
    let path = manifest_path(dir);
    // The temp name starts with a dot like the manifest, so a scan that skips
    // dotfiles can never pick it up mid-write.
    let tmp = dir.join(format!("{ROLL_MANIFEST_FILE}.tmp"));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

/// Drops edit entries whose file name no longer exists in the roll's scan, so
/// a recycled name can never reattach stale edits.
pub fn reconcile(manifest: &mut RollManifest, files: &[String]) {
    manifest.edits.retain(|name, _| files.contains(name));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique scratch directory under the OS temp dir, caller-created and
    /// caller-cleaned.
    fn temp_dir(label: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "exposure-edit-manifest-{}-{label}-{seq}",
            std::process::id()
        ))
    }

    #[test]
    fn defaults_are_identity() {
        let manifest = RollManifest::default();
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.tone("missing.DNG"), ToneEdit::identity());
        assert_eq!(manifest.tone("missing.DNG").white, DEFAULT_WHITE);
        assert_eq!(manifest.tone("missing.DNG").contrast, 1.0);
        assert_eq!(manifest.base_or_default(), film::DEFAULT_FILM_BASE);
        assert_eq!(manifest.calibration_frame(), None);
    }

    #[test]
    fn tone_builds_a_develop_with_the_roll_base() {
        // `to_develop` pins the roll's resolved clear-film base and carries the
        // stored shape controls straight through.
        let tone = ToneEdit {
            exposure_ev: -0.4,
            contrast: 1.6,
            black: 0.15,
            white: 2.9,
            pivot_offset: -0.1,
        };
        let develop = tone.to_develop(0.71);
        assert!((develop.base - 0.71).abs() < 1e-6);
        assert!((develop.exposure_ev - (-0.4)).abs() < 1e-6);
        assert!((develop.contrast - 1.6).abs() < 1e-6);
        assert!((develop.black - 0.15).abs() < 1e-6);
        assert!((develop.white - 2.9).abs() < 1e-6);
        assert!((develop.pivot_offset - (-0.1)).abs() < 1e-6);
        // The identity tone yields a develop at that base whose shape controls
        // are identity; only the exposure carries the default `+0.7 EV`.
        let identity = ToneEdit::identity().to_develop(0.8);
        assert!((identity.base - 0.8).abs() < 1e-6);
        assert!((identity.exposure_ev - DEFAULT_EXPOSURE_EV).abs() < 1e-6);
        assert!((identity.contrast - 1.0).abs() < 1e-6);
        assert!(identity.black.abs() < 1e-6);
        assert!((identity.white - DEFAULT_WHITE).abs() < 1e-6);
        assert!(identity.pivot_offset.abs() < 1e-6);
    }

    #[test]
    fn round_trip_preserves_edits_and_name() {
        let dir = temp_dir("roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_name(Some("My Roll".to_owned()));
        manifest.set_exposure("IMG_0001.DNG", 0.40);
        manifest.set_exposure("IMG_0002.RAW", -0.75);
        manifest.set_develop(
            "IMG_0001.DNG",
            ToneEdit {
                exposure_ev: 0.40,
                contrast: 0.5,
                black: -0.2,
                white: 3.0,
                pivot_offset: 0.15,
            },
        );
        manifest.set_rotation("IMG_0001.DNG", 1);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.name(), Some("My Roll"));
        // The stored real values round-trip exactly through the storage grid.
        assert_eq!(loaded.tone("IMG_0001.DNG").exposure_ev, quantize(0.40));
        assert_eq!(loaded.tone("IMG_0002.RAW").exposure_ev, quantize(-0.75));
        let tone = loaded.tone("IMG_0001.DNG");
        assert_eq!(tone.contrast, 0.5);
        assert_eq!(tone.black, quantize(-0.2));
        assert_eq!(tone.white, quantize(3.0));
        assert_eq!(tone.pivot_offset, quantize(0.15));
        assert_eq!(loaded.rotation("IMG_0001.DNG"), 1);
        // The second image carried only an exposure edit → shape identity.
        assert_eq!(
            loaded.tone("IMG_0002.RAW"),
            ToneEdit {
                exposure_ev: -0.75,
                ..ToneEdit::identity()
            }
        );
    }

    #[test]
    fn manifest_stores_real_values() {
        let dir = temp_dir("real");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.4);
        manifest.set_develop(
            "a.DNG",
            ToneEdit {
                exposure_ev: 0.4,
                contrast: 2.0,
                black: 0.25,
                white: 2.0,
                pivot_offset: -0.1,
            },
        );
        save_roll_manifest(&dir, &manifest).unwrap();
        let text = std::fs::read_to_string(manifest_path(&dir)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        // Stored as the clean real control values, not tick counts. The
        // newline-terminated match is deliberate: a prefix match would also
        // accept the full-precision form (`0.4000000059604645`).
        assert!(text.contains("exposure_ev = 0.4\n"), "{text}");
        assert!(text.contains("contrast_lift = 1.0\n"), "{text}");
        assert!(text.contains("black = 0.25\n"), "{text}");
        assert!(text.contains("white = 2.0\n"), "{text}");
        assert!(text.contains("pivot_offset = -0.1\n"), "{text}");
        assert!(!text.contains("_ticks"), "legacy tick keys leaked: {text}");
    }

    #[test]
    fn f32_values_serialize_as_clean_decimals() {
        // `toml` widens f32 to f64, so without rounding a stored value would
        // leak its full binary form; these are the exact strings that bug
        // produced for the defaults and a calibration base.
        let dir = temp_dir("decimals");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        // A fresh entry carrying the field defaults (white = 2.4).
        manifest
            .edits
            .insert("a.DNG".to_owned(), EditData::default());
        manifest.set_calibrated_base(0.71);
        save_roll_manifest(&dir, &manifest).unwrap();
        let text = std::fs::read_to_string(manifest_path(&dir)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains("exposure_ev = 0.7\n"), "{text}");
        assert!(text.contains("contrast_lift = 0.0\n"), "{text}");
        assert!(text.contains("black = 0.0\n"), "{text}");
        assert!(text.contains("white = 2.4\n"), "{text}");
        assert!(text.contains("pivot_offset = 0.0\n"), "{text}");
        assert!(text.contains("base = 0.71\n"), "{text}");
        for widened in [
            "2.4000000953674316",
            "0.699999988079071",
            "0.7099999785423279",
        ] {
            assert!(!text.contains(widened), "leaked full precision: {text}");
        }
    }

    #[test]
    fn steps_are_whole_multiples_of_the_storage_grid() {
        // The control ladder rests on this: every UI step lands exactly on the
        // storage grid, so a drag or key can never persist a value that the
        // manifest would round (a fine step is never lost). Coarse is a whole
        // number of fine steps, so the Shift rung is a pure refinement of the
        // coarse one.
        for (coarse, fine) in [(EV_STEP, EV_TICK), (TONE_STEP, TONE_TICK)] {
            for step in [coarse, fine] {
                let grid = step * STORAGE_SCALE;
                assert!(
                    (grid - grid.round()).abs() < 1e-3,
                    "step {step} is not a whole number of storage-grid units"
                );
            }
            let rungs = coarse / fine;
            assert!(
                (rungs - rungs.round()).abs() < 1e-6 && rungs >= 1.0,
                "coarse step {coarse} is not a whole number of fine steps {fine}"
            );
        }
    }

    #[test]
    fn control_ranges_match_their_bounds_and_the_storage_grid() {
        // Each slider's range is exactly the control's real domain, and its end
        // points sit on the storage grid, so the slider can never hand an edit a
        // value the manifest would have to round.
        for (range, min, max) in [
            (exposure_range(), EXPOSURE_EV_MIN, EXPOSURE_EV_MAX),
            (contrast_lift_range(), CONTRAST_LIFT_MIN, CONTRAST_LIFT_MAX),
            (black_range(), BLACK_MIN, BLACK_MAX),
            (white_range(), WHITE_MIN, WHITE_MAX),
            (pivot_range(), PIVOT_MIN, PIVOT_MAX),
        ] {
            assert_eq!(*range.start(), min);
            assert_eq!(*range.end(), max);
            for bound in [min, max] {
                let grid = bound * STORAGE_SCALE;
                assert!(
                    (grid - grid.round()).abs() < 1e-3,
                    "bound {bound} is off the storage grid"
                );
            }
        }
        // The contrast slider shows stops, the develop functions take a power;
        // both come from one symmetric pair of lift bounds.
        let power = contrast_power_range();
        assert_eq!(*power.start(), 0.125);
        assert_eq!(*power.end(), 8.0);
        assert_eq!(contrast_lift_range(), -3.0..=3.0);
    }

    #[test]
    fn set_develop_updates_in_place_and_creates_on_fresh_file() {
        let dir = temp_dir("setdevelop");
        let mut manifest = RollManifest::default();
        manifest.set_develop(
            "a.DNG",
            ToneEdit {
                contrast: 2.0,
                black: 0.1,
                white: 2.0,
                pivot_offset: 0.2,
                ..ToneEdit::identity()
            },
        );
        assert_eq!(manifest.edits.len(), 1);
        manifest.set_develop(
            "a.DNG",
            ToneEdit {
                contrast: 1.0,
                black: 0.0,
                white: DEFAULT_WHITE,
                pivot_offset: 0.0,
                ..ToneEdit::identity()
            },
        );
        assert_eq!(manifest.edits.len(), 1);
        let tone = manifest.tone("a.DNG");
        assert!((tone.contrast - 1.0).abs() < 1e-6);
        assert!(tone.black.abs() < 1e-6);
        let _ = dir;
    }

    #[test]
    fn set_tone_replaces_the_shape_and_preserves_crop_and_rotation() {
        let mut manifest = RollManifest::default();
        manifest.set_crop(
            "a.DNG",
            CropMargins {
                top: 1,
                right: 2,
                bottom: 3,
                left: 4,
            },
        );
        manifest.set_rotation("a.DNG", 2);
        let pasted = ToneEdit {
            exposure_ev: -0.5,
            contrast: 2.0,
            black: 0.2,
            white: 3.0,
            pivot_offset: -0.2,
        };
        manifest.set_tone("a.DNG", pasted);
        assert_eq!(manifest.tone("a.DNG"), pasted);
        assert_eq!(manifest.rotation("a.DNG"), 2);
        assert_eq!(
            manifest.crop("a.DNG"),
            CropMargins {
                top: 1,
                right: 2,
                bottom: 3,
                left: 4,
            }
        );
    }

    #[test]
    fn crop_and_rotation_round_trip() {
        let dir = temp_dir("crop");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_crop(
            "a.DNG",
            CropMargins {
                top: 5,
                right: 6,
                bottom: 7,
                left: 8,
            },
        );
        manifest.set_rotation("a.DNG", 3);
        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded.crop("a.DNG").top, 5);
        assert_eq!(loaded.crop("a.DNG").left, 8);
        assert_eq!(loaded.rotation("a.DNG"), 3);
    }

    #[test]
    fn calibration_base_and_frame_round_trip() {
        let dir = temp_dir("calibration");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_calibration_frame("IMG_0001.DNG");
        manifest.set_calibrated_base(0.71);
        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded.calibration_frame(), Some("IMG_0001.DNG"));
        assert_eq!(loaded.calibrated_base(), Some(0.71));
        assert!((loaded.base_or_default() - 0.71).abs() < 1e-6);
    }

    #[test]
    fn clear_calibration_drops_frame_and_base() {
        let mut manifest = RollManifest::default();
        manifest.set_calibration_frame("a.DNG");
        manifest.set_calibrated_base(0.7);
        manifest.clear_calibration();
        assert_eq!(manifest.calibration_frame(), None);
        assert_eq!(manifest.calibrated_base(), None);
        assert_eq!(manifest.base_or_default(), film::DEFAULT_FILM_BASE);
    }

    #[test]
    fn dates_round_trip() {
        let dir = temp_dir("dates");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_dates(Some("2026-05-30".to_owned()), Some("2026-06-02".to_owned()));
        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded.start_date(), Some("2026-05-30"));
        assert_eq!(loaded.end_date(), Some("2026-06-02"));
    }

    #[test]
    fn roll_meta_round_trips() {
        let dir = temp_dir("roll-meta");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_meta(RollMeta {
            film: Some("Kodak Tri-X 400".to_owned()),
            location: Some("Chicago".to_owned()),
            camera: Some("Nikon FM2".to_owned()),
            lens: Some("50mm f/1.4".to_owned()),
            developer_notes: Some("HC-110 dil B, 6:30".to_owned()),
        });
        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded.meta(), manifest.meta());
        assert_eq!(loaded.meta().film.as_deref(), Some("Kodak Tri-X 400"));
        assert_eq!(
            loaded.meta().developer_notes.as_deref(),
            Some("HC-110 dil B, 6:30")
        );
    }

    #[test]
    fn default_manifest_omits_empty_roll_meta() {
        let text = toml::to_string_pretty(&RollManifest::default()).unwrap();
        for key in ["film", "location", "camera", "lens", "developer_notes"] {
            assert!(
                !text.contains(key),
                "default manifest unexpectedly contains `{key}`:\n{text}"
            );
        }
    }

    #[test]
    fn roll_meta_normalized_trims_and_clears_whitespace() {
        let draft = RollMeta {
            film: Some("  Kodak Tri-X  ".to_owned()),
            location: Some("   ".to_owned()),
            camera: Some("Nikon FM2".to_owned()),
            lens: None,
            developer_notes: Some("\t HC-110 \n".to_owned()),
        };
        assert_eq!(
            draft.normalized(),
            RollMeta {
                film: Some("Kodak Tri-X".to_owned()),
                location: None,
                camera: Some("Nikon FM2".to_owned()),
                lens: None,
                developer_notes: Some("HC-110".to_owned()),
            }
        );
    }

    #[test]
    fn unknown_fields_and_legacy_tone_keys_are_tolerated() {
        let dir = temp_dir("unknown");
        std::fs::create_dir_all(&dir).unwrap();
        let text = r#"
version = 1
fortune = "ignored"
base = 0.66
calibration_frame = "a.DNG"
[edits."a.DNG"]
exposure_ticks = 150
curve_contrast_lift_ticks = 8
curve_highlights_ticks = 5
curve_shadows_ticks = -3
"#;
        std::fs::write(manifest_path(&dir), text).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        // Unknown key ignored; the legacy `*_ticks` keys are unknown now and load
        // the field defaults (no migration), so exposure falls back to 0.7.
        assert_eq!(loaded.tone("a.DNG").exposure_ev, DEFAULT_EXPOSURE_EV);
        assert!((loaded.tone("a.DNG").contrast - 1.0).abs() < 1e-6);
        assert_eq!(loaded.calibrated_base(), Some(0.66));
    }

    #[test]
    fn malformed_manifest_falls_back_to_default() {
        let dir = temp_dir("malformed");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "not = [valid toml").unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded, RollManifest::default());
    }

    #[test]
    fn parse_manifest_accepts_valid_and_rejects_malformed_toml() {
        assert!(parse_manifest("version = 1").is_ok());
        assert!(parse_manifest("not = [valid toml").is_err());
    }

    #[tokio::test]
    async fn async_load_matches_the_sync_loader() {
        let dir = temp_dir("async-load");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_name(Some("Async Roll".to_owned()));
        manifest.set_exposure("a.DNG", 0.5);
        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest_async(&dir).await;
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn reconcile_drops_stale_entries() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("keep.DNG", 0.4);
        manifest.set_exposure("stale.DNG", 0.2);
        reconcile(&mut manifest, &["keep.DNG".to_owned()]);
        assert_eq!(manifest.tone("keep.DNG").exposure_ev, 0.4);
        assert_eq!(manifest.tone("stale.DNG"), ToneEdit::identity());
    }

    #[test]
    fn values_round_trip_through_storage_quantization() {
        // Values that already sit on the storage grid survive quantize()
        // unchanged (a clean `0.7` stays `0.7`, never `0.70000005`), and
        // quantize is idempotent.
        for value in [
            -3.0_f32, -0.75, -0.5, -0.2, -0.1, 0.0, 0.1, 0.15, 0.25, 0.4, 0.5, 0.7, 1.0, 2.0, 2.4,
            3.0, 4.0, 5.0,
        ] {
            assert_eq!(quantize(value), value, "quantize({value}) is not fixed");
            assert_eq!(quantize(quantize(value)), quantize(value));
        }
    }

    #[test]
    fn off_grid_values_snap_and_clamp() {
        // Off-grid values round to the nearest storage-grid step.
        assert_eq!(quantize(0.4004), 0.4);
        assert_eq!(quantize(-0.4004), -0.4);

        // A write clamps into the control range (the develop contrast is first
        // converted to its stop lift).
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 100.0);
        assert_eq!(manifest.edits["a.DNG"].exposure_ev, EXPOSURE_EV_MAX);
        manifest.set_exposure("b.DNG", -100.0);
        assert_eq!(manifest.edits["b.DNG"].exposure_ev, EXPOSURE_EV_MIN);
        manifest.set_develop(
            "c.DNG",
            ToneEdit {
                black: 100.0,
                white: 0.0,
                ..ToneEdit::identity()
            },
        );
        assert_eq!(manifest.edits["c.DNG"].black, BLACK_MAX);
        assert_eq!(manifest.edits["c.DNG"].white, WHITE_MIN);

        // A non-positive contrast power is clamped before the log, so it can
        // never store NaN/-inf (power 0 clamps to 0.125 -> -3 stops).
        for power in [0.0_f32, -1.0] {
            manifest.set_develop(
                "e.DNG",
                ToneEdit {
                    contrast: power,
                    ..ToneEdit::identity()
                },
            );
            assert_eq!(manifest.edits["e.DNG"].contrast_lift, CONTRAST_LIFT_MIN);
        }

        // Reading clamps to the current range in RAM without touching the stored
        // value, so widening the control later restores it.
        manifest.edits.insert(
            "d.DNG".to_owned(),
            EditData {
                exposure_ev: 4.5,
                ..EditData::default()
            },
        );
        assert_eq!(manifest.tone("d.DNG").exposure_ev, EXPOSURE_EV_MAX);
        assert_eq!(manifest.edits["d.DNG"].exposure_ev, 4.5);
    }
}
