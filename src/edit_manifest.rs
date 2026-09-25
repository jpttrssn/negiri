// SPDX-License-Identifier: GPL-3.0-or-later

use crate::film::{BaseMode, FilmPreset};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// File name of the per-roll edit manifest inside a roll directory.
///
/// Deliberately names the file after its function rather than the application,
/// so an application rename never orphans existing manifests.
pub const ROLL_MANIFEST_FILE: &str = ".film-roll.toml";

/// Exposure applied to an image until an edit records otherwise.
///
/// Matches Darktable's default +0.7 EV starting point: with the auto-expose
/// (`normalize_positive`) removed, EV 0 is true sensor exposure and untouched
/// frames open at this visible base rather than black. The slider spans
/// −3..+4 so the user can drag down to honest sensor exposure.
pub const DEFAULT_EXPOSURE_EV: f32 = 0.7;

/// Contrast power applied until an edit records otherwise (identity `1.0`).
pub const DEFAULT_CURVE_CONTRAST: f32 = 1.0;

/// Highlights power (pivoted at the image's measured shadow anchor) applied
/// until an edit records otherwise (identity `1.0`).
pub const DEFAULT_CURVE_HIGHLIGHTS: f32 = 1.0;

/// Shadows power (pivoted at the image's measured white point) applied until an
/// edit records otherwise (identity `1.0`).
pub const DEFAULT_CURVE_SHADOWS: f32 = 1.0;

/// Fixed-point quantum for the stored exposure (EV). The manifest stores an
/// integer tick count so on-disk values are exact and carry no f32 noise;
/// `0.05 EV` matches the keyboard nudge and keeps the `0.7 EV` default at 14.
pub const EV_TICK: f32 = 0.05;

/// Fixed-point quantum for the stored tone controls: all three (Contrast,
/// Highlights, Shadows) are stop-based lifts. Matches every tone slider's step.
pub const TONE_TICK: f32 = 0.05;

/// Stored exposure tick bounds (the exposure slider's −3..+4 EV range).
const EXPOSURE_TICK_MIN: i16 = -60;
const EXPOSURE_TICK_MAX: i16 = 80;
/// Stored contrast lift-tick bounds (the centered `−3..=3` stop track, power
/// `0.125..=8.0`).
const CONTRAST_TICK_MIN: i16 = -60;
const CONTRAST_TICK_MAX: i16 = 60;
/// Stored Highlights/Shadows lift-tick bounds (the centered `−2..=2` track).
const TONE_LIFT_TICK_MIN: i16 = -40;
const TONE_LIFT_TICK_MAX: i16 = 40;

/// `+0.7 EV` default in ticks (`0.7 / EV_TICK`).
const DEFAULT_EXPOSURE_TICKS: i16 = 14;
/// Identity contrast lift in ticks (power `1.0` ⇔ `0` stops).
const DEFAULT_CONTRAST_LIFT_TICKS: i16 = 0;

/// Rounds `value` to the nearest tick and clamps to `[min, max]`, so every
/// stored edit lands on the fixed-point grid.
#[allow(clippy::cast_possible_truncation)]
fn quantize_tick(value: f32, tick: f32, min: i16, max: i16) -> i16 {
    ((value / tick).round() as i32).clamp(i32::from(min), i32::from(max)) as i16
}

/// The stored exposure tick for an EV value.
fn exposure_ticks(ev: f32) -> i16 {
    quantize_tick(ev, EV_TICK, EXPOSURE_TICK_MIN, EXPOSURE_TICK_MAX)
}

/// The exposure EV a stored tick decodes to.
fn exposure_ev(ticks: i16) -> f32 {
    f32::from(ticks) * EV_TICK
}

/// The stored contrast lift tick for a mid-pivoted power. The lift is
/// `+log2(power)` (a rightward drag raises contrast), stored in the lift domain
/// so the `0.05` slider grid is exact through the integer round-trip.
fn contrast_lift_ticks(power: f32) -> i16 {
    quantize_tick(
        power.clamp(0.125, 8.0).log2(),
        TONE_TICK,
        CONTRAST_TICK_MIN,
        CONTRAST_TICK_MAX,
    )
}

/// The contrast power a stored lift tick decodes to.
fn contrast_power(ticks: i16) -> f32 {
    (f32::from(ticks) * TONE_TICK).exp2()
}

/// The stored Highlights lift tick for a shadow-pivoted power. The lift is
/// `+log2(power)` (a rightward drag brightens); storing it in the lift domain
/// keeps the `0.05` slider grid exact through the integer round-trip.
fn highlight_lift_ticks(power: f32) -> i16 {
    quantize_tick(
        power.clamp(0.25, 4.0).log2(),
        TONE_TICK,
        TONE_LIFT_TICK_MIN,
        TONE_LIFT_TICK_MAX,
    )
}

/// The Highlights power a stored lift tick decodes to.
fn highlight_power(ticks: i16) -> f32 {
    (f32::from(ticks) * TONE_TICK).exp2()
}

/// The stored Shadows lift tick for a white-pivoted power. The lift is
/// `-log2(power)` (a rightward drag brightens), the opposite power direction
/// from Highlights.
fn shadow_lift_ticks(power: f32) -> i16 {
    quantize_tick(
        -power.clamp(0.25, 4.0).log2(),
        TONE_TICK,
        TONE_LIFT_TICK_MIN,
        TONE_LIFT_TICK_MAX,
    )
}

/// The Shadows power a stored lift tick decodes to.
fn shadow_power(ticks: i16) -> f32 {
    (-f32::from(ticks) * TONE_TICK).exp2()
}

/// Serializable per-file edits, stored as exact integer ticks (see [`EV_TICK`]
/// / [`TONE_TICK`]) rather than f32.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditData {
    /// Exposure compensation in [`EV_TICK`] ticks (`14` = `+0.70 EV`).
    #[serde(default = "default_exposure_ticks")]
    pub exposure_ticks: i16,
    /// Contrast lift in stop ticks (mid-pivoted power, `+log2`), `0` =
    /// identity.
    #[serde(default = "default_contrast_lift_ticks")]
    pub curve_contrast_lift_ticks: i16,
    /// Highlights lift in stop ticks (shadow-pivoted power, `+log2`), `0` =
    /// identity.
    #[serde(default = "default_identity_lift_ticks")]
    pub curve_highlights_ticks: i16,
    /// Shadows lift in stop ticks (white-pivoted power, `-log2`), `0` =
    /// identity.
    #[serde(default = "default_identity_lift_ticks")]
    pub curve_shadows_ticks: i16,
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
            exposure_ticks: DEFAULT_EXPOSURE_TICKS,
            curve_contrast_lift_ticks: DEFAULT_CONTRAST_LIFT_TICKS,
            curve_highlights_ticks: 0,
            curve_shadows_ticks: 0,
            crop: CropMargins::default(),
            rotation: 0,
        }
    }
}

/// `#[serde(default)]` target so a legacy manifest entry without the exposure
/// field loads the current base exposure rather than a raw `0`.
#[allow(clippy::unnecessary_wraps)]
fn default_exposure_ticks() -> i16 {
    DEFAULT_EXPOSURE_TICKS
}

/// `#[serde(default)]` target so a legacy manifest entry without the contrast
/// field loads the identity contrast lift.
#[allow(clippy::unnecessary_wraps)]
fn default_contrast_lift_ticks() -> i16 {
    DEFAULT_CONTRAST_LIFT_TICKS
}

/// `#[serde(default)]` target for the identity Highlights/Shadows lift.
#[allow(clippy::unnecessary_wraps)]
fn default_identity_lift_ticks() -> i16 {
    0
}

/// A per-file edit aggregated into one value the decode/thumbnail pipelines
/// can thread through without a 5-tuple or per-field reads.
///
/// This is the runtime `f32` view decoded from the manifest's exact integer
/// ticks ([`EditData`]); the shader, LUT, and bakes consume these powers
/// directly. All fields are identities at `Default`: default exposure, identity
/// powers. `Default` must match the un-edited rendering exactly so a manifest
/// entry without an edit paints identical to `EditData::default()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneEdit {
    pub exposure_ev: f32,
    pub curve_contrast: f32,
    pub curve_highlights: f32,
    pub curve_shadows: f32,
}

impl Default for ToneEdit {
    fn default() -> Self {
        Self {
            exposure_ev: DEFAULT_EXPOSURE_EV,
            curve_contrast: DEFAULT_CURVE_CONTRAST,
            curve_highlights: DEFAULT_CURVE_HIGHLIGHTS,
            curve_shadows: DEFAULT_CURVE_SHADOWS,
        }
    }
}

impl ToneEdit {
    /// The value stored for a file carrying no edits.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
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
    pub version: u32,
    /// Optional human-readable roll label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Edits keyed by file name within the roll.
    #[serde(default)]
    pub edits: HashMap<String, EditData>,
    /// The roll's film-inversion preset, stored as its choice key. Absent means
    /// the default [`FilmPreset::None`] (a non-negative/regular-RAW roll). The
    /// base strategy (preset base / auto per frame / auto selected frame) is a
    /// separate roll-level choice, [`Self::base_mode`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// How the roll's black point is resolved, stored as its choice key. Absent
    /// means the default [`BaseMode::Preset`] (the chosen film's preset base).
    /// `AutoPerFrame` measures each frame, `AutoSelectedFrame` pins the whole
    /// roll to the measured [`Self::base`] of [`Self::calibration_frame`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_mode: Option<String>,
    /// The roll's resolved black point: the measured clear-film transmission of
    /// [`Self::calibration_frame`]. Meaningful only while the roll's base mode
    /// is [`BaseMode::AutoSelectedFrame`]; every frame inverts against this
    /// same base. Absent falls back to the film's preset base (preset-first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<f32>,
    /// The frame name whose measured clear-film plateau is the roll's black
    /// point under [`BaseMode::AutoSelectedFrame`]. Defaults to the roll's
    /// first sorted frame when that base mode is chosen; `None` under any other
    /// base mode (and never a decode input — only the dot indicator and the
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
}

impl Default for RollManifest {
    fn default() -> Self {
        Self {
            version: 1,
            name: None,
            edits: HashMap::new(),
            preset: None,
            base_mode: None,
            base: None,
            calibration_frame: None,
            start_date: None,
            end_date: None,
        }
    }
}

impl RollManifest {
    /// Records the exposure for `name`, updating an existing entry in place.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_exposure(&mut self, name: &str, exposure_ev: f32) {
        let ticks = exposure_ticks(exposure_ev);
        if let Some(edit) = self.edits.get_mut(name) {
            edit.exposure_ticks = ticks;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    exposure_ticks: ticks,
                    ..Default::default()
                },
            );
        }
    }

    /// Records the tone curve for `name` (contrast + highlights + shadows),
    /// updating an existing entry in place. Powers are quantized to
    /// [`TONE_TICK`]; the identities are contrast `1.0`, both lifts `0.0`.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_curve(&mut self, name: &str, contrast: f32, highlights: f32, shadows: f32) {
        let contrast = contrast_lift_ticks(contrast);
        let highlights = highlight_lift_ticks(highlights);
        let shadows = shadow_lift_ticks(shadows);
        if let Some(edit) = self.edits.get_mut(name) {
            edit.curve_contrast_lift_ticks = contrast;
            edit.curve_highlights_ticks = highlights;
            edit.curve_shadows_ticks = shadows;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    curve_contrast_lift_ticks: contrast,
                    curve_highlights_ticks: highlights,
                    curve_shadows_ticks: shadows,
                    ..EditData::default()
                },
            );
        }
    }

    /// The full edit for `name` as a single [`ToneEdit`], merging exposure
    /// and the curve powers (decoded from their stored ticks). Files (or
    /// manifest fields) never touched fall back to their identities.
    ///
    /// The crop margins are intentionally NOT part of this aggregate: it is
    /// the copy/paste payload, and a crop must never be copied/pasted.
    #[must_use]
    pub fn tone(&self, name: &str) -> ToneEdit {
        self.edits
            .get(name)
            .map_or_else(ToneEdit::identity, |edit| ToneEdit {
                exposure_ev: exposure_ev(edit.exposure_ticks),
                curve_contrast: contrast_power(edit.curve_contrast_lift_ticks),
                curve_highlights: highlight_power(edit.curve_highlights_ticks),
                curve_shadows: shadow_power(edit.curve_shadows_ticks),
            })
    }

    /// Replaces the full edit for `name` with `tone` (exposure + curve powers
    /// in one step — copy/paste), updating an existing entry in place.
    ///
    /// A paste never touches the target's crop or rotation: the crop margins
    /// and the user rotation survive [`Self::set_tone`] unchanged (or stay
    /// default on a fresh file). Powers are quantized to [`TONE_TICK`].
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
                exposure_ticks: exposure_ticks(tone.exposure_ev),
                curve_contrast_lift_ticks: contrast_lift_ticks(tone.curve_contrast),
                curve_highlights_ticks: highlight_lift_ticks(tone.curve_highlights),
                curve_shadows_ticks: shadow_lift_ticks(tone.curve_shadows),
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

    /// The film-inversion preset recorded for this roll. A manifest with no
    /// preset key recorded (or an unknown key) falls back to the default
    /// [`FilmPreset::None`].
    #[must_use]
    pub fn preset(&self) -> FilmPreset {
        self.preset
            .as_deref()
            .map_or_else(FilmPreset::default, FilmPreset::from_key)
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

    /// Records the film-inversion preset for the roll. Only a non-default
    /// preset is written: `Generic` or any stock preset stores its choice key,
    /// [`FilmPreset::None`] clears it back to the implicit default (the key
    /// disappears on the next save, which keeps the default manifest clean for
    /// raw scans).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_preset(&mut self, preset: FilmPreset) {
        match preset {
            FilmPreset::None => self.preset = None,
            _ => self.preset = Some(preset.choice_key().to_owned()),
        }
    }

    /// How the roll's black point is resolved. The absent/unknown key falls
    /// back to the default [`BaseMode::Preset`].
    #[must_use]
    pub fn base_mode(&self) -> BaseMode {
        self.base_mode
            .as_deref()
            .map_or_else(BaseMode::default, BaseMode::from_key)
    }

    /// Records the roll's base strategy. Only a non-default mode is written:
    /// [`BaseMode::AutoPerFrame`]/[`BaseMode::AutoSelectedFrame`] store their
    /// choice key, [`BaseMode::Preset`] clears it back to the implicit default
    /// (the key disappears on the next save).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_base_mode(&mut self, mode: BaseMode) {
        match mode {
            BaseMode::Preset => self.base_mode = None,
            BaseMode::AutoPerFrame | BaseMode::AutoSelectedFrame => {
                self.base_mode = Some(mode.choice_key().to_owned());
            }
        }
    }

    /// The roll's calibrated black point (the measured clear-film transmission
    /// of [`Self::calibration_frame`]), if the auto-selected calibration frame
    /// was measured and recorded.
    #[must_use]
    pub const fn calibrated_base(&self) -> Option<f32> {
        self.base
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

    /// The frame name designated as the roll's auto-calibration reference
    /// (the black-point source under the [`BaseMode::AutoSelectedFrame`] base
    /// mode), if any. `None` under any other base mode or before a frame is
    /// designated.
    #[must_use]
    pub fn calibration_frame(&self) -> Option<&str> {
        self.calibration_frame.as_deref()
    }

    /// Designates `name` as the roll's auto-calibration frame: under the
    /// [`BaseMode::AutoSelectedFrame`] base mode its measured clear-film
    /// plateau is the roll's black point. The numeric base itself is recorded
    /// separately via [`Self::set_calibrated_base`].
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
        self.base = Some(base);
    }

    /// Clears the auto-selected calibration reference: both the designated
    /// frame and its measured base. Used when the roll leaves the
    /// [`BaseMode::AutoSelectedFrame`] base mode (a stale reference must not
    /// linger for other modes).
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn clear_calibration(&mut self) {
        self.base = None;
        self.calibration_frame = None;
    }

    /// The roll's base-resolution state as a single [`film::BaseConfig`], the
    /// unit threaded into every decode and bake so a rendering reflects the
    /// roll's base mode. The film preset supplies the stock (and falls back to
    /// its preset base); a non-inverted roll (`None`) resolves the default (the
    /// base is irrelevant with no inversion). The base mode then decides where
    /// the black point comes from: `AutoPerFrame` opts into per-frame
    /// measurement, `AutoSelectedFrame` pins the whole roll to its measured
    /// calibration, and `Preset` uses the film's preset base.
    #[must_use]
    pub fn base_config(&self) -> crate::film::BaseConfig {
        if self.preset().stock().is_none() {
            return crate::film::BaseConfig::default();
        }
        match self.base_mode() {
            BaseMode::Preset => crate::film::BaseConfig {
                calibrated: None,
                auto: false,
            },
            BaseMode::AutoPerFrame => crate::film::BaseConfig {
                calibrated: None,
                auto: true,
            },
            BaseMode::AutoSelectedFrame => crate::film::BaseConfig {
                calibrated: self.calibrated_base(),
                auto: false,
            },
        }
    }
}

/// Full path to the roll manifest inside `dir`.
#[must_use]
pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join(ROLL_MANIFEST_FILE)
}

/// Loads the roll manifest for `dir`.
///
/// A missing manifest maps to a default roll. An unreadable or malformed
/// manifest likewise falls back to a default roll and is reported to stderr,
/// so a broken file never blocks scanning the library.
#[must_use]
pub fn load_roll_manifest(dir: &Path) -> RollManifest {
    let path = manifest_path(dir);
    match std::fs::read(&path) {
        Ok(bytes) => match std::str::from_utf8(&bytes) {
            Ok(text) => match toml::from_str(text) {
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

    /// A Contrast power on the stored lift-tick grid (mid-pivoted, `+log2`).
    fn cpow(tick: i16) -> f32 {
        contrast_power(tick)
    }

    /// A Highlights power on the stored tick grid (shadow-pivoted, `+log2`).
    fn hpow(tick: i16) -> f32 {
        highlight_power(tick)
    }

    /// A Shadows power on the stored tick grid (white-pivoted, `-log2`).
    fn spow(tick: i16) -> f32 {
        shadow_power(tick)
    }

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
    fn round_trip_preserves_edits_and_name() {
        let dir = temp_dir("roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_exposure("IMG_0001.DNG", 0.40);
        manifest.set_exposure("IMG_0002.RAW", -0.75);
        manifest.set_curve("IMG_0001.DNG", cpow(-4), hpow(4), spow(-3));
        manifest.set_rotation("IMG_0001.DNG", 1);

        save_roll_manifest(&dir, &manifest).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.tone("IMG_0001.DNG").exposure_ev, 0.40);
        assert_eq!(loaded.tone("IMG_0002.RAW").exposure_ev, -0.75);
        let tone_one = loaded.tone("IMG_0001.DNG");
        assert_eq!(tone_one.curve_contrast, cpow(-4));
        assert_eq!(tone_one.curve_highlights, hpow(4));
        assert_eq!(tone_one.curve_shadows, spow(-3));
        // The second image carried only an exposure edit → curve identity.
        assert_eq!(
            loaded.tone("IMG_0002.RAW"),
            ToneEdit {
                exposure_ev: -0.75,
                ..ToneEdit::identity()
            }
        );
    }

    #[test]
    fn missing_manifest_is_default() {
        let dir = temp_dir("missing");
        std::fs::create_dir_all(&dir).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, RollManifest::default());
        assert_eq!(loaded.version, 1);
    }

    #[test]
    fn dates_round_trip() {
        let dir = temp_dir("dates");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_dates(Some("2024-05-09".to_owned()), Some("2024-05-12".to_owned()));

        save_roll_manifest(&dir, &manifest).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.start_date(), Some("2024-05-09"));
        assert_eq!(loaded.end_date(), Some("2024-05-12"));
    }

    #[test]
    fn name_round_trips_and_clears() {
        let dir = temp_dir("name");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        assert_eq!(manifest.name(), None);

        manifest.set_name(Some("Rollerskates".to_owned()));
        save_roll_manifest(&dir, &manifest).unwrap();
        let mut loaded = load_roll_manifest(&dir);
        assert_eq!(loaded.name(), Some("Rollerskates"));

        // Clearing the label writes it back to None (defaulted on load), so a
        // renamed roll reverts to its directory leaf.
        loaded.set_name(None);
        save_roll_manifest(&dir, &loaded).unwrap();
        let cleared = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(cleared.name(), None);
    }

    #[test]
    fn undated_default_round_trip_stays_clean() {
        let dir = temp_dir("undated");
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = RollManifest::default();

        save_roll_manifest(&dir, &manifest).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.start_date(), None);
        assert_eq!(loaded.end_date(), None);
    }

    #[test]
    fn malformed_manifest_falls_back_to_default() {
        let dir = temp_dir("malformed");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "[edits".as_bytes()).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, RollManifest::default());
    }

    #[test]
    fn unicode_file_names_round_trip() {
        let dir = temp_dir("unicode");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_exposure("Rolle 1 IMG_10.DNG", 0.3);

        save_roll_manifest(&dir, &manifest).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.tone("Rolle 1 IMG_10.DNG").exposure_ev, 0.3);
    }

    #[test]
    fn unknown_fields_and_tables_are_tolerated() {
        let dir = temp_dir("unknown");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            manifest_path(&dir),
            "version = 1\nfortune = 42\n\n[edits.\"a.DNG\"]\nexposure_ticks = 30\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.tone("a.DNG").exposure_ev, 1.5);
        // The on-disk version is preserved verbatim: loading tolerates unknown
        // fields, so a hand-edited manifest survives a round trip.
        assert_eq!(loaded.version, 1);
    }

    #[test]
    fn edit_without_exposure_field_is_identity() {
        let dir = temp_dir("bare-edit");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "version = 1\n\n[edits.\"a.DNG\"]\n").unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.tone("a.DNG").exposure_ev, DEFAULT_EXPOSURE_EV);
    }

    #[test]
    fn edit_without_curve_fields_is_identity_curve() {
        // A manifest that predates the tone curve (or one written by this same
        // schema without the curve on a bare exposure edit) must load the curve
        // as identity, so exposure-only edits never pick up a phantom curve.
        let dir = temp_dir("bare-curve");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            manifest_path(&dir),
            "version = 1\n\n[edits.\"a.DNG\"]\nexposure_ticks = 15\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        let tone = loaded.tone("a.DNG");
        assert_eq!(tone.exposure_ev, 0.75);
        assert_eq!(tone.curve_contrast, DEFAULT_CURVE_CONTRAST);
        assert_eq!(tone.curve_highlights, DEFAULT_CURVE_HIGHLIGHTS);
        assert_eq!(tone.curve_shadows, DEFAULT_CURVE_SHADOWS);
    }

    #[test]
    fn curve_fields_round_trip() {
        let dir = temp_dir("curve-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_exposure("IMG_0001.DNG", 0.40);
        manifest.set_curve("IMG_0001.DNG", cpow(5), hpow(5), spow(-1));

        save_roll_manifest(&dir, &manifest).unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        let tone = loaded.tone("IMG_0001.DNG");
        assert_eq!(tone.curve_contrast, cpow(5));
        assert_eq!(tone.curve_highlights, hpow(5));
        assert_eq!(tone.curve_shadows, spow(-1));
        // Exposure survives alongside the curve on the same edit.
        assert_eq!(tone.exposure_ev, 0.40);
    }

    #[test]
    fn set_curve_updates_in_place() {
        let mut manifest = RollManifest::default();
        manifest.set_curve("a.DNG", cpow(-2), 1.1, 1.0);
        manifest.set_curve("a.DNG", cpow(0), 1.0, 1.0);

        assert_eq!(manifest.edits.len(), 1);
        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.curve_contrast, 1.0);
        assert_eq!(tone.curve_highlights, 1.0);
        assert_eq!(tone.curve_shadows, 1.0);
    }

    #[test]
    fn curve_and_exposure_coexist_on_one_edit() {
        // A file edited through both APIs keeps all values on one entry, and
        // a bare exposure edit never disturbs the curve defaults (or vice
        // versa).
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.3);
        assert_eq!(manifest.tone("a.DNG").curve_contrast, 1.0);

        manifest.set_curve("a.DNG", cpow(4), 0.9, 1.1);
        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.exposure_ev, 0.3);
        assert_eq!(manifest.edits.len(), 1);
    }

    #[test]
    fn tone_merges_exposure_and_curve() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", -0.5);
        manifest.set_curve("a.DNG", cpow(6), hpow(-6), spow(3));

        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.exposure_ev, -0.5);
        assert_eq!(tone.curve_contrast, cpow(6));
        assert_eq!(tone.curve_highlights, hpow(-6));
        assert_eq!(tone.curve_shadows, spow(3));
    }

    #[test]
    fn set_tone_replaces_the_full_edit_in_one_step() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.3);
        manifest.set_curve("a.DNG", cpow(6), hpow(2), spow(-2));

        // A copy/paste replaces every field at once.
        manifest.set_tone(
            "a.DNG",
            ToneEdit {
                exposure_ev: -1.2,
                curve_contrast: cpow(-16),
                curve_highlights: hpow(10),
                curve_shadows: spow(-8),
            },
        );

        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.exposure_ev, -1.2);
        assert_eq!(tone.curve_contrast, cpow(-16));
        assert_eq!(tone.curve_highlights, hpow(10));
        assert_eq!(tone.curve_shadows, spow(-8));
        assert_eq!(manifest.edits.len(), 1);
    }

    #[test]
    fn set_tone_creates_an_entry_on_a_fresh_file() {
        let mut manifest = RollManifest::default();
        manifest.set_tone(
            "b.DNG",
            ToneEdit {
                exposure_ev: 0.4,
                curve_contrast: cpow(2),
                curve_highlights: hpow(-3),
                curve_shadows: spow(0),
            },
        );
        assert_eq!(manifest.tone("b.DNG").exposure_ev, 0.4);
    }

    #[test]
    fn curve_of_unknown_file_is_identity() {
        let manifest = RollManifest::default();
        assert_eq!(manifest.tone("missing.DNG"), ToneEdit::identity());
    }

    #[test]
    fn reconcile_drops_stale_entries() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("gone.DNG", 0.3);
        manifest.set_exposure("kept.DNG", -0.4);

        reconcile(&mut manifest, &["kept.DNG".to_owned()]);

        assert_eq!(manifest.tone("gone.DNG").exposure_ev, DEFAULT_EXPOSURE_EV);
        assert_eq!(manifest.tone("kept.DNG").exposure_ev, -0.4);
        assert!(!manifest.edits.contains_key("gone.DNG"));
    }

    #[test]
    fn set_exposure_updates_in_place() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.5);
        manifest.set_exposure("a.DNG", -1.25);

        assert_eq!(manifest.edits.len(), 1);
        assert_eq!(manifest.tone("a.DNG").exposure_ev, -1.25);
    }

    #[test]
    fn crop_round_trips_and_defaults_to_zero() {
        let dir = temp_dir("crop-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        let crop = CropMargins {
            top: 10,
            right: 20,
            bottom: 30,
            left: 40,
        };
        manifest.set_crop("IMG_0001.DNG", crop);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.crop("IMG_0001.DNG"), crop);
        // An untouched file has no crop.
        assert_eq!(loaded.crop("IMG_0002.DNG"), CropMargins::default());
    }

    #[test]
    fn legacy_manifest_without_crop_loads_zero_margins() {
        let dir = temp_dir("legacy-crop");
        std::fs::create_dir_all(&dir).unwrap();
        // A manifest without the crop field.
        std::fs::write(
            manifest_path(&dir),
            "version = 1\n\n[edits.\"a.DNG\"]\nexposure_ticks = 15\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.crop("a.DNG"), CropMargins::default());
        // And the surviving tone fields still load.
        assert_eq!(loaded.tone("a.DNG").exposure_ev, 0.75);
    }

    #[test]
    fn crop_coexists_with_tone_edits_on_one_entry() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.3);
        manifest.set_curve("a.DNG", cpow(6), 0.9, 1.1);
        let crop = CropMargins {
            top: 5,
            right: 6,
            bottom: 7,
            left: 8,
        };
        manifest.set_crop("a.DNG", crop);

        assert_eq!(manifest.edits.len(), 1, "all edits on one entry");
        assert_eq!(manifest.crop("a.DNG"), crop);
        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.exposure_ev, 0.3);
        assert_eq!(tone.curve_contrast, cpow(6));
    }

    #[test]
    fn rotation_round_trips_and_defaults_to_zero() {
        let dir = temp_dir("rotation-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_rotation("IMG_0001.DNG", 2);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.rotation("IMG_0001.DNG"), 2);
        // An untouched file has no user rotation.
        assert_eq!(loaded.rotation("IMG_0002.DNG"), 0);
    }

    #[test]
    fn legacy_manifest_without_rotation_loads_zero() {
        let dir = temp_dir("legacy-rotation");
        std::fs::create_dir_all(&dir).unwrap();
        // A manifest without the rotation field.
        std::fs::write(
            manifest_path(&dir),
            "version = 1\n\n[edits.\"a.DNG\"]\nexposure_ticks = 15\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.rotation("a.DNG"), 0);
        // And the surviving tone fields still load.
        assert_eq!(loaded.tone("a.DNG").exposure_ev, 0.75);
    }

    #[test]
    fn rotation_coexists_with_tone_and_crop_on_one_entry() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.3);
        manifest.set_curve("a.DNG", cpow(6), 0.9, 1.1);
        let crop = CropMargins {
            top: 5,
            right: 6,
            bottom: 7,
            left: 8,
        };
        manifest.set_crop("a.DNG", crop);
        manifest.set_rotation("a.DNG", 3);

        assert_eq!(manifest.edits.len(), 1, "all edits on one entry");
        assert_eq!(manifest.rotation("a.DNG"), 3);
        assert_eq!(manifest.crop("a.DNG"), crop);
        let tone = manifest.tone("a.DNG");
        assert_eq!(tone.exposure_ev, 0.3);
        assert_eq!(tone.curve_contrast, cpow(6));
    }

    #[test]
    fn set_rotation_updates_in_place_and_creates_on_fresh_file() {
        let mut manifest = RollManifest::default();
        manifest.set_rotation("a.DNG", 0);
        assert_eq!(manifest.rotation("a.DNG"), 0);

        manifest.set_rotation("a.DNG", 1);
        manifest.set_rotation("a.DNG", 2);
        assert_eq!(manifest.edits.len(), 1);
        assert_eq!(manifest.rotation("a.DNG"), 2);
        // The fresh-file branch keeps the tone identities.
        assert_eq!(manifest.tone("a.DNG"), ToneEdit::identity());
    }

    #[test]
    fn paste_does_not_clobber_an_existing_crop_or_rotation() {
        let mut manifest = RollManifest::default();
        let crop = CropMargins {
            top: 5,
            right: 6,
            bottom: 7,
            left: 8,
        };
        manifest.set_crop("a.DNG", crop);
        manifest.set_rotation("a.DNG", 3);
        manifest.set_exposure("a.DNG", 0.3);

        // Copy/paste writes a fresh ToneEdit onto the same file; the crop and
        // the rotation must survive untouched (neither is part of the
        // copy/paste payload).
        manifest.set_tone(
            "a.DNG",
            ToneEdit {
                exposure_ev: -1.2,
                curve_contrast: cpow(8),
                curve_highlights: hpow(7),
                curve_shadows: spow(-6),
            },
        );

        assert_eq!(manifest.crop("a.DNG"), crop, "paste keeps the target's crop");
        assert_eq!(manifest.rotation("a.DNG"), 3, "paste keeps the target's rotation");
        assert_eq!(manifest.tone("a.DNG").exposure_ev, -1.2);
    }

    #[test]
    fn set_crop_updates_in_place_and_creates_on_fresh_file() {
        let mut manifest = RollManifest::default();
        manifest.set_crop("a.DNG", CropMargins::default());
        assert_eq!(manifest.crop("a.DNG"), CropMargins::default());

        let crop = CropMargins {
            top: 2,
            right: 2,
            bottom: 2,
            left: 2,
        };
        manifest.set_crop("a.DNG", crop);
        assert_eq!(manifest.edits.len(), 1);
        assert_eq!(manifest.crop("a.DNG"), crop);
    }

    #[test]
    fn preset_absent_loads_default_none() {
        // A manifest predating the preset field (or a default roll) renders as
        // a non-inverted scan: nothing recorded → FilmPreset::None.
        let mut manifest = RollManifest::default();
        assert_eq!(manifest.preset(), FilmPreset::None);

        manifest.set_exposure("a.DNG", 0.5);
        assert_eq!(manifest.preset(), FilmPreset::None);

        let dir = temp_dir("preset-absent");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "version = 1\n").unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::None);
    }

    #[test]
    fn preset_round_trips_non_default_and_coexists_with_edits() {
        let dir = temp_dir("preset-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_preset(FilmPreset::Hp5Plus);
        manifest.set_exposure("IMG_0001.DNG", 0.40);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::Hp5Plus);
        assert_eq!(loaded.tone("IMG_0001.DNG").exposure_ev, 0.40);
    }

    #[test]
    fn set_preset_default_clears_the_stored_key() {
        let mut manifest = RollManifest::default();
        manifest.set_preset(FilmPreset::Hp5Plus);
        assert_eq!(manifest.preset(), FilmPreset::Hp5Plus);
        assert_eq!(manifest.preset.as_deref(), Some("hp5"));

        // Switching back to the default clears the key so a save omits it.
        manifest.set_preset(FilmPreset::None);
        assert_eq!(manifest.preset(), FilmPreset::None);
        assert_eq!(manifest.preset, None);
    }

    #[test]
    fn unknown_preset_key_resolves_to_default() {
        let dir = temp_dir("preset-unknown");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "version = 1\npreset = \"delta\"\n").unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::None);
    }

    #[test]
    fn preset_key_string_loads_verbatim() {
        let dir = temp_dir("preset-key");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(manifest_path(&dir), "version = 1\npreset = \"hp5\"\n").unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::Hp5Plus);
    }

    #[test]
    fn calibrated_base_round_trips() {
        let dir = temp_dir("base-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_calibrated_base(0.71);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded, manifest);
        assert_eq!(loaded.calibrated_base(), Some(0.71));
    }

    #[test]
    fn preset_first_default_has_no_base_recording() {
        // Preset-first: a fresh roll records neither a calibration frame nor a
        // base.
        let mut manifest = RollManifest::default();
        assert_eq!(manifest.calibrated_base(), None);
        assert_eq!(manifest.calibration_frame(), None);
    }

    #[test]
    fn calibration_frame_round_trips_with_its_base() {
        let dir = temp_dir("calib-frame");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_preset(FilmPreset::Hp5Plus);
        manifest.set_base_mode(BaseMode::AutoSelectedFrame);
        manifest.set_calibration_frame("IMG_0007.DNG");
        manifest.set_calibrated_base(0.63);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::Hp5Plus);
        assert_eq!(loaded.base_mode(), BaseMode::AutoSelectedFrame);
        assert_eq!(loaded.calibration_frame(), Some("IMG_0007.DNG"));
        assert_eq!(loaded.calibrated_base(), Some(0.63));
    }

    #[test]
    fn clear_calibration_drops_frame_and_base() {
        let mut manifest = RollManifest::default();
        manifest.set_calibration_frame("IMG_0007.DNG");
        manifest.set_calibrated_base(0.63);
        manifest.clear_calibration();
        assert_eq!(manifest.calibration_frame(), None);
        assert_eq!(manifest.calibrated_base(), None);
    }

    #[test]
    fn legacy_manifest_without_base_loads_preset_default() {
        let dir = temp_dir("legacy-base");
        std::fs::create_dir_all(&dir).unwrap();
        // A manifest without the base fields.
        std::fs::write(
            manifest_path(&dir),
            "version = 1\n\n[edits.\"a.DNG\"]\nexposure_ticks = 15\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.version, 1, "the on-disk version is preserved");
        assert_eq!(loaded.calibrated_base(), None);
        assert_eq!(loaded.calibration_frame(), None);
        assert_eq!(loaded.base_mode(), BaseMode::Preset);
        assert_eq!(loaded.tone("a.DNG").exposure_ev, 0.75);
    }

    #[test]
    fn manifest_without_a_base_mode_keeps_its_preset() {
        let dir = temp_dir("legacy-no-auto");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            manifest_path(&dir),
            "version = 1\npreset = \"hp5\"\nbase = 0.71\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::Hp5Plus);
        assert_eq!(loaded.base_mode(), BaseMode::Preset);
        assert_eq!(loaded.calibrated_base(), Some(0.71));
    }

    #[test]
    fn base_mode_keys_round_trip() {
        let dir = temp_dir("base-mode-keys");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_preset(FilmPreset::Hp5Plus);
        manifest.set_base_mode(BaseMode::AutoPerFrame);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.base_mode(), BaseMode::AutoPerFrame);
        assert_eq!(loaded.base_mode.as_deref(), Some("auto-per-frame"));

        // The default (Preset) clears the key so a save omits it.
        let mut defaulted = manifest;
        defaulted.set_base_mode(BaseMode::Preset);
        assert_eq!(defaulted.base_mode(), BaseMode::Preset);
        assert_eq!(defaulted.base_mode, None);
    }

    #[test]
    fn base_config_maps_the_preset_and_base_mode() {
        // The film preset supplies the stock; the base mode decides where the
        // black point comes from. A non-inverted roll ignores the base mode.
        let mut manifest = RollManifest::default();
        assert_eq!(
            manifest.base_config(),
            crate::film::BaseConfig {
                calibrated: None,
                auto: false
            }
        );

        manifest.set_preset(FilmPreset::Generic);
        manifest.set_base_mode(BaseMode::AutoPerFrame);
        assert_eq!(
            manifest.base_config(),
            crate::film::BaseConfig {
                calibrated: None,
                auto: true
            }
        );

        manifest.set_base_mode(BaseMode::AutoSelectedFrame);
        manifest.set_calibrated_base(0.63);
        assert_eq!(
            manifest.base_config(),
            crate::film::BaseConfig {
                calibrated: Some(0.63),
                auto: false
            }
        );

        manifest.set_base_mode(BaseMode::Preset);
        assert_eq!(
            manifest.base_config(),
            crate::film::BaseConfig {
                calibrated: None,
                auto: false
            }
        );

        // A non-inverted roll resolves the default regardless of base mode.
        manifest.set_preset(FilmPreset::None);
        manifest.set_base_mode(BaseMode::AutoPerFrame);
        assert_eq!(
            manifest.base_config(),
            crate::film::BaseConfig {
                calibrated: None,
                auto: false
            }
        );
    }

    #[test]
    fn base_calibration_coexists_with_preset_and_edits() {
        let dir = temp_dir("base-coexist");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_preset(FilmPreset::Hp5Plus);
        manifest.set_base_mode(BaseMode::AutoSelectedFrame);
        manifest.set_calibrated_base(0.7);
        manifest.set_exposure("IMG_0001.DNG", 0.40);

        save_roll_manifest(&dir, &manifest).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.preset(), FilmPreset::Hp5Plus);
        assert_eq!(loaded.base_mode(), BaseMode::AutoSelectedFrame);
        assert_eq!(loaded.calibrated_base(), Some(0.7));
        assert_eq!(loaded.tone("IMG_0001.DNG").exposure_ev, 0.40);
    }

    #[test]
    fn manifest_stores_integer_ticks() {
        // The on-disk manifest is exact integers, not f32: the whole point of
        // the fixed-point storage.
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.7);
        manifest.set_curve("a.DNG", cpow(8), hpow(5), spow(-3));

        let dir = temp_dir("integer-ticks");
        std::fs::create_dir_all(&dir).unwrap();
        save_roll_manifest(&dir, &manifest).unwrap();
        let text = std::fs::read_to_string(manifest_path(&dir)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains("exposure_ticks = 14"), "{text}");
        assert!(text.contains("curve_contrast_lift_ticks = 8"), "{text}");
        assert!(text.contains("curve_highlights_ticks = 5"), "{text}");
        assert!(text.contains("curve_shadows_ticks = -3"), "{text}");
    }

    #[test]
    fn old_contrast_key_is_ignored_and_loads_identity() {
        // The pre-lift key carried a raw-power tick count; the renamed key must
        // not be read, so a stale sidecar loads the identity contrast.
        let dir = temp_dir("old-contrast-key");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            manifest_path(&dir),
            "version = 1\n\n[edits.\"a.DNG\"]\ncurve_contrast_ticks = 40\n",
        )
        .unwrap();

        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded.tone("a.DNG").curve_contrast, DEFAULT_CURVE_CONTRAST);
    }

    #[test]
    fn tick_conversions_round_trip_exactly_on_the_grid() {
        // The hardcoded identity ticks must track the f32 default constants.
        assert_eq!(exposure_ticks(DEFAULT_EXPOSURE_EV), DEFAULT_EXPOSURE_TICKS);
        assert_eq!(
            contrast_lift_ticks(DEFAULT_CURVE_CONTRAST),
            DEFAULT_CONTRAST_LIFT_TICKS
        );
        for ticks in [-60_i16, -7, 0, 14, 80] {
            assert_eq!(exposure_ticks(exposure_ev(ticks)), ticks);
        }
        for ticks in [-60_i16, -8, 0, 8, 60] {
            assert_eq!(contrast_lift_ticks(contrast_power(ticks)), ticks);
        }
        for ticks in [-40_i16, -3, 0, 4, 40] {
            assert_eq!(highlight_lift_ticks(highlight_power(ticks)), ticks);
            assert_eq!(shadow_lift_ticks(shadow_power(ticks)), ticks);
        }
    }

    #[test]
    fn off_grid_values_snap_to_the_nearest_tick_and_clamp() {
        // 0.42 EV is 8.4 ticks → 8 (0.40 EV); a 1.4 highlights power is
        // +0.485 stops → tick 10 (2^0.5); a 3.0 contrast power is +1.585 stops
        // → tick 32.
        assert_eq!(exposure_ticks(0.42), 8);
        assert_eq!(highlight_lift_ticks(1.4), 10);
        assert_eq!(contrast_lift_ticks(3.0), 32);
        // Out-of-range powers clamp to the endpoints (contrast above 8.0 and
        // below 0.125, shadows clamped to its 0.25 floor).
        assert_eq!(exposure_ticks(99.0), 80);
        assert_eq!(contrast_lift_ticks(99.0), 60);
        assert_eq!(contrast_lift_ticks(0.0), -60);
        assert_eq!(highlight_lift_ticks(99.0), 40);
        assert_eq!(shadow_lift_ticks(0.0), 40);
    }
}
