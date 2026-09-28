// SPDX-License-Identifier: GPL-3.0-or-later

use crate::film::{self, Develop};
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
pub const DEFAULT_CONTRAST: f32 = 1.0;

/// Black density anchor applied until an edit records otherwise (identity `0`).
pub const DEFAULT_BLACK: f32 = 0.0;

/// White density anchor applied until an edit records otherwise
/// ([`film::DEFAULT_D_MAX`]).
pub const DEFAULT_WHITE: f32 = film::DEFAULT_D_MAX;

/// Midtone pivot offset applied until an edit records otherwise (identity `0`).
pub const DEFAULT_PIVOT: f32 = 0.0;

/// Fixed-point quantum for the stored exposure (EV). The manifest stores an
/// integer tick count so on-disk values are exact and carry no f32 noise;
/// `0.05 EV` matches the keyboard nudge and keeps the `0.7 EV` default at 14.
pub const EV_TICK: f32 = 0.05;

/// Fixed-point quantum for the stored develop controls (contrast lift, black /
/// white density, midtone pivot). Matches every develop slider's step.
pub const TONE_TICK: f32 = 0.05;

/// Stored exposure tick bounds (the exposure slider's −3..+4 EV range).
const EXPOSURE_TICK_MIN: i16 = -60;
const EXPOSURE_TICK_MAX: i16 = 80;
/// Stored contrast lift-tick bounds (the centered `−3..=3` stop track, power
/// `0.125..=8.0`).
const CONTRAST_TICK_MIN: i16 = -60;
const CONTRAST_TICK_MAX: i16 = 60;
/// Stored black density anchor bounds (±1.0 density).
const BLACK_TICK_MIN: i16 = -20;
const BLACK_TICK_MAX: i16 = 20;
/// Stored white density anchor bounds (`0.5..5.0` density).
const WHITE_TICK_MIN: i16 = 10;
const WHITE_TICK_MAX: i16 = 100;
/// Stored midtone pivot offset bounds (the `−0.5..=0.5` window-fraction range).
const PIVOT_TICK_MIN: i16 = -10;
const PIVOT_TICK_MAX: i16 = 10;

/// `+0.7 EV` default in ticks (`0.7 / EV_TICK`).
const DEFAULT_EXPOSURE_TICKS: i16 = 14;
/// Identity contrast lift in ticks (power `1.0` ⇔ `0` stops).
const DEFAULT_CONTRAST_LIFT_TICKS: i16 = 0;
/// Default white density in ticks ([`DEFAULT_WHITE`] / [`TONE_TICK`]).
const DEFAULT_WHITE_TICKS: i16 = 48;

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

/// The stored black density anchor tick.
fn black_ticks(black: f32) -> i16 {
    quantize_tick(black, TONE_TICK, BLACK_TICK_MIN, BLACK_TICK_MAX)
}

/// The black density anchor a stored tick decodes to.
fn black_density(ticks: i16) -> f32 {
    f32::from(ticks) * TONE_TICK
}

/// The stored white density anchor tick.
fn white_ticks(white: f32) -> i16 {
    quantize_tick(white, TONE_TICK, WHITE_TICK_MIN, WHITE_TICK_MAX)
}

/// The white density anchor a stored tick decodes to.
fn white_density(ticks: i16) -> f32 {
    f32::from(ticks) * TONE_TICK
}

/// The stored midtone pivot offset tick.
fn pivot_ticks(pivot: f32) -> i16 {
    quantize_tick(pivot, TONE_TICK, PIVOT_TICK_MIN, PIVOT_TICK_MAX)
}

/// The midtone pivot offset a stored tick decodes to.
fn pivot_offset(ticks: i16) -> f32 {
    f32::from(ticks) * TONE_TICK
}

/// Serializable per-file edits, stored as exact integer ticks (see [`EV_TICK`]
/// / [`TONE_TICK`]) rather than f32.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EditData {
    /// Exposure compensation in [`EV_TICK`] ticks (`14` = `+0.70 EV`).
    #[serde(default = "default_exposure_ticks")]
    pub exposure_ticks: i16,
    /// Contrast lift in stop ticks (about the midtone pivot, `+log2`), `0` =
    /// identity.
    #[serde(default = "default_contrast_lift_ticks")]
    pub contrast_lift_ticks: i16,
    /// Black density anchor in [`TONE_TICK`] ticks, `0` = identity.
    #[serde(default = "default_identity_tone_ticks")]
    pub black_ticks: i16,
    /// White density anchor in [`TONE_TICK`] ticks ([`DEFAULT_WHITE`] default).
    #[serde(default = "default_white_ticks")]
    pub white_ticks: i16,
    /// Midtone pivot offset in [`TONE_TICK`] ticks, `0` = identity.
    #[serde(default = "default_identity_tone_ticks")]
    pub pivot_ticks: i16,
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
            contrast_lift_ticks: DEFAULT_CONTRAST_LIFT_TICKS,
            black_ticks: 0,
            white_ticks: DEFAULT_WHITE_TICKS,
            pivot_ticks: 0,
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

/// `#[serde(default)]` target for an identity develop tick (black / pivot).
#[allow(clippy::unnecessary_wraps)]
fn default_identity_tone_ticks() -> i16 {
    0
}

/// `#[serde(default)]` target so a legacy manifest entry without the white
/// field loads the default white density anchor.
#[allow(clippy::unnecessary_wraps)]
fn default_white_ticks() -> i16 {
    DEFAULT_WHITE_TICKS
}

/// A per-file edit aggregated into one value the decode/thumbnail pipelines
/// can thread through without a tuple or per-field reads.
///
/// This is the runtime `f32` view decoded from the manifest's exact integer
/// ticks ([`EditData`]): the four develop shape controls plus exposure. The
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

    /// Records the develop shape for `name` (contrast, black, white, midtone
    /// pivot), updating an existing entry in place. Values are quantized to
    /// [`TONE_TICK`]; the identities are contrast `1.0`, black `0.0`, white
    /// [`DEFAULT_WHITE`], pivot `0.0`.
    ///
    /// RAM-only: the caller flushes to disk via [`save_roll_manifest`].
    pub fn set_develop(&mut self, name: &str, tone: ToneEdit) {
        let contrast = contrast_lift_ticks(tone.contrast);
        let black = black_ticks(tone.black);
        let white = white_ticks(tone.white);
        let pivot = pivot_ticks(tone.pivot_offset);
        if let Some(edit) = self.edits.get_mut(name) {
            edit.contrast_lift_ticks = contrast;
            edit.black_ticks = black;
            edit.white_ticks = white;
            edit.pivot_ticks = pivot;
        } else {
            self.edits.insert(
                name.to_owned(),
                EditData {
                    contrast_lift_ticks: contrast,
                    black_ticks: black,
                    white_ticks: white,
                    pivot_ticks: pivot,
                    ..EditData::default()
                },
            );
        }
    }

    /// The full edit for `name` as a single [`ToneEdit`], merging exposure and
    /// the develop shape (decoded from their stored ticks). Files (or manifest
    /// fields) never touched fall back to their identities.
    ///
    /// The crop margins are intentionally NOT part of this aggregate: it is
    /// the copy/paste payload, and a crop must never be copied/pasted.
    #[must_use]
    pub fn tone(&self, name: &str) -> ToneEdit {
        self.edits
            .get(name)
            .map_or_else(ToneEdit::identity, |edit| ToneEdit {
                exposure_ev: exposure_ev(edit.exposure_ticks),
                contrast: contrast_power(edit.contrast_lift_ticks),
                black: black_density(edit.black_ticks),
                white: white_density(edit.white_ticks),
                pivot_offset: pivot_offset(edit.pivot_ticks),
            })
    }

    /// Replaces the full edit for `name` with `tone` (exposure + develop shape
    /// in one step — copy/paste), updating an existing entry in place.
    ///
    /// A paste never touches the target's crop or rotation: the crop margins
    /// and the user rotation survive [`Self::set_tone`] unchanged (or stay
    /// default on a fresh file). Values are quantized to [`TONE_TICK`].
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
                contrast_lift_ticks: contrast_lift_ticks(tone.contrast),
                black_ticks: black_ticks(tone.black),
                white_ticks: white_ticks(tone.white),
                pivot_ticks: pivot_ticks(tone.pivot_offset),
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
        self.base = Some(base);
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
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A Contrast power on the stored lift-tick grid (mid-pivoted, `+log2`).
    fn cpow(tick: i16) -> f32 {
        contrast_power(tick)
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
                contrast: cpow(-4),
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
        assert_eq!(loaded.tone("IMG_0001.DNG").exposure_ev, 0.40);
        assert_eq!(loaded.tone("IMG_0002.RAW").exposure_ev, -0.75);
        let tone = loaded.tone("IMG_0001.DNG");
        assert_eq!(tone.contrast, cpow(-4));
        assert_eq!(tone.black, black_density(black_ticks(-0.2)));
        assert_eq!(tone.white, white_density(white_ticks(3.0)));
        assert_eq!(tone.pivot_offset, pivot_offset(pivot_ticks(0.15)));
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
    fn manifest_stores_integer_ticks() {
        let dir = temp_dir("ticks");
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = RollManifest::default();
        manifest.set_exposure("a.DNG", 0.4);
        manifest.set_develop(
            "a.DNG",
            ToneEdit {
                exposure_ev: 0.4,
                contrast: (8.0_f32).powf(1.0 / 3.0),
                black: 0.25,
                white: 2.0,
                pivot_offset: -0.1,
            },
        );
        save_roll_manifest(&dir, &manifest).unwrap();
        let text = std::fs::read_to_string(manifest_path(&dir)).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();

        assert!(text.contains("exposure_ticks = 8"), "{text}");
        assert!(text.contains("contrast_lift_ticks ="), "{text}");
        assert!(text.contains("black_ticks = 5"), "{text}");
        assert!(text.contains("white_ticks = 40"), "{text}");
        assert!(text.contains("pivot_ticks = -2"), "{text}");
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
    fn unknown_fields_and_legacy_tone_keys_are_tolerated() {
        let dir = temp_dir("unknown");
        std::fs::create_dir_all(&dir).unwrap();
        let text = r#"
version = 1
fortune = "ignored"
base = 0.66
calibration_frame = "a.DNG"
[edits."a.DNG"]
exposure_ticks = 30
curve_contrast_lift_ticks = 8
curve_highlights_ticks = 5
curve_shadows_ticks = -3
"#;
        std::fs::write(manifest_path(&dir), text).unwrap();
        let loaded = load_roll_manifest(&dir);
        std::fs::remove_dir_all(&dir).unwrap();
        // Unknown key ignored; legacy curve_* keys are ignored and load identity.
        assert!((loaded.tone("a.DNG").exposure_ev - 1.5).abs() < 1e-6);
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
    fn reconcile_drops_stale_entries() {
        let mut manifest = RollManifest::default();
        manifest.set_exposure("keep.DNG", 0.4);
        manifest.set_exposure("stale.DNG", 0.2);
        reconcile(&mut manifest, &["keep.DNG".to_owned()]);
        assert_eq!(manifest.tone("keep.DNG").exposure_ev, 0.4);
        assert_eq!(manifest.tone("stale.DNG"), ToneEdit::identity());
    }

    #[test]
    fn tick_conversions_round_trip_on_the_grid() {
        for ev in [-3.0_f32, 0.0, 0.7, 2.0, 4.0] {
            assert!((exposure_ev(exposure_ticks(ev)) - ev).abs() < 1e-5);
        }
        for tick in [-60_i16, -10, 0, 10, 60] {
            assert_eq!(contrast_lift_ticks(contrast_power(tick)), tick);
        }
        for tick in [-20_i16, -5, 0, 5, 20] {
            assert_eq!(black_ticks(black_density(tick)), tick);
        }
        for tick in [10_i16, 48, 100] {
            assert_eq!(white_ticks(white_density(tick)), tick);
        }
        for tick in [-10_i16, 0, 10] {
            assert_eq!(pivot_ticks(pivot_offset(tick)), tick);
        }
    }

    #[test]
    fn off_grid_values_snap_and_clamp() {
        assert_eq!(exposure_ticks(100.0), EXPOSURE_TICK_MAX);
        assert_eq!(exposure_ticks(-100.0), EXPOSURE_TICK_MIN);
        assert_eq!(black_ticks(100.0), BLACK_TICK_MAX);
        assert_eq!(white_ticks(0.0), WHITE_TICK_MIN);
        assert_eq!(pivot_ticks(100.0), PIVOT_TICK_MAX);
    }
}
