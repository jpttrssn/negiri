<p align="center">
  <img src="resources/icons/hicolor/scalable/apps/io.github.jpttrssn.negiri.svg" alt="Negiri" width="180">
</p>

# Negiri
A film roll library and non-destructive RAW editor focused on digital camera scans of black and white film rolls developed for the COSMIC desktop.

- Manage your film rolls
- Non-destructive per-frame edits, stored in a hand-editable `.film-roll.toml` sidecar
- Film-negative inversion in the optical-density domain: exposure, contrast, black, white, midtone pivot
- One designated calibration frame per roll sets its clear-film base
- Histogram with an output/input axis toggle, plus a classic `T(p)` curve overlay
- Crop mode tuned for removing edges, plus quarter-turn display rotation
- Post-gain 2×2 CFA averaging for lower read noise on monochrome negatives
- Keyboard-centric: VIM navigation, a `?` shortcut overlay, Shift for fine steps
- Set and export original roll date with chronological frame order
- Search by roll metadata
- Export to JPEG or 16-bit PNG with EXIF capture-time stamping

## Installation

A [justfile](./justfile) is included by default for the [casey/just][just] command runner.

- `just` builds the application with the default `just build-release` recipe
- `just run` builds and runs the application
- `just install` installs the project into the system
- `just vendor` creates a vendored tarball
- `just build-vendored` compiles with vendored dependencies from that tarball
- `just check` runs clippy on the project to check for linter warnings
- `just check-json` can be used by IDEs that support LSP

## Translators

[Fluent][fluent] is used for localization of the software. Fluent's translation files are found in the [i18n directory](./i18n). New translations may copy the [English (en) localization](./i18n/en) of the project, rename `en` to the desired [ISO 639-1 language code][iso-codes], and then translations can be provided for each [message identifier][fluent-guide]. If no translation is necessary, the message may be omitted.

## Packaging

If packaging for a Linux distribution, vendor dependencies locally with the `vendor` rule, and build with the vendored sources using the `build-vendored` rule. When installing files, use the `rootdir` and `prefix` variables to change installation paths.

```sh
just vendor
just build-vendored
just rootdir=debian/negiri prefix=/usr install
```

It is recommended to build a source tarball with the vendored dependencies, which can typically be done by running `just vendor` on the host system before it enters the build environment.

## Developers

Developers should install [rustup][rustup] and configure their editor to use [rust-analyzer][rust-analyzer]. To improve compilation times, disable LTO in the release profile, install the [mold][mold] linker, and configure [sccache][sccache] for use with Rust. The [mold][mold] linker will only improve link times if LTO is disabled.

[fluent]: https://projectfluent.org/
[fluent-guide]: https://projectfluent.org/fluent/guide/hello.html
[iso-codes]: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
[just]: https://github.com/casey/just
[rustup]: https://rustup.rs/
[rust-analyzer]: https://rust-analyzer.github.io/
[mold]: https://github.com/rui314/mold
[sccache]: https://github.com/mozilla/sccache
