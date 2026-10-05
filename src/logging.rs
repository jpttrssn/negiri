// SPDX-License-Identifier: GPL-3.0-or-later

//! The app's logging sink: the `log` facade with a minimal stderr writer, so
//! diagnostics go through the standard macros without pulling in a full
//! subscriber tree.
//!
//! By default only **this crate's own** records are emitted (all of the app's
//! diagnostics are `log::error!`): the dependency tree's chatter — libcosmic,
//! iced, wgpu, i18n-embed, zbus — stays quiet unless `RUST_LOG` opts into it.
//! `RUST_LOG=debug` (a bare level) raises the threshold for every crate;
//! `RUST_LOG=negiri=trace` raises only ours. The
//! `EXPOSURE_TRACE_DETAIL`/`EXPOSURE_TRACE_REBAKE` env vars force our crate to
//! `trace`, preserving the codebase's existing detail/rebake tracing knobs.
//!
//! Output keeps the established "prefix-less" style: `failed to decode
//! /path: reason` — the message is written exactly as passed to the macro.

use log::{LevelFilter, Log, Metadata, Record};
use std::sync::LazyLock;

/// Records whose target starts with this are the app's own.
const OWN_CRATE: &str = env!("CARGO_PKG_NAME");

/// Per-scope thresholds: `(own crate, every other crate)`.
static SCOPES: LazyLock<(LevelFilter, LevelFilter)> = LazyLock::new(scopes_from_env);

/// A `log` sink that writes the records its scoped thresholds admit to stderr.
struct StderrLogger;

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        let (own, dep) = &*SCOPES;
        let threshold = if metadata.target().starts_with(OWN_CRATE) {
            own
        } else {
            dep
        };
        metadata.level().to_level_filter() <= *threshold
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;

/// Installs the stderr logger once. The global max level is left at `trace` so
/// every record reaches the sink, which filters by target + scope level; the
/// thresholds come from `RUST_LOG` and the `EXPOSURE_TRACE_*` knobs. No
/// dependency calls `log::set_max_level` (verified against the lock tree), so
/// nothing overrides this. Idempotent; safe to call from `main`.
pub(crate) fn init() {
    log::set_logger(&LOGGER).ok();
    log::set_max_level(LevelFilter::Trace);
    // Force the env-derived thresholds to be read at startup, before any work.
    let _ = &*SCOPES;
}

/// The per-scope thresholds from the environment.
///
/// Defaults: our crate `error` (the app's diagnostics are all `error!`), every
/// dependency `off`. The `EXPOSURE_TRACE_*` knobs force our crate to `trace`.
/// `RUST_LOG` overrides: a bare level (`debug`) applies to every scope; a
/// `negiri=<level>` form applies only to our crate. Other targets are
/// ignored.
fn scopes_from_env() -> (LevelFilter, LevelFilter) {
    let mut own = LevelFilter::Error;
    let mut dep = LevelFilter::Off;

    if std::env::var("EXPOSURE_TRACE_DETAIL").is_ok()
        || std::env::var("EXPOSURE_TRACE_REBAKE").is_ok()
    {
        own = LevelFilter::Trace;
    }

    if let Ok(spec) = std::env::var("RUST_LOG") {
        match spec.split_once('=') {
            Some((target, level)) if target == OWN_CRATE || target.is_empty() => {
                own = parse_level(level).unwrap_or(own);
            }
            Some(_) => {}
            None => {
                let level = parse_level(&spec).unwrap_or(own);
                own = level;
                dep = level;
            }
        }
    }

    (own, dep)
}

/// Parses a level name (`trace`/`debug`/`info`/`warn`/`error`), case-insensitive.
fn parse_level(name: &str) -> Option<LevelFilter> {
    match name.to_ascii_lowercase().as_str() {
        "trace" => Some(LevelFilter::Trace),
        "debug" => Some(LevelFilter::Debug),
        "info" => Some(LevelFilter::Info),
        "warn" => Some(LevelFilter::Warn),
        "error" => Some(LevelFilter::Error),
        _ => None,
    }
}