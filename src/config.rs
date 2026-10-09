// SPDX-License-Identifier: GPL-3.0-or-later

use cosmic::cosmic_config::{self, CosmicConfigEntry, cosmic_config_derive::CosmicConfigEntry};

/// User configuration: the film-roll directories shown in the library.
#[derive(Debug, Default, Clone, CosmicConfigEntry, Eq, PartialEq)]
#[version = 1]
pub struct Config {
    /// The roll directories the library lists, as paths **relative to the
    /// user's Pictures directory** (`dirs::picture_dir()`, `/`-separated).
    /// Relative storage keeps every roll inside Pictures by construction (the
    /// app's sandbox grant) and survives a relocatable XDG Pictures dir; the
    /// absolute path is resolved at load and validated on add. Pre-release, so
    /// the older absolute-path form is not migrated — invalid entries are
    /// dropped on load.
    pub rolls: Vec<String>,
}
