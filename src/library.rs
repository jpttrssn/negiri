// SPDX-License-Identifier: GPL-3.0-or-later

//! Library (roll-list) logic: ISO date helpers, the date-aware search
//! (`MonthAliases`/`parse_search_query`), the date-sort order, the unified
//! grid cells + arrow-key navigation, and the roll-metadata manifest setters.

use std::cmp::Ordering;
use std::path::Path;

use crate::app::{LibrarySelection, MoveDir, Roll};
use crate::edit_manifest;

/// The roll-info drawer's editable values, written together in one manifest
/// update. `name`/`start_date`/`end_date` are the *committed* forms (an empty
/// name is `None`, so the label falls back to the directory leaf; an empty date
/// is `None`), and `meta` carries the five free-form film-metadata fields.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct RollDrawer {
    pub name: Option<String>,
    pub start_date: Option<String>,
    pub end_date: Option<String>,
    pub meta: edit_manifest::RollMeta,
}

/// Persists the roll-info drawer's values to the roll's edit manifest, the
/// on-disk source of truth across restarts. All fields are written in one
/// load+save so a debounced commit is a single atomic write; a cleared field is
/// recorded as absent (there is no implicit-default shortcut here).
pub(crate) fn record_roll_drawer(dir: &Path, drawer: RollDrawer) {
    let mut manifest = edit_manifest::load_roll_manifest(dir);
    manifest.set_name(drawer.name);
    manifest.set_dates(drawer.start_date, drawer.end_date);
    manifest.set_meta(drawer.meta);
    if let Err(err) = edit_manifest::save_roll_manifest(dir, &manifest) {
        log::error!(
            "failed to write roll manifest {}: {err}",
            edit_manifest::manifest_path(dir).display()
        );
    }
}

/// Parses a zero-padded ISO date (`YYYY-MM-DD`) with a real calendar month and
/// day (leap-year aware) into `(year, month, day)`. `None` for anything else —
/// including a non-zero-padded shape like `2024-5-9`.
pub(crate) fn parse_iso_date(s: &str) -> Option<(u32, u32, u32)> {
    let (year, rest) = s.split_once('-')?;
    let (month, day) = rest.split_once('-')?;
    let (Ok(year), Ok(month), Ok(day)) = (
        year.parse::<u32>(),
        month.parse::<u32>(),
        day.parse::<u32>(),
    ) else {
        return None;
    };
    // Require the zero-padded `YYYY-MM-DD` shape, not `2024-5-9`.
    if s.len() != 10 || month > 12 || month == 0 || day == 0 {
        return None;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => return None,
    };
    (day <= days).then_some((year, month, day))
}

/// Whether `s` is a valid zero-padded ISO date (`YYYY-MM-DD`) with a real
/// calendar month and day (leap-year aware). The pure validation gate for the
/// roll-info drawer's date fields; an empty string is handled as "clear" by
/// the caller and never reaches this check.
pub(crate) fn valid_iso_date(s: &str) -> bool {
    parse_iso_date(s).is_some()
}

/// Resolves a date field's draft text against the committed value for a
/// debounced commit: an empty draft clears the date (`None`), a valid ISO
/// `YYYY-MM-DD` draft becomes that date, and anything else (still-being-typed or
/// malformed) keeps the committed value so a partial edit is never persisted.
#[must_use]
pub(crate) fn resolve_date_draft(draft: &str, committed: Option<&str>) -> Option<String> {
    let trimmed = draft.trim();
    if trimmed.is_empty() {
        None
    } else if valid_iso_date(trimmed) {
        Some(trimmed.to_owned())
    } else {
        committed.map(str::to_owned)
    }
}

/// Whether a roll's committed dates are coherent: either may be absent, but when
/// both are set the start date must not be after the end date. ISO `YYYY-MM-DD`
/// strings compare lexicographically == chronologically (zero-padded), so this
/// is a plain ordering check.
#[must_use]
pub(crate) fn roll_dates_valid(start: Option<&str>, end: Option<&str>) -> bool {
    match (start, end) {
        (Some(start), Some(end)) => start <= end,
        _ => true,
    }
}

/// Formats a roll's start/end ISO dates for the library card, de-duplicated:
/// `May 3 - 4 2026` (month- and year-unique), `May 30 - Jun 2 2026`
/// (year-unique, each side keeps its month), `Dec 30 2025 - Jan 2 2026`
/// (each side keeps its year), or a single `May 3 2026` when there is no end
/// date (or it equals the start). `months` supplies the localized short month
/// names; `None` is returned for any malformed date so the caller can fall
/// back to raw ISO text.
pub(crate) fn format_roll_card_dates(months: &[&str; 12], start: &str, end: Option<&str>) -> Option<String> {
    let (start_year, start_month, start_day) = parse_iso_date(start)?;
    let (end_year, end_month, end_day) = match end {
        Some(end) => parse_iso_date(end)?,
        None => (start_year, start_month, start_day),
    };
    let start_name = months[(start_month - 1) as usize];
    let end_name = months[(end_month - 1) as usize];
    let single =
        end.is_none() || (start_year, start_month, start_day) == (end_year, end_month, end_day);
    Some(
        match (single, start_year == end_year, start_month == end_month) {
            (true, _, _) => format!("{start_name} {start_day} {start_year}"),
            (false, true, true) => format!("{start_name} {start_day} - {end_day} {start_year}"),
            (false, true, false) => {
                format!("{start_name} {start_day} - {end_name} {end_day} {start_year}")
            }
            (false, false, _) => {
                format!("{start_name} {start_day} {start_year} - {end_name} {end_day} {end_year}")
            }
        },
    )
}

/// Localized month-name aliases for the library search, mapping a typed
/// month token (lowercased) to its month number. In practice this is the
/// localized short names from `month-01`…`month-12` plus the English full and
/// 3-letter names, so `May`, `may`, `September`, and `Sep` all resolve. Built
/// once at startup (the fluent locale is fixed for the session) and reused by
/// the view and the pure search helpers.
pub(crate) struct MonthAliases {
    /// Lowercased alias → month 1–12. Linear scan is fine: ~20 entries.
    names: Vec<(String, u32)>,
}

impl MonthAliases {
    /// `Some(month)` when `token` names a month; `None` otherwise. Matches case-
    /// insensitively (`May` and `may` both resolve to 5).
    fn lookup(&self, token: &str) -> Option<u32> {
        let token = token.to_lowercase();
        self.names
            .iter()
            .find(|(alias, _)| alias == &token)
            .map(|&(_, month)| month)
    }

    /// Builds the aliases: the given localized short month names (from
    /// `month-01`…`month-12`) plus the always-recognized English full and
    /// 3-letter names. `localized_short` must be in month order 1–12.
    pub(crate) fn new(localized_short: &[String; 12]) -> Self {
        let mut names: Vec<(String, u32)> = localized_short
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let month = u32::try_from(i)
                    .expect("the 12-element month array index always fits in u32")
                    + 1;
                (name.to_lowercase(), month)
            })
            .collect();
        names.extend(Self::english_aliases());
        MonthAliases { names }
    }

    /// The English full and 3-letter month names, always recognized regardless
    /// of the active locale (the universal typing fallback for dates).
    fn english_aliases() -> Vec<(String, u32)> {
        let pairs: [(&str, u32); 12] = [
            ("january", 1),
            ("february", 2),
            ("march", 3),
            ("april", 4),
            ("may", 5),
            ("june", 6),
            ("july", 7),
            ("august", 8),
            ("september", 9),
            ("october", 10),
            ("november", 11),
            ("december", 12),
        ];
        let mut names: Vec<(String, u32)> = pairs
            .into_iter()
            .flat_map(|(full, month)| {
                let short = &full[..3];
                [(full.to_owned(), month), (short.to_owned(), month)]
            })
            .collect();
        // "sept" is a common extra September spelling beside the "sep" slice.
        names.push(("sept".to_owned(), 9));
        names
    }
}

/// One whitespace-separated library search token. `Text` matches the roll name
/// or any of its free-form film-metadata fields as a case-insensitive
/// substring; `Date` matches those the same way OR a committed roll date.
enum SearchTerm {
    /// A plain term matched against the roll name and film metadata
    /// (case-insensitive substring).
    Text(String),
    /// A date-shaped term matched against the roll name and film metadata as a
    /// substring OR the committed dates (`None` predicate fields are
    /// wildcards). `2026` → year only; `may` → month only; adjacent tokens stay
    /// independent (`may 2026` is a May AND a 2026 predicate, per the chosen
    /// simple semantics).
    Date {
        /// The lowercased source token, for the name/metadata-substring fallback.
        text: String,
        month: Option<u32>,
        year: Option<u32>,
    },
}

/// Splits `query` into independent search terms: whitespace-separated, each
/// token classified as a 4-digit year, a month-name, or plain text. An empty
/// (or whitespace-only) query yields `[]`, which matches every roll.
fn parse_search_query(query: &str, months: &MonthAliases) -> Vec<SearchTerm> {
    query
        .split_whitespace()
        .map(|token| {
            let token = token.to_lowercase();
            // A 4-digit number is a calendar year; other lengths stay text.
            if token.len() == 4
                && let Some(year) = token.parse::<u32>().ok()
            {
                return SearchTerm::Date {
                    text: token,
                    month: None,
                    year: Some(year),
                };
            }
            if let Some(month) = months.lookup(&token) {
                return SearchTerm::Date {
                    text: token,
                    month: Some(month),
                    year: None,
                };
            }
            SearchTerm::Text(token)
        })
        .collect()
}

/// Whether the lowercased `text` appears in the roll's name or any of its
/// free-form film-metadata fields (film, location, camera, lens, developer
/// notes) as a case-insensitive substring.
fn roll_text_matches(roll: &Roll, text: &str) -> bool {
    roll.name.to_lowercase().contains(text)
        || [
            roll.meta.film.as_deref(),
            roll.meta.location.as_deref(),
            roll.meta.camera.as_deref(),
            roll.meta.lens.as_deref(),
            roll.meta.developer_notes.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| value.to_lowercase().contains(text))
}

/// Whether `roll` satisfies every term in `terms` (AND-across-tokens). `[]`
/// matches every roll. A `Text` term matches the roll name or its film metadata
/// (case-insensitive substring). A `Date` term matches when the source token
/// appears in the name or metadata (so an undated roll named "Trip 2026" answers
/// `2026`, and "Mayfield" answers `may`) OR when either committed date satisfies
/// the month/year predicate (`parse_iso_date`) — a roll spanning two months
/// answers both month queries.
fn roll_matches_terms(roll: &Roll, terms: &[SearchTerm]) -> bool {
    terms.iter().all(|term| match term {
        SearchTerm::Text(text) => roll_text_matches(roll, text),
        SearchTerm::Date { text, month, year } => {
            roll_text_matches(roll, text)
                || [roll.start_date.as_deref(), roll.end_date.as_deref()]
                    .into_iter()
                    .flatten()
                    .any(|iso| {
                        let Some((y, m, _)) = parse_iso_date(iso) else {
                            return false;
                        };
                        (year.is_none_or(|wanted| wanted == y))
                            && (month.is_none_or(|wanted| wanted == m))
                    })
        }
    })
}

/// Rolls matching the toolbar query: every token must match the roll name or
/// its film metadata (case-insensitive substring) or a committed roll date
/// (year / month). Shared by the library view and arrow-key navigation so both
/// move over the same visible set.
pub(crate) fn filtered_rolls<'a>(rolls: &'a [Roll], query: &str, months: &MonthAliases) -> Vec<&'a Roll> {
    let terms = parse_search_query(query, months);
    rolls
        .iter()
        .filter(|roll| roll_matches_terms(roll, &terms))
        .collect()
}

/// Orders two rolls for the library grid: undated rolls lead (by name), then
/// dated rolls newest-start-date first. ISO `YYYY-MM-DD` start dates compare
/// lexicographically == chronologically; name decides ties so the order is
/// deterministic. The sort is derived per-view, so a roll's committed date
/// reorders it on the next render — never mutating `rolls`.
fn roll_date_cmp(a: &Roll, b: &Roll) -> Ordering {
    match (a.start_date.as_deref(), b.start_date.as_deref()) {
        (Some(a_date), Some(b_date)) => b_date.cmp(a_date).then_with(|| a.name.cmp(&b.name)),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => a.name.cmp(&b.name),
    }
}

/// A selectable cell in the library grid: the always-first Add Roll tile, or a
/// search-filtered roll. One type so rendering and arrow-key navigation walk
/// the same set, keeping the Add Roll tile selectable just like a roll card.
pub(crate) enum LibraryCell<'a> {
    /// The Add Roll tile (grid cell 0).
    AddRoll,
    /// A real roll card.
    Roll(&'a Roll),
}

impl LibraryCell<'_> {
    /// The [`LibrarySelection`] this cell carries, for storing a picked cell.
    pub(crate) fn selection(&self) -> LibrarySelection {
        match self {
            LibraryCell::AddRoll => LibrarySelection::AddRoll,
            LibraryCell::Roll(roll) => LibrarySelection::Roll(roll.dir.clone()),
        }
    }
}

/// Every selectable cell in the library grid: the Add Roll tile always first,
/// then the rolls whose name or dates match the query — undated rolls leading,
/// the dated ones newest-first (see [`roll_date_cmp`]). Since the add tile is
/// always present, the returned slice is never empty.
pub(crate) fn library_cells<'a>(
    rolls: &'a [Roll],
    query: &str,
    months: &MonthAliases,
) -> Vec<LibraryCell<'a>> {
    // The Add Roll tile leads the grid only while no search is active: while
    // searching, only the matching rolls are shown (and `cells` may be empty).
    let mut cells: Vec<LibraryCell<'a>> = Vec::with_capacity(rolls.len().saturating_add(1));
    if query.trim().is_empty() {
        cells.push(LibraryCell::AddRoll);
    }
    let mut matching = filtered_rolls(rolls, query, months);
    matching.sort_by(|a, b| roll_date_cmp(a, b));
    cells.extend(matching.into_iter().map(LibraryCell::Roll));
    cells
}

/// The index of `selection` within `cells`, if it names a cell in the set.
/// None means nothing is selected (or the selection left the set).
pub(crate) fn library_cell_index(
    selection: Option<&LibrarySelection>,
    cells: &[LibraryCell<'_>],
) -> Option<usize> {
    let selection = selection?;
    cells.iter().position(|cell| cell.selection() == *selection)
}

/// The index arrow-key navigation moves the selection to: `selected` as an
/// index into the visible cells (None = nothing selected yet), `len` visible
/// cells, `cols` grid columns. Left/Right step within the row and never wrap;
/// Up/Down step a full row, clamped to the first/last item.
pub(crate) fn nav_target(selected: Option<usize>, len: usize, cols: usize, dir: MoveDir) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let cols = cols.max(1);
    Some(match selected {
        None => 0,
        Some(selected) => {
            let selected = selected.min(len - 1);
            match dir {
                MoveDir::Left => selected.saturating_sub(1),
                MoveDir::Right => (selected + 1).min(len - 1),
                MoveDir::Up => selected.saturating_sub(cols),
                MoveDir::Down => (selected + cols).min(len - 1),
            }
        }
    })
}

/// When a detail view is open, Left/Right step one frame through the set. The
/// step is clamped at both ends (no wrap): `None` when there is no current
/// frame anchored (or the list is empty), matching the selection-driven grid
/// nav. Up/Down never page.
pub(crate) fn paginate(current: usize, len: usize, dir: MoveDir) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let current = current.min(len - 1);
    match dir {
        MoveDir::Left => Some(current.saturating_sub(1)),
        MoveDir::Right => Some((current + 1).min(len - 1)),
        MoveDir::Up | MoveDir::Down => None,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::app::Thumb;

    fn roll(dir: &str, name: &str) -> Roll {
        Roll {
            dir: PathBuf::from(dir),
            name: name.to_string(),
            cover: None,
            frame_count: 0,
            start_date: None,
            end_date: None,
            meta: edit_manifest::RollMeta::default(),
            thumb: Thumb::Loading,
        }
    }
    #[test]
    fn valid_iso_date_accepts_real_dates() {
        assert!(valid_iso_date("2024-05-09"));
        assert!(valid_iso_date("2024-02-29")); // leap year
        assert!(valid_iso_date("2000-02-29")); // 400-year leap
        assert!(valid_iso_date("1900-12-31"));
    }

    #[test]
    fn valid_iso_date_rejects_impossible_dates() {
        assert!(!valid_iso_date("2023-02-29")); // not a leap year
        assert!(!valid_iso_date("1900-02-29")); // 100-year non-leap
        assert!(!valid_iso_date("2024-13-01")); // month 13
        assert!(!valid_iso_date("2024-00-01")); // month 0
        assert!(!valid_iso_date("2024-04-31")); // April has 30 days
        assert!(!valid_iso_date("2024-01-00")); // day 0
    }

    /// The English short month table, exercising the localized-format path
    /// with the fallback locale's values.
    fn months_jan_to_dec() -> [&'static str; 12] {
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
    }

    /// A test `MonthAliases` with the English short names plus the full/3-letter
    /// aliases (which the app builds from `fl!` at startup).
    fn test_aliases() -> MonthAliases {
        let localized: [String; 12] = std::array::from_fn(|i| months_jan_to_dec()[i].to_owned());
        MonthAliases::new(&localized)
    }

    #[test]
    fn format_roll_card_dates_single_date_and_equal_range() {
        let months = months_jan_to_dec();
        assert_eq!(
            format_roll_card_dates(&months, "2026-05-03", None),
            Some("May 3 2026".to_owned())
        );
        // A one-day range collapses to the single-date form.
        assert_eq!(
            format_roll_card_dates(&months, "2026-05-03", Some("2026-05-03")),
            Some("May 3 2026".to_owned())
        );
    }

    #[test]
    fn format_roll_card_dates_same_month_share_month_and_year() {
        let months = months_jan_to_dec();
        assert_eq!(
            format_roll_card_dates(&months, "2026-05-03", Some("2026-05-04")),
            Some("May 3 - 4 2026".to_owned())
        );
    }

    #[test]
    fn format_roll_card_dates_same_year_keep_both_months() {
        let months = months_jan_to_dec();
        assert_eq!(
            format_roll_card_dates(&months, "2026-05-30", Some("2026-06-02")),
            Some("May 30 - Jun 2 2026".to_owned())
        );
    }

    #[test]
    fn format_roll_card_dates_cross_year_keep_both_years() {
        let months = months_jan_to_dec();
        assert_eq!(
            format_roll_card_dates(&months, "2025-12-30", Some("2026-01-02")),
            Some("Dec 30 2025 - Jan 2 2026".to_owned())
        );
    }

    #[test]
    fn format_roll_card_dates_rejects_malformed_input() {
        let months = months_jan_to_dec();
        assert_eq!(format_roll_card_dates(&months, "not-a-date", None), None);
        assert_eq!(
            format_roll_card_dates(&months, "2026-05-03", Some("not-a-date")),
            None
        );
        assert_eq!(format_roll_card_dates(&months, "2026-02-30", None), None);
    }

    #[test]
    fn valid_iso_date_rejects_malformed_input() {
        assert!(!valid_iso_date("2024-5-9")); // not zero-padded
        assert!(!valid_iso_date("2024/05/09")); // wrong separator
        assert!(!valid_iso_date("may 9 2024"));
        assert!(!valid_iso_date("2024-05-09-10")); // trailing noise
        assert!(!valid_iso_date(""));
    }

    #[test]
    fn resolve_date_draft_clears_validates_or_keeps_committed() {
        // An empty (or whitespace) draft clears the date.
        assert_eq!(resolve_date_draft("", Some("2024-05-09")), None);
        assert_eq!(resolve_date_draft("   ", Some("2024-05-09")), None);
        // A valid ISO date is trimmed and adopted.
        assert_eq!(
            resolve_date_draft(" 2024-06-01 ", None),
            Some("2024-06-01".to_owned())
        );
        // A partial or malformed draft keeps the committed value, so a
        // mid-typing debounce never persists a bad date.
        assert_eq!(
            resolve_date_draft("2024-06", Some("2024-05-09")),
            Some("2024-05-09".to_owned())
        );
        assert_eq!(resolve_date_draft("nope", None), None);
        assert_eq!(
            resolve_date_draft("2024-5-9", Some("2024-05-09")),
            Some("2024-05-09".to_owned())
        );
    }

    #[test]
    fn roll_dates_require_start_before_end_when_both_set() {
        // Either date alone is always coherent.
        assert!(roll_dates_valid(None, None));
        assert!(roll_dates_valid(Some("2024-05-09"), None));
        assert!(roll_dates_valid(None, Some("2024-05-09")));
        // Equal dates are a valid single-day roll.
        assert!(roll_dates_valid(Some("2024-05-09"), Some("2024-05-09")));
        // A start before its end is the healthy case.
        assert!(roll_dates_valid(Some("2024-05-09"), Some("2024-05-11")));
        // The roll must not end before it starts.
        assert!(!roll_dates_valid(Some("2024-05-11"), Some("2024-05-09")));
        // Cross-year comparisons are plain ISO ordering, not calendar math.
        assert!(roll_dates_valid(Some("2023-12-31"), Some("2024-01-01")));
    }

    #[test]
    fn filtered_rolls_matches_case_insensitive_substring() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "chicago")];

        let matched = filtered_rolls(&rolls, "CHI", &test_aliases());

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "chicago");
    }

    #[test]
    fn filtered_rolls_returns_all_on_empty_query() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "Beta")];

        assert_eq!(filtered_rolls(&rolls, "", &test_aliases()).len(), 2);
        assert_eq!(filtered_rolls(&rolls, "   ", &test_aliases()).len(), 2);
    }

    #[test]
    fn month_aliases_lookup_full_and_abbreviated_names() {
        let aliases = test_aliases();

        assert_eq!(aliases.lookup("may"), Some(5));
        assert_eq!(aliases.lookup("May"), Some(5));
        assert_eq!(aliases.lookup("september"), Some(9));
        assert_eq!(aliases.lookup("sep"), Some(9));
        assert_eq!(aliases.lookup("sept"), Some(9));
        assert_eq!(aliases.lookup("dec"), Some(12));
        assert_eq!(aliases.lookup("nope"), None);
    }

    #[test]
    fn parse_search_query_classifies_year_month_and_text() {
        let aliases = test_aliases();

        let terms = parse_search_query("chi 2026", &aliases);
        assert!(matches!(&terms[0], SearchTerm::Text(t) if t == "chi"));
        assert!(matches!(
            &terms[1],
            SearchTerm::Date { text, month: None, year: Some(2026) } if text == "2026"
        ));

        let terms = parse_search_query("may", &aliases);
        assert!(matches!(
            &terms[0],
            SearchTerm::Date { text, month: Some(5), year: None } if text == "may"
        ));

        let terms = parse_search_query("may 2026", &aliases);
        assert!(matches!(
            &terms[0],
            SearchTerm::Date {
                month: Some(5),
                year: None,
                ..
            }
        ));
        assert!(matches!(
            &terms[1],
            SearchTerm::Date {
                month: None,
                year: Some(2026),
                ..
            }
        ));

        // Shorter numbers stay text (a name can contain "400", "2880", etc.).
        let terms = parse_search_query("trip  400", &aliases);
        assert!(matches!(&terms[0], SearchTerm::Text(t) if t == "trip"));
        assert!(matches!(&terms[1], SearchTerm::Text(t) if t == "400"));

        // Empty query parses to nothing (matches everything).
        assert!(parse_search_query("", &aliases).is_empty());
        assert!(parse_search_query("   ", &aliases).is_empty());
    }

    #[test]
    fn search_matches_year_only() {
        let aliases = test_aliases();
        let mut dated = roll("/a", "Archives");
        dated.start_date = Some("2024-06-01".into());
        let mut other = roll("/b", "Older");
        other.start_date = Some("2023-12-31".into());
        let undated = roll("/c", "No-roll");
        let rolls = [dated, other, undated];

        let matched = filtered_rolls(&rolls, "2024", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Archives");
    }

    #[test]
    fn search_matches_month_only() {
        let aliases = test_aliases();
        let mut may_roll = roll("/a", "Spring");
        may_roll.start_date = Some("2024-05-10".into());
        let mut june_roll = roll("/b", "Summer");
        june_roll.start_date = Some("2024-06-03".into());
        let rolls = [may_roll, june_roll];

        let matched = filtered_rolls(&rolls, "may", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Spring");
    }

    #[test]
    fn search_matches_month_and_year() {
        let aliases = test_aliases();
        let mut may26 = roll("/a", "May2026");
        may26.start_date = Some("2026-05-03".into());
        let mut may25 = roll("/b", "May2025");
        may25.start_date = Some("2025-05-09".into());
        let mut mar26 = roll("/c", "Mar2026");
        mar26.start_date = Some("2026-03-15".into());
        let rolls = [may26, may25, mar26];

        // Only the May 2026 roll satisfies both independent tokens.
        let matched = filtered_rolls(&rolls, "may 2026", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "May2026");
    }

    #[test]
    fn search_matches_name_plus_year() {
        let aliases = test_aliases();
        let mut chicago = roll("/a", "chicago trip");
        chicago.start_date = Some("2026-05-03".into());
        let mut orlando = roll("/b", "orlando trip");
        orlando.start_date = Some("2026-06-01".into());
        let rolls = [chicago, orlando];

        let matched = filtered_rolls(&rolls, "chi 2026", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "chicago trip");
    }

    #[test]
    fn search_matches_end_date() {
        let aliases = test_aliases();
        let mut spanning = roll("/a", "Trip");
        spanning.start_date = Some("2026-05-30".into());
        spanning.end_date = Some("2026-06-02".into());
        let rolls = [spanning.clone()];

        // A range spans both months: it answers both "may" and "june", and the
        // "june 2026" query via the end date alone.
        assert_eq!(filtered_rolls(&rolls, "may", &aliases).len(), 1);
        assert_eq!(filtered_rolls(&rolls, "june", &aliases).len(), 1);
        assert_eq!(filtered_rolls(&rolls, "june 2026", &aliases).len(), 1);
    }

    #[test]
    fn search_matches_film_metadata_fields() {
        let aliases = test_aliases();
        let mut r = roll("/a", "Roll");
        r.meta.film = Some("Kodak Tri-X 400".into());
        r.meta.location = Some("Chicago".into());
        r.meta.camera = Some("Nikon FM2".into());
        r.meta.lens = Some("50mm f/1.4".into());
        r.meta.developer_notes = Some("HC-110 dilution B".into());
        let rolls = [r];

        // Each metadata field is searchable as a case-insensitive substring.
        for query in ["tri-x", "chicago", "nikon", "50mm", "hc-110"] {
            assert_eq!(filtered_rolls(&rolls, query, &aliases).len(), 1, "{query}");
        }
        // Matching is case-insensitive both ways.
        assert_eq!(filtered_rolls(&rolls, "KODAK", &aliases).len(), 1);
        // A term no field contains matches nothing.
        assert!(filtered_rolls(&rolls, "leica", &aliases).is_empty());
    }

    #[test]
    fn search_metadata_terms_are_and_composed_with_name() {
        let aliases = test_aliases();
        let mut r = roll("/a", "Summer");
        r.meta.camera = Some("Nikon FM2".into());
        r.meta.film = Some("Kodak Tri-X".into());
        let rolls = [r];

        // Name AND metadata must both match the same roll.
        assert_eq!(filtered_rolls(&rolls, "summer nikon", &aliases).len(), 1);
        assert!(filtered_rolls(&rolls, "summer leica", &aliases).is_empty());
        // Two metadata terms AND together.
        assert_eq!(filtered_rolls(&rolls, "nikon tri-x", &aliases).len(), 1);
        assert!(filtered_rolls(&rolls, "nikon portra", &aliases).is_empty());
    }

    #[test]
    fn search_date_token_falls_back_to_metadata_substring() {
        let aliases = test_aliases();
        // "may" is a date-shaped token; a roll with no date but a "Mayfair"
        // location still answers it through the metadata substring fallback.
        let mut r = roll("/a", "Roll");
        r.meta.location = Some("Mayfair".into());
        let rolls = [r];

        assert_eq!(filtered_rolls(&rolls, "may", &aliases).len(), 1);
    }

    #[test]
    fn search_tokens_are_and_composed() {
        let aliases = test_aliases();
        let mut dated = roll("/a", "Trip");
        dated.start_date = Some("2026-05-03".into());
        let rolls = [dated.clone()];

        // Both tokens must match the same roll: a date AND a name term.
        assert_eq!(filtered_rolls(&rolls, "trip may 2026", &aliases).len(), 1);
        assert!(filtered_rolls(&rolls, "trip june", &aliases).is_empty());
        // A contradiction (two years) matches nothing.
        assert!(filtered_rolls(&rolls, "2026 2025", &aliases).is_empty());
    }

    #[test]
    fn search_month_like_name_wins_via_name() {
        let aliases = test_aliases();
        // A roll named "Mayfield" dated in June still matches "may" — the name
        // substring counts alongside any date interpretation.
        let mut mayfield = roll("/a", "Mayfield");
        mayfield.start_date = Some("2026-06-03".into());
        let rolls = [mayfield];

        let matched = filtered_rolls(&rolls, "may", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Mayfield");
    }

    #[test]
    fn search_digit_in_name_matches_via_name() {
        let aliases = test_aliases();
        // A roll literally named "Trip 2026" (undated) matches the year token
        // through its name even with no committed date.
        let undated = roll("/a", "Trip 2026");
        let rolls = [undated];

        let matched = filtered_rolls(&rolls, "2026", &aliases);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].name, "Trip 2026");
    }

    #[test]
    fn library_cells_keep_the_add_tile_first() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "Beta")];

        let cells = library_cells(&rolls, "", &test_aliases());

        assert_eq!(cells.len(), 3);
        assert!(matches!(cells[0], LibraryCell::AddRoll));
        // Rolls follow in filtered order.
        assert!(matches!(cells[1], LibraryCell::Roll(r) if r.name == "Alpha"));
        assert!(matches!(cells[2], LibraryCell::Roll(r) if r.name == "Beta"));
        // A whitespace-only query is an inactive search: the add tile stays.
        assert_eq!(library_cells(&rolls, "   ", &test_aliases()).len(), 3);
    }

    #[test]
    fn library_cells_hide_the_add_tile_while_searching() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "Beta")];

        // A non-empty query drops the Add Roll tile: only matches remain.
        let cells = library_cells(&rolls, "beta", &test_aliases());
        assert_eq!(cells.len(), 1);
        assert!(matches!(cells[0], LibraryCell::Roll(r) if r.name == "Beta"));

        // A query matching nothing yields an entirely empty cell set.
        let cells = library_cells(&rolls, "zzz", &test_aliases());
        assert!(cells.is_empty());
        // No rolls at all: still the add tile while idle.
        assert_eq!(library_cells(&[], "", &test_aliases()).len(), 1);
    }

    #[test]
    fn library_cells_put_undated_first_then_newest_dated_rolls() {
        let mut newest = roll("/a", "Started");
        newest.start_date = Some("2024-06-01".into());
        let mut older = roll("/b", "Winter");
        older.start_date = Some("2023-12-31".into());
        let mut same_date_zeta = roll("/c", "Zulu");
        same_date_zeta.start_date = Some("2024-05-09".into());
        let mut same_date_alpha = roll("/d", "Alpha");
        same_date_alpha.start_date = Some("2024-05-09".into());
        let undated = roll("/e", "No-roll");

        let mixed = [
            same_date_alpha.clone(),
            older.clone(),
            same_date_zeta.clone(),
            newest.clone(),
            undated.clone(),
        ];
        let cells = library_cells(&mixed, "", &test_aliases());

        // Add Roll cell 0, then undated roll, then dated newest-first, with
        // same-day rolls tied by name.
        assert!(matches!(cells[0], LibraryCell::AddRoll));
        assert!(matches!(cells[1], LibraryCell::Roll(r) if r.name == "No-roll"));
        assert!(matches!(cells[2], LibraryCell::Roll(r) if r.name == "Started"));
        assert!(matches!(cells[3], LibraryCell::Roll(r) if r.name == "Alpha"));
        assert!(matches!(cells[4], LibraryCell::Roll(r) if r.name == "Zulu"));
        assert!(matches!(cells[5], LibraryCell::Roll(r) if r.name == "Winter"));

        // The sort is derived: committing a date to the undated roll reorders
        // it on the next render without touching the backing slice.
        let mut undated = undated.clone();
        undated.start_date = Some("2025-01-01".into());
        let reordered = [newest.clone(), undated.clone(), same_date_alpha];
        let cells = library_cells(&reordered, "", &test_aliases());
        assert!(matches!(cells[1], LibraryCell::Roll(r) if r.name == "No-roll"));
        assert!(matches!(cells[2], LibraryCell::Roll(r) if r.name == "Started"));
    }

    #[test]
    fn library_cell_index_finds_add_and_roll_slots() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "Beta")];
        let cells = library_cells(&rolls, "", &test_aliases());

        assert_eq!(
            library_cell_index(Some(&LibrarySelection::AddRoll), &cells),
            Some(0)
        );
        assert_eq!(
            library_cell_index(Some(&LibrarySelection::Roll(PathBuf::from("/b"))), &cells),
            Some(2)
        );
        // No selection, or one not in the (filtered) set, yields None.
        assert_eq!(library_cell_index(None, &cells), None);
        assert_eq!(
            library_cell_index(Some(&LibrarySelection::Roll(PathBuf::from("/x"))), &cells),
            None
        );
    }

    #[test]
    fn nav_can_land_on_and_leave_the_add_tile() {
        let rolls = vec![roll("/a", "Alpha"), roll("/b", "Beta")];
        let cells = library_cells(&rolls, "", &test_aliases());

        // From nothing, Down selects cell 0 (the add tile).
        let target = nav_target(None, cells.len(), 2, MoveDir::Down).unwrap();
        assert_eq!(target, 0);
        assert!(matches!(&cells[target], LibraryCell::AddRoll));

        // From the add tile, Right moves to the first roll (cell 1).
        let target = nav_target(Some(0), cells.len(), 2, MoveDir::Right).unwrap();
        assert_eq!(target, 1);
        assert!(matches!(&cells[target], LibraryCell::Roll(r) if r.name == "Alpha"));

        // From cell 1, Left returns to the add tile.
        let target = nav_target(Some(1), cells.len(), 2, MoveDir::Left).unwrap();
        assert_eq!(target, 0);
        assert!(matches!(&cells[target], LibraryCell::AddRoll));
        // Left on the add tile does not wrap.
        assert_eq!(nav_target(Some(0), cells.len(), 2, MoveDir::Left), Some(0));
    }

    #[test]
    fn nav_target_steps_within_the_row() {
        assert_eq!(nav_target(Some(1), 8, 3, MoveDir::Left), Some(0));
        assert_eq!(nav_target(Some(6), 8, 3, MoveDir::Right), Some(7));
        // No wrap at the last item.
        assert_eq!(nav_target(Some(7), 8, 3, MoveDir::Right), Some(7));
    }

    #[test]
    fn nav_target_jumps_full_rows_vertically() {
        assert_eq!(nav_target(Some(4), 8, 3, MoveDir::Up), Some(1));
        assert_eq!(nav_target(Some(4), 8, 3, MoveDir::Down), Some(7));
    }

    #[test]
    fn nav_target_clamps_to_grid_edges() {
        // First row: Up clamps to the top item.
        assert_eq!(nav_target(Some(1), 8, 3, MoveDir::Up), Some(0));
        // Last row: Down clamps to the last item.
        assert_eq!(nav_target(Some(7), 8, 3, MoveDir::Down), Some(7));
    }

    #[test]
    fn nav_target_selects_the_first_item_from_no_selection() {
        assert_eq!(nav_target(None, 4, 3, MoveDir::Down), Some(0));
    }

    #[test]
    fn nav_target_handles_a_single_column_grid() {
        assert_eq!(nav_target(Some(1), 4, 1, MoveDir::Up), Some(0));
        assert_eq!(nav_target(Some(1), 4, 1, MoveDir::Down), Some(2));
    }

    #[test]
    fn nav_target_empty_list_has_no_target() {
        assert_eq!(nav_target(Some(0), 0, 3, MoveDir::Left), None);
    }

    #[test]
    fn paginate_steps_left_and_right_within_visible_set() {
        assert_eq!(paginate(1, 5, MoveDir::Left), Some(0));
        assert_eq!(paginate(1, 5, MoveDir::Right), Some(2));
    }

    #[test]
    fn paginate_clamps_at_both_ends() {
        assert_eq!(paginate(0, 5, MoveDir::Left), Some(0));
        assert_eq!(paginate(4, 5, MoveDir::Right), Some(4));
    }

    #[test]
    fn paginate_never_moves_vertically() {
        assert_eq!(paginate(2, 5, MoveDir::Up), None);
        assert_eq!(paginate(2, 5, MoveDir::Down), None);
    }

    #[test]
    fn paginate_empty_list_has_no_target() {
        assert_eq!(paginate(0, 0, MoveDir::Right), None);
    }

}
