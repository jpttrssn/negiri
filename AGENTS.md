# AGENTS.md

COSMIC desktop app (libcosmic, Rust edition 2024) for film roll library management and film negative RAW image editing. Single-binary crate; early stage.

- `NOTES.md` tracks current working state and next-step targets for in-progress feature work. Read it before continuing work; update it when the state changes.

## Commands

- Lint/verify: `just check` — runs `cargo clippy --all-features --locked -- -W clippy::pedantic`. Use this to validate changes.
- Tests: `cargo test --locked` — unit tests for pure pipeline helpers in `src/app.rs`.
- Run the app: `just run` — builds and runs in **release** profile with `RUST_BACKTRACE=full` (not debug).
- Build: `just` (= `build-release`) or `just build-debug`.
- Edition 2024 needs a recent stable toolchain (rustup).
- Just recipes use `--locked`; the vendored build recipe (`build-vendored`) uses `--frozen --offline` since it must not touch the network.

## Dependencies

- `libcosmic` is a **git dependency on pop-os master**, not crates.io. `Cargo.lock` pins the commit; always build/check with `--locked` (the just recipes already do).

## Codegen and i18n

- `build.rs` runs `xdgen` at compile time: generates `target/xdgen/app.desktop` and `app.metainfo.xml` from the templates in `resources/` combined with fluent strings from `i18n/`. Edit templates in `resources/`, never generated output (`target/` is gitignored).
- User-facing strings use the `fl!` macro with message IDs from `i18n/en/curvectrl.ftl`. Add new messages there; missing translations fall back to English.

## Conventions

- Every `.rs` file starts with `// SPDX-License-Identifier: GPL-3.0-or-later` (repo license is GPL-3.0-or-later).
- Distro packaging/vendoring flow is documented in README (`just vendor` → `just build-vendored`; `install` honors `rootdir`/`prefix`).

### Docs drift

`NOTES.md` is a reference for agents. Treat its current-state and decisions prose as assertions to be re-verified, never as context you may assume — it accumulates drift through commits that each look correct in isolation.

- After any rename, removal, or literal-value change in `src/`, read the whole of `NOTES.md` end-to-end and grep the source for every identifier you touched. Verifying only the lines you remember is not enough; partial reads miss most of the drift. Check the identifier *and* its value — a correct name with a stale number fails just as hard as a missing one.
- Don't link a doc that does not exist. Before citing a `docs/*.md` path, confirm the file is there.
- `NOTES.md` has no change log by design — the record lives in `git log`. Do not add one: prose history drifts silently and reads like current state.
- `## Decisions` entries are ADR-style and read as authoritative. When a decision's subject is later removed, delete or re-point its ADR; never leave it "Accepted" while pointing at a deleted symbol.
