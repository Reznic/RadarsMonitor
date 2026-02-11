# Hebrew Localization Design (`en`/`he`) for RadarsMonitor

## Summary
This design introduces runtime bilingual localization with Hebrew-first behavior and RTL support for user-facing frontend UI.

Decisions validated:
- Runtime language switching between English and Hebrew.
- Default language is Hebrew (`he`) for all users.
- User choice is persisted in `localStorage`.
- Translation scope for v1 includes UI chrome and domain strings shown to users.
- Language switch UI is placed in the Debug Menu.

## Goals
- Localize the existing frontend without changing backend APIs.
- Ensure first-load experience is Hebrew + RTL.
- Allow switching to English at runtime without reload.
- Centralize user-visible strings to prevent future hardcoded text.

## Non-Goals
- Backend localization.
- Translating internal-only console logs in v1 (unless surfaced to users).
- Changing radar/canvas coordinate semantics.

## Approach Options Considered
1. In-house lightweight i18n module (selected).
2. External library (`i18next`) with heavier setup/runtime.
3. Compile-time split pages (`index.he.html` / `index.en.html`) with lower flexibility.

Selected approach: lightweight in-house i18n, because it matches the project size, keeps dependencies minimal, and supports runtime switching cleanly.

## Architecture
Create `src/fe/src/i18n.ts`:
- `type Language = "en" | "he"`
- `const DEFAULT_LANGUAGE: Language = "he"`
- `const STORAGE_KEY = "ui-language"`
- Dictionary object by semantic translation keys.
- API surface:
  - `getLanguage(): Language`
  - `setLanguage(lang: Language): void`
  - `t(key: string, params?: Record<string, string | number>): string`
  - `subscribeLanguageChange(listener: () => void): () => void`
  - `applyI18nToDocument(): void` (for static DOM `data-i18n` nodes)

Runtime behavior:
1. Start with default `he`.
2. Read persisted language from `localStorage` and override if valid.
3. On language set:
   - update in-memory language,
   - persist to `localStorage`,
   - set `<html lang>` and `<html dir>` (`rtl` for `he`, `ltr` for `en`),
   - notify subscribers for dynamic re-render.

## Translation Coverage (v1)
Convert these user-facing strings to translation keys:
- `src/fe/index.html`
  - Title, tab labels, debug title, fullscreen title/aria label, side labels, alert title, alert button, camera list toggle labels/titles.
- `src/fe/src/main.ts`
  - Dynamic fullscreen button labels (`Toggle/Exit Fullscreen`).
- `src/fe/src/view/camera.ts`
  - Camera panel title, show/hide button text and aria labels.
- `src/fe/src/debugMenu.ts`
  - Section titles (`Tooltip Settings`, `System Settings`) and visible UI labels.
- `src/fe/src/network.ts`
  - HUD status texts (`CHECKING`, `ALL RADARS OK`, `NO RADARS`, `MALFUNCTION`, inactive radar count text).
- `src/fe/src/view/alert.ts`
  - Radar badge prefix (e.g. localized `RADAR {id}`), alert dismiss UI text where applicable.
- Domain strings:
  - Camera display names currently defined as `Camera 1...` in config should be localized in UI rendering.

## UI Placement
Language control is added to Debug Menu:
- Control label localized (`Language` / `שפה`).
- Options:
  - `עברית` (`he`)
  - `English` (`en`)
- Selecting value immediately updates UI text and layout direction.

## RTL/LTR Strategy
Use directional CSS overrides, not duplicated markup:
- Add `[dir="rtl"]` rule blocks for affected layout areas.
- Mirror ordering/placement in tab bar and side panel controls where needed.
- Flip directional icons only when they convey direction (e.g. collapse arrows).
- Keep fixed technical tokens (IDs/serials) readable with targeted `dir="ltr"` wrappers if required.
- Do not alter radar canvas math; only textual overlays and surrounding DOM layout mirror.

## Error Handling & Fallbacks
In `i18n.ts`:
- Invalid language input => fallback to `he`.
- Missing key in active language => fallback to `en`, then key string.
- Log missing translations once per key per session to avoid spam.
- Wrap `localStorage` access in `try/catch` for restricted environments.

## Testing Plan
Automated tests:
- Unit tests for `i18n.ts`:
  - default language is `he`,
  - valid persisted language overrides default,
  - invalid persisted language is ignored,
  - `setLanguage()` updates document `lang` and `dir`,
  - translation fallback chain works.
- Integration tests:
  - camera list `Show/Hide` localizes by language,
  - network HUD statuses localize correctly,
  - fullscreen title/aria labels localize across state changes.

Manual verification:
1. First launch: Hebrew + RTL by default.
2. Switch to English in Debug Menu; reload and confirm persistence.
3. Switch back to Hebrew; verify mirrored layout and readable controls.
4. Verify radar/camera functionality unchanged (rendering, polling, controls).

## Implementation Phases
1. Add i18n infrastructure and translation dictionaries.
2. Replace in-scope hardcoded strings with translation keys/usages.
3. Add RTL CSS overrides and icon direction fixes.
4. Verify with:
   - `bun test`
   - `tsc --noEmit`
   - `biome check .`
   - manual smoke test in both languages.

## Risks & Mitigations
- Risk: missed hardcoded strings.
  - Mitigation: search-based pass (`rg`) for user-visible literals before completion.
- Risk: RTL regressions in complex layouts.
  - Mitigation: explicit `[dir="rtl"]` review and manual visual checks for radar/camera tabs.
- Risk: inconsistent terminology between views.
  - Mitigation: centralized key naming and reuse of canonical terms.
