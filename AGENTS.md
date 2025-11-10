# Repository Guidelines

## Project Structure & Module Organization

- `src/fe/` holds the browser UI (HTML/CSS under `index.html` and `style.css`, TypeScript modules under `scripts/`). Entry point: `scripts/main.ts`.
- `src/radar_tracks_server.py` exposes the Python backend that streams radar tracks.
- `src/run.ts` orchestrates full-stack startup via Bun, while `src/fe/serve.ts` runs the frontend-only dev server.
- Type definitions live in `types.ts`; shared config for the frontend is in `scripts/config.ts`, debugging helpers in `debugConfig.ts`.
- Node/Bun tooling config: `package.json`, `bun.lock`, `biome.json`, `tsconfig.json`. Python deps: `requirements.txt`.

## Build, Test, and Development Commands

- `bun src/fe/serve.ts` — launch the Vite-style frontend dev server.
- `bun src/run.ts` / `bun src/run.ts --prod` — start the integrated stack (frontend proxy + backend); `--prod` serves optimized assets.
- `bun src/radar_tracks_server.py` (through `npm run be`) — run only the Python backend API.
- `bun test` — execute frontend/bun unit tests.
- `tsc --noEmit` — TypeScript type checking.
- `biome check .` / `biome check --write .` — lint/format enforcement (read-only vs. autofix).

## Coding Style & Naming Conventions

- TypeScript/JavaScript: 2-space indentation, ES modules (`import … from "./file.ts"`). Favor descriptive camelCase for variables/functions, PascalCase for types/interfaces.
- CSS follows BEM-inspired class names (`.sensor-grid`, `.debug-menu`). Keep palette-consistent rgba values.
- Python backend uses standard 4-space indentation; follow PEP 8 naming.
- Run Biome before committing; it enforces lint + formatting across TS/CSS/JSON.

## Testing Guidelines

- Frontend unit/integration tests live wherever Bun’s test runner expects (co-located `*.test.ts`). Name suites after the module under test, e.g., `radar.test.ts`.
- Use `bun test --watch` locally for rapid feedback. No formal coverage gate, but cover sensor state updates, network polling, and canvas math helpers when modifying them.
- Manual verification: `bun src/run.ts` should show accurate sensor HUD + radar plotting after protocol changes.

## Commit & Pull Request Guidelines

- Commits follow practical, present-tense summaries (e.g., “Add HUD accordion toggle”). Include scoped body lines for context when touching multiple areas.
- Keep diffs focused; separate UI tweaks from backend/math changes where possible.
- PRs should describe motivation, summarize key changes, list test commands/results, and attach screenshots/gifs for UI updates (HUD, debug menu, radar canvas). Link related issues or TODOs.

## Security & Configuration Tips

- Never commit `.env` or credentials; backend uses plain HTTP endpoints configured via `API_BASE` in `scripts/config.ts`.
- When adding network calls, honor existing polling intervals/constants to avoid overwhelming the Python server.
