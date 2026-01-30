# Fullscreen Button Design

**Date:** 2026-01-30
**Status:** Approved

## Overview

Add a fullscreen toggle button to the RadarsMonitor web interface, allowing users to enter/exit fullscreen mode on demand.

## Requirements

- Simple UI button for fullscreen toggle
- No PWA/installation complexity
- Manual user control (not automatic)

## Design

### UI Placement

**Location:** Top-right area, near the debug menu

**Rationale:**
- Groups controls together
- Doesn't interfere with radar visualization
- Maintains clean UI layout

### Components

**1. HTML Button (`src/fe/index.html`)**
```html
<button class="fullscreen-btn" id="fullscreenBtn" title="Toggle Fullscreen">
  <span class="fullscreen-icon">⛶</span>
</button>
```

**2. CSS Styling (`src/fe/style.css`)**
- Match existing dark theme with green accents
- Icon updates based on fullscreen state
- Fixed position in top-right corner

**3. JavaScript Handler (`src/fe/scripts/main.ts`)**
- Click handler toggles fullscreen state
- Enter: `document.documentElement.requestFullscreen()`
- Exit: `document.exitFullscreen()`
- Listen to `fullscreenchange` event to update icon
- Handle browser compatibility (webkit prefixes if needed)

### State Management

- Track fullscreen state via `document.fullscreenElement`
- Update button icon:
  - Not fullscreen: `⛶` (expand icon)
  - In fullscreen: `✕` (exit icon)

### Error Handling

- Graceful degradation if Fullscreen API not supported
- Hide button if browser doesn't support fullscreen
- Silent failure on user denial (browser shows native message)

## Implementation Files

1. `src/fe/index.html` - Add button element
2. `src/fe/style.css` - Style fullscreen button
3. `src/fe/scripts/main.ts` - Add click handler and state management

## Browser Support

Uses standard Fullscreen API (supported in all modern browsers):
- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (with webkit prefix fallback)
