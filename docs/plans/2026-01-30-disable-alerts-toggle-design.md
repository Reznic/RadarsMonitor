# Disable Alerts Toggle - Design Document

**Date:** 2026-01-30
**Status:** Approved

## Overview

Add a toggle in the Debug Settings to completely disable the radar trigger (alert view) logic for system testing purposes.

## Requirements

- Add checkbox labeled "Disable Alerts" to existing Debug Menu
- Persist state in localStorage (like other debug settings)
- When disabled, prevent new alert overlays from appearing
- Leave currently open alerts visible (less code, debug menu inaccessible during alerts anyway)
- Track detection and radar display continue normally - only suppress alert popup

## Architecture

### Components Modified

1. **debugConfig.ts** - Add `disableAlerts` boolean field with getter/setter
2. **debugMenu.ts** - No changes (auto-renders new field)
3. **network.ts** - Add conditional check before `showTrackAlert()`

### Design Decisions

- Reuse existing debug menu infrastructure (checkbox rendering, localStorage, state management)
- Store alongside tooltip field configuration for consistency
- Default to `false` (alerts enabled by default)
- Zero new UI code - checkbox appears automatically via existing field renderer

## Implementation Details

### 1. Configuration State (debugConfig.ts)

Extend interface:
```typescript
export interface DebugConfig {
  tooltipFields: TooltipFieldConfig;
  disableAlerts: boolean;
}
```

Add accessor functions:
```typescript
export function isAlertsDisabled(): boolean {
  return debugConfig.disableAlerts;
}

export function setAlertsDisabled(disabled: boolean): void {
  debugConfig.disableAlerts = disabled;
}
```

Add to `getAvailableFields()` return array to register checkbox.

### 2. Alert Blocking Logic (network.ts)

Import getter:
```typescript
import { isAlertsDisabled } from "./debugConfig.ts";
```

Wrap `showTrackAlert()` call in `pollRadarData()` (line ~437):
```typescript
if (newTrackRadarIds.length > 0) {
  if (!isAlertsDisabled()) {
    showTrackAlert(newTrackRadarIds);
  }
}
```

## Behavior

**When Enabled (default):**
- System behaves as normal
- New tracks trigger alert overlay with camera feeds

**When Disabled:**
- Track detection continues (dots appear on radar)
- Track history and trails render normally
- Alert overlay popup is suppressed
- Currently open alerts remain visible until manually dismissed

## Testing

1. Enable toggle → verify no new alerts appear when tracks detected
2. Disable toggle → verify alerts resume appearing
3. Verify setting persists across page reloads
4. Verify radar display continues working regardless of toggle state
