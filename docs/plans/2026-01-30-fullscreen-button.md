# Fullscreen Button Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fullscreen toggle button to allow users to enter/exit fullscreen mode on demand.

**Architecture:** Simple UI button with click handler using the Fullscreen API. Button is positioned in the top-right corner near the debug menu. Icon updates based on fullscreen state via `fullscreenchange` event listener.

**Tech Stack:** TypeScript, Fullscreen API, CSS3

---

## Task 1: Add Fullscreen Button HTML

**Files:**
- Modify: `src/fe/index.html:30-45` (after debug menu, before radar container)

**Step 1: Add fullscreen button element**

Insert the button element after the debug menu closing tag (after line 45):

```html
    </div>

    <!-- Fullscreen Button -->
    <button class="fullscreen-btn" id="fullscreenBtn" title="Toggle Fullscreen">
      <span class="fullscreen-icon">⛶</span>
    </button>

    <!-- Radar Display -->
```

**Step 2: Verify HTML structure**

Run: `cat src/fe/index.html | grep -A 2 "fullscreen-btn"`

Expected: Button element with correct class and id

**Step 3: Commit**

```bash
git add src/fe/index.html
git commit -m "feat: add fullscreen button HTML element

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Style Fullscreen Button

**Files:**
- Modify: `src/fe/style.css:623-690` (after debug menu styles, before radar container)

**Step 1: Add fullscreen button styles**

Insert after the debug menu styles (around line 690, before "/* Radar Container */"):

```css
/* Fullscreen Button */
.fullscreen-btn {
  position: absolute;
  top: 20px;
  right: 280px;
  width: 40px;
  height: 40px;
  background: rgba(0, 0, 0, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  transition: all 0.2s ease;
}

.fullscreen-btn:hover {
  background: rgba(0, 0, 0, 1);
  border-color: rgba(255, 255, 255, 0.5);
  transform: scale(1.05);
}

.fullscreen-btn:active {
  transform: scale(0.95);
}

.fullscreen-icon {
  font-size: 20px;
  color: rgba(255, 255, 255, 0.9);
  user-select: none;
}

/* Hide button if fullscreen API not supported */
.fullscreen-btn.unsupported {
  display: none;
}

/* Responsive adjustments */
@media (max-width: 1200px) {
  .fullscreen-btn {
    right: 230px;
    width: 36px;
    height: 36px;
  }

  .fullscreen-icon {
    font-size: 18px;
  }
}

@media (max-width: 700px) {
  .fullscreen-btn {
    top: 10px;
    right: 200px;
    width: 32px;
    height: 32px;
  }

  .fullscreen-icon {
    font-size: 16px;
  }
}
```

**Step 2: Verify CSS**

Run: `grep -A 5 "fullscreen-btn" src/fe/style.css`

Expected: Fullscreen button styles present

**Step 3: Commit**

```bash
git add src/fe/style.css
git commit -m "feat: add fullscreen button styles

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement Fullscreen Handler

**Files:**
- Modify: `src/fe/src/main.ts:1-118`

**Step 1: Add fullscreen initialization function**

Add this function after `initTabBar()` (around line 80):

```typescript
// Initialize fullscreen button
function initFullscreenButton(): void {
  const fullscreenBtn = document.getElementById("fullscreenBtn");
  const fullscreenIcon = fullscreenBtn?.querySelector(".fullscreen-icon");

  if (!fullscreenBtn || !fullscreenIcon) return;

  // Check if Fullscreen API is supported
  if (!document.fullscreenEnabled) {
    fullscreenBtn.classList.add("unsupported");
    return;
  }

  // Toggle fullscreen on button click
  fullscreenBtn.addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) {
        // Enter fullscreen
        await document.documentElement.requestFullscreen();
      } else {
        // Exit fullscreen
        await document.exitFullscreen();
      }
    } catch (err) {
      console.warn("Fullscreen request failed:", err);
    }
  });

  // Update icon when fullscreen state changes
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement) {
      // In fullscreen - show exit icon
      fullscreenIcon.textContent = "✕";
    } else {
      // Not in fullscreen - show expand icon
      fullscreenIcon.textContent = "⛶";
    }
  });
}
```

**Step 2: Call initialization in init()**

Modify the `init()` function to call `initFullscreenButton()`:

```typescript
function init(): void {
  initCanvas();
  initNetworkDOM();
  initDebugMenu();
  initCameraView();
  initAlertView();
  initTabBar();
  initFullscreenButton(); // Add this line
  startHealthCheck();
  startRadarPolling();
}
```

**Step 3: Verify TypeScript compiles**

Run: `tsc --noEmit`

Expected: No errors

**Step 4: Test manually**

Run: `bun run fe`

Expected:
- Fullscreen button visible in top-right
- Clicking enters fullscreen mode
- Icon changes to "✕"
- Clicking again exits fullscreen
- Icon changes back to "⛶"

**Step 5: Commit**

```bash
git add src/fe/src/main.ts
git commit -m "feat: implement fullscreen toggle functionality

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Final Verification

**Files:**
- All modified files

**Step 1: Run full type check**

Run: `tsc --noEmit`

Expected: No type errors

**Step 2: Run linter**

Run: `biome check --write .`

Expected: All files formatted and linted

**Step 3: Test in browser**

Run: `bun run fe`

Test scenarios:
1. Button is visible in top-right
2. Button has correct hover effects
3. Clicking enters fullscreen mode
4. Icon updates to exit icon (✕)
5. Pressing ESC exits fullscreen
6. Icon updates back to expand icon (⛶)
7. Button hidden if Fullscreen API unsupported

**Step 4: Final commit if any formatting changes**

```bash
git add .
git commit -m "chore: format code with biome

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Implementation Complete

The fullscreen button is now fully functional with:
- Clean UI integration matching existing dark theme
- Proper icon state management
- Graceful degradation for unsupported browsers
- Responsive design for mobile devices
- Full keyboard support (ESC to exit)
