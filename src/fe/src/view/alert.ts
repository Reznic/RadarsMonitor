import { CAMERAS, type CameraMode } from "../config.ts";

let alertOverlay: HTMLElement | null = null;
let alertGrid: HTMLElement | null = null;
let alertCloseButton: HTMLElement | null = null;

// Track which radar IDs have active alerts
const activeAlerts: Set<number> = new Set();

// Track camera modes for alert cameras (separate from main camera view)
const alertCameraModes: Map<number, CameraMode> = new Map();

// Determine if it's currently nighttime (between 6 PM and 6 AM)
function isNightTime(): boolean {
	const hour = new Date().getHours();
	return hour >= 18 || hour < 6;
}

// Get the default mode based on current time
function getDefaultMode(): CameraMode {
	return isNightTime() ? "night" : "day";
}

export function initAlertView(): void {
	alertOverlay = document.getElementById("trackAlertOverlay");
	alertGrid = document.getElementById("trackAlertGrid");
	alertCloseButton = document.getElementById("trackAlertClose");

	if (alertCloseButton) {
		alertCloseButton.addEventListener("click", dismissAllAlerts);
	}

	// Handle individual dismiss buttons and mode toggles via event delegation
	if (alertGrid) {
		alertGrid.addEventListener("click", (event) => {
			const target = event.target as HTMLElement;
			if (target.classList.contains("track-alert-dismiss")) {
				const radarId = Number.parseInt(
					target.getAttribute("data-radar-id") || "0",
					10,
				);
				dismissAlert(radarId);
			} else {
				const toggleBtn = target.closest(
					".camera-mode-toggle",
				) as HTMLElement | null;
				if (toggleBtn) {
					const radarId = Number.parseInt(
						toggleBtn.getAttribute("data-camera-id") || "0",
						10,
					);
					toggleAlertCameraMode(radarId);
				}
			}
		});
	}
}

function toggleAlertCameraMode(radarId: number): void {
	const currentMode = alertCameraModes.get(radarId) || getDefaultMode();
	const newMode: CameraMode = currentMode === "day" ? "night" : "day";
	alertCameraModes.set(radarId, newMode);

	updateAlertCameraModeUI(radarId, newMode);
}

function updateAlertCameraModeUI(radarId: number, mode: CameraMode): void {
	const cameraCell = alertGrid?.querySelector(
		`.track-alert-camera[data-radar-id="${radarId}"]`,
	);
	if (!cameraCell) return;

	const toggleBtn = cameraCell.querySelector(
		".camera-mode-toggle",
	) as HTMLElement;
	if (toggleBtn) {
		toggleBtn.classList.toggle("night", mode === "night");
	}

	const camera = CAMERAS.find((c) => c.id === radarId);
	if (camera) {
		const ipDisplay = cameraCell.querySelector(".camera-ip");
		if (ipDisplay) {
			const streamPath =
				mode === "day" ? camera.dayStreamPath : camera.nightStreamPath;
			ipDisplay.textContent = `${camera.ip}:${camera.port}${streamPath}`;
		}
	}
}

// Show alert for new tracks from specific radars
export function showTrackAlert(radarIds: number[]): void {
	if (!alertOverlay || !alertGrid) return;

	// Filter out radars that already have alerts
	const newRadarIds = radarIds.filter((id) => !activeAlerts.has(id));
	if (newRadarIds.length === 0) return;

	// Add new radar IDs to active alerts and set their default mode based on time
	const defaultMode = getDefaultMode();
	for (const id of newRadarIds) {
		activeAlerts.add(id);
		// Only set mode if not already set (preserve user's choice)
		if (!alertCameraModes.has(id)) {
			alertCameraModes.set(id, defaultMode);
		}
	}

	// Render alert cameras
	renderAlertGrid();

	// Show overlay
	alertOverlay.classList.remove("hidden");
}

function renderAlertGrid(): void {
	if (!alertGrid) return;

	const alertHtml = Array.from(activeAlerts)
		.map((radarId) => {
			const camera = CAMERAS.find((c) => c.id === radarId);
			const mode = alertCameraModes.get(radarId) || getDefaultMode();
			const isNight = mode === "night";

			if (!camera) {
				// Create placeholder for radars without configured cameras
				return `
          <div class="track-alert-camera" data-radar-id="${radarId}">
            <div class="track-alert-camera-header">
              <span class="track-alert-camera-name">CAMERA ${radarId}</span>
              <span class="track-alert-camera-radar">RADAR ${radarId}</span>
            </div>
            <div class="track-alert-camera-feed">
              <button class="track-alert-dismiss" data-radar-id="${radarId}">✕</button>
              <div class="camera-placeholder">
                <div class="camera-placeholder-icon">📷</div>
                <div>No Camera Configured</div>
              </div>
            </div>
          </div>
        `;
			}

			const streamPath = isNight
				? camera.nightStreamPath
				: camera.dayStreamPath;

			return `
        <div class="track-alert-camera" data-radar-id="${radarId}">
          <div class="track-alert-camera-header">
            <span class="track-alert-camera-name">${camera.name}</span>
            <button class="camera-mode-toggle${isNight ? " night" : ""}" data-camera-id="${radarId}">
              <span class="mode-day">DAY</span>
              <span class="mode-night">NIGHT</span>
            </button>
            <span class="track-alert-camera-radar">RADAR ${radarId}</span>
          </div>
          <div class="track-alert-camera-feed">
            <button class="track-alert-dismiss" data-radar-id="${radarId}">✕</button>
            <div class="camera-placeholder">
              <div class="camera-placeholder-icon">📷</div>
              <div>Connecting...</div>
              <div class="camera-ip">${camera.ip}:${camera.port}${streamPath}</div>
            </div>
          </div>
        </div>
      `;
		})
		.join("");

	alertGrid.innerHTML = alertHtml;
}

function dismissAlert(radarId: number): void {
	activeAlerts.delete(radarId);
	alertCameraModes.delete(radarId);

	if (activeAlerts.size === 0) {
		dismissAllAlerts();
	} else {
		renderAlertGrid();
	}
}

function dismissAllAlerts(): void {
	activeAlerts.clear();
	alertCameraModes.clear();
	if (alertOverlay) {
		alertOverlay.classList.add("hidden");
	}
	if (alertGrid) {
		alertGrid.innerHTML = "";
	}
}

// Check if alert overlay is currently visible
export function isAlertVisible(): boolean {
	return alertOverlay ? !alertOverlay.classList.contains("hidden") : false;
}
