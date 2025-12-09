import { CAMERAS } from "../config.ts";

let alertOverlay: HTMLElement | null = null;
let alertGrid: HTMLElement | null = null;
let alertCloseButton: HTMLElement | null = null;

// Track which radar IDs have active alerts
const activeAlerts: Set<number> = new Set();

export function initAlertView(): void {
	alertOverlay = document.getElementById("trackAlertOverlay");
	alertGrid = document.getElementById("trackAlertGrid");
	alertCloseButton = document.getElementById("trackAlertClose");

	if (alertCloseButton) {
		alertCloseButton.addEventListener("click", dismissAllAlerts);
	}

	// Handle individual dismiss buttons via event delegation
	if (alertGrid) {
		alertGrid.addEventListener("click", (event) => {
			const target = event.target as HTMLElement;
			if (target.classList.contains("track-alert-dismiss")) {
				const radarId = Number.parseInt(
					target.getAttribute("data-radar-id") || "0",
					10,
				);
				dismissAlert(radarId);
			}
		});
	}
}

// Show alert for new tracks from specific radars
export function showTrackAlert(radarIds: number[]): void {
	if (!alertOverlay || !alertGrid) return;

	// Filter out radars that already have alerts
	const newRadarIds = radarIds.filter((id) => !activeAlerts.has(id));
	if (newRadarIds.length === 0) return;

	// Add new radar IDs to active alerts
	for (const id of newRadarIds) {
		activeAlerts.add(id);
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

			return `
        <div class="track-alert-camera" data-radar-id="${radarId}">
          <div class="track-alert-camera-header">
            <span class="track-alert-camera-name">${camera.name}</span>
            <span class="track-alert-camera-radar">RADAR ${radarId}</span>
          </div>
          <div class="track-alert-camera-feed">
            <button class="track-alert-dismiss" data-radar-id="${radarId}">✕</button>
            <div class="camera-placeholder">
              <div class="camera-placeholder-icon">📷</div>
              <div>Connecting...</div>
              <div class="camera-ip">${camera.ip}:${camera.port}</div>
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

	if (activeAlerts.size === 0) {
		dismissAllAlerts();
	} else {
		renderAlertGrid();
	}
}

function dismissAllAlerts(): void {
	activeAlerts.clear();
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
