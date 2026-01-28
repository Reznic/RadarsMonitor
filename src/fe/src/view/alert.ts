import { CAMERAS, type CameraMode, getCameraStreamUrl } from "../config.ts";
import { connectWebRTC, disconnectWebRTC } from "../stream/index.ts";
import { getCameraStreamIfActive } from "./camera.ts";

let alertOverlay: HTMLElement | null = null;
let alertGrid: HTMLElement | null = null;
let alertCloseButton: HTMLElement | null = null;

// Track which radar IDs have active alerts
const activeAlerts: Set<number> = new Set();

// Track camera modes for alert cameras (separate from main camera view)
const alertCameraModes: Map<number, CameraMode> = new Map();

// Track which alert streams are reused (don't disconnect these)
const reusedStreams: Set<number> = new Set();

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
		const streamUrl = getCameraStreamUrl(camera, mode);

		const video = cameraCell.querySelector(".camera-video") as HTMLVideoElement;
		if (video) {
			// Check if we can reuse existing stream from main view
			const existingStream = getCameraStreamIfActive(radarId, mode);

			if (existingStream) {
				// Reuse existing stream - no new WebRTC connection needed
				const cameraId = `alert-${radarId}`;
				disconnectWebRTC(cameraId); // Disconnect if we had created one
				reusedStreams.add(radarId);
				video.srcObject = existingStream;
				video.play().catch(console.error);
			} else {
				// No existing stream available, create new connection
				reusedStreams.delete(radarId);
				const cameraId = `alert-${radarId}`;
				disconnectWebRTC(cameraId);
				video.srcObject = null;

				video.dataset.streamUrl = streamUrl;
				connectWebRTC(video, streamUrl).catch(console.error);
			}
		}

		const ipDisplay = cameraCell.querySelector(".camera-ip");
		if (ipDisplay) {
			ipDisplay.textContent = streamUrl;
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

			const streamUrl = getCameraStreamUrl(camera, mode);

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
            <video class="camera-video" data-camera-id="alert-${radarId}" data-stream-url="${streamUrl}" autoplay muted playsinline></video>
          </div>
          <div class="camera-ip">${streamUrl}</div>
        </div>
      `;
		})
		.join("");

	alertGrid.innerHTML = alertHtml;

	// Connect streams for all alert videos (reuse if available from main view)
	const videos = alertGrid.querySelectorAll<HTMLVideoElement>(".camera-video");
	videos.forEach((video) => {
		const radarIdAttr = video.closest(".track-alert-camera")?.getAttribute(
			"data-radar-id",
		);
		if (!radarIdAttr) return;

		const radarId = Number.parseInt(radarIdAttr, 10);
		const mode = alertCameraModes.get(radarId) || getDefaultMode();

		// Check if we can reuse existing stream from main camera view
		const existingStream = getCameraStreamIfActive(radarId, mode);

		if (existingStream) {
			// Reuse existing stream - no new WebRTC connection needed!
			reusedStreams.add(radarId);
			video.srcObject = existingStream;
			video.play().catch(console.error);
			console.log(
				`[Alert] Reusing existing stream for camera ${radarId} (${mode} mode)`,
			);
		} else {
			// Create new connection only if stream not available
			reusedStreams.delete(radarId);
			const streamUrl = video.dataset.streamUrl;
			if (streamUrl) {
				connectWebRTC(video, streamUrl).catch(console.error);
			}
		}
	});
}

function dismissAlert(radarId: number): void {
	// Only disconnect if we created the stream (not reused from main view)
	if (!reusedStreams.has(radarId)) {
		const cameraId = `alert-${radarId}`;
		disconnectWebRTC(cameraId);
	} else {
		// Clear the srcObject reference but don't disconnect the underlying stream
		const cameraCell = alertGrid?.querySelector(
			`.track-alert-camera[data-radar-id="${radarId}"]`,
		);
		const video = cameraCell?.querySelector(".camera-video") as HTMLVideoElement;
		if (video) {
			video.srcObject = null;
		}
	}

	reusedStreams.delete(radarId);
	activeAlerts.delete(radarId);
	alertCameraModes.delete(radarId);

	if (activeAlerts.size === 0) {
		dismissAllAlerts();
	} else {
		renderAlertGrid();
	}
}

function dismissAllAlerts(): void {
	// Disconnect only streams we created (not reused from main view)
	for (const radarId of activeAlerts) {
		if (!reusedStreams.has(radarId)) {
			const cameraId = `alert-${radarId}`;
			disconnectWebRTC(cameraId);
		} else {
			// Clear srcObject reference for reused streams
			const cameraCell = alertGrid?.querySelector(
				`.track-alert-camera[data-radar-id="${radarId}"]`,
			);
			const video = cameraCell?.querySelector(
				".camera-video",
			) as HTMLVideoElement;
			if (video) {
				video.srcObject = null;
			}
		}
	}

	reusedStreams.clear();
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
