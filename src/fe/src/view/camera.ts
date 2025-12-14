import {
	CAMERAS,
	type CameraConfig,
	type CameraMode,
	getCameraStreamUrl,
} from "../config.ts";
import { connectMSE } from "../stream.ts";

let cameraGrid: HTMLElement | null = null;
const cameraModes: Map<number, CameraMode> = new Map();

export function initCameraView(): void {
	cameraGrid = document.getElementById("cameraGrid");

	// Initialize all cameras to day mode
	for (const camera of CAMERAS) {
		cameraModes.set(camera.id, "day");
	}

	if (cameraGrid) {
		renderCameraGrid();
		setupModeToggleListeners();
		connectAllCameras();
	}
}

function connectAllCameras(): void {
	const videos = document.querySelectorAll<HTMLVideoElement>(".camera-video");
	videos.forEach((video) => {
		const streamUrl = video.dataset.streamUrl;
		if (streamUrl) {
			connectMSE(video, streamUrl).catch(console.error);
		}
	});
}

function renderCameraGrid(): void {
	if (!cameraGrid) return;

	cameraGrid.innerHTML = CAMERAS.map((camera) => createCameraCell(camera)).join(
		"",
	);
}

function setupModeToggleListeners(): void {
	if (!cameraGrid) return;

	cameraGrid.addEventListener("click", (event) => {
		const target = event.target as HTMLElement;
		const toggleBtn = target.closest(".camera-mode-toggle") as HTMLElement;

		if (toggleBtn) {
			const cameraId = Number(toggleBtn.dataset.cameraId);
			toggleCameraMode(cameraId);
		}
	});
}

function toggleCameraMode(cameraId: number): void {
	const currentMode = cameraModes.get(cameraId) || "day";
	const newMode: CameraMode = currentMode === "day" ? "night" : "day";
	cameraModes.set(cameraId, newMode);

	updateCameraModeUI(cameraId, newMode);
}

function updateCameraModeUI(cameraId: number, mode: CameraMode): void {
	const cameraCell = document.querySelector(
		`.camera-cell[data-camera-id="${cameraId}"]`,
	);
	if (!cameraCell) return;

	const toggleBtn = cameraCell.querySelector(
		".camera-mode-toggle",
	) as HTMLElement;
	if (toggleBtn) {
		toggleBtn.classList.toggle("night", mode === "night");
	}

	const camera = CAMERAS.find((c) => c.id === cameraId);
	if (camera) {
		const streamUrl = getCameraStreamUrl(camera, mode);

		const video = cameraCell.querySelector(".camera-video") as HTMLVideoElement;
		if (video) {
			video.dataset.streamUrl = streamUrl;
			connectMSE(video, streamUrl).catch(console.error);
		}

		const ipDisplay = cameraCell.querySelector(".camera-ip");
		if (ipDisplay) {
			ipDisplay.textContent = streamUrl;
		}
	}
}

function createCameraCell(camera: CameraConfig): string {
	const mode = cameraModes.get(camera.id) || "day";
	const streamUrl = getCameraStreamUrl(camera, mode);

	return `
    <div class="camera-cell" data-camera-id="${camera.id}">
      <div class="camera-header">
        <span class="camera-name">${camera.name}</span>
        <button class="camera-mode-toggle${mode === "night" ? " night" : ""}" data-camera-id="${camera.id}">
          <span class="mode-day">DAY</span>
          <span class="mode-night">NIGHT</span>
        </button>
        <span class="camera-status offline">OFFLINE</span>
      </div>
      <div class="camera-feed">
        <video class="camera-video" data-camera-id="cam-${camera.id}" data-stream-url="${streamUrl}" autoplay muted playsinline></video>
        <div class="camera-placeholder">
          <div class="camera-placeholder-icon">📷</div>
          <div>No Signal</div>
          <div class="camera-ip">${streamUrl}</div>
        </div>
      </div>
    </div>
  `;
}

export function getCameraMode(cameraId: number): CameraMode {
	return cameraModes.get(cameraId) || "day";
}
