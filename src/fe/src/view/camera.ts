import { CAMERAS, type CameraConfig, type CameraMode } from "../config.ts";

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
	}
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
		const ipDisplay = cameraCell.querySelector(".camera-ip");
		if (ipDisplay) {
			const streamPath =
				mode === "day" ? camera.dayStreamPath : camera.nightStreamPath;
			ipDisplay.textContent = `${camera.ip}:${camera.port}${streamPath}`;
		}
	}
}

function createCameraCell(camera: CameraConfig): string {
	const mode = cameraModes.get(camera.id) || "day";
	const streamPath =
		mode === "day" ? camera.dayStreamPath : camera.nightStreamPath;

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
        <div class="camera-placeholder">
          <div class="camera-placeholder-icon">📷</div>
          <div>No Signal</div>
          <div class="camera-ip">${camera.ip}:${camera.port}${streamPath}</div>
        </div>
      </div>
    </div>
  `;
}

export function getCameraStreamUrl(
	camera: CameraConfig,
	mode?: CameraMode,
): string {
	const cameraMode = mode || cameraModes.get(camera.id) || "day";
	const streamPath =
		cameraMode === "day" ? camera.dayStreamPath : camera.nightStreamPath;
	return `http://${camera.ip}:${camera.port}${streamPath}`;
}

export function getCameraMode(cameraId: number): CameraMode {
	return cameraModes.get(cameraId) || "day";
}
