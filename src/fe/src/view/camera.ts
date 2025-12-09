import { CAMERAS, type CameraConfig } from "../config.ts";

let cameraGrid: HTMLElement | null = null;

export function initCameraView(): void {
	cameraGrid = document.getElementById("cameraGrid");

	if (cameraGrid) {
		renderCameraGrid();
	}
}

function renderCameraGrid(): void {
	if (!cameraGrid) return;

	cameraGrid.innerHTML = CAMERAS.map((camera) => createCameraCell(camera)).join(
		"",
	);
}

function createCameraCell(camera: CameraConfig): string {
	return `
    <div class="camera-cell" data-camera-id="${camera.id}">
      <div class="camera-header">
        <span class="camera-name">${camera.name}</span>
        <span class="camera-status offline">OFFLINE</span>
      </div>
      <div class="camera-feed">
        <div class="camera-placeholder">
          <div class="camera-placeholder-icon">📷</div>
          <div>No Signal</div>
          <div class="camera-ip">${camera.ip}:${camera.port}</div>
        </div>
      </div>
    </div>
  `;
}

export function getCameraStreamUrl(camera: CameraConfig): string {
	return `http://${camera.ip}:${camera.port}/stream`;
}
