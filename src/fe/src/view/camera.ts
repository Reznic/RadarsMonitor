import { CAMERAS } from "../config.ts";

let cameraGrid: HTMLElement | null = null;

export function initCameraView(): void {
	cameraGrid = document.getElementById("cameraGrid");
	if (!cameraGrid) return;
	renderCameraGrid();
}

function renderCameraGrid(): void {
	if (!cameraGrid) return;

	cameraGrid.innerHTML = CAMERAS.map(
		(camera) =>
			`<radars-camera-feed camera-id="${camera.id}" variant="gallery"></radars-camera-feed>`,
	).join("");
}
