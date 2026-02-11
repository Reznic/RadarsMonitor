import { CAMERAS } from "../config.ts";
import {
	isCameraVisible,
	setCameraVisible,
} from "../store/cameraVisibilityStore.ts";

let cameraGrid: HTMLElement | null = null;
let cameraList: HTMLElement | null = null;

export function initCameraView(): void {
	cameraGrid = document.getElementById("cameraGrid");
	cameraList = document.getElementById("cameraList");
	if (!cameraGrid || !cameraList) return;
	renderCameraList();
	renderCameraGrid();
}

function renderCameraList(): void {
	if (!cameraList) return;

	cameraList.innerHTML = `
		<div class="camera-list-title">CAMERAS</div>
		${CAMERAS.map(
			(camera) => `
			<div class="camera-list-item" data-camera-id="${camera.id}">
				<span class="camera-list-name">${escapeHtml(camera.name)}</span>
				<button type="button" class="camera-list-toggle" data-camera-id="${camera.id}" aria-label="${isCameraVisible(camera.id) ? "Hide" : "Show"} ${escapeHtml(camera.name)}">
					${isCameraVisible(camera.id) ? "Hide" : "Show"}
				</button>
			</div>
		`,
		).join("")}
	`;

	for (const btn of Array.from(cameraList.querySelectorAll(".camera-list-toggle"))) {
		btn.addEventListener("click", () => {
			const cameraId = Number.parseInt(
				(btn as HTMLElement).dataset.cameraId ?? "",
				10,
			);
			if (!Number.isFinite(cameraId)) return;
			const next = !isCameraVisible(cameraId);
			setCameraVisible(cameraId, next);
			renderCameraList();
			renderCameraGrid();
		});
	}
}

function escapeHtml(text: string): string {
	const div = document.createElement("div");
	div.textContent = text;
	return div.innerHTML;
}

function renderCameraGrid(): void {
	if (!cameraGrid) return;

	const visibleCameras = CAMERAS.filter((c) => isCameraVisible(c.id));
	cameraGrid.innerHTML = visibleCameras
		.map(
			(camera) =>
				`<radars-camera-feed camera-id="${camera.id}" variant="gallery"></radars-camera-feed>`,
		)
		.join("");
}
