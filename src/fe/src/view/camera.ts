import { CAMERAS } from "../config.ts";
import {
	isCameraVisible,
	setCameraVisible,
} from "../store/cameraVisibilityStore.ts";

let cameraGrid: HTMLElement | null = null;
let cameraList: HTMLElement | null = null;
let cameraListWrapper: HTMLElement | null = null;

export function initCameraView(): void {
	cameraGrid = document.getElementById("cameraGrid");
	cameraList = document.getElementById("cameraList");
	cameraListWrapper = document.getElementById("cameraListWrapper");
	if (!cameraGrid || !cameraList) return;
	initCameraListCollapse();
	renderCameraList();
	renderCameraGrid();
}

function initCameraListCollapse(): void {
	const btn = document.getElementById("cameraListCollapseBtn");
	if (!cameraListWrapper || !btn) return;

	btn.addEventListener("click", () => {
		cameraListWrapper?.classList.toggle("collapsed");
		btn.setAttribute(
			"aria-label",
			cameraListWrapper?.classList.contains("collapsed")
				? "Expand camera list"
				: "Collapse camera list",
		);
		btn.setAttribute(
			"title",
			cameraListWrapper?.classList.contains("collapsed")
				? "Expand camera list"
				: "Collapse camera list",
		);
	});
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
			updateCameraGridVisibility();
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

	cameraGrid.innerHTML = CAMERAS.map(
		(camera) =>
			`<radars-camera-feed camera-id="${camera.id}" variant="gallery"></radars-camera-feed>`,
	).join("");

	updateCameraGridVisibility();
}

function updateCameraGridVisibility(): void {
	if (!cameraGrid) return;

	let visibleCount = 0;
	for (const feed of Array.from(cameraGrid.querySelectorAll("radars-camera-feed"))) {
		const cameraId = Number.parseInt(feed.getAttribute("camera-id") ?? "", 10);
		if (!Number.isFinite(cameraId)) continue;
		const visible = isCameraVisible(cameraId);
		if (visible) visibleCount++;
		(feed as HTMLElement).style.display = visible ? "" : "none";
	}

	cameraGrid.classList.toggle("cols-1", visibleCount === 1 || visibleCount === 2);
	cameraGrid.classList.toggle("cols-2", visibleCount >= 3 && visibleCount < 5);
}
