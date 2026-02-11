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
	renderCameraList();
	renderCameraGrid();
}

function renderCameraList(): void {
	if (!cameraList) return;
	const collapsed = cameraListWrapper?.classList.contains("collapsed") ?? false;

	cameraList.innerHTML = `
		<div class="camera-list-header">
			<div class="camera-list-title">CAMERAS</div>
			<button
				type="button"
				class="camera-list-collapse-btn"
				id="cameraListCollapseBtn"
				aria-label="${collapsed ? "Expand camera list" : "Collapse camera list"}"
				title="${collapsed ? "Expand camera list" : "Collapse camera list"}"
			>▼</button>
		</div>
		<div class="camera-list-items">
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
		</div>
	`;

	const collapseBtn = cameraList.querySelector(
		"#cameraListCollapseBtn",
	) as HTMLButtonElement | null;
	const header = cameraList.querySelector(
		".camera-list-header",
	) as HTMLElement | null;
	header?.addEventListener("click", () => {
		cameraListWrapper?.classList.toggle("collapsed");
		const isCollapsed = cameraListWrapper?.classList.contains("collapsed");
		if (collapseBtn) {
			collapseBtn.setAttribute(
				"aria-label",
				isCollapsed ? "Expand camera list" : "Collapse camera list",
			);
			collapseBtn.setAttribute(
				"title",
				isCollapsed ? "Expand camera list" : "Collapse camera list",
			);
		}
	});

	for (const btn of Array.from(
		cameraList.querySelectorAll(".camera-list-toggle"),
	)) {
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
	for (const feed of Array.from(
		cameraGrid.querySelectorAll("radars-camera-feed"),
	)) {
		const cameraId = Number.parseInt(feed.getAttribute("camera-id") ?? "", 10);
		if (!Number.isFinite(cameraId)) continue;
		const visible = isCameraVisible(cameraId);
		if (visible) visibleCount++;
		(feed as HTMLElement).style.display = visible ? "" : "none";
	}

	cameraGrid.classList.toggle(
		"cols-1",
		visibleCount === 1 || visibleCount === 2,
	);
	cameraGrid.classList.toggle("cols-2", visibleCount >= 3 && visibleCount < 5);
	cameraGrid.classList.toggle("cols-4", visibleCount >= 5);
}
