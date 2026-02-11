import { HEALTH_CHECK_INTERVAL } from "./config.ts";
import "./components/radars-camera-feed.ts";
import "./components/radars-radar-display.ts";
import { initDebugMenu } from "./debugMenu.ts";
import {
	checkServerAvailability,
	initNetworkDOM,
	radarDots,
	radarStatuses,
	startHealthCheck,
	startRadarPolling,
	trackHistory,
} from "./network.ts";
import { initAlertView } from "./view/alert.ts";
import { initCameraView } from "./view/camera.ts";
import {
	drawInactiveRadarAreas,
	drawRadarBase,
	drawRadarDots,
	drawRadarTrails,
	drawRadarUnitIndices,
	drawSweepLine,
	drawVehicleOverlay,
	initCanvas,
	updateSweepLine,
} from "./view/radar.ts";

type ViewType = "radar" | "camera";
let currentView: ViewType = "radar";

// Target render frame rate (lower to reduce CPU/GPU load)
const TARGET_FPS = 30;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
let lastFrameTime = 0;

// Initialize application
function init(): void {
	syncRadarCanvas(); // Creates offscreen canvas and draws static base
	initNetworkDOM();
	initDebugMenu(); // Initialize debug menu controls
	initCameraView(); // Initialize camera view
	initAlertView(); // Initialize alert overlay
	initTabBar(); // Initialize tab bar controls
	initFullscreenButton();
	startHealthCheck();
	startRadarPolling();
}

function getActiveRadarCanvas(
	view: ViewType = currentView,
): HTMLCanvasElement | null {
	const target = view === "radar" ? "main" : "mini";
	return document.querySelector(
		`.radar-canvas[data-radar-canvas="${target}"]`,
	) as HTMLCanvasElement | null;
}

function syncRadarCanvas(view: ViewType = currentView): void {
	const canvas = getActiveRadarCanvas(view);
	if (!canvas) return;
	initCanvas(canvas);
}

// Initialize tab bar
function initTabBar(): void {
	const tabButtons = document.querySelectorAll(".tab-btn");
	const radarView = document.getElementById("radarView");
	const cameraView = document.getElementById("cameraView");
	const hud = document.getElementById("hud");
	const debugMenu = document.getElementById("debugMenu");

	tabButtons.forEach((btn) => {
		btn.addEventListener("click", () => {
			const view = btn.getAttribute("data-view") as ViewType;
			if (view === currentView) return;

			currentView = view;

			// Update tab button states
			for (const b of Array.from(tabButtons)) {
				b.classList.remove("active");
			}
			btn.classList.add("active");

			// Toggle views
			if (view === "radar") {
				radarView?.classList.remove("hidden");
				cameraView?.classList.add("hidden");
				hud?.classList.remove("hidden");
				debugMenu?.classList.remove("hidden");
			} else {
				radarView?.classList.add("hidden");
				cameraView?.classList.remove("hidden");
				hud?.classList.add("hidden");
				debugMenu?.classList.add("hidden");
			}

			syncRadarCanvas(view);
		});
	});
}

// Initialize fullscreen button
function initFullscreenButton(): void {
	const fullscreenBtn = document.getElementById("fullscreenBtn");
	const iconExpand = fullscreenBtn?.querySelector<SVGElement>(
		".fullscreen-icon-expand",
	);
	const iconExit = fullscreenBtn?.querySelector<SVGElement>(
		".fullscreen-icon-exit",
	);

	if (!fullscreenBtn || !iconExpand || !iconExit) return;

	// Check if Fullscreen API is supported
	if (!document.fullscreenEnabled) {
		fullscreenBtn.classList.add("unsupported");
		return;
	}

	// Toggle fullscreen on button click
	fullscreenBtn.addEventListener("click", async () => {
		try {
			if (!document.fullscreenElement) {
				// Enter fullscreen
				await document.documentElement.requestFullscreen();
			} else {
				// Exit fullscreen
				await document.exitFullscreen();
			}
		} catch (err) {
			console.warn("Fullscreen request failed:", err);
		}
	});

	// Update icon when fullscreen state changes
	document.addEventListener("fullscreenchange", () => {
		const isFullscreen = !!document.fullscreenElement;
		iconExpand.classList.toggle("hidden", isFullscreen);
		iconExit.classList.toggle("hidden", !isFullscreen);
		fullscreenBtn.setAttribute(
			"title",
			isFullscreen ? "Exit Fullscreen" : "Toggle Fullscreen",
		);
		fullscreenBtn.setAttribute(
			"aria-label",
			isFullscreen ? "Exit full screen" : "Toggle full screen",
		);
	});
}

// Render frame (throttled to TARGET_FPS to reduce compositor usage)
function render(timestamp: number): void {
	// Throttle to desired FPS to avoid overloading VizCompositorTh
	if (timestamp - lastFrameTime < FRAME_INTERVAL_MS) {
		requestAnimationFrame(render);
		return;
	}
	lastFrameTime = timestamp;

	checkServerAvailability();
	updateSweepLine(HEALTH_CHECK_INTERVAL); // Update sweep line animation
	drawRadarBase();
	drawInactiveRadarAreas(radarStatuses); // Draw greyed areas for inactive radars
	drawSweepLine(); // Draw sweep line after base, before dots
	drawRadarTrails(trackHistory); // Draw fading trails before current dots
	drawRadarDots(radarDots);
	drawVehicleOverlay(); // Ensure vehicle and markers render above dots/trails
	drawRadarUnitIndices(radarStatuses);
	requestAnimationFrame(render);
}

// Handle window resize
let resizeTimeout: ReturnType<typeof setTimeout> | undefined;
window.addEventListener("resize", () => {
	if (resizeTimeout) {
		clearTimeout(resizeTimeout);
	}
	resizeTimeout = setTimeout(() => {
		syncRadarCanvas(); // Reinitialize active canvas with new dimensions
	}, 250);
});

// Start application
init();
requestAnimationFrame(render);
