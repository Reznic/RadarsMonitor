import { HEALTH_CHECK_INTERVAL } from "./config.ts";
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
	drawSweepLine,
	drawVehicleOverlay,
	initCanvas,
	updateSweepLine,
} from "./view/radar.ts";

type ViewType = "radar" | "camera";
let currentView: ViewType = "radar";

// Initialize application
function init(): void {
	initCanvas(); // Creates offscreen canvas and draws static base
	initNetworkDOM();
	initDebugMenu(); // Initialize debug menu controls
	initCameraView(); // Initialize camera view
	initAlertView(); // Initialize alert overlay
	initTabBar(); // Initialize tab bar controls
	startHealthCheck();
	startRadarPolling();
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
		});
	});
}

// Render frame
function render(): void {
	if (currentView === "radar") {
		checkServerAvailability();
		updateSweepLine(HEALTH_CHECK_INTERVAL); // Update sweep line animation
		drawRadarBase();
		drawInactiveRadarAreas(radarStatuses); // Draw greyed areas for inactive radars
		drawSweepLine(); // Draw sweep line after base, before dots
		drawRadarTrails(trackHistory); // Draw fading trails before current dots
		drawRadarDots(radarDots);
		drawVehicleOverlay(); // Ensure vehicle and markers render above dots/trails
	}
	requestAnimationFrame(render);
}

// Handle window resize
let resizeTimeout: ReturnType<typeof setTimeout> | undefined;
window.addEventListener("resize", () => {
	if (resizeTimeout) {
		clearTimeout(resizeTimeout);
	}
	resizeTimeout = setTimeout(() => {
		initCanvas(); // Reinitialize canvas with new dimensions
	}, 250);
});

// Start application
init();
render();
