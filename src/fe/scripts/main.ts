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
import {
	drawInactiveRadarAreas,
	drawPulsatingCenter,
	drawRadarBase,
	drawRadarDots,
	drawRadarTrails,
	drawSweepLine,
	initCanvas,
	updateSweepLine,
} from "./radar.ts";

// Initialize application
function init(): void {
	initCanvas(); // Creates offscreen canvas and draws static base
	initNetworkDOM();
	initDebugMenu(); // Initialize debug menu controls
	startHealthCheck();
	startRadarPolling();
}

// Render frame
function render(): void {
	checkServerAvailability();
	updateSweepLine(HEALTH_CHECK_INTERVAL); // Update sweep line animation
	drawRadarBase();
	drawInactiveRadarAreas(radarStatuses); // Draw greyed areas for inactive radars
	drawSweepLine(); // Draw sweep line after base, before dots
	drawRadarTrails(trackHistory); // Draw fading trails before current dots
	drawRadarDots(radarDots);
	drawPulsatingCenter(); // Draw pulsating center dot on top
	requestAnimationFrame(render);
}

// Handle window resize
let resizeTimeout: number;
window.addEventListener("resize", () => {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(() => {
		initCanvas(); // Reinitialize canvas with new dimensions
	}, 250);
});

// Start application
init();
render();
