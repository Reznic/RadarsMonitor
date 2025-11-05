import {
	checkServerAvailability,
	initNetworkDOM,
	radarDots,
	startHealthCheck,
	startRadarPolling,
	trackHistory,
} from "./network.ts";
import { drawRadarBase, drawRadarDots, drawRadarTrails, drawSweepLine, initCanvas, updateSweepLine } from "./radar.ts";
import { initDebugMenu } from "./debugMenu.ts";
import { HEALTH_CHECK_INTERVAL } from "./config.ts";

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
	drawSweepLine(); // Draw sweep line after base, before dots
	drawRadarTrails(trackHistory); // Draw fading trails before current dots
	drawRadarDots(radarDots);
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
