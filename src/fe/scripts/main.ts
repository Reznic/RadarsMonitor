import {
	checkServerAvailability,
	initNetworkDOM,
	radarDots,
	startHealthCheck,
	startRadarPolling,
} from "./network.ts";
import { drawRadarBase, drawRadarDots, initCanvas } from "./radar.ts";
import { initDebugMenu } from "./debugMenu.ts";

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
	drawRadarBase();
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
