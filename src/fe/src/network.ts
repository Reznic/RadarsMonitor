import type {
	RadarDot,
	RadarsStatusResponse,
	TracksResponse,
} from "../../types.ts";
import {
	API_BASE,
	HEALTH_CHECK_INTERVAL,
	MAX_DOTS,
	RADAR_CHECK_INTERVAL,
	SERVER_TIMEOUT,
} from "./config.ts";
import { showTrackAlert } from "./view/alert.ts";
import { cartesianToCanvas } from "./view/radar.ts";

// DOM elements
let statusElement: HTMLElement | null;
let sensorGridElement: HTMLElement | null;
let hudElement: HTMLElement | null;
let hudHeaderElement: HTMLElement | null;
let hudContentElement: HTMLElement | null;
let hudToggleButton: HTMLButtonElement | null;
const sensorElements: Map<string, HTMLElement> = new Map();
const azimuthElements: Map<string, HTMLElement> = new Map();
let currentSensorOrder: string[] = [];

const HUD_COLLAPSE_KEY = "hud-menu-collapsed";

// State
export const radarDots: RadarDot[] = [];
export let lastDataReceived: number = Date.now();
export let radarStatuses: RadarsStatusResponse = {}; // Store radar status data

// Track history for trail effect
export const trackHistory: Map<number, RadarDot[]> = new Map(); // track_id -> array of positions

// Known track IDs to detect new tracks
const knownTrackIds: Set<number> = new Set();

// Initialize DOM references
export function initNetworkDOM(): void {
	statusElement = document.getElementById("status");
	sensorGridElement = document.getElementById("sensorGrid");
	hudElement = document.getElementById("hud");
	hudHeaderElement = document.getElementById("hudHeader");
	hudContentElement = document.getElementById("hudContent");
	hudToggleButton = document.getElementById(
		"hudToggle",
	) as HTMLButtonElement | null;

	// Initialize radar control button event listeners
	initRadarControls();
	initHudAccordion();
}

function initHudAccordion(): void {
	if (!hudElement || !hudHeaderElement || !hudContentElement) {
		return;
	}

	const toggleHudMenu = (): void => {
		if (!hudElement) return;
		const collapsed = hudElement.classList.toggle("collapsed");
		saveHudCollapseState(collapsed);
	};

	hudHeaderElement.addEventListener("click", toggleHudMenu);

	if (hudToggleButton) {
		hudToggleButton.addEventListener("click", (event) => {
			event.stopPropagation();
			toggleHudMenu();
		});
	}

	loadHudCollapseState();
}

function saveHudCollapseState(collapsed: boolean): void {
	try {
		localStorage.setItem(HUD_COLLAPSE_KEY, JSON.stringify(collapsed));
	} catch (error) {
		console.warn("Failed to save HUD collapse state", error);
	}
}

function loadHudCollapseState(): void {
	if (!hudElement) return;
	try {
		const value = localStorage.getItem(HUD_COLLAPSE_KEY);
		if (value) {
			const collapsed = JSON.parse(value);
			if (collapsed) {
				hudElement.classList.add("collapsed");
			}
		}
	} catch (error) {
		console.warn("Failed to load HUD collapse state", error);
	}
}

// Initialize radar control buttons
function initRadarControls(): void {
	if (!sensorGridElement) return;

	sensorGridElement.addEventListener("click", async (event) => {
		const target = (event.target as HTMLElement)?.closest<HTMLButtonElement>(
			".sensor-btn",
		);

		if (!target) return;

		const radarId = target.getAttribute("data-radar-id");
		const action = target.getAttribute("data-action");

		if (!radarId || !action) return;

		target.disabled = true;

		try {
			if (action === "on") {
				await turnRadarOn(radarId);
			} else if (action === "off") {
				await turnRadarOff(radarId);
			}
			await checkHealth();
		} catch (error) {
			console.error(`Failed to ${action} radar ${radarId}:`, error);
		} finally {
			target.disabled = false;
		}
	});
}

function sortRadarIds(radarIds: string[]): string[] {
	return radarIds
		.slice()
		.sort((a, b) =>
			a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }),
		);
}

function renderSensorGrid(radarIds: string[]): void {
	if (!sensorGridElement) return;

	const needsUpdate =
		radarIds.length !== currentSensorOrder.length ||
		radarIds.some((id, index) => id !== currentSensorOrder[index]);

	if (!needsUpdate) return;

	currentSensorOrder = radarIds.slice();
	sensorElements.clear();
	azimuthElements.clear();
	sensorGridElement.innerHTML = "";

	if (radarIds.length === 0) {
		const emptyState = document.createElement("div");
		emptyState.className = "sensor-empty";
		emptyState.textContent = "No sensors available";
		sensorGridElement.appendChild(emptyState);
		return;
	}

	const fragment = document.createDocumentFragment();

	radarIds.forEach((radarId) => {
		const sensorItem = document.createElement("div");
		sensorItem.className = "sensor-item";
		sensorItem.dataset.radarId = radarId;

		const sensorHeader = document.createElement("div");
		sensorHeader.className = "sensor-header";

		const label = document.createElement("div");
		label.className = "sensor-label";
		label.textContent = radarId.toUpperCase();

		const statusDiv = document.createElement("div");
		statusDiv.className = "sensor-status";
		statusDiv.textContent = "N/A";
		sensorElements.set(radarId, statusDiv);

		sensorHeader.appendChild(label);
		sensorHeader.appendChild(statusDiv);

		const azimuthDiv = document.createElement("div");
		azimuthDiv.className = "sensor-azimuth";
		azimuthDiv.textContent = "Angles — N/A";
		azimuthElements.set(radarId, azimuthDiv);

		const controls = document.createElement("div");
		controls.className = "sensor-controls";
		controls.appendChild(createControlButton("on", radarId));
		controls.appendChild(createControlButton("off", radarId));

		sensorItem.appendChild(sensorHeader);
		sensorItem.appendChild(azimuthDiv);
		sensorItem.appendChild(controls);
		fragment.appendChild(sensorItem);
	});

	sensorGridElement.appendChild(fragment);
}

function createControlButton(
	action: "on" | "off",
	radarId: string,
): HTMLButtonElement {
	const button = document.createElement("button");
	const actionClass = action === "on" ? "sensor-btn-on" : "sensor-btn-off";
	button.type = "button";
	button.className = `sensor-btn ${actionClass}`;
	button.dataset.radarId = radarId;
	button.dataset.action = action;
	button.textContent = action.toUpperCase();
	return button;
}

// Turn radar on
async function turnRadarOn(radarId: string): Promise<void> {
	try {
		const response = await fetch(`${API_BASE}/radar/on`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ radar_id: radarId }),
		});

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		const data = await response.json();
		console.log(`Radar ${radarId} turned on:`, data);
	} catch (error) {
		console.error(`Error turning radar ${radarId} on:`, error);
		throw error;
	}
}

// Turn radar off
async function turnRadarOff(radarId: string): Promise<void> {
	try {
		const response = await fetch(`${API_BASE}/radar/off`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ radar_id: radarId }),
		});

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		const data = await response.json();
		console.log(`Radar ${radarId} turned off:`, data);
	} catch (error) {
		console.error(`Error turning radar ${radarId} off:`, error);
		throw error;
	}
}

// Check if server is responding and clear dots if needed
export function checkServerAvailability(): void {
	const timeSinceLastData: number = Date.now() - lastDataReceived;
	if (timeSinceLastData > SERVER_TIMEOUT && radarDots.length > 0) {
		radarDots.length = 0; // Clear array
	}
}

// Health check function (now using /radars_status endpoint)
async function checkHealth(): Promise<void> {
	try {
		const response: Response = await fetch(`${API_BASE}/radars_status`);

		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`);
		}

		const data: RadarsStatusResponse = await response.json();

		// Store radar statuses for visualization
		radarStatuses = data;

		// Count active and inactive radars
		const radarIds = sortRadarIds(Object.keys(data));
		const inactiveRadars = radarIds.filter((id) => {
			const status = data[id];
			return status && !status.is_active;
		});

		renderSensorGrid(radarIds);

		// Update overall status
		if (statusElement) {
			if (inactiveRadars.length === 0 && radarIds.length > 0) {
				statusElement.textContent = "ALL RADARS OK";
				statusElement.className = "hud-status healthy";
			} else if (radarIds.length === 0) {
				statusElement.textContent = "NO RADARS";
				statusElement.className = "hud-status unhealthy";
			} else {
				statusElement.textContent = `${inactiveRadars.length} RADAR${
					inactiveRadars.length > 1 ? "S" : ""
				} DOWN`;
				statusElement.className = "hud-status unhealthy";
			}
		}

		// Update individual radar statuses
		for (const radarId of radarIds) {
			const radarStatus = data[radarId];
			const sensorElement = sensorElements.get(radarId);
			const azimuthElement = azimuthElements.get(radarId);
			if (sensorElement && radarStatus) {
				if (radarStatus.is_active) {
					sensorElement.textContent = `ON`;
					sensorElement.className = "sensor-status sensor-ok";
				} else {
					sensorElement.textContent = `OFF`;
					sensorElement.className = "sensor-status sensor-error";
				}

				// Update azimuth range display
				if (azimuthElement) {
					const orientationAngle = radarStatus.orientation_angle;
					const startAngle = (orientationAngle - 35 + 360) % 360; // Handle negative angles
					const endAngle = (orientationAngle + 35) % 360;
					azimuthElement.textContent = `Angles — ${startAngle.toFixed(
						0,
					)}° to ${endAngle.toFixed(0)}°`;
					azimuthElement.className = "sensor-azimuth";
				}
			}
		}
	} catch (error) {
		// Server is unreachable - mark as malfunction
		if (statusElement) {
			statusElement.textContent = "MALFUNCTION";
			statusElement.className = "hud-status unhealthy";
		}

		// Mark all sensors as malfunction
		sensorElements.forEach((sensorElement) => {
			sensorElement.textContent = "MALFUNCTION";
			sensorElement.className = "sensor-status sensor-malfunction";
		});
		azimuthElements.forEach((azimuthElement) => {
			azimuthElement.textContent = "Angles — ---";
			azimuthElement.className = "sensor-azimuth";
		});

		console.error("Health check failed:", error);
	}
}

// Radar data polling function (now using /tracks endpoint)
async function pollRadarData(): Promise<void> {
	try {
		const response: Response = await fetch(`${API_BASE}/tracks`);
		const data: TracksResponse = await response.json();

		// Update last data received timestamp
		lastDataReceived = Date.now();

		// Clear the current dots array
		radarDots.length = 0;

		// Track new detections for alerting
		const newTrackRadarIds: number[] = [];

		// Process each radar's track data
		const currentTrackIds = new Set<number>();
		for (const [radarId, trackData] of Object.entries(data)) {
			// Convert azimuth from degrees to radians
			const azimuthRad = (trackData.azimuth * Math.PI) / 180;

			// Radar azimuths use screen-style polar coordinates: 0° on the positive Y axis,
			// increasing clockwise. Convert that to cartesian space (X right, Y up) before
			// projecting to canvas coordinates.
			const x = trackData.range * Math.sin(azimuthRad);
			const y = trackData.range * Math.cos(azimuthRad);

			// Convert cartesian to canvas coordinates
			const canvasPos = cartesianToCanvas(x, y);

			// Extract radar ID number
			const radarIdNum = Number.parseInt(radarId.replace(/\D/g, ""), 10) || 0;

			// Create dot with all necessary data
			const dot: RadarDot = {
				track_id: trackData.track_id,
				radar_id: radarIdNum,
				x: x,
				y: y,
				canvasX: canvasPos.x,
				canvasY: canvasPos.y,
				range: trackData.range,
				azimuth: trackData.azimuth, // Store azimuth in degrees
				class: trackData.class_name ?? "unknown",
				timestamp: Date.now() / 1000, // Current timestamp in seconds
			};

			// Check if this is a new track
			if (!knownTrackIds.has(trackData.track_id)) {
				knownTrackIds.add(trackData.track_id);
				newTrackRadarIds.push(radarIdNum);
			}

			// Add to current dots
			radarDots.push(dot);
			currentTrackIds.add(trackData.track_id);

			// Update track history for trail effect
			if (!trackHistory.has(trackData.track_id)) {
				trackHistory.set(trackData.track_id, []);
			}
			const history = trackHistory.get(trackData.track_id);
			if (history) {
				history.push(dot);

				// Limit history size per track
				if (history.length > MAX_DOTS) {
					history.shift();
				}
			}
		}

		// Trigger alert for new tracks
		if (newTrackRadarIds.length > 0) {
			showTrackAlert(newTrackRadarIds);
		}

		// Clean up history and known tracks for tracks that are no longer present
		for (const [trackId, history] of trackHistory.entries()) {
			if (!currentTrackIds.has(trackId)) {
				// Track is gone, gradually remove from history
				if (history.length > 0) {
					history.shift();
					if (history.length === 0) {
						trackHistory.delete(trackId);
						knownTrackIds.delete(trackId); // Remove from known tracks so it can re-alert
					}
				}
			}
		}
	} catch (error) {
		console.error("Radar data fetch failed:", error);
		// Don't update lastDataReceived on error
	}
}

// Start health check polling
export function startHealthCheck(): void {
	checkHealth(); // Initial check
	setInterval(checkHealth, HEALTH_CHECK_INTERVAL);
}

// Start radar polling
export function startRadarPolling(): void {
	setInterval(pollRadarData, RADAR_CHECK_INTERVAL);
}
