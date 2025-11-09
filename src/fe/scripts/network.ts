import type { RadarDot, RadarsStatusResponse, TracksResponse } from "../../types.ts";
import {
	API_BASE,
	HEALTH_CHECK_INTERVAL,
	MAX_DOTS,
	RADAR_CHECK_INTERVAL,
	SERVER_TIMEOUT,
} from "./config.ts";
import { cartesianToCanvas } from "./radar.ts";

// DOM elements
let statusElement: HTMLElement | null;
const sensorElements: Map<number, HTMLElement> = new Map();

// State
export const radarDots: RadarDot[] = [];
export let lastDataReceived: number = Date.now();

// Track history for trail effect
export const trackHistory: Map<number, RadarDot[]> = new Map(); // track_id -> array of positions

// Initialize DOM references
export function initNetworkDOM(): void {
	statusElement = document.getElementById("status");

	// Initialize sensor element references (for radar status display)
	for (let i = 1; i <= 4; i++) {
		const element = document.getElementById(`sensor${i}`);
		if (element) {
			sensorElements.set(i, element);
		}
	}

	// Initialize radar control button event listeners
	initRadarControls();
}

// Initialize radar control buttons
function initRadarControls(): void {
	const buttons = document.querySelectorAll('.sensor-btn');
	buttons.forEach(button => {
		button.addEventListener('click', async (e) => {
			const target = e.target as HTMLButtonElement;
			const radarId = target.getAttribute('data-radar-id');
			const action = target.getAttribute('data-action');

			if (!radarId || !action) return;

			// Disable button during request
			target.disabled = true;

			try {
				if (action === 'on') {
					await turnRadarOn(radarId);
				} else if (action === 'off') {
					await turnRadarOff(radarId);
				}
				// Immediately check health after action
				await checkHealth();
			} catch (error) {
				console.error(`Failed to ${action} radar ${radarId}:`, error);
			} finally {
				// Re-enable button
				target.disabled = false;
			}
		});
	});
}

// Turn radar on
async function turnRadarOn(radarId: string): Promise<void> {
	try {
		const response = await fetch(`${API_BASE}/radar/on`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
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
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
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

		// Count active and inactive radars
		const radarIds = Object.keys(data);
		const inactiveRadars = radarIds.filter(id => {
			const status = data[id];
			return status && !status.is_active;
		});

		// Update overall status
		if (statusElement) {
			if (inactiveRadars.length === 0 && radarIds.length > 0) {
				statusElement.textContent = "ALL RADARS OK";
				statusElement.className = "hud-status healthy";
			} else if (radarIds.length === 0) {
				statusElement.textContent = "NO RADARS";
				statusElement.className = "hud-status unhealthy";
			} else {
				statusElement.textContent = `${inactiveRadars.length} RADAR${inactiveRadars.length > 1 ? "S" : ""} DOWN`;
				statusElement.className = "hud-status unhealthy";
			}
		}

		// Track which radars we've displayed
		const displayedRadarIds = new Set<string>();

		// Update individual radar statuses (map radars to sensor display slots)
		let sensorIndex = 1;
		for (const radarId of radarIds.slice(0, 4)) { // Only show first 4 radars
			const radarStatus = data[radarId];
			const sensorElement = sensorElements.get(sensorIndex);
			if (sensorElement && radarStatus) {
				displayedRadarIds.add(radarId);
				const radarNumber = radarId.replace(/\D/g, ''); // Extract number from radar ID

				if (radarStatus.is_active) {
					sensorElement.textContent = `${radarNumber}: ON`;
					sensorElement.className = "sensor-status sensor-ok";
				} else {
					sensorElement.textContent = `${radarNumber}: OFF`;
					sensorElement.className = "sensor-status sensor-error";
				}
			}
			sensorIndex++;
		}

		// Clear remaining sensor slots or mark as unavailable
		for (let i = sensorIndex; i <= 4; i++) {
			const sensorElement = sensorElements.get(i);
			if (sensorElement) {
				sensorElement.textContent = "---";
				sensorElement.className = "sensor-status";
			}
		}
	} catch (error) {
		// Server is unreachable - mark as malfunction
		if (statusElement) {
			statusElement.textContent = "MALFUNCTION";
			statusElement.className = "hud-status unhealthy";
		}

		// Mark all sensors as malfunction
		for (let i = 1; i <= 4; i++) {
			const sensorElement = sensorElements.get(i);
			if (sensorElement) {
				sensorElement.textContent = "MALFUNCTION";
				sensorElement.className = "sensor-status sensor-malfunction";
			}
		}

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

		// Process each radar's track data
		const currentTrackIds = new Set<number>();
		for (const [radarId, trackData] of Object.entries(data)) {
			// Convert azimuth from degrees to radians
			const azimuthRad = (trackData.azimuth * Math.PI) / 180;

			// Convert polar coordinates (azimuth, range) to cartesian (x, y)
			// Note: azimuth 0° is typically North (pointing up), increasing clockwise
			// For standard math: 0° is East (right), increasing counter-clockwise
			// We need to adjust: math_angle = 90° - azimuth
			const mathAngleRad = (Math.PI / 2) - azimuthRad;
			const x = trackData.range * Math.cos(mathAngleRad);
			const y = trackData.range * Math.sin(mathAngleRad);

			// Convert cartesian to canvas coordinates
			const canvasPos = cartesianToCanvas(x, y, trackData.range);

			// Create dot with all necessary data
			const dot: RadarDot = {
				track_id: trackData.track_id,
				radar_id: Number.parseInt(radarId.replace(/\D/g, ''), 10) || 0, // Extract number from radar ID
				x: x,
				y: y,
				canvasX: canvasPos.x,
				canvasY: canvasPos.y,
				range: trackData.range,
				azimuth: trackData.azimuth, // Store azimuth in degrees
				class: "unknown", // Not provided by Python server
				timestamp: Date.now() / 1000, // Current timestamp in seconds
			};

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

		// Clean up history for tracks that are no longer present
		for (const [trackId, history] of trackHistory.entries()) {
			if (!currentTrackIds.has(trackId)) {
				// Track is gone, gradually remove from history
				if (history.length > 0) {
					history.shift();
					if (history.length === 0) {
						trackHistory.delete(trackId);
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
