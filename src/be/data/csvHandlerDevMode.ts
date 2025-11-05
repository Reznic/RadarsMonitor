/**
 * Development mode handler for CSV data
 *
 * In dev mode, the handler:
 * - Loads all data at once
 * - Loops through timestamps automatically
 * - Simulates real-time data by advancing through timestamps
 */

import type { CSVRow, PlaybackInfo, RadarTarget } from "../../types.ts";

// Dev mode playback state
let csvData: CSVRow[] = [];
let timestamps: number[] = [];
let currentTimestampIndex: number = 0;
let isPlaying: boolean = false;
let playbackInterval: Timer | null = null;

// Playback configuration
export const PLAYBACK_INTERVAL_MS = 50;

/**
 * Load and initialize dev mode with all CSV data
 */
export function initDevMode(allRows: CSVRow[]): {
	timestamps: number[];
	success: boolean;
} {
	csvData = allRows;

	// Extract unique timestamps and sort them
	const uniqueTimestamps = new Set(csvData.map((row) => row.timestamp));
	timestamps = Array.from(uniqueTimestamps).sort((a, b) => a - b);

	console.log(
		`✅ [DEV MODE] Loaded ${csvData.length} tracks with ${timestamps.length} unique timestamps`,
	);

	// Auto-start playback loop
	startPlayback();

	return { timestamps, success: true };
}

/**
 * Start automatic playback through timestamps
 */
function startPlayback(): void {
	if (isPlaying) return;

	isPlaying = true;
	console.log("▶️  Starting CSV playback");

	// Advance timestamp at configured interval
	playbackInterval = setInterval(() => {
		currentTimestampIndex = (currentTimestampIndex + 1) % timestamps.length;

		// Log when we loop back
		if (currentTimestampIndex === 0) {
			console.log("🔄 Playback loop - restarting from beginning");
		}
	}, PLAYBACK_INTERVAL_MS);
}

/**
 * Stop playback and cleanup
 */
export function stopDevModePlayback(): void {
	if (playbackInterval) {
		clearInterval(playbackInterval);
		playbackInterval = null;
	}
	isPlaying = false;
	console.log("⏸️  Stopped CSV playback");
}

/**
 * Get targets for current timestamp
 */
export function getDevModeTargets(): RadarTarget[] {
	if (timestamps.length === 0) {
		return [];
	}

	const currentTimestamp = timestamps[currentTimestampIndex];

	// Get all tracks at current timestamp
	const targets = csvData
		.filter((row) => row.timestamp === currentTimestamp)
		.map((row) => ({
			track_id: row.track_id,
			radar_id: row["Radar id"] as number, // CSV column is "Radar id" with space
			timestamp: row.timestamp,
			x: row.x,
			y: row.y,
			velocity: Math.abs(row.doppler), // Use doppler as velocity
			doppler: row.doppler,
			range: row.range,
			class: row.class,
		}));

	return targets;
}

/**
 * Get playback info for dev mode
 */
export function getDevModePlaybackInfo(): PlaybackInfo {
	return {
		currentTimestamp: timestamps[currentTimestampIndex] ?? 0,
		totalTimestamps: timestamps.length,
		currentIndex: currentTimestampIndex,
		isPlaying,
	};
}

/**
 * Get dev mode state (for testing)
 */
export function getDevModeState() {
	return {
		csvData: [...csvData],
		timestamps: [...timestamps],
		currentTimestampIndex,
		isPlaying,
	};
}

/**
 * Reset dev mode state (for testing)
 */
export function resetDevModeState(): void {
	csvData = [];
	timestamps = [];
	currentTimestampIndex = 0;
	isPlaying = false;

	if (playbackInterval) {
		clearInterval(playbackInterval);
		playbackInterval = null;
	}
}
