// API Configuration
export const API_BASE: string = "http://localhost:1337";

// Polling intervals
export const HEALTH_CHECK_INTERVAL: number = 1000; // 1 second
export const RADAR_CHECK_INTERVAL: number = 50; // 50 milliseconds (matching data sampling rate)

// Timeout settings
export const SERVER_TIMEOUT: number = 1000; // Clear dots if no data for 1 second

// Radar settings
export const MAX_DOTS: number = 200; // Maximum number of dots to display (10 seconds of history at 50ms intervals)

// Camera Configuration
export type CameraMode = "day" | "night";

export const STREAM_BASE_URL = "http://127.0.0.1:8083";

export interface CameraConfig {
	id: number;
	name: string;
	streamId: string;
}

// Channel IDs: 0 = day, 1 = night
export function getCameraStreamUrl(
	camera: CameraConfig,
	mode: CameraMode,
): string {
	const channelId = mode === "day" ? 0 : 1;
	return `${STREAM_BASE_URL}/stream/${camera.streamId}/channel/${channelId}/hlsll/live/index.m3u8`;
}

export const CAMERAS: CameraConfig[] = [
	{ id: 1, name: "Camera 1", streamId: "cam1" },
	{ id: 2, name: "Camera 2", streamId: "cam2" },
	{ id: 3, name: "Camera 3", streamId: "cam3" },
	{ id: 4, name: "Camera 4", streamId: "cam4" },
	{ id: 5, name: "Camera 5", streamId: "cam5" },
	{ id: 6, name: "Camera 6", streamId: "cam6" },
	{ id: 7, name: "Camera 7", streamId: "cam7" },
	{ id: 8, name: "Camera 8", streamId: "cam8" },
];
