// API Configuration
export const API_BASE: string = "http://localhost:1337";

// Polling intervals
export const HEALTH_CHECK_INTERVAL: number = 1000; // 1 second
export const RADAR_CHECK_INTERVAL: number = 50; // 50 milliseconds (matching data sampling rate)

// Timeout settings
export const SERVER_TIMEOUT: number = 1000; // Clear dots if no data for 1 second

// Radar settings
// Lower history length to reduce per-frame drawing work (CPU/GPU)
// 100 dots - 5 seconds of history at 50ms intervals
export const MAX_DOTS: number = 100;

// Camera Configuration
export type CameraMode = "day" | "night";

export const STREAM_BASE_URL = "ws://127.0.0.1:8083";

export interface CameraConfig {
	id: number;
	name: string;
	streamId: string;
	radarSerial: string; // Backend radar ID (serial) associated with this camera
}

// Channel IDs: 0 = day, 1 = night
export function getCameraStreamUrl(
	camera: CameraConfig,
	mode: CameraMode,
): string {
	const channelId = mode === "day" ? 0 : 1;
	return `${STREAM_BASE_URL}/stream/${camera.streamId}/channel/${channelId}/mse?uuid=${camera.streamId}&channel=${channelId}`;
}

export const CAMERAS: CameraConfig[] = [
	{ id: 1, name: "Camera 1", streamId: "cam1", radarSerial: "00ED24D1" },
	{ id: 2, name: "Camera 2", streamId: "cam2", radarSerial: "00ED248C" },
	{ id: 3, name: "Camera 3", streamId: "cam3", radarSerial: "016C2377" },
	{ id: 4, name: "Camera 4", streamId: "cam4", radarSerial: "016A5BCC" },
	{ id: 5, name: "Camera 5", streamId: "cam5", radarSerial: "016C4AB6" },
	{ id: 6, name: "Camera 6", streamId: "cam6", radarSerial: "016A5874" },
	{ id: 7, name: "Camera 7", streamId: "cam7", radarSerial: "00D20CBB" },
	{ id: 8, name: "Camera 8", streamId: "cam8", radarSerial: "00D20CD7" },
];

// Helper to get camera ID by radar serial
export function getCameraIdByRadarSerial(radarSerial: string): number | undefined {
	const camera = CAMERAS.find((c) => c.radarSerial === radarSerial);
	return camera?.id;
}
