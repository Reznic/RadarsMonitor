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

// Mapping from backend radar IDs (serials) to camera IDs (1-8).
// This is used so that when a specific radar reports a track,
// the alert overlay knows which camera stream to show.
// NOTE: Adjust these mappings to match your actual radar serials - cameras.
export const RADAR_TO_CAMERA_ID: Record<string, number> = {
	"00ED24D1": 1,
	"00ED248C": 2,
	"016C2377": 3,
	"016A5BCC": 4,
	"016C4AB6": 5,
	"016A5874": 6,
	"00D20CBB": 7,
	"00D20CD7": 8,
};

// Camera Configuration
export type CameraMode = "day" | "night";

export const STREAM_BASE_URL = "ws://127.0.0.1:8083";

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
	return `${STREAM_BASE_URL}/stream/${camera.streamId}/channel/${channelId}/mse?uuid=${camera.streamId}&channel=${channelId}`;
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
