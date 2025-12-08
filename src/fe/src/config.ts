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
export interface CameraConfig {
	id: number;
	name: string;
	ip: string;
	port: number;
}

export const CAMERAS: CameraConfig[] = [
	{ id: 1, name: "Camera 1", ip: "192.168.1.101", port: 554 },
	{ id: 2, name: "Camera 2", ip: "192.168.1.102", port: 554 },
	{ id: 3, name: "Camera 3", ip: "192.168.1.103", port: 554 },
	{ id: 4, name: "Camera 4", ip: "192.168.1.104", port: 554 },
	{ id: 5, name: "Camera 5", ip: "192.168.1.105", port: 554 },
	{ id: 6, name: "Camera 6", ip: "192.168.1.106", port: 554 },
	{ id: 7, name: "Camera 7", ip: "192.168.1.107", port: 554 },
	{ id: 8, name: "Camera 8", ip: "192.168.1.108", port: 554 },
];
