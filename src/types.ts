// Radar Target Types
export interface RadarTarget {
	track_id: number;
	radar_id: number;
	timestamp: number;
	x: number;
	y: number;
	velocity: number;
	doppler: number;
	range: number;
	class: string;
}

// CSV Row Type - represents a single row from the CSV file
export interface CSVRow {
	track_id: number;
	radar_id: number;
	timestamp: number;
	x: number;
	y: number;
	doppler: number;
	range: number;
	class: string;
	[key: string]: string | number; // Allow additional fields
}

// Canvas Coordinates
export interface CanvasCoordinates {
	x: number;
	y: number;
}

// Radar Dot with canvas position (used in frontend)
export interface RadarDot extends RadarTarget {
	canvasX: number;
	canvasY: number;
}

// Playback Information
export interface PlaybackInfo {
	currentTimestamp: number;
	totalTimestamps: number;
	currentIndex: number;
	isPlaying: boolean;
}

// Sensor Status Types
export interface SensorStatus {
	id: number;
	name: string;
	healthy: boolean;
	lastUpdate?: number;
}

// API Response Types
export interface RadarResponse {
	targets: RadarTarget[];
}

export interface HealthResponse {
	sensors: SensorStatus[];
	overallHealthy: boolean;
}

// Configuration Types
export interface ServerConfig {
	PORT: number;
	CORS_HEADERS: Record<string, string>;
}

export interface FrontendConfig {
	API_BASE: string;
	HEALTH_CHECK_INTERVAL: number;
	RADAR_CHECK_INTERVAL: number;
	SERVER_TIMEOUT: number;
	MAX_DOTS: number;
}
