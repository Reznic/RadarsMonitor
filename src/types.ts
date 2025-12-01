// Canvas Coordinates
export interface CanvasCoordinates {
	x: number;
	y: number;
}

// Radar Dot with canvas position (used in frontend)
export interface RadarDot extends TrackData {
	radar_id: number;
	timestamp: number;
	x: number;
	y: number;
	class: string;
	canvasX: number;
	canvasY: number;
}

// Python Server API Types
export interface TrackData {
	track_id: number;
	azimuth: number; // in degrees
	range: number; // in meters
	class_name?: string;
}

export interface RadarStatus {
	is_active: boolean;
	orientation_angle: number; // in degrees
}

// API Response Types (from Python server)
export interface TracksResponse {
	[radar_id: string]: TrackData;
}

export interface RadarsStatusResponse {
	[radar_id: string]: RadarStatus;
}
