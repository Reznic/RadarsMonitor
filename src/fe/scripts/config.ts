// API Configuration
export const API_BASE: string = "http://localhost:1337";

// Polling intervals
export const HEALTH_CHECK_INTERVAL: number = 1000; // 1 second
export const RADAR_CHECK_INTERVAL: number = 50; // 50 milliseconds (matching data sampling rate)

// Timeout settings
export const SERVER_TIMEOUT: number = 1000; // Clear dots if no data for 1 second

// Radar settings
export const MAX_DOTS: number = 200; // Maximum number of dots to display (10 seconds of history at 50ms intervals)
