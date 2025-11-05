// Debug configuration for tooltip display
export interface TooltipField {
	key: keyof TooltipFieldConfig;
	label: string;
	enabled: boolean;
}

export interface TooltipFieldConfig {
	track_id: boolean;
	class: boolean;
	x: boolean;
	y: boolean;
	range: boolean;
	velocity: boolean;
	doppler: boolean;
	timestamp: boolean;
}

// Default configuration - all fields enabled
const defaultConfig: TooltipFieldConfig = {
	track_id: true,
	class: true,
	x: true,
	y: true,
	range: true,
	velocity: true,
	doppler: false,
	timestamp: false,
};

// Current configuration state
let currentConfig: TooltipFieldConfig = { ...defaultConfig };

// Get current configuration
export function getTooltipConfig(): TooltipFieldConfig {
	return { ...currentConfig };
}

// Update a specific field
export function setTooltipField(field: keyof TooltipFieldConfig, enabled: boolean): void {
	currentConfig[field] = enabled;
}

// Reset to defaults
export function resetTooltipConfig(): void {
	currentConfig = { ...defaultConfig };
}

// Get all available fields with metadata
export function getAvailableFields(): TooltipField[] {
	return [
		{ key: "track_id", label: "Track ID", enabled: currentConfig.track_id },
		{ key: "class", label: "Class", enabled: currentConfig.class },
		{ key: "x", label: "X Coordinate", enabled: currentConfig.x },
		{ key: "y", label: "Y Coordinate", enabled: currentConfig.y },
		{ key: "range", label: "Range", enabled: currentConfig.range },
		{ key: "velocity", label: "Velocity", enabled: currentConfig.velocity },
		{ key: "doppler", label: "Doppler", enabled: currentConfig.doppler },
		{ key: "timestamp", label: "Timestamp", enabled: currentConfig.timestamp },
	];
}
