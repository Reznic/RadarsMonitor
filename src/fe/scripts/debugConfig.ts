// Debug configuration for tooltip display
export interface TooltipField {
	key: keyof TooltipFieldConfig;
	label: string;
	enabled: boolean;
}

export interface TooltipFieldConfig {
	show_tooltips: boolean;
	track_id: boolean;
	class: boolean;
	x: boolean;
	y: boolean;
	range: boolean;
	azimuth: boolean;
	timestamp: boolean;
}

// Default configuration - all fields enabled
const defaultConfig: TooltipFieldConfig = {
	show_tooltips: true,
	track_id: true,
	class: true,
	x: true,
	y: true,
	range: true,
	azimuth: true,
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
		{ key: "show_tooltips", label: "Show Tooltips", enabled: currentConfig.show_tooltips },
		{ key: "track_id", label: "Track ID", enabled: currentConfig.track_id },
		{ key: "class", label: "Class", enabled: currentConfig.class },
		{ key: "x", label: "X Coordinate", enabled: currentConfig.x },
		{ key: "y", label: "Y Coordinate", enabled: currentConfig.y },
		{ key: "range", label: "Range", enabled: currentConfig.range },
		{ key: "azimuth", label: "Azimuth", enabled: currentConfig.azimuth },
		{ key: "timestamp", label: "Timestamp", enabled: currentConfig.timestamp },
	];
}
