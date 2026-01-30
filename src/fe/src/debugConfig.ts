// Debug configuration for tooltip display
export interface ConfigField {
	key: string;
	label: string;
	enabled: boolean;
	section: string;
}

export interface TooltipFieldConfig {
	show_tooltips: boolean;
	track_id: boolean;
	class: boolean;
	range: boolean;
	azimuth: boolean;
	timestamp: boolean;
}

// Default configuration - all fields enabled
const defaultConfig: TooltipFieldConfig = {
	show_tooltips: true,
	track_id: true,
	class: true,
	range: true,
	azimuth: true,
	timestamp: false,
};

// Current configuration state
let currentConfig: TooltipFieldConfig = { ...defaultConfig };

// Alerts configuration state
let disableAlerts = false;

// Get current configuration
export function getTooltipConfig(): TooltipFieldConfig {
	return { ...currentConfig };
}

// Update a specific field
export function setTooltipField(
	field: keyof TooltipFieldConfig,
	enabled: boolean,
): void {
	currentConfig[field] = enabled;
}

// Reset to defaults
export function resetTooltipConfig(): void {
	currentConfig = { ...defaultConfig };
}

// Get alerts disabled state
export function isAlertsDisabled(): boolean {
	return disableAlerts;
}

// Set alerts disabled state
export function setAlertsDisabled(disabled: boolean): void {
	disableAlerts = disabled;
}

// Get all available fields with metadata
export function getAvailableFields(): ConfigField[] {
	return [
		{
			key: "show_tooltips",
			label: "Show Tooltips",
			enabled: currentConfig.show_tooltips,
		},
		{ key: "track_id", label: "Track ID", enabled: currentConfig.track_id },
		{ key: "class", label: "Class", enabled: currentConfig.class },
		{ key: "range", label: "Range", enabled: currentConfig.range },
		{ key: "azimuth", label: "Azimuth", enabled: currentConfig.azimuth },
		{ key: "timestamp", label: "Timestamp", enabled: currentConfig.timestamp },
		{
			key: "disable_alerts",
			label: "Disable Alerts",
			enabled: disableAlerts,
		},
	];
}
