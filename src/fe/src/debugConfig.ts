// Debug configuration for tooltip display
export interface ConfigField {
	key: string;
	label: string;
	enabled: boolean;
	section: "tooltip" | "alerts" | "cameras";
}

export interface TooltipFieldConfig {
	show_tooltips: boolean;
	track_id: boolean;
	class: boolean;
	range: boolean;
	azimuth: boolean;
	timestamp: boolean;
}

export interface DebugConfigSnapshot extends TooltipFieldConfig {
	disable_alerts: boolean;
	show_camera_urls: boolean;
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
let showCameraUrls = false;
const listeners = new Set<(config: DebugConfigSnapshot) => void>();

function getDebugConfigSnapshot(): DebugConfigSnapshot {
	return {
		...currentConfig,
		disable_alerts: disableAlerts,
		show_camera_urls: showCameraUrls,
	};
}

function notifyListeners(): void {
	const snapshot = getDebugConfigSnapshot();
	for (const listener of listeners) {
		listener(snapshot);
	}
}

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
	notifyListeners();
}

// Reset to defaults
export function resetTooltipConfig(): void {
	currentConfig = { ...defaultConfig };
	notifyListeners();
}

// Get alerts disabled state
export function isAlertsDisabled(): boolean {
	return disableAlerts;
}

// Set alerts disabled state
export function setAlertsDisabled(disabled: boolean): void {
	disableAlerts = disabled;
	notifyListeners();
}

export function isCameraUrlsVisible(): boolean {
	return showCameraUrls;
}

export function setCameraUrlsVisible(visible: boolean): void {
	showCameraUrls = visible;
	notifyListeners();
}

export function subscribeDebugConfig(
	listener: (config: DebugConfigSnapshot) => void,
): () => void {
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
	};
}

// Get all available fields with metadata
export function getAvailableFields(): ConfigField[] {
	return [
		{
			key: "show_tooltips",
			label: "Show Tooltips",
			enabled: currentConfig.show_tooltips,
			section: "tooltip",
		},
		{
			key: "track_id",
			label: "Track ID",
			enabled: currentConfig.track_id,
			section: "tooltip",
		},
		{
			key: "class",
			label: "Class",
			enabled: currentConfig.class,
			section: "tooltip",
		},
		{
			key: "range",
			label: "Range",
			enabled: currentConfig.range,
			section: "tooltip",
		},
		{
			key: "azimuth",
			label: "Azimuth",
			enabled: currentConfig.azimuth,
			section: "tooltip",
		},
		{
			key: "timestamp",
			label: "Timestamp",
			enabled: currentConfig.timestamp,
			section: "tooltip",
		},
		{
			key: "disable_alerts",
			label: "Disable Alerts",
			enabled: disableAlerts,
			section: "alerts",
		},
		{
			key: "show_camera_urls",
			label: "Show Camera URLs",
			enabled: showCameraUrls,
			section: "cameras",
		},
	];
}
