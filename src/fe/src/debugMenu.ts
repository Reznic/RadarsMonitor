// Debug menu UI management
import {
	type ConfigField,
	getAvailableFields,
	setAlertsDisabled,
	setTooltipField,
	type TooltipFieldConfig,
} from "./debugConfig.ts";

let debugMenu: HTMLElement;
let debugToggle: HTMLButtonElement;
let debugContent: HTMLElement;
let debugFields: HTMLElement;

// Initialize debug menu DOM elements and event listeners
export function initDebugMenu(): void {
	debugMenu = document.getElementById("debugMenu") as HTMLElement;
	debugToggle = document.getElementById("debugToggle") as HTMLButtonElement;
	debugContent = document.getElementById("debugContent") as HTMLElement;
	debugFields = document.getElementById("debugFields") as HTMLElement;

	if (!debugMenu || !debugToggle || !debugContent || !debugFields) {
		console.error("Debug menu elements not found");
		return;
	}

	// Set up collapse/expand functionality
	const debugHeader = debugMenu.querySelector(".debug-header") as HTMLElement;
	debugHeader?.addEventListener("click", toggleDebugMenu);

	// Generate field checkboxes
	renderFieldCheckboxes();

	// Load saved state from localStorage
	loadDebugMenuState();
}

// Toggle debug menu collapse/expand
function toggleDebugMenu(): void {
	const isCollapsed = debugMenu.classList.toggle("collapsed");
	saveCollapseState(isCollapsed);
}

// Render checkboxes for all available fields
function renderFieldCheckboxes(): void {
	const fields = getAvailableFields();

	debugFields.innerHTML = "";

	// Group fields by section
	const sections = new Map<string, ConfigField[]>();
	for (const field of fields) {
		if (!sections.has(field.section)) {
			sections.set(field.section, []);
		}
		sections.get(field.section)?.push(field);
	}

	// Section title mapping
	const sectionTitles: Record<string, string> = {
		tooltip: "Tooltip Settings",
		alerts: "System Settings",
	};

	// Render each section
	for (const [sectionKey, sectionFields] of sections) {
		// Add section header
		const sectionTitle = document.createElement("div");
		sectionTitle.className = "debug-section-title";
		sectionTitle.textContent = sectionTitles[sectionKey] || sectionKey;
		debugFields.appendChild(sectionTitle);

		// Add fields in this section
		for (const field of sectionFields) {
			const fieldDiv = document.createElement("div");
			fieldDiv.className = "debug-field";

			const checkbox = document.createElement("input");
			checkbox.type = "checkbox";
			checkbox.id = `debug-field-${field.key}`;
			checkbox.checked = field.enabled;
			checkbox.addEventListener("change", () => {
				handleFieldToggle(field.key, checkbox.checked);
			});

			const label = document.createElement("label");
			label.htmlFor = `debug-field-${field.key}`;
			label.textContent = field.label;

			fieldDiv.appendChild(checkbox);
			fieldDiv.appendChild(label);
			debugFields.appendChild(fieldDiv);
		}
	}
}

// Handle field checkbox toggle
function handleFieldToggle(fieldKey: string, enabled: boolean): void {
	if (fieldKey === "disable_alerts") {
		setAlertsDisabled(enabled);
	} else {
		setTooltipField(fieldKey as keyof TooltipFieldConfig, enabled);
	}
	saveFieldState(fieldKey, enabled);
}

// Save collapse state to localStorage
function saveCollapseState(collapsed: boolean): void {
	try {
		localStorage.setItem("debug-menu-collapsed", JSON.stringify(collapsed));
	} catch (e) {
		console.warn("Failed to save debug menu collapse state", e);
	}
}

// Save field state to localStorage
function saveFieldState(fieldKey: string, enabled: boolean): void {
	try {
		const saved = localStorage.getItem("debug-menu-fields");
		const fields = saved ? JSON.parse(saved) : {};
		fields[fieldKey] = enabled;
		localStorage.setItem("debug-menu-fields", JSON.stringify(fields));
	} catch (e) {
		console.warn("Failed to save field state", e);
	}
}

// Load debug menu state from localStorage
function loadDebugMenuState(): void {
	try {
		// Load collapse state
		const collapsedStr = localStorage.getItem("debug-menu-collapsed");
		if (collapsedStr !== null) {
			const collapsed = JSON.parse(collapsedStr);
			if (collapsed) {
				debugMenu.classList.add("collapsed");
			}
		}

		// Load field states
		const fieldsStr = localStorage.getItem("debug-menu-fields");
		if (fieldsStr !== null) {
			const fields = JSON.parse(fieldsStr);
			for (const [key, enabled] of Object.entries(fields)) {
				if (key === "disable_alerts") {
					setAlertsDisabled(Boolean(enabled));
				} else {
					setTooltipField(key as keyof TooltipFieldConfig, Boolean(enabled));
				}
				const checkbox = document.getElementById(
					`debug-field-${key}`,
				) as HTMLInputElement;
				if (checkbox) {
					checkbox.checked = enabled as boolean;
				}
			}
		}
	} catch (e) {
		console.warn("Failed to load debug menu state", e);
	}
}
