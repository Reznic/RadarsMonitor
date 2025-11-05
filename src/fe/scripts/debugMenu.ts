// Debug menu UI management
import { getAvailableFields, setTooltipField } from "./debugConfig.ts";

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

	for (const field of fields) {
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

// Handle field checkbox toggle
function handleFieldToggle(fieldKey: string, enabled: boolean): void {
	setTooltipField(fieldKey as any, enabled);
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
				setTooltipField(key as any, enabled as boolean);
				const checkbox = document.getElementById(`debug-field-${key}`) as HTMLInputElement;
				if (checkbox) {
					checkbox.checked = enabled as boolean;
				}
			}
		}
	} catch (e) {
		console.warn("Failed to load debug menu state", e);
	}
}
