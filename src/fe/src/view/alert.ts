let alertOverlay: HTMLElement | null = null;
let alertGrid: HTMLElement | null = null;
let alertCloseButton: HTMLElement | null = null;

// Track which radar IDs have active alerts
const activeAlerts: Set<number> = new Set();
const alertCards: Map<number, HTMLElement> = new Map();

export function initAlertView(): void {
	alertOverlay = document.getElementById("trackAlertOverlay");
	alertGrid = document.getElementById("trackAlertGrid");
	alertCloseButton = document.getElementById("trackAlertClose");

	if (alertCloseButton) {
		alertCloseButton.addEventListener("click", dismissAllAlerts);
	}

	if (alertGrid) {
		// Handle dismiss buttons via event delegation
		alertGrid.addEventListener("click", (event) => {
			const target = event.target as HTMLElement;
			if (!target.classList.contains("track-alert-dismiss")) return;

			const radarId = Number.parseInt(
				target.getAttribute("data-radar-id") || "0",
				10,
			);
			dismissAlert(radarId);
		});
	}
}

// Show alert for new tracks from specific radars
export function showTrackAlert(radarIds: number[]): void {
	if (!alertOverlay || !alertGrid) return;

	// Filter out radars that already have alerts
	const newRadarIds = radarIds.filter((id) => !activeAlerts.has(id));
	if (newRadarIds.length === 0) return;

	for (const id of newRadarIds) {
		activeAlerts.add(id);
		addAlertCard(id);
	}

	// Show overlay
	alertOverlay.classList.remove("hidden");
}

function addAlertCard(radarId: number): void {
	if (!alertGrid) return;
	if (alertCards.has(radarId)) return;

	const el = document.createElement("radars-camera-feed");
	el.setAttribute("camera-id", String(radarId));
	el.setAttribute("variant", "alert");

	const badge = document.createElement("span");
	badge.setAttribute("slot", "header-right");
	badge.className = "track-alert-camera-radar";
	badge.textContent = `RADAR ${radarId}`;

	const dismiss = document.createElement("button");
	dismiss.setAttribute("slot", "overlay-top-right");
	dismiss.className = "track-alert-dismiss";
	dismiss.setAttribute("data-radar-id", String(radarId));
	dismiss.textContent = "✕";

	el.appendChild(badge);
	el.appendChild(dismiss);

	alertCards.set(radarId, el);
	alertGrid.appendChild(el);
}

function dismissAlert(radarId: number): void {
	activeAlerts.delete(radarId);
	const card = alertCards.get(radarId);
	if (card) {
		card.remove();
		alertCards.delete(radarId);
	}

	if (activeAlerts.size === 0) {
		dismissAllAlerts();
	}
}

function dismissAllAlerts(): void {
	activeAlerts.clear();
	alertCards.clear();

	if (alertOverlay) {
		alertOverlay.classList.add("hidden");
	}
	if (alertGrid) {
		alertGrid.innerHTML = "";
	}
}

// Check if alert overlay is currently visible
export function isAlertVisible(): boolean {
	return alertOverlay ? !alertOverlay.classList.contains("hidden") : false;
}
