let alertOverlay: HTMLElement | null = null;
let alertGrid: HTMLElement | null = null;
let alertCloseButton: HTMLElement | null = null;

// Track which radar IDs have active alerts
const activeAlerts: Set<number> = new Set();
const alertCards: Map<number, HTMLElement> = new Map();

export interface TrackAlert {
	cameraId: number;
	threatAzimuthDeg?: number;
}

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
export function showTrackAlert(alerts: TrackAlert[]): void {
	if (!alertOverlay || !alertGrid) return;

	const newAlerts = alerts.filter((alert) => {
		return !activeAlerts.has(alert.cameraId);
	});
	if (newAlerts.length === 0) return;

	for (const alert of newAlerts) {
		activeAlerts.add(alert.cameraId);
		addAlertCard(alert.cameraId, alert.threatAzimuthDeg ?? null);
	}

	// Show overlay
	alertOverlay.classList.remove("hidden");
}

function addAlertCard(cameraId: number, threatAzimuthDeg: number | null): void {
	if (!alertGrid) return;
	if (alertCards.has(cameraId)) return;

	const el = document.createElement("radars-camera-feed");
	el.setAttribute("camera-id", String(cameraId));
	el.setAttribute("variant", "alert");
	if (threatAzimuthDeg !== null) {
		el.setAttribute("threat-azimuth", String(threatAzimuthDeg));
	}

	const badge = document.createElement("span");
	badge.setAttribute("slot", "header-right");
	badge.className = "track-alert-camera-radar";
	badge.textContent = `RADAR ${cameraId}`;

	const dismiss = document.createElement("button");
	dismiss.setAttribute("slot", "overlay-top-right");
	dismiss.className = "track-alert-dismiss";
	dismiss.setAttribute("data-radar-id", String(cameraId));
	dismiss.textContent = "✕";

	el.appendChild(badge);
	el.appendChild(dismiss);

	alertCards.set(cameraId, el);
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
