import { CAMERAS, type CameraMode, getCameraStreamUrl } from "../config.ts";
import { createSharedWhepManager } from "../stream/sharedWebrtc.ts";
import {
	ensureCameraMode,
	setCameraMode,
	subscribeCameraMode,
} from "./cameraModeStore.ts";

const whepManager = createSharedWhepManager();

type Variant = "gallery" | "alert";
type Status = "online" | "offline";

function parseCameraId(value: string | null): number | null {
	if (!value) return null;
	const id = Number.parseInt(value, 10);
	return Number.isFinite(id) ? id : null;
}

function parseMode(value: string | null): CameraMode {
	return value === "night" ? "night" : "day";
}

function parseVariant(value: string | null): Variant {
	return value === "alert" ? "alert" : "gallery";
}

function parseStatus(value: string | null): Status {
	return value === "online" ? "online" : "offline";
}

export class RadarsCameraFeedElement extends HTMLElement {
	static observedAttributes = [
		"camera-id",
		"mode",
		"variant",
		"name",
		"status",
	];

	#detachStream: (() => void) | null = null;
	#attachedUrl: string | null = null;
	#unsubscribeMode: (() => void) | null = null;
	#subscribedCameraId: number | null = null;
	#suppressModeStoreUpdate = false;
	#video: HTMLVideoElement | null = null;
	#nameEl: HTMLElement | null = null;
	#statusEl: HTMLElement | null = null;
	#ipEl: HTMLElement | null = null;
	#modeToggle: HTMLButtonElement | null = null;

	connectedCallback(): void {
		this.#ensureShadowDom();
		this.#ensureModeSync();
		this.#syncUi();
		this.#syncStream();
	}

	disconnectedCallback(): void {
		this.#detach();
		this.#unsubscribeFromModeStore();
	}

	attributeChangedCallback(): void {
		if (!this.isConnected) return;
		this.#ensureModeSync();
		this.#syncUi();
		this.#syncStream();
	}

	#ensureShadowDom(): void {
		if (this.shadowRoot) return;

		const shadow = this.attachShadow({ mode: "open" });
		shadow.innerHTML = `
      <style>
        :host {
          position: relative;
          display: flex;
          flex-direction: column;
          width: 100%;
          height: 100%;
          background: rgba(10, 10, 10, 0.9);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 8px;
          overflow: hidden;
          box-sizing: border-box;
          font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        }

        :host([variant="alert"]) {
          background: rgba(10, 10, 10, 0.95);
          border: 4px solid #ff0000;
          border-radius: 12px;
          min-height: 300px;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.6);
          border-bottom: 1px solid rgba(255, 255, 255, 0.2);
          gap: 10px;
        }

        :host([variant="alert"]) .header {
          padding: 12px 16px;
          background: rgba(255, 0, 0, 0.2);
          border-bottom: 2px solid #ff0000;
        }

        .name {
          font-size: 12px;
          font-weight: 600;
          letter-spacing: 1px;
          color: rgba(255, 255, 255, 0.9);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        :host([variant="alert"]) .name {
          font-size: 18px;
          font-weight: bold;
          letter-spacing: 2px;
          color: #ff0000;
        }

        .header-right {
          display: flex;
          align-items: center;
          justify-content: flex-end;
          gap: 10px;
          min-width: 0;
        }

        .status {
          font-size: 10px;
          font-weight: bold;
          padding: 2px 6px;
          border-radius: 3px;
          user-select: none;
        }

        .status.online {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.15);
        }

        .status.offline {
          color: #666666;
          background: rgba(102, 102, 102, 0.15);
        }

        .mode-toggle {
          display: flex;
          padding: 0;
          font-size: 10px;
          font-weight: bold;
          letter-spacing: 0.5px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 4px;
          cursor: pointer;
          overflow: hidden;
          font-family: inherit;
        }

        .mode-toggle:hover {
          border-color: rgba(255, 255, 255, 0.4);
        }

        .mode-day,
        .mode-night {
          padding: 3px 6px;
          color: rgba(255, 255, 255, 0.5);
          background: transparent;
          user-select: none;
        }

        .mode-toggle:not(.night) .mode-day {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.2);
        }

        .mode-toggle.night .mode-night {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.2);
        }

        .feed {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #000;
          position: relative;
          min-height: 120px;
        }

        :host([variant="alert"]) .feed {
          min-height: 300px;
        }

        video {
          width: 100%;
          height: 100%;
          object-fit: contain;
          position: absolute;
          top: 0;
          left: 0;
          z-index: 1;
          background: #000;
        }

        .placeholder {
          color: rgba(255, 255, 255, 0.3);
          font-size: 12px;
          letter-spacing: 1px;
          text-align: center;
          padding: 10px;
          z-index: 0;
        }

        .placeholder-icon {
          font-size: 48px;
          margin-bottom: 8px;
          opacity: 0.3;
        }

        .ip {
          font-size: 10px;
          color: rgba(255, 255, 255, 0.4);
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.5);
          text-align: center;
          word-break: break-all;
        }

        :host([variant="alert"]) .ip {
          color: rgba(255, 255, 255, 0.5);
        }

        .overlay-slot {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 10;
        }

        .overlay-slot ::slotted(*) {
          pointer-events: auto;
        }
      </style>

      <div class="header">
        <span class="name"></span>
        <div class="header-right">
          <button class="mode-toggle" type="button">
            <span class="mode-day">DAY</span>
            <span class="mode-night">NIGHT</span>
          </button>
          <slot name="header-right"></slot>
          <span class="status offline">OFFLINE</span>
        </div>
      </div>

      <div class="feed">
        <video autoplay muted playsinline></video>
        <div class="placeholder">
          <div class="placeholder-icon">📷</div>
          <div>No Signal</div>
          <div class="placeholder-ip"></div>
        </div>
        <div class="overlay-slot">
          <slot name="overlay-top-right"></slot>
        </div>
      </div>

      <div class="ip"></div>
    `;

		this.#video = shadow.querySelector("video");
		this.#nameEl = shadow.querySelector(".name");
		this.#statusEl = shadow.querySelector(".status");
		this.#ipEl = shadow.querySelector(".ip");
		this.#modeToggle = shadow.querySelector(".mode-toggle");

		this.#modeToggle?.addEventListener("click", () => {
			const cameraId = this.cameraId;
			if (cameraId === null) return;

			const currentMode = this.mode;
			const nextMode: CameraMode = currentMode === "day" ? "night" : "day";
			setCameraMode(cameraId, nextMode);

			this.dispatchEvent(
				new CustomEvent("modechange", {
					detail: { cameraId, mode: nextMode },
					bubbles: true,
					composed: true,
				}),
			);
		});
	}

	get cameraId(): number | null {
		return parseCameraId(this.getAttribute("camera-id"));
	}

	get mode(): CameraMode {
		return parseMode(this.getAttribute("mode"));
	}

	set mode(value: CameraMode) {
		this.setAttribute("mode", value);
	}

	get variant(): Variant {
		return parseVariant(this.getAttribute("variant"));
	}

	get status(): Status {
		return parseStatus(this.getAttribute("status"));
	}

	#unsubscribeFromModeStore(): void {
		if (this.#unsubscribeMode) {
			this.#unsubscribeMode();
			this.#unsubscribeMode = null;
		}
		this.#subscribedCameraId = null;
	}

	#ensureModeSync(): void {
		const cameraId = this.cameraId;
		if (cameraId === null) {
			this.#unsubscribeFromModeStore();
			return;
		}

		if (this.#subscribedCameraId !== cameraId) {
			this.#unsubscribeFromModeStore();
			this.#subscribedCameraId = cameraId;

			this.#unsubscribeMode = subscribeCameraMode(cameraId, (mode) => {
				if (this.mode === mode) return;
				this.#suppressModeStoreUpdate = true;
				this.mode = mode;
				this.#suppressModeStoreUpdate = false;
			});
		}

		// If the element has an explicit mode, make it the source of truth.
		// Otherwise, adopt the stored mode (defaulting to "day") and set attribute.
		if (this.hasAttribute("mode")) {
			if (!this.#suppressModeStoreUpdate) {
				setCameraMode(cameraId, this.mode);
			}
		} else {
			this.#suppressModeStoreUpdate = true;
			this.mode = ensureCameraMode(cameraId);
			this.#suppressModeStoreUpdate = false;
		}
	}

	#syncUi(): void {
		const cameraId = this.cameraId;
		const mode = this.mode;
		const variant = this.variant;

		if (!this.hasAttribute("variant")) {
			this.setAttribute("variant", variant);
		}

		const camera =
			cameraId === null ? undefined : CAMERAS.find((c) => c.id === cameraId);
		const displayName =
			this.getAttribute("name") ??
			camera?.name ??
			(cameraId !== null ? `CAMERA ${cameraId}` : "CAMERA");

		const streamUrl =
			camera && cameraId !== null ? getCameraStreamUrl(camera, mode) : "";

		if (this.#nameEl) this.#nameEl.textContent = displayName;
		if (this.#ipEl) this.#ipEl.textContent = streamUrl;

		const placeholderIp = this.shadowRoot?.querySelector(".placeholder-ip");
		if (placeholderIp) placeholderIp.textContent = streamUrl;

		if (this.#modeToggle) {
			this.#modeToggle.classList.toggle("night", mode === "night");
		}

		if (this.#statusEl) {
			const status = this.status;
			this.#statusEl.classList.toggle("online", status === "online");
			this.#statusEl.classList.toggle("offline", status !== "online");
			this.#statusEl.textContent = status === "online" ? "ONLINE" : "OFFLINE";
		}
	}

	#syncStream(): void {
		const cameraId = this.cameraId;
		const mode = this.mode;
		const camera =
			cameraId === null ? undefined : CAMERAS.find((c) => c.id === cameraId);
		const url = camera ? getCameraStreamUrl(camera, mode) : null;

		if (!url || !this.#video) {
			this.#detach();
			return;
		}

		if (this.#attachedUrl === url) return;

		this.#detach();
		this.#detachStream = whepManager.attach(this.#video, url);
		this.#attachedUrl = url;
	}

	#detach(): void {
		if (this.#detachStream) {
			this.#detachStream();
			this.#detachStream = null;
		}
		this.#attachedUrl = null;
		if (this.#video) {
			this.#video.srcObject = null;
		}
	}
}

export function defineRadarsCameraFeed(
	registry: CustomElementRegistry | undefined = globalThis.customElements,
): void {
	if (!registry) return;
	if (registry.get("radars-camera-feed")) return;
	registry.define("radars-camera-feed", RadarsCameraFeedElement);
}

defineRadarsCameraFeed();
