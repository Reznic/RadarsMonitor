import { CAMERAS, type CameraMode, getCameraStreamUrl } from "../config.ts";
import { isCameraUrlsVisible, subscribeDebugConfig } from "../debugConfig.ts";
import {
	getLocalizedCameraName,
	subscribeLanguageChange,
	t,
} from "../i18n/index.ts";
import {
	ensureCameraMode,
	setCameraMode,
	subscribeCameraMode,
} from "../store/cameraModeStore.ts";
import { createSharedWhepManager } from "../stream/sharedWebrtc.ts";
import {
	azimuthDegToCardinal8,
	normalizeAzimuthDeg,
} from "../view/threatDirection.ts";
import { configureCameraFeedVideo } from "./cameraFeedVideo.ts";

const whepManager = createSharedWhepManager();

type Variant = "gallery" | "alert";

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

function parseThreatAzimuth(value: string | null): number | null {
	if (value === null) return null;
	const trimmed = value.trim();
	if (trimmed === "") return null;
	const parsed = Number.parseFloat(trimmed);
	return Number.isFinite(parsed) ? normalizeAzimuthDeg(parsed) : null;
}

export class RadarsCameraFeedElement extends HTMLElement {
	static observedAttributes = [
		"camera-id",
		"mode",
		"variant",
		"name",
		"threat-azimuth",
	];

	#detachStream: (() => void) | null = null;
	#attachedUrl: string | null = null;
	#unsubscribeMode: (() => void) | null = null;
	#subscribedCameraId: number | null = null;
	#suppressModeStoreUpdate = false;
	#video: HTMLVideoElement | null = null;
	#nameEl: HTMLElement | null = null;
	#ipEl: HTMLElement | null = null;
	#modeToggle: HTMLButtonElement | null = null;
	#placeholderIpEl: HTMLElement | null = null;
	#placeholderStatusEl: HTMLElement | null = null;
	#threatLabelEl: HTMLElement | null = null;
	#fullscreenBtn: HTMLButtonElement | null = null;
	#iconExpand: SVGElement | null = null;
	#iconExit: SVGElement | null = null;
	#unsubscribeDebugConfig: (() => void) | null = null;
	#unsubscribeLanguage: (() => void) | null = null;

	connectedCallback(): void {
		this.#ensureShadowDom();
		this.#ensureModeSync();
		if (!this.#unsubscribeDebugConfig) {
			this.#unsubscribeDebugConfig = subscribeDebugConfig(() => {
				this.#syncUrlVisibility();
			});
		}
		if (!this.#unsubscribeLanguage) {
			this.#unsubscribeLanguage = subscribeLanguageChange(() => {
				this.#syncUi();
				this.#onFullscreenChange();
			});
		}
		this.#syncUi();
		this.#syncStream();
	}

	disconnectedCallback(): void {
		document.removeEventListener("fullscreenchange", this.#onFullscreenChange);
		this.#detach();
		this.#unsubscribeFromModeStore();
		if (this.#unsubscribeDebugConfig) {
			this.#unsubscribeDebugConfig();
			this.#unsubscribeDebugConfig = null;
		}
		if (this.#unsubscribeLanguage) {
			this.#unsubscribeLanguage();
			this.#unsubscribeLanguage = null;
		}
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
          --threat-rotation: 0deg;
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
          min-height: 400px;
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

        .threat-label {
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.5px;
          color: rgba(255, 255, 255, 0.85);
          padding: 2px 6px;
          border-radius: 3px;
          background: rgba(255, 0, 0, 0.25);
          border: 1px solid rgba(255, 0, 0, 0.4);
          user-select: none;
          white-space: nowrap;
        }

        .threat-label.hidden {
          display: none;
        }

        .mode-button {
          position: absolute;
          left: 10px;
          bottom: 10px;
          padding: 6px 12px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.8px;
          background: rgba(0, 0, 0, 0.55);
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 8px;
          color: rgba(255, 255, 255, 0.9);
          cursor: pointer;
          z-index: 12;
          font-family: inherit;
        }

        .mode-button:hover {
          border-color: rgba(255, 255, 255, 0.5);
          background: rgba(0, 0, 0, 0.75);
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
          min-height: 340px;
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
          pointer-events: none;
        }

        .placeholder {
          color: rgba(255, 255, 255, 0.9);
          font-size: 12px;
          letter-spacing: 1px;
          text-align: center;
          padding: 10px;
          z-index: 0;
        }

        .placeholder-icon {
          font-size: 48px;
          margin-bottom: 8px;
          opacity: 0.7;
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

        .threat-mini-map {
          display: none;
          position: absolute;
          left: 14px;
          bottom: 14px;
          width: 86px;
          height: 86px;
          border-radius: 12px;
          background: rgba(0, 0, 0, 0.55);
          border: 1px solid rgba(255, 0, 0, 0.35);
          backdrop-filter: blur(6px);
          box-shadow: 0 0 0 1px rgba(255, 0, 0, 0.25) inset;
          z-index: 11;
          align-items: center;
          justify-content: center;
          pointer-events: none;
        }

        :host([variant="alert"]) .threat-mini-map {
          display: flex;
        }

        .threat-mini-map svg {
          width: 100%;
          height: 100%;
          padding: 8px;
          box-sizing: border-box;
        }

        .threat-arrow {
          transform-origin: 50% 50%;
          transform: rotate(var(--threat-rotation));
        }

        .threat-label {
          position: absolute;
          right: 6px;
          bottom: 6px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.5px;
          color: #ff4d4d;
          background: rgba(0, 0, 0, 0.45);
          border: 1px solid rgba(255, 0, 0, 0.25);
          border-radius: 6px;
          padding: 1px 5px;
          line-height: 1.4;
        }

        .hidden {
          display: none;
        }

        .fullscreen-btn {
          width: 30px;
          height: 30px;
          padding: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 0, 0, 0.35);
          border: 1px solid rgba(255, 255, 255, 0.25);
          border-radius: 6px;
          cursor: pointer;
          color: rgba(255, 255, 255, 0.85);
          transition: background 0.15s, border-color 0.15s;
        }

        .fullscreen-btn:hover {
          background: rgba(0, 0, 0, 0.7);
          border-color: rgba(255, 255, 255, 0.4);
        }

        .fullscreen-btn svg {
          width: 16px;
          height: 16px;
        }
      </style>

      <div class="header">
        <span class="name"></span>
        <div class="header-right">
          <button class="fullscreen-btn" type="button" title="${t("fullscreen.toggle")}" aria-label="${t("fullscreen.ariaToggle")}">
            <svg class="icon-expand" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
            </svg>
            <svg class="icon-exit hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3" />
            </svg>
          </button>
          <span class="threat-label hidden"></span>
          <slot name="header-right"></slot>
        </div>
      </div>

      <div class="feed">
        <video autoplay muted playsinline></video>
        <div class="placeholder">
          <div class="placeholder-icon">📷</div>
          <div class="placeholder-status">${t("camera.noSignal")}</div>
          <div class="placeholder-ip"></div>
        </div>
        <div class="threat-mini-map" data-role="threat-mini-map" aria-hidden="true">
          <svg viewBox="0 0 100 100" role="presentation">
            <circle cx="50" cy="50" r="46" fill="rgba(0,0,0,0)" stroke="rgba(255,255,255,0.2)" stroke-width="2" />
            <circle cx="50" cy="50" r="30" fill="rgba(0,0,0,0)" stroke="rgba(255,255,255,0.1)" stroke-width="2" />
            <line x1="50" y1="8" x2="50" y2="92" stroke="rgba(255,255,255,0.08)" stroke-width="2" />
            <line x1="8" y1="50" x2="92" y2="50" stroke="rgba(255,255,255,0.08)" stroke-width="2" />
            <g class="threat-arrow">
              <line x1="50" y1="50" x2="50" y2="12" stroke="#ff4d4d" stroke-width="4" stroke-linecap="round" />
              <polygon points="50,6 42,18 58,18" fill="#ff4d4d" />
            </g>
          </svg>
          <div class="threat-label"></div>
        </div>
        <div class="overlay-slot">
          <slot name="overlay-top-right"></slot>
        </div>
        <button class="mode-button" type="button" title="${t("camera.modeSwitch")}" aria-label="${t("camera.dayNightSwitch")}"></button>
      </div>

      <div class="ip"></div>
    `;

		this.#video = shadow.querySelector("video");
		this.#nameEl = shadow.querySelector(".name");
		this.#ipEl = shadow.querySelector(".ip");
		this.#modeToggle = shadow.querySelector(".mode-button");
		this.#placeholderIpEl = shadow.querySelector(".placeholder-ip");
		this.#placeholderStatusEl = shadow.querySelector(".placeholder-status");
		this.#threatLabelEl = shadow.querySelector(".threat-label");
		if (this.#video) configureCameraFeedVideo(this.#video);

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

		this.#fullscreenBtn = shadow.querySelector(".fullscreen-btn");
		this.#iconExpand = shadow.querySelector(".icon-expand");
		this.#iconExit = shadow.querySelector(".icon-exit");

		this.#fullscreenBtn?.addEventListener("click", () =>
			this.#toggleFullscreen(),
		);

		document.addEventListener("fullscreenchange", this.#onFullscreenChange);
	}

	#onFullscreenChange = (): void => {
		const isFullscreen = document.fullscreenElement === this;
		this.#iconExpand?.classList.toggle("hidden", isFullscreen);
		this.#iconExit?.classList.toggle("hidden", !isFullscreen);
		this.#fullscreenBtn?.setAttribute(
			"title",
			isFullscreen ? t("fullscreen.exit") : t("fullscreen.toggle"),
		);
		this.#fullscreenBtn?.setAttribute(
			"aria-label",
			isFullscreen ? t("fullscreen.ariaExit") : t("fullscreen.ariaToggle"),
		);
	};

	#toggleFullscreen(): void {
		if (document.fullscreenElement === this) {
			document.exitFullscreen?.();
		} else {
			this.requestFullscreen?.();
		}
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
			(camera ? getLocalizedCameraName(camera.name, camera.id) : null) ??
			(cameraId !== null
				? t("camera.defaultNameWithId", { id: cameraId })
				: t("camera.defaultName"));

		const streamUrl =
			camera && cameraId !== null ? getCameraStreamUrl(camera, mode) : "";

		if (this.#nameEl) this.#nameEl.textContent = displayName;
		if (this.#ipEl) this.#ipEl.textContent = streamUrl;

		if (this.#placeholderIpEl) {
			this.#placeholderIpEl.textContent = streamUrl;
		}
		if (this.#placeholderStatusEl) {
			this.#placeholderStatusEl.textContent = t("camera.noSignal");
		}

		if (this.#modeToggle) {
			this.#modeToggle.textContent =
				mode === "day" ? t("camera.mode.day") : t("camera.mode.night");
			this.#modeToggle.setAttribute("title", t("camera.modeSwitch"));
			this.#modeToggle.setAttribute("aria-label", t("camera.dayNightSwitch"));
		}
		this.#syncUrlVisibility();

		if (this.#threatLabelEl) {
			const threatAzimuthDeg = parseThreatAzimuth(
				this.getAttribute("threat-azimuth"),
			);
			if (threatAzimuthDeg === null || threatAzimuthDeg === undefined) {
				this.style.setProperty("--threat-rotation", "0deg");
				this.#threatLabelEl.classList.add("hidden");
				this.#threatLabelEl.textContent = "";
			} else {
				const cardinal = azimuthDegToCardinal8(threatAzimuthDeg);
				this.style.setProperty("--threat-rotation", `${threatAzimuthDeg}deg`);
				this.#threatLabelEl.textContent = `${Math.round(
					threatAzimuthDeg,
				)}° ${cardinal}`;
				this.#threatLabelEl.classList.remove("hidden");
			}
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

	#syncUrlVisibility(): void {
		const showUrls = isCameraUrlsVisible();
		this.#ipEl?.classList.toggle("hidden", !showUrls);
		this.#placeholderIpEl?.classList.toggle("hidden", !showUrls);
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
