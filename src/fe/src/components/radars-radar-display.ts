import { subscribeLanguageChange, t } from "../i18n/index.ts";

function parseTarget(value: string | null): "main" | "mini" {
	return value === "mini" ? "mini" : "main";
}

export class RadarsRadarDisplayElement extends HTMLElement {
	#unsubscribeLanguage: (() => void) | null = null;

	connectedCallback(): void {
		this.#render();
		this.#applyDirectionLabels();
		if (!this.#unsubscribeLanguage) {
			this.#unsubscribeLanguage = subscribeLanguageChange(() => {
				this.#applyDirectionLabels();
			});
		}
	}

	disconnectedCallback(): void {
		if (this.#unsubscribeLanguage) {
			this.#unsubscribeLanguage();
			this.#unsubscribeLanguage = null;
		}
	}

	#render(): void {
		const target = parseTarget(this.getAttribute("target"));
		const compact = this.hasAttribute("compact");
		this.classList.toggle("compact", compact);
		this.classList.add("radar-display");

		this.innerHTML = `
			<div class="radar-container">
				<canvas class="radar-canvas" data-radar-canvas="${target}"></canvas>
				<div class="label label-top" data-radar-dir="top"></div>
				<div class="label label-right" data-radar-dir="right"></div>
				<div class="label label-bottom" data-radar-dir="bottom"></div>
				<div class="label label-left" data-radar-dir="left"></div>
			</div>
		`;
	}

	#applyDirectionLabels(): void {
		const top = this.querySelector('[data-radar-dir="top"]');
		const right = this.querySelector('[data-radar-dir="right"]');
		const bottom = this.querySelector('[data-radar-dir="bottom"]');
		const left = this.querySelector('[data-radar-dir="left"]');
		if (top) top.textContent = t("radar.direction.top");
		if (right) right.textContent = t("radar.direction.right");
		if (bottom) bottom.textContent = t("radar.direction.bottom");
		if (left) left.textContent = t("radar.direction.left");
	}
}

export function defineRadarsRadarDisplay(
	registry: CustomElementRegistry | undefined = globalThis.customElements,
): void {
	if (!registry) return;
	if (registry.get("radars-radar-display")) return;
	registry.define("radars-radar-display", RadarsRadarDisplayElement);
}

defineRadarsRadarDisplay();
