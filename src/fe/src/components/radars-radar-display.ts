function parseTarget(value: string | null): "main" | "mini" {
	return value === "mini" ? "mini" : "main";
}

export class RadarsRadarDisplayElement extends HTMLElement {
	connectedCallback(): void {
		this.#render();
	}

	#render(): void {
		const target = parseTarget(this.getAttribute("target"));
		const compact = this.hasAttribute("compact");
		this.classList.toggle("compact", compact);
		this.classList.add("radar-display");

		this.innerHTML = `
			<div class="radar-container">
				<canvas class="radar-canvas" data-radar-canvas="${target}"></canvas>
				<div class="label label-top">TOP</div>
				<div class="label label-right">RIGHT</div>
				<div class="label label-bottom">BOTTOM</div>
				<div class="label label-left">LEFT</div>
			</div>
		`;
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
