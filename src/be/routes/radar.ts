import type { RadarTarget } from "../../types.ts";
import { CORS_HEADERS } from "../config.ts";
import { getCurrentTargets } from "../data/csvHandler.ts";

// Radar data endpoint handler - returns all current targets
export function handleRadar(): Response {
	const targets: RadarTarget[] = getCurrentTargets();

	return new Response(JSON.stringify({ targets }), { headers: CORS_HEADERS });
}
