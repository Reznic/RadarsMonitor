export function normalizeAzimuthDeg(value: number): number {
	if (!Number.isFinite(value)) return 0;
	const mod = value % 360;
	return mod < 0 ? mod + 360 : mod;
}

export function azimuthDegToCardinal8(value: number): string {
	const azimuth = normalizeAzimuthDeg(value);
	const labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;
	const index = Math.round(azimuth / 45) % labels.length;
	return labels[index] ?? "N";
}
