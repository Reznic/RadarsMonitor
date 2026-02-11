import { t } from "../i18n/index.ts";

export function normalizeAzimuthDeg(value: number): number {
	if (!Number.isFinite(value)) return 0;
	const mod = value % 360;
	return mod < 0 ? mod + 360 : mod;
}

export function azimuthDegToCardinal8(value: number): string {
	const azimuth = normalizeAzimuthDeg(value);
	const labels = [
		t("radar.direction.n"),
		t("radar.direction.ne"),
		t("radar.direction.e"),
		t("radar.direction.se"),
		t("radar.direction.s"),
		t("radar.direction.sw"),
		t("radar.direction.w"),
		t("radar.direction.nw"),
	] as const;
	const index = Math.round(azimuth / 45) % labels.length;
	return labels[index] ?? t("radar.direction.n");
}
