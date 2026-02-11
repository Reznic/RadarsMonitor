import type { CameraMode } from "../config.ts";

type Listener = (mode: CameraMode) => void;

const modeByCameraId: Map<number, CameraMode> = new Map();
const listenersByCameraId: Map<number, Set<Listener>> = new Map();

export function getCameraMode(cameraId: number): CameraMode | undefined {
	return modeByCameraId.get(cameraId);
}

export function ensureCameraMode(cameraId: number): CameraMode {
	const existing = modeByCameraId.get(cameraId);
	if (existing) return existing;
	modeByCameraId.set(cameraId, "day");
	return "day";
}

export function setCameraMode(cameraId: number, mode: CameraMode): void {
	const prev = modeByCameraId.get(cameraId);
	if (prev === mode) return;

	modeByCameraId.set(cameraId, mode);
	const listeners = listenersByCameraId.get(cameraId);
	if (!listeners) return;
	for (const listener of listeners) {
		listener(mode);
	}
}

export function subscribeCameraMode(
	cameraId: number,
	listener: Listener,
): () => void {
	let set = listenersByCameraId.get(cameraId);
	if (!set) {
		set = new Set();
		listenersByCameraId.set(cameraId, set);
	}
	set.add(listener);

	return () => {
		const current = listenersByCameraId.get(cameraId);
		if (!current) return;
		current.delete(listener);
		if (current.size === 0) {
			listenersByCameraId.delete(cameraId);
		}
	};
}
