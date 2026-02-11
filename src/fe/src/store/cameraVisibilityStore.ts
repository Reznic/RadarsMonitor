type Listener = (visible: boolean) => void;

const visibleByCameraId: Map<number, boolean> = new Map();
const listenersByCameraId: Map<number, Set<Listener>> = new Map();

export function isCameraVisible(cameraId: number): boolean {
	const value = visibleByCameraId.get(cameraId);
	return value === undefined ? true : value;
}

export function setCameraVisible(cameraId: number, visible: boolean): void {
	const prev = visibleByCameraId.get(cameraId);
	if (prev === visible) return;

	visibleByCameraId.set(cameraId, visible);
	const listeners = listenersByCameraId.get(cameraId);
	if (listeners) {
		for (const listener of listeners) {
			listener(visible);
		}
	}
}

export function subscribeCameraVisibility(
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
