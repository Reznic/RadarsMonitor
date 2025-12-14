// MSE stream connection for go2rtc

const activeConnections: Map<string, WebSocket> = new Map();

export async function connectMSE(
	video: HTMLVideoElement,
	url: string,
): Promise<void> {
	const cameraId = video.dataset.cameraId || "";

	// Close existing connection
	disconnectStream(cameraId);

	const mediaSource = new MediaSource();
	video.src = URL.createObjectURL(mediaSource);

	mediaSource.addEventListener("sourceopen", () => {
		console.log(`[MSE ${cameraId}] MediaSource opened`);

		const ws = new WebSocket(url);
		ws.binaryType = "arraybuffer";

		if (cameraId) {
			activeConnections.set(cameraId, ws);
		}

		let sourceBuffer: SourceBuffer | null = null;
		const queue: ArrayBuffer[] = [];
		let isFirstMessage = true;

		ws.onmessage = (event) => {
			if (typeof event.data === "string") {
				console.log(`[MSE ${cameraId}] String message:`, event.data);
				return;
			}

			const data = event.data as ArrayBuffer;

			// First message contains codec info: [length byte][codec string]
			if (isFirstMessage) {
				isFirstMessage = false;
				const bytes = new Uint8Array(data);
				// Skip first byte (length), decode rest as codec string
				const codecStr = new TextDecoder().decode(bytes.slice(1));
				const mimeType = `video/mp4; codecs="${codecStr}"`;
				console.log(`[MSE ${cameraId}] Codec from server:`, mimeType);

				if (MediaSource.isTypeSupported(mimeType)) {
					sourceBuffer = mediaSource.addSourceBuffer(mimeType);
					sourceBuffer.mode = "segments";

					sourceBuffer.addEventListener("updateend", () => {
						if (sourceBuffer && sourceBuffer.buffered.length > 0) {
							const end = sourceBuffer.buffered.end(0);
							if (end > 0.1 && video.paused) {
								video.play().catch(console.error);
							}
						}
						if (queue.length > 0 && sourceBuffer && !sourceBuffer.updating) {
							sourceBuffer.appendBuffer(queue.shift()!);
						}
					});

					sourceBuffer.addEventListener("error", (e) => {
						console.error(`[MSE ${cameraId}] SourceBuffer error:`, e);
					});
				} else {
					console.error(`[MSE ${cameraId}] Unsupported codec:`, mimeType);
					ws.close();
				}
				return;
			}

			// Subsequent messages are MP4 fragments
			if (!sourceBuffer) return;

			if (sourceBuffer.updating || queue.length > 0) {
				queue.push(data);
			} else {
				try {
					sourceBuffer.appendBuffer(data);
				} catch (e) {
					console.error(`[MSE ${cameraId}] Buffer error:`, e);
				}
			}
		};

		ws.onopen = () => {
			console.log(`[MSE ${cameraId}] Connected`);
		};

		ws.onerror = (e) => {
			console.error(`[MSE ${cameraId}] WebSocket error:`, e);
		};

		ws.onclose = () => {
			console.log(`[MSE ${cameraId}] Disconnected`);
			activeConnections.delete(cameraId);
		};
	});

	video.play().catch(console.error);
}

export function disconnectStream(cameraId: string): void {
	const ws = activeConnections.get(cameraId);
	if (ws) {
		ws.close();
		activeConnections.delete(cameraId);
	}
}
