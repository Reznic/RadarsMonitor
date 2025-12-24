// WebRTC stream connection for MediaMTX using WHEP protocol
// API: POST /{path}/whep with SDP offer in plain text (application/sdp)

const activeConnections: Map<string, RTCPeerConnection> = new Map();
const reconnectTimers: Map<string, number> = new Map();

export async function connectWebRTC(
	video: HTMLVideoElement,
	url: string,
): Promise<void> {
	const cameraId = video.dataset.cameraId || "";

	// Close existing connection
	disconnectWebRTC(cameraId);

	const pc = new RTCPeerConnection({
		iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
	});

	if (cameraId) {
		activeConnections.set(cameraId, pc);
	}

	// Handle incoming tracks - use the stream from the event directly
	pc.ontrack = (event) => {
		console.log(`[WebRTC ${cameraId}] Track received:`, event.track.kind);
		// Use the stream from the event directly (RTSPtoWEB sends a single stream)
		if (event.streams && event.streams[0]) {
			video.srcObject = event.streams[0];
			video.play().catch(() => {
				// Autoplay might be blocked
			});
		}
	};

	pc.oniceconnectionstatechange = () => {
		console.log(`[WebRTC ${cameraId}] ICE state:`, pc.iceConnectionState);
		if (pc.iceConnectionState === "connected") {
			console.log(`[WebRTC ${cameraId}] Stream active`);
		}
		// Auto-reconnect on disconnect/failure
		if (
			pc.iceConnectionState === "disconnected" ||
			pc.iceConnectionState === "failed"
		) {
			console.log(`[WebRTC ${cameraId}] Scheduling reconnect...`);
			scheduleReconnect(video, url, cameraId);
		}
	};

	try {
		// Create offer with receive-only for audio/video
		const offer = await pc.createOffer({
			offerToReceiveAudio: true,
			offerToReceiveVideo: true,
		});
		await pc.setLocalDescription(offer);

		// Wait for ICE gathering to complete (RTSPtoWEB doesn't support trickle ICE)
		if (pc.iceGatheringState !== "complete") {
			await new Promise<void>((resolve) => {
				const checkState = () => {
					if (pc.iceGatheringState === "complete") {
						pc.removeEventListener("icegatheringstatechange", checkState);
						resolve();
					}
				};
				pc.addEventListener("icegatheringstatechange", checkState);
				checkState();
				setTimeout(resolve, 5000);
			});
		}

		console.log(`[WebRTC ${cameraId}] Sending offer to:`, url);

		// MediaMTX WHEP protocol: send SDP as plain text with application/sdp content type
		const response = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/sdp",
			},
			body: pc.localDescription?.sdp || "",
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		// MediaMTX returns SDP answer as plain text
		const answerSdp = await response.text();

		console.log(`[WebRTC ${cameraId}] Received answer`);

		await pc.setRemoteDescription(
			new RTCSessionDescription({ type: "answer", sdp: answerSdp }),
		);

		console.log(`[WebRTC ${cameraId}] Connection established`);
	} catch (error) {
		console.error(`[WebRTC ${cameraId}] Connection failed:`, error);
		disconnectWebRTC(cameraId);
		throw error;
	}
}

function scheduleReconnect(
	video: HTMLVideoElement,
	url: string,
	cameraId: string,
): void {
	// Clear existing timer
	const existingTimer = reconnectTimers.get(cameraId);
	if (existingTimer) {
		clearTimeout(existingTimer);
	}

	// Schedule reconnect in 2 seconds
	const timer = window.setTimeout(() => {
		reconnectTimers.delete(cameraId);
		console.log(`[WebRTC ${cameraId}] Reconnecting...`);
		connectWebRTC(video, url).catch((err) => {
			console.error(`[WebRTC ${cameraId}] Reconnect failed:`, err);
		});
	}, 2000);

	reconnectTimers.set(cameraId, timer);
}

export function disconnectWebRTC(cameraId: string): void {
	// Clear reconnect timer
	const timer = reconnectTimers.get(cameraId);
	if (timer) {
		clearTimeout(timer);
		reconnectTimers.delete(cameraId);
	}

	const pc = activeConnections.get(cameraId);
	if (pc) {
		pc.close();
		activeConnections.delete(cameraId);
	}
}
