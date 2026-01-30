export type VideoSink = {
	srcObject: MediaProvider | null;
	play: () => Promise<void>;
};

type ConnectArgs = {
	pc: RTCPeerConnection;
	url: string;
	onStream: (stream: MediaStream) => void;
	onConnectionProblem: () => void;
};

type SharedWhepManagerOptions = {
	createPeerConnection?: () => RTCPeerConnection;
	connect?: (args: ConnectArgs) => Promise<void>;
	setTimeout?: typeof globalThis.setTimeout;
	clearTimeout?: typeof globalThis.clearTimeout;
};

type ConnectionState = {
	pc: RTCPeerConnection;
	url: string;
	subscribers: Set<VideoSink>;
	stream: MediaStream | null;
	connecting: Promise<void> | null;
	reconnectTimer: ReturnType<typeof globalThis.setTimeout> | null;
};

export function createSharedWhepManager(
	options: SharedWhepManagerOptions = {},
): {
	attach: (video: VideoSink, url: string) => () => void;
} {
	const createPeerConnection =
		options.createPeerConnection ??
		(() =>
			new RTCPeerConnection({
				iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
			}));
	const connect = options.connect ?? defaultConnectWhep;
	const setTimeoutFn = options.setTimeout ?? globalThis.setTimeout;
	const clearTimeoutFn = options.clearTimeout ?? globalThis.clearTimeout;

	const connections: Map<string, ConnectionState> = new Map();

	function applyStreamToSubscriber(
		video: VideoSink,
		stream: MediaStream,
	): void {
		video.srcObject = stream;
		video.play().catch(() => {
			// Autoplay might be blocked; best-effort.
		});
	}

	function closeConnection(url: string): void {
		const conn = connections.get(url);
		if (!conn) return;

		if (conn.reconnectTimer) {
			clearTimeoutFn(conn.reconnectTimer);
			conn.reconnectTimer = null;
		}

		conn.pc.close();
		connections.delete(url);
	}

	function scheduleReconnect(conn: ConnectionState): void {
		if (conn.reconnectTimer) return;
		if (conn.subscribers.size === 0) return;

		conn.reconnectTimer = setTimeoutFn(() => {
			conn.reconnectTimer = null;
			if (conn.subscribers.size === 0) {
				closeConnection(conn.url);
				return;
			}

			try {
				conn.pc.close();
			} catch {
				// Ignore.
			}

			conn.pc = createPeerConnection();
			conn.stream = null;
			conn.connecting = null;
			ensureConnected(conn);
		}, 2000);
	}

	function ensureConnected(conn: ConnectionState): void {
		if (conn.connecting) return;

		conn.connecting = connect({
			pc: conn.pc,
			url: conn.url,
			onStream: (stream) => {
				conn.stream = stream;
				for (const subscriber of conn.subscribers) {
					applyStreamToSubscriber(subscriber, stream);
				}
			},
			onConnectionProblem: () => scheduleReconnect(conn),
		})
			.catch(() => {
				scheduleReconnect(conn);
			})
			.finally(() => {
				conn.connecting = null;
			});
	}

	function attach(video: VideoSink, url: string): () => void {
		let conn = connections.get(url);
		if (!conn) {
			conn = {
				pc: createPeerConnection(),
				url,
				subscribers: new Set(),
				stream: null,
				connecting: null,
				reconnectTimer: null,
			};
			connections.set(url, conn);
		}

		conn.subscribers.add(video);

		if (conn.stream) {
			applyStreamToSubscriber(video, conn.stream);
		} else {
			ensureConnected(conn);
		}

		return () => {
			const current = connections.get(url);
			if (!current) return;
			current.subscribers.delete(video);

			if (current.subscribers.size === 0) {
				closeConnection(url);
			}
		};
	}

	return { attach };
}

async function defaultConnectWhep({
	pc,
	url,
	onStream,
	onConnectionProblem,
}: ConnectArgs): Promise<void> {
	pc.ontrack = (event) => {
		const stream = event.streams?.[0];
		if (stream) onStream(stream);
	};

	pc.oniceconnectionstatechange = () => {
		if (
			pc.iceConnectionState === "disconnected" ||
			pc.iceConnectionState === "failed"
		) {
			onConnectionProblem();
		}
	};

	const offer = await pc.createOffer({
		offerToReceiveAudio: true,
		offerToReceiveVideo: true,
	});
	await pc.setLocalDescription(offer);

	await waitForIceGatheringComplete(pc, 5000);

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

	const answerSdp = await response.text();

	await pc.setRemoteDescription(
		new RTCSessionDescription({ type: "answer", sdp: answerSdp }),
	);
}

function waitForIceGatheringComplete(
	pc: RTCPeerConnection,
	timeoutMs: number,
): Promise<void> {
	if (pc.iceGatheringState === "complete") return Promise.resolve();

	return new Promise((resolve) => {
		let done = false;

		const finish = () => {
			if (done) return;
			done = true;
			pc.removeEventListener("icegatheringstatechange", onChange);
			resolve();
		};

		const onChange = () => {
			if (pc.iceGatheringState === "complete") finish();
		};

		pc.addEventListener("icegatheringstatechange", onChange);
		globalThis.setTimeout(finish, timeoutMs);
	});
}
