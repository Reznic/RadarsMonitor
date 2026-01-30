// Stream module - exports both MSE (legacy) and WebRTC (default)

export { connectMSE, disconnectStream } from "./mse.ts";
// Default export uses WebRTC
export {
	connectWebRTC,
	connectWebRTC as connectStream,
	disconnectWebRTC,
	disconnectWebRTC as disconnectStreamConnection,
} from "./webrtc.ts";
