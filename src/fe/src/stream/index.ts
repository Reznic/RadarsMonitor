// Stream module - exports both MSE (legacy) and WebRTC (default)

export { connectMSE, disconnectStream } from "./mse.ts";
export { connectWebRTC, disconnectWebRTC } from "./webrtc.ts";

// Default export uses WebRTC
export { connectWebRTC as connectStream } from "./webrtc.ts";
export { disconnectWebRTC as disconnectStreamConnection } from "./webrtc.ts";
