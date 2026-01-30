type VideoElementLike = Pick<
	HTMLVideoElement,
	| "autoplay"
	| "muted"
	| "playsInline"
	| "controls"
	| "disablePictureInPicture"
	| "disableRemotePlayback"
	| "setAttribute"
	| "removeAttribute"
>;

export function configureCameraFeedVideo(video: VideoElementLike): void {
	video.autoplay = true;
	video.muted = true;
	video.playsInline = true;
	video.setAttribute("playsinline", "");

	// Avoid native UI surfaces inside our custom component chrome.
	video.controls = false;
	video.removeAttribute("controls");

	// Reduce ways a user can take the feed out of the dashboard.
	video.disablePictureInPicture = true;
	video.setAttribute("disablepictureinpicture", "");
	video.disableRemotePlayback = true;
	video.setAttribute("disableremoteplayback", "");

	// Hint to browsers to hide extra video UI affordances.
	video.setAttribute("controlslist", "nodownload noplaybackrate noremoteplayback");
}
