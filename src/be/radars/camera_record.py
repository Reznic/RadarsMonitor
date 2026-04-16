import subprocess
import signal
import time
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, quote

class CameraRecorder:
    def __init__(
        self,
        cam_id,
        rtsp_url,
        *,
        stream_name="main",
        output_dir=None,
        rtsp_user=None,
        rtsp_password=None,
    ):
        self.cam_id = cam_id
        self.stream_name = stream_name
        self.rtsp_url = self._with_rtsp_auth(rtsp_url, rtsp_user, rtsp_password)
        self.process = None
        self._log_file_handle = None
        self._log_path = None

        if output_dir is None:
            output_dir = os.environ.get("CAMERA_RECORDINGS_DIR", "/mnt/nvme/recordings")
        self.output_dir = Path(output_dir).expanduser()

    @staticmethod
    def _with_rtsp_auth(rtsp_url: str, user: str | None, password: str | None) -> str:
        if not user and not password:
            return rtsp_url

        parts = urlsplit(rtsp_url)
        if parts.scheme and parts.scheme.lower() != "rtsp":
            return rtsp_url

        # If credentials are already present in the URL, keep them.
        if "@" in parts.netloc:
            return rtsp_url

        safe_user = quote(user or "", safe="")
        safe_pass = quote(password or "", safe="")
        auth = safe_user
        if password is not None:
            auth = f"{safe_user}:{safe_pass}"

        netloc = f"{auth}@{parts.netloc}" if auth else parts.netloc
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    def _ensure_output_dir(self) -> Path:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir
        except PermissionError:
            fallback = Path("./recordings").resolve()
            fallback.mkdir(parents=True, exist_ok=True)
            print(
                f"Camera {self.cam_id}: no permission to write to '{self.output_dir}'. "
                f"Falling back to '{fallback}'."
            )
            self.output_dir = fallback
            return fallback

    @staticmethod
    def _read_last_bytes(path: Path, max_bytes: int = 4096) -> str:
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - max_bytes), os.SEEK_SET)
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def start_recording(self, *, session_dir: Path | None = None, session_timestamp: str | None = None):
        if self.process is not None:
            print(f"Camera {self.cam_id} ({self.stream_name}) is already recording.")
            return

        timestamp = session_timestamp or time.strftime("%Y%m%d_%H%M%S")
        if session_dir is None:
            base_dir = self._ensure_output_dir()
            cam_dir = base_dir / str(self.cam_id)
            cam_dir.mkdir(parents=True, exist_ok=True)
            session_dir = cam_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        output_file = str(session_dir / f"{self.stream_name}.mp4")
        self._log_path = session_dir / f"{self.stream_name}.ffmpeg.log"

        # The magic is '-c copy' which prevents decoding/encoding
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp", # Forces TCP to prevent packet loss
            "-i", self.rtsp_url,
            "-c", "copy",             # Direct stream copy
            "-f", "mp4",
            output_file
        ]

        # Start FFmpeg as a background process
        self._log_file_handle = self._log_path.open("wb")
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log_file_handle)

        time.sleep(0.5)
        exit_code = self.process.poll()
        if exit_code is not None:
            self.process = None
            if self._log_file_handle is not None:
                try:
                    self._log_file_handle.close()
                finally:
                    self._log_file_handle = None
            tail = self._read_last_bytes(self._log_path)
            print(f"Failed to start recording Camera {self.cam_id} (ffmpeg exit code {exit_code}).")
            if tail.strip():
                print("--- ffmpeg log (tail) ---")
                print(tail.rstrip())
            return

        print(f"Started recording Camera {self.cam_id} ({self.stream_name}) to {output_file}")
        print(f"FFmpeg log: {self._log_path}")

    def stop_recording(self):
        if self.process:
            # Send SIGINT (Ctrl+C) so FFmpeg closes the MP4 atom/header properly
            self.process.send_signal(signal.SIGINT)
            self.process.wait() # Wait for process to exit cleanly
            self.process = None
            if self._log_file_handle is not None:
                try:
                    self._log_file_handle.close()
                finally:
                    self._log_file_handle = None
            print(f"Stopped recording Camera {self.cam_id} ({self.stream_name})")


class MultiStreamCameraRecorder:
    def __init__(self, cam_id, streams: dict[str, str], *, output_dir=None, rtsp_user=None, rtsp_password=None):
        self.cam_id = cam_id
        self.recorders: dict[str, CameraRecorder] = {}
        for name, url in streams.items():
            self.recorders[name] = CameraRecorder(
                cam_id=cam_id,
                rtsp_url=url,
                stream_name=name,
                output_dir=output_dir,
                rtsp_user=rtsp_user,
                rtsp_password=rtsp_password,
            )

    def start_recording(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Ensure a single shared session directory for all streams of this camera.
        any_recorder = next(iter(self.recorders.values()), None)
        if any_recorder is None:
            return
        base_dir = any_recorder._ensure_output_dir()
        session_dir = base_dir / str(self.cam_id) / timestamp
        for recorder in self.recorders.values():
            recorder.start_recording(session_dir=session_dir, session_timestamp=timestamp)

    def stop_recording(self):
        for recorder in self.recorders.values():
            recorder.stop_recording()

if __name__ == "__main__":
    # 8 cameras: 192.168.1.35 -> 192.168.1.42
    cameras: dict[int, MultiStreamCameraRecorder] = {}
    for last_octet in range(35, 43):
        ip = f"192.168.1.{last_octet}"
        cam_id = last_octet
        cameras[cam_id] = MultiStreamCameraRecorder(
            cam_id=cam_id,
            streams={
                # FC465T visible stream
                "visible": f"rtsp://{ip}:554/cam/realmonitor?channel=1&subtype=0",
                # FC465T thermal stream
                "thermal": f"rtsp://{ip}:554/cam/realmonitor?channel=2&subtype=0",
            },
            rtsp_user="admin",
            rtsp_password="password",
        )

    for cam in cameras.values():
        cam.start_recording()

    time.sleep(10)  # Record for 10 seconds...

    for cam in cameras.values():
        cam.stop_recording()