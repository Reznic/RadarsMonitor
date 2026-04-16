import os
import time
import threading
import shutil
import subprocess
from datetime import datetime
from urllib.parse import quote


class CameraSnapshotManager:
    def __init__(
        self,
        camera_id,
        ip,
        username,
        password,
        interval=10,
        day_rtsp_url=None,
        radar_serial=None,
    ):
        """
        Capture JPEG snapshots from the camera day stream via ffmpeg (RTSP).
        :param camera_id: ID for folder naming (often the camera display name).
        :param ip: Camera IP (used if day_rtsp_url is not set).
        :param username: RTSP user (used if day_rtsp_url is not set).
        :param password: RTSP password (used if day_rtsp_url is not set).
        :param interval: Seconds between snapshots in the background loop.
        :param day_rtsp_url: Full day-channel RTSP URL (from config.json); preferred.
        :param radar_serial: Radar ID from FE config.ts (streamId ↔ radarSerial); optional.
        """
        self.camera_id = camera_id
        self.ip = ip
        self.username = username
        self.password = password
        self.interval = interval
        self.radar_serial = radar_serial
        self._interval_lock = threading.Lock()

        self.folder_name = f"cameras_snapshots/camera_{self.camera_id}"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.is_running = False
        self.thread = None
        self._ffmpeg_bin: str | None = None
        self._rtsp_url = day_rtsp_url or self._default_rtsp_url()

        self._setup_ffmpeg()

    def _default_rtsp_url(self) -> str:
        u = quote(self.username, safe="")
        p = quote(self.password, safe="")
        return f"rtsp://{u}:{p}@{self.ip}:554/live?channel=0&subtype=0"

    def _setup_ffmpeg(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print(
                f"[Cam {self.camera_id}] ffmpeg not found in PATH; "
                "install ffmpeg for RTSP snapshots."
            )
            return
        self._ffmpeg_bin = ffmpeg
        print(f"[Cam {self.camera_id}] Snapshots via RTSP → JPEG (ffmpeg).")

    def _snapshot_ready(self) -> bool:
        return bool(self._ffmpeg_bin and self._rtsp_url)

    def _snapshot_via_ffmpeg(self, filename: str) -> None:
        assert self._ffmpeg_bin and self._rtsp_url
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self._rtsp_url,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-y",
            filename,
        ]
        try:
            subprocess.run(
                cmd,
                timeout=45,
                check=True,
                capture_output=True,
            )
            if os.path.isfile(filename) and os.path.getsize(filename) > 100:
                print(f"[Cam {self.camera_id}] Saved: {filename}")
            else:
                print(f"[Cam {self.camera_id}] ffmpeg produced no usable JPEG.")
        except subprocess.TimeoutExpired:
            print(f"[Cam {self.camera_id}] ffmpeg snapshot timed out.")
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode(errors="replace").strip()
            tail = (err[-200:] if err else str(e))
            print(f"[Cam {self.camera_id}] ffmpeg failed: {tail}")
        except OSError as e:
            print(f"[Cam {self.camera_id}] ffmpeg could not run: {e}")

    def _capture_loop(self) -> None:
        print(f"[Cam {self.camera_id}] Started snapshot loop every {self.interval}s.")
        while self.is_running:
            if self._snapshot_ready():
                self._take_snapshot()
            with self._interval_lock:
                sleep_interval = self.interval
            time.sleep(sleep_interval)

    def _take_snapshot(self) -> None:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(self.folder_name, f"{timestamp}.jpg")
            if self._snapshot_ready():
                self._snapshot_via_ffmpeg(filename)
        except Exception as e:
            print(f"[Cam {self.camera_id}] Error capturing snapshot: {e}")

    def start_sampling(self) -> None:
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def stop_sampling(self) -> None:
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1)
        print(f"[Cam {self.camera_id}] Sampling stopped.")

    def set_interval(self, interval_seconds: float) -> None:
        with self._interval_lock:
            self.interval = interval_seconds

    def take_snapshot_now(self) -> None:
        if self._snapshot_ready():
            self._take_snapshot()


if __name__ == "__main__":
    _user = os.environ.get("CAM_USER", "admin")
    _pw = os.environ.get("CAM_PASSWORD", "password")
    _u = quote(_user, safe="")
    _p = quote(_pw, safe="")
    _cams = [
        ("192.168.1.38", f"rtsp://{_u}:{_p}@192.168.1.38:554/live?channel=0&subtype=0"),
        ("192.168.1.40", f"rtsp://{_u}:{_p}@192.168.1.40:554/live?channel=0&subtype=0"),
    ]
    managers = []

    for i, (ip, rtsp_day) in enumerate(_cams):
        cam = CameraSnapshotManager(
            camera_id=i + 1,
            ip=ip,
            username=_user,
            password=_pw,
            interval=60,
            day_rtsp_url=rtsp_day,
        )
        cam.start_sampling()
        managers.append(cam)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for cam in managers:
            cam.stop_sampling()
