import json
import socket
import threading
import urllib.request
from typing import Callable, Dict, List, Optional


class RadarNodeClient:
    def __init__(self, radar_id: str, host: str, config_port: int, data_port: int) -> None:
        self.radar_id = radar_id
        self.host = host
        self.config_http_port = config_port
        self.data_tcp_port = data_port
        self._events_thread: Optional[threading.Thread] = None
        self._events_stop = threading.Event()

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.config_http_port}{path}"

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self._url('/health'), timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_serial_number(self, timeout: float = 3.0) -> Optional[str]:
        try:
            with urllib.request.urlopen(self._url('/serial'), timeout=timeout) as resp:
                raw = resp.read().decode('utf-8')
                obj = json.loads(raw)
                return obj.get('serial_number')
        except Exception:
            return None

    def send_command(self, command: str, timeout: float = 3.0) -> str:
        data = command.encode('utf-8')
        req = urllib.request.Request(self._url('/command'), data=data, method='POST', headers={'Content-Type': 'text/plain'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
            try:
                obj = json.loads(raw)
                return obj.get('response', '')
            except Exception:
                return raw

    def start_events(self, on_event: Callable[[str, Dict], None]) -> None:
        if self._events_thread and self._events_thread.is_alive():
            return
        self._events_stop.clear()

        def run():
            try:
                with urllib.request.urlopen(self._url('/events'), timeout=5) as resp:
                    for raw in resp:
                        if self._events_stop.is_set():
                            break
                        try:
                            line = raw.decode('utf-8').strip()
                        except Exception:
                            continue
                        if not line:
                            continue
                        if line.startswith('event:'):
                            current_event = line.split(':', 1)[1].strip()
                        elif line.startswith('data:'):
                            payload_str = line.split(':', 1)[1].strip()
                            try:
                                data = eval(payload_str) if payload_str.startswith('{') else {"data": payload_str}
                            except Exception:
                                data = {"data": payload_str}
                            on_event(current_event if 'current_event' in locals() else 'message', data)
            except Exception:
                pass
        self._events_thread = threading.Thread(target=run, name=f"AdapterEvents@{self.host}", daemon=True)
        self._events_thread.start()

    def stop_events(self) -> None:
        self._events_stop.set()
        if self._events_thread and self._events_thread.is_alive():
            self._events_thread.join(timeout=2)

    def stop(self) -> None:
        self.stop_events()
        self._events_thread = None