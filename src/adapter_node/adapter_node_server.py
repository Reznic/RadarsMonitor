from sys import argv
import threading
import time
import socket
import socketserver
import urllib.request
import json
from typing import List, Optional
from flask import Flask, Response, request, jsonify, abort

# Adapter Node Server version
ADAPTER_VERSION = "0.1.0"

from radar_device import RadarDevice
from device_scanner import scan_radar_devices



class _TCPClientRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: List[socket.socket] = []

    def add(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.append(sock)

    def remove(self, sock: socket.socket) -> None:
        with self._lock:
            if sock in self._clients:
                self._clients.remove(sock)
        try:
            sock.close()
        except Exception:
            pass

    def broadcast(self, data: bytes) -> None:
        to_drop: List[socket.socket] = []
        with self._lock:
            for s in self._clients:
                try:
                    s.sendall(data)
                except Exception:
                    to_drop.append(s)
            for s in to_drop:
                if s in self._clients:
                    self._clients.remove(s)
        for s in to_drop:
            try:
                s.close()
            except Exception:
                pass


class RadarAdapterNodeServer:
    BOOT_MSG_RETRY_INTERVAL = 10

    def __init__(self, radar: RadarDevice, http_port: int, tcp_port: int) -> None:
        self.radar = radar
        self.http_port = http_port
        self.tcp_port = tcp_port

        self._flask_app: Optional[Flask] = None
        self._http_thread: Optional[threading.Thread] = None
        self._tcp_server: Optional[socketserver.ThreadingTCPServer] = None
        self._tcp_thread: Optional[threading.Thread] = None
        self._data_reader_thread: Optional[threading.Thread] = None
        self._boot_message_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

        self._tcp_clients = _TCPClientRegistry()

    def start(self, manager_host: Optional[str] = None, manager_port: int = 9090) -> None:
        self._running.set()
        self._start_http()
        self._start_tcp_stream()
        self._start_data_reader()
        
        # Send boot message to manager if host is provided
        if manager_host:
            self._start_boot_message_thread(manager_host, manager_port)

    def stop(self) -> None:
        self._running.clear()
        try:
            if self._flask_app:
                # Flask doesn't have a shutdown method, so we use a context manager pattern
                pass
        except Exception:
            pass
        try:
            if self._tcp_server:
                self._tcp_server.shutdown()
                self._tcp_server.server_close()
                self._tcp_server = None
        except Exception:
            print("Failed to shutdown TCP server")
        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join(timeout=2)
        if self._tcp_thread and self._tcp_thread.is_alive():
            self._tcp_thread.join(timeout=2)
        if self._data_reader_thread and self._data_reader_thread.is_alive():
            self._data_reader_thread.join(timeout=2)
        if self._boot_message_thread and self._boot_message_thread.is_alive():
            self._boot_message_thread.join(timeout=2)
        try:
            self.radar.close()
        except Exception:
            pass

    def join(self):
        if self._http_thread and self._http_thread.is_alive():
            self._http_thread.join()

    def _start_http(self) -> None:
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET', 'OPTIONS'])
        def health():
            return jsonify({"ok": True})

        @app.route('/version', methods=['GET'])
        def version():
            return jsonify({"version": ADAPTER_VERSION})
        
        @app.route('/serial', methods=['GET'])
        def get_serial():
            return jsonify({"serial_number": self.radar.serial_number})
        
        @app.route('/events', methods=['GET'])
        def events():
            def generate():
                yield ": keepalive\n\n"
                try:
                    while self._running.is_set():
                        yield ": keepalive\n\n"
                        time.sleep(self.BOOT_MSG_RETRY_INTERVAL)
                except GeneratorExit:
                    pass
            return Response(generate(), mimetype='text/event-stream')
        
        @app.route('/command', methods=['POST', 'OPTIONS'])
        def command():
            cmd = request.data.decode('utf-8').strip()
            response_text = self._handle_command(cmd)
            return jsonify({"response": response_text})
        
        @app.after_request
        def after_request(resp):
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return resp
        
        self._flask_app = app
        self._http_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=self.http_port, use_reloader=False),
            name="RadarHTTP", daemon=True
        )
        self._http_thread.start()

    def _start_tcp_stream(self) -> None:
        registry = self._tcp_clients

        class TCPHandler(socketserver.BaseRequestHandler):
            def handle(self):
                registry.add(self.request)
                try:
                    while True:
                        data = self.request.recv(1)
                        if not data:
                            break
                except Exception:
                    pass
                finally:
                    registry.remove(self.request)

        self._tcp_server = socketserver.ThreadingTCPServer(("0.0.0.0", self.tcp_port), TCPHandler)
        self._tcp_server.daemon_threads = True
        self._tcp_thread = threading.Thread(target=self._tcp_server.serve_forever, name="RadarTCP", daemon=True)
        self._tcp_thread.start()

    def _start_data_reader(self) -> None:
        # Todo: run in process instead of thread?
        def run() -> None:
            while self._running.is_set():
                try:
                    chunk = self.radar.read_data()
                    if chunk:
                        self._tcp_clients.broadcast(chunk)
                except Exception as e:
                    print(f"Error reading data from radar {self.radar.serial_number}: {e}")
        self._data_reader_thread = threading.Thread(target=run, name=f"Radar{self.radar.serial_number}DataReader", daemon=True)
        self._data_reader_thread.start()

    def _handle_command(self, command: str) -> str:
        if not command:
            return "\"\""
        try:
            response = self.radar.send_command(command)
            if not response:
                return _json_escape_str("")
            try:
                s = response.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                s = repr(response)
            return _json_escape_str(s)
        except Exception as e:
            print(f"error sending command: {command} to radar {self.radar.serial_number}: {e}")
            return _json_escape_str(f"error: {e}")
    
    def _start_boot_message_thread(self, manager_host: str, manager_port: int) -> None:
        """Start thread to send boot messages with retry logic"""
        def run():
            registered = False
            while not registered and self._running.is_set():
                if self._send_boot_message(manager_host, manager_port):
                    registered = True
                else:
                    time.sleep(self.BOOT_MSG_RETRY_INTERVAL)  # Wait 10 seconds before retry
        
        self._boot_message_thread = threading.Thread(target=run, name="BootMessageSender", daemon=True)
        self._boot_message_thread.start()
    
    def _send_boot_message(self, manager_host: str, manager_port: int) -> bool:
        """Send boot message to RadarsManager server. Returns True if successful."""
        try:
            boot_data = {
                "radar_serial": self.radar.serial_number,
                "http_port": self.http_port,
                "tcp_port": self.tcp_port
            }
            url = f"http://{manager_host}:{manager_port}/boot"
            req = urllib.request.Request(url, data=json.dumps(boot_data).encode('utf-8'), 
                                        headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=5) as resp:
                response = json.loads(resp.read().decode('utf-8'))
                if response.get('status') == 'registered':
                    print(f"✓ Registered radar {response.get('radar_id')} with manager")
                    return True
                else:
                    print(f"Warning: Boot message failed: {response}")
                    return False
        except Exception as e:
            print(f"Warning: Could not send boot message to {manager_host}:{manager_port}: {e}")
            return False


def _json_escape_str(s: str) -> str:
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _json_escape(s: str) -> str:
    s = s or ""
    return _json_escape_str(s)


def wait_for_manager_ping(manager_host: str, manager_port: int, retry_interval: int = 2, timeout: Optional[int] = None) -> bool:
    """Wait for radars_manager to be available by checking ping endpoint.
    Returns True when manager is available, False if timeout is reached.
    
    Args:
        manager_host: Hostname or IP of the radars_manager
        manager_port: Port of the radars_manager boot server
        retry_interval: Seconds between ping attempts
        timeout: Maximum seconds to wait (None = wait indefinitely)
    """
    start_time = time.time()
    print(f"Waiting for radars_manager at {manager_host}:{manager_port}...")
    
    while True:
        try:
            url = f"http://{manager_host}:{manager_port}/ping"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as resp:
                response = json.loads(resp.read().decode('utf-8'))
                if response.get('status') == 'ok':
                    print(f"✓ radars_manager is ready at {manager_host}:{manager_port}")
                    return True
        except Exception:
            # Manager not ready yet, continue waiting
            pass
        
        # Check timeout
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                print(f"✗ Timeout waiting for radars_manager after {timeout} seconds")
                return False
        
        time.sleep(retry_interval)


def load_radars(manager_host: Optional[str] = None, manager_port: int = 9090) -> None:
    radars = scan_radar_devices()
    # Create a server for each radar device
    servers = []
    
    for i, radar in enumerate(radars):
        # Calculate ports based on config device index
        # Each node gets unique ports: 8080 + cfg_idx, 9100 + cfg_idx
        http_port = 8080 + i
        tcp_port = 9100 + i
        
        print(f"Starting radar adapter {i+1}/{len(radars)}:")
        print(f"  serial number: {radar.serial_number}")
        print(f"  Configure serial: {radar.configure_port}")
        print(f"  Data serial: {radar.data_port}")
        print(f"  HTTP port for control: {http_port}, TCP data stream port: {tcp_port}")
        
        try:
            server = RadarAdapterNodeServer(radar, http_port=http_port, tcp_port=tcp_port)
            server.start(manager_host=manager_host, manager_port=manager_port)
            servers.append(server)
            print(f"  ✓ Server running on http://0.0.0.0:{http_port} and TCP stream on port {tcp_port}")
        except Exception as e:
            print(f"  ✗ Failed to start server: {e}")
    
    if not servers:
        print("No servers started successfully")
        return
    
    print(f"\n{len(servers)} server(s) running. Press Ctrl+C to stop...")
    
    try:
        for server in servers:
            server.join()

    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            try:
                server.stop()
            except Exception:
                pass


def main():
    radars_manager_ip = argv[1] if len(argv) > 1 else "192.168.1.100"
    radars_manager_port = int(argv[2]) if len(argv) > 2 else 9090
    
    # Wait for radars_manager to be ready before loading radars
    wait_for_manager_ping(radars_manager_ip, radars_manager_port)
    
    load_radars(radars_manager_ip, radars_manager_port)


if __name__ == "__main__":
    main()
