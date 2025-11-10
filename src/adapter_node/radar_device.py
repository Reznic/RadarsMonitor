try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    serial = None  # type: ignore

if serial is None:
    raise RuntimeError("pyserial is required! Install it with: pip install pyserial")

BAUDRATE_CONFIG = 115200
BAUDRATE_DATA = 921600


class RadarDevice:
    def __init__(self, serial, configure_port: str, data_port: str):
        self.serial_number = serial
        self.configure_port = configure_port
        self.data_port = data_port
        self.ser_config = None
        self.ser_data = None
        self._connect_serial()

    def _connect_serial(self):
        self.ser_config = serial.Serial(self.configure_port, BAUDRATE_CONFIG, timeout=0.5)
        self.ser_data = serial.Serial(self.data_port, BAUDRATE_DATA, timeout=0.5)

    def send_command(self, command):
        try:
            for command_line in command.split("\n"):
                self.ser_config.reset_input_buffer()
                self.ser_config.write((command_line + "\n").encode())
                self.ser_config.flush()
                response = b""
                while len(response) < 2:
                    response += self.ser_config.read(self.ser_config.in_waiting)
        except Exception as e:
            print(f"error sending command to radar {self.serial_number}: {e}")
            return "config serial error"

        return response

    def read_data(self):
        data = b""
        while len(data) == 0:
            try:
                data = self.ser_data.read(self.ser_data.in_waiting)
            except Exception as e:
                pass
        return data

    def close(self):
        self.ser_config.close()
        self.ser_data.close()