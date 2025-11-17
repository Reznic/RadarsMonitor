try:
    import serial  # pyserial
    from serial import SerialException
except ImportError:  # pragma: no cover
    serial = None  # type: ignore
    SerialException = Exception  # type: ignore

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

    def is_connected(self) -> bool:
        """Check if both serial ports are still connected and open.
        
        Returns:
            True if both ports are open and appear to be connected, False otherwise.
        """
        try:
            # Check if ports are open
            if not (self.ser_config and self.ser_config.is_open):
                return False
            if not (self.ser_data and self.ser_data.in_waiting is not None):
                return False
            
            # Try a non-blocking read to verify the port is still accessible
            # This will raise an exception if the device is disconnected
            try:
                _ = self.ser_data.in_waiting
                _ = self.ser_config.in_waiting
            except (SerialException, OSError, IOError):
                return False
            
            return True
        except (AttributeError, SerialException, OSError, IOError):
            return False

    def send_command(self, command):
        try:
            # Check if device is still connected before attempting to send
            if not self.is_connected():
                raise SerialException("Device disconnected")
            
            for command_line in command.split("\n"):
                self.ser_config.reset_input_buffer()
                self.ser_config.write((command_line + "\n").encode())
                self.ser_config.flush()
                response = b""
                while len(response) < 2:
                    response += self.ser_config.read(self.ser_config.in_waiting)
        except (SerialException, OSError, IOError) as e:
            print(f"error sending command to radar {self.serial_number}: {e}")
            # Mark as disconnected
            self._mark_disconnected()
            return "config serial error"
        except Exception as e:
            print(f"error sending command to radar {self.serial_number}: {e}")
            return "config serial error"

        return response

    def read_data(self):
        data = b""
        while len(data) == 0:
            try:
                # Check if device is still connected before attempting to read
                if not self.is_connected():
                    raise SerialException("Device disconnected")
                
                data = self.ser_data.read(self.ser_data.in_waiting)
            except (SerialException, OSError, IOError) as e:
                # Mark as disconnected and re-raise to let caller handle it
                self._mark_disconnected()
                raise
            except Exception as e:
                # Other exceptions, continue trying
                pass
        return data

    def _mark_disconnected(self):
        """Mark the serial ports as disconnected (close them safely)."""
        try:
            if self.ser_config and self.ser_config.is_open:
                self.ser_config.close()
        except Exception:
            pass
        try:
            if self.ser_data and self.ser_data.is_open:
                self.ser_data.close()
        except Exception:
            pass

    def close(self):
        """Close both serial ports."""
        self._mark_disconnected()