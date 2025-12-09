"""
Jetson GPIO Controller for MOSFET Transistor Control

This module provides a class to control GPIO pins on NVIDIA Jetson Orin devices,
specifically designed for controlling MOSFET transistors for on/off switching.

Requirements:
    - Jetson.GPIO library (usually pre-installed on Jetson devices)
    - Run with sudo privileges for GPIO access

Example usage:
    from jetson_gpio_controller import JetsonGPIOController
    
    # Initialize controller with default MOSFET pin (GPIO 18)
    controller = JetsonGPIOController()
    
    # Turn MOSFET ON
    controller.mosfet_on()
    
    # Turn MOSFET OFF
    controller.mosfet_off()
    
    # Start warning alarm (beeping pattern, runs in background thread)
    controller.start_warning_alarm()
    
    # Stop warning alarm
    controller.stop_warning_alarm()
    
    # Cleanup when done
    controller.cleanup()
"""

import logging
import threading
import time
from typing import Optional

# Try to import Jetson.GPIO, fallback for non-Jetson systems
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    GPIO = None
    print("Warning: Jetson.GPIO not available. Running in simulation mode.")


class JetsonGPIOController:
    """
    GPIO Controller for NVIDIA Jetson Orin devices.
    
    Provides methods to control GPIO pins, specifically designed for MOSFET
    transistor control. Uses GPIO pin 18 by default (physical pin 12 on 40-pin header).
    
    Attributes:
        mosfet_pin: GPIO pin number used for MOSFET control (default: 18)
        is_initialized: Whether GPIO has been initialized
        simulation_mode: Whether running in simulation mode (non-Jetson system)
    """
    
    # Default MOSFET control pin: GPIO 18 (Physical pin 12 on 40-pin header)
    # This pin is safe to use and not reserved for other functions
    DEFAULT_MOSFET_PIN = 18
    
    def __init__(self, mosfet_pin: int = DEFAULT_MOSFET_PIN, simulation_mode: bool = False):
        """
        Initialize the GPIO controller.
        
        Args:
            mosfet_pin: GPIO pin number for MOSFET control (default: 18)
            simulation_mode: Force simulation mode even if GPIO is available
        """
        self.mosfet_pin = mosfet_pin
        self.is_initialized = False
        self.simulation_mode = simulation_mode or not GPIO_AVAILABLE
        
        # Warning alarm thread control
        self._alarm_thread: Optional[threading.Thread] = None
        self._alarm_stop_event = threading.Event()
        self._alarm_active = False
        self._alarm_lock = threading.Lock()
        
        # Alarm timing configuration (beep pattern)
        self.alarm_on_duration = 0.5  # seconds - horn ON time
        self.alarm_off_duration = 0.5  # seconds - horn OFF time
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        if not self.simulation_mode:
            try:
                # Set GPIO mode to BOARD (use physical pin numbering)
                # Alternative: GPIO.BCM (use Broadcom pin numbering)
                GPIO.setmode(GPIO.BOARD)
                
                # Setup MOSFET pin as output
                GPIO.setup(self.mosfet_pin, GPIO.OUT, initial=GPIO.LOW)
                
                self.is_initialized = True
                self.logger.info(f"GPIO initialized. MOSFET control on pin {self.mosfet_pin}")
                print(f"GPIO Controller initialized. MOSFET pin: {self.mosfet_pin}")
            except Exception as e:
                self.logger.error(f"Failed to initialize GPIO: {e}")
                print(f"Warning: Failed to initialize GPIO: {e}. Running in simulation mode.")
                self.simulation_mode = True
        else:
            self.logger.info("Running in simulation mode (GPIO not available)")
            print("GPIO Controller running in simulation mode")
    
    def mosfet_on(self) -> bool:
        """
        Turn the MOSFET transistor ON.
        
        Sets the GPIO pin to HIGH, which turns on the MOSFET.
        For N-channel MOSFET: HIGH = ON, LOW = OFF
        For P-channel MOSFET: LOW = ON, HIGH = OFF (invert logic if needed)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.simulation_mode and self.is_initialized:
                GPIO.output(self.mosfet_pin, GPIO.HIGH)
                self.logger.info(f"MOSFET turned ON (pin {self.mosfet_pin} = HIGH)")
                print(f"MOSFET turned ON (pin {self.mosfet_pin})")
                return True
            else:
                self.logger.info(f"MOSFET ON (simulation mode, pin {self.mosfet_pin})")
                print(f"MOSFET ON (simulation mode, pin {self.mosfet_pin})")
                return True
        except Exception as e:
            self.logger.error(f"Failed to turn MOSFET ON: {e}")
            print(f"Error: Failed to turn MOSFET ON: {e}")
            return False
    
    def mosfet_off(self) -> bool:
        """
        Turn the MOSFET transistor OFF.
        
        Sets the GPIO pin to LOW, which turns off the MOSFET.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.simulation_mode and self.is_initialized:
                GPIO.output(self.mosfet_pin, GPIO.LOW)
                self.logger.info(f"MOSFET turned OFF (pin {self.mosfet_pin} = LOW)")
                print(f"MOSFET turned OFF (pin {self.mosfet_pin})")
                return True
            else:
                self.logger.info(f"MOSFET OFF (simulation mode, pin {self.mosfet_pin})")
                print(f"MOSFET OFF (simulation mode, pin {self.mosfet_pin})")
                return True
        except Exception as e:
            self.logger.error(f"Failed to turn MOSFET OFF: {e}")
            print(f"Error: Failed to turn MOSFET OFF: {e}")
            return False
    
    def get_mosfet_state(self) -> Optional[bool]:
        """
        Get the current state of the MOSFET control pin.
        
        Returns:
            True if MOSFET is ON (HIGH), False if OFF (LOW), None if error
        """
        try:
            if not self.simulation_mode and self.is_initialized:
                state = GPIO.input(self.mosfet_pin)
                return bool(state)
            else:
                # In simulation mode, return None (unknown state)
                return None
        except Exception as e:
            self.logger.error(f"Failed to read MOSFET state: {e}")
            return None
    
    def toggle_mosfet(self) -> bool:
        """
        Toggle the MOSFET state (ON -> OFF or OFF -> ON).
        
        Returns:
            True if successful, False otherwise
        """
        current_state = self.get_mosfet_state()
        if current_state is None:
            # In simulation mode, just toggle
            print("Toggling MOSFET (simulation mode)")
            return True
        
        if current_state:
            return self.mosfet_off()
        else:
            return self.mosfet_on()
    
    def cleanup(self) -> None:
        """
        Clean up GPIO resources.
        
        Stops any active alarm and resets all GPIO pins to their default state.
        Should be called when done with GPIO operations.
        """
        # Stop alarm if active
        if self._alarm_active:
            self.stop_warning_alarm()
        
        try:
            if not self.simulation_mode and self.is_initialized:
                GPIO.cleanup(self.mosfet_pin)
                self.is_initialized = False
                self.logger.info("GPIO cleanup completed")
                print("GPIO cleanup completed")
            else:
                self.logger.info("GPIO cleanup (simulation mode)")
                print("GPIO cleanup (simulation mode)")
        except Exception as e:
            self.logger.error(f"Error during GPIO cleanup: {e}")
            print(f"Warning: Error during GPIO cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        self.cleanup()
    
    def start_warning_alarm(self) -> bool:
        """
        Start the warning alarm (horn) in a separate thread.
        
        The alarm will continue playing (beeping pattern) until stop_warning_alarm() is called.
        Creates a beeping pattern: ON for alarm_on_duration, OFF for alarm_off_duration, repeat.
        
        Returns:
            True if alarm started successfully, False otherwise
        """
        with self._alarm_lock:
            if self._alarm_active:
                self.logger.warning("Warning alarm is already active")
                print("Warning alarm is already active")
                return True
            
            # Clear the stop event
            self._alarm_stop_event.clear()
            self._alarm_active = True
            
            # Create and start the alarm thread
            self._alarm_thread = threading.Thread(
                target=self._alarm_thread_worker,
                name="WarningAlarmThread",
                daemon=True
            )
            self._alarm_thread.start()
            
            self.logger.info("Warning alarm started")
            print("Warning alarm started - horn is beeping")
            return True
    
    def stop_warning_alarm(self) -> bool:
        """
        Stop the warning alarm (horn).
        
        Signals the alarm thread to stop and waits for it to finish.
        The horn will be turned OFF.
        
        Returns:
            True if alarm stopped successfully, False otherwise
        """
        with self._alarm_lock:
            if not self._alarm_active:
                self.logger.info("Warning alarm is not active")
                return True
            
            # Signal the thread to stop
            self._alarm_stop_event.set()
            self._alarm_active = False
            
            # Wait for thread to finish (with timeout)
            if self._alarm_thread and self._alarm_thread.is_alive():
                self._alarm_thread.join(timeout=2.0)
                if self._alarm_thread.is_alive():
                    self.logger.warning("Alarm thread did not stop within timeout")
            
            # Ensure MOSFET is off
            self.mosfet_off()
            
            self.logger.info("Warning alarm stopped")
            print("Warning alarm stopped - horn is OFF")
            return True
    
    def is_alarm_active(self) -> bool:
        """
        Check if the warning alarm is currently active.
        
        Returns:
            True if alarm is active, False otherwise
        """
        with self._alarm_lock:
            return self._alarm_active
    
    def _alarm_thread_worker(self) -> None:
        """
        Worker thread function for the warning alarm.
        
        Continuously toggles the MOSFET ON and OFF to create a beeping pattern
        until the stop event is set.
        """
        try:
            self.logger.info("Alarm thread started")
            
            while not self._alarm_stop_event.is_set():
                # Turn horn ON
                self.mosfet_on()
                
                # Wait for ON duration or until stop event
                if self._alarm_stop_event.wait(timeout=self.alarm_on_duration):
                    # Stop event was set, break immediately
                    break
                
                # Turn horn OFF
                self.mosfet_off()
                
                # Wait for OFF duration or until stop event
                if self._alarm_stop_event.wait(timeout=self.alarm_off_duration):
                    # Stop event was set, break immediately
                    break
            
            # Ensure MOSFET is off when thread exits
            self.mosfet_off()
            self.logger.info("Alarm thread stopped")
            
        except Exception as e:
            self.logger.error(f"Error in alarm thread: {e}")
            # Ensure MOSFET is off on error
            try:
                self.mosfet_off()
            except:
                pass


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("Jetson GPIO Controller - MOSFET Control Test")
    print("=" * 50)
    
    # Initialize controller
    controller = JetsonGPIOController(mosfet_pin=18)
    
    try:
        # Test MOSFET control
        print("\n1. Turning MOSFET ON...")
        controller.mosfet_on()
        time.sleep(2)
        
        # Check state
        state = controller.get_mosfet_state()
        print(f"   MOSFET state: {'ON' if state else 'OFF' if state is False else 'Unknown'}")
        
        print("\n2. Turning MOSFET OFF...")
        controller.mosfet_off()
        time.sleep(2)
        
        # Check state
        state = controller.get_mosfet_state()
        print(f"   MOSFET state: {'ON' if state else 'OFF' if state is False else 'Unknown'}")
        
        print("\n3. Toggling MOSFET...")
        controller.toggle_mosfet()
        time.sleep(1)
        controller.toggle_mosfet()
        
        print("\n4. Testing warning alarm...")
        print("   Starting alarm (will beep for 5 seconds)...")
        controller.start_warning_alarm()
        time.sleep(5)
        print("   Stopping alarm...")
        controller.stop_warning_alarm()
        
        print("\nTest completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Cleanup
        controller.cleanup()
        print("\nGPIO resources cleaned up")

