import pyudev

from radar_device import RadarDevice

# TI IWR6843ISK Radars
VENDOR_ID = "10c4" # Silicon Lab
MODEL_ID = "ea70"


def scan_radar_devices(filter_by_model=True):
    if filter_by_model:
        devices = scan_usb_devices(filter_by_vendor=VENDOR_ID, filter_by_model=MODEL_ID)
    else:
        devices = scan_usb_devices()
    radar_devices = []
    for serial, device_serial_ports in devices:
        if len(device_serial_ports) == 2:
            radar_devices.append(RadarDevice(serial, 
                                            configure_port=device_serial_ports[0], 
                                            data_port=device_serial_ports[1]))
        else:
            print(f"Error: Expected 2 serial ports for radar device {serial}, found {len(device_serial_ports)} ports!")
    return radar_devices


def scan_usb_devices(filter_by_vendor=None, filter_by_model=None):
    context = pyudev.Context()
    devices = []
    for device in context.list_devices(subsystem='usb', DEVTYPE='usb_device'):
        if filter_by_vendor and device.get('ID_VENDOR_ID') != filter_by_vendor:
            continue
        if filter_by_model and device.get('ID_MODEL_ID') != filter_by_model:
            continue
        serial_num = device.get('ID_SERIAL_SHORT')
        device_node = pyudev.Device.from_device_file(context, device.device_node)
        device_serial_ports = [child.device_node for child in device_node.children 
                               if child.subsystem == 'tty']

        devices.append((serial_num, device_serial_ports))
    return devices