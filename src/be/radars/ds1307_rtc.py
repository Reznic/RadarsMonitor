"""
DS1307 RTC (I2C 0x68) helper for Jetson/Linux.

Wiring you described (Jetson 40-pin header):
- Pin 1: 3.3V
- Pin 3: I2C SDA
- Pin 5: I2C SCL
- Pin 6: GND

Notes:
- DS1307 stores time as local time with no timezone information. This module uses
  naive `datetime.datetime` values (no tzinfo).
- The DS1307 typically runs at 5V. Many breakout boards include level shifting,
  but if yours does not, ensure the electrical interface is safe for Jetson 3.3V I2C.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import re
import subprocess
from typing import Iterable, Optional, Protocol, Sequence


class _SMBusLike(Protocol):
    def read_i2c_block_data(self, i2c_addr: int, register: int, length: int) -> list[int]: ...

    def write_i2c_block_data(self, i2c_addr: int, register: int, data: Sequence[int]) -> None: ...

    def close(self) -> None: ...


def _bcd_to_int(x: int) -> int:
    return ((x >> 4) * 10) + (x & 0x0F)


def _int_to_bcd(x: int) -> int:
    if x < 0 or x > 99:
        raise ValueError(f"Value out of BCD range: {x}")
    return ((x // 10) << 4) | (x % 10)


def _open_bus(bus: int) -> _SMBusLike:
    """
    Open an I2C bus using smbus2 if available, otherwise smbus.
    """
    try:
        from smbus2 import SMBus  # type: ignore
    except Exception:  # pragma: no cover
        from smbus import SMBus  # type: ignore
    return SMBus(bus)  # type: ignore[return-value]


def _decode_hour(hour_reg: int) -> int:
    """
    DS1307 hour register supports 12h/24h.
    Bit 6: 1=12h mode, 0=24h mode
    """
    is_12h = (hour_reg & 0x40) != 0
    if not is_12h:
        return _bcd_to_int(hour_reg & 0x3F)

    # 12-hour mode:
    # bit 5: AM/PM (1=PM)
    # bits 4..0: hour (1..12) in BCD
    pm = (hour_reg & 0x20) != 0
    hour_12 = _bcd_to_int(hour_reg & 0x1F)
    if hour_12 < 1 or hour_12 > 12:
        raise ValueError(f"Invalid 12-hour value in DS1307: {hour_12}")

    if hour_12 == 12:
        return 12 if pm else 0
    return hour_12 + (12 if pm else 0)


def _encode_hour_24h(hour: int) -> int:
    if hour < 0 or hour > 23:
        raise ValueError(f"Hour out of range: {hour}")
    # force 24h mode (bit 6 = 0)
    return _int_to_bcd(hour) & 0x3F


@dataclass(frozen=True)
class DS1307RTC:
    """
    Minimal DS1307 RTC driver.

    Example:
        rtc = DS1307RTC(bus=1)
        now = rtc.get_time()
        rtc.set_time(datetime.now())
    """

    bus: int = 1
    address: int = 0x68
    sync_on_init: bool = False
    sync_cutoff_year: int = 2026

    def __post_init__(self) -> None:
        if self.sync_on_init:
            self.sync_with_system_time(cutoff_year=self.sync_cutoff_year)

    def get_time(self) -> datetime:
        return get_time(bus=self.bus, address=self.address)

    def set_time(self, dt: datetime) -> None:
        set_time(dt, bus=self.bus, address=self.address)

    def sync_with_system_time(self, *, cutoff_year: int = 2026) -> None:
        """
        Sync policy:
        - If system year >= cutoff_year: write system time -> RTC
        - Else: read RTC -> attempt to set system time
        """
        system_now = datetime.now()
        if system_now.year >= cutoff_year:
            self.set_time(system_now)
            return

        rtc_now = self.get_time()
        _try_set_system_time(rtc_now)


def _try_set_system_time(dt: datetime) -> bool:
    """
    Best-effort system time update.

    This may require root (or passwordless sudo) depending on how the service is started.
    Returns True if any command succeeded.
    """
    timestr = dt.strftime("%Y-%m-%d %H:%M:%S")
    for cmd in (
        ["timedatectl", "set-time", timestr],
        ["sudo", "-n", "timedatectl", "set-time", timestr],
        ["date", "-s", timestr],
        ["sudo", "-n", "date", "-s", timestr],
    ):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue
    return False


def list_i2c_buses() -> list[int]:
    """
    Return detected I2C bus numbers from /dev/i2c-*.
    """
    buses: list[int] = []
    try:
        for name in os.listdir("/dev"):
            m = re.fullmatch(r"i2c-(\d+)", name)
            if m:
                buses.append(int(m.group(1)))
    except FileNotFoundError:
        return []
    buses.sort()
    return buses


def probe_device(*, bus: int, address: int = 0x68) -> bool:
    """
    Try a minimal read from the device. Returns True if it ACKs.
    """
    b = _open_bus(bus)
    try:
        b.read_i2c_block_data(address, 0x00, 1)
        return True
    except OSError:
        return False
    finally:
        b.close()


def find_ds1307_bus(*, buses: Optional[Iterable[int]] = None, address: int = 0x68) -> Optional[int]:
    """
    Scan I2C buses to find which bus has a DS1307 responding at the given address.
    """
    scan_buses = list(buses) if buses is not None else list_i2c_buses()
    for bus in scan_buses:
        if probe_device(bus=bus, address=address):
            return bus
    return None


def get_time(*, bus: int = 1, address: int = 0x68) -> datetime:
    """
    Read current time from DS1307.

    Returns:
        datetime: Naive local datetime (no timezone).
    """
    b = _open_bus(bus)
    try:
        # Registers 0x00..0x06: sec, min, hour, day-of-week, date, month, year
        data = b.read_i2c_block_data(address, 0x00, 7)
    except OSError as e:
        raise RuntimeError(
            "I2C read failed for DS1307. Common causes: wrong I2C bus number, wrong wiring, "
            "device not powered, or a DS1307 breakout that requires 5V/pullups. "
            f"Tried bus={bus} address=0x{address:02x}. Original error: {e!r}"
        ) from e
    finally:
        b.close()

    sec_reg, min_reg, hour_reg, _dow_reg, date_reg, month_reg, year_reg = data

    # seconds bit7 = CH (clock halt). If set, oscillator is stopped; seconds are in bits 6..0.
    seconds = _bcd_to_int(sec_reg & 0x7F)
    minutes = _bcd_to_int(min_reg & 0x7F)
    hours = _decode_hour(hour_reg)
    day = _bcd_to_int(date_reg & 0x3F)
    month = _bcd_to_int(month_reg & 0x1F)
    year = 2000 + _bcd_to_int(year_reg)

    return datetime(year, month, day, hours, minutes, seconds)


def set_time(dt: datetime, *, bus: int = 1, address: int = 0x68, day_of_week: Optional[int] = None) -> None:
    """
    Set DS1307 time.

    Args:
        dt: Naive datetime to write (treated as local time).
        bus: I2C bus number (Jetson usually 1).
        address: I2C address (default 0x68).
        day_of_week: Optional 1..7 (DS1307 convention). If not provided, derived from dt.isoweekday().
    """
    if dt.tzinfo is not None:
        raise ValueError("DS1307 stores no timezone; pass a naive datetime (tzinfo=None).")

    dow = day_of_week if day_of_week is not None else dt.isoweekday()
    if dow < 1 or dow > 7:
        raise ValueError("day_of_week must be in range 1..7")

    sec = _int_to_bcd(dt.second) & 0x7F  # ensure CH=0 (oscillator running)
    minute = _int_to_bcd(dt.minute)
    hour = _encode_hour_24h(dt.hour)
    dow_bcd = _int_to_bcd(dow)
    date = _int_to_bcd(dt.day)
    month = _int_to_bcd(dt.month)
    year = _int_to_bcd(dt.year % 100)

    b = _open_bus(bus)
    try:
        b.write_i2c_block_data(address, 0x00, [sec, minute, hour, dow_bcd, date, month, year])
    except OSError as e:
        raise RuntimeError(
            "I2C write failed for DS1307. Common causes: wrong I2C bus number, wrong wiring, "
            "device not powered, or a DS1307 breakout that requires 5V/pullups. "
            f"Tried bus={bus} address=0x{address:02x}. Original error: {e!r}"
        ) from e
    finally:
        b.close()


if __name__ == "__main__":
    # Allow overriding via environment:
    #   DS1307_I2C_BUS=7 DS1307_I2C_ADDR=0x68 python3 ds1307_rtc.py
    bus_env = os.getenv("DS1307_I2C_BUS")
    addr_env = os.getenv("DS1307_I2C_ADDR")
    address = int(addr_env, 0) if addr_env else 0x68

    detected = find_ds1307_bus(address=address)
    if bus_env:
        bus = int(bus_env)
    else:
        bus = detected if detected is not None else 1
        if detected is not None and detected != 1:
            print(f"Note: DS1307 responded on bus {detected}. Using DS1307_I2C_BUS={detected}.")

    rtc = DS1307RTC(bus=bus, address=address)

    # Set RTC to current system time, then read back and print.
    rtc.set_time(datetime.now())
    print("DS1307 time:", rtc.get_time().isoformat(sep=" ", timespec="seconds"))
