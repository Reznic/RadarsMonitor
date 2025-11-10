import struct
import zlib


def create_data(tgts, magic_cookie=b'\x01\x02\x04\x03\x06\x05\x08\x07'):
    """
    Create the data payload with the structure:
    1. Magic cookie (8 bytes)
    2. Data length (4 bytes)
    3. Targets (x_tgt, y_tgt, ,z_tgt, range, dopp_tgt, time to arrive, class_byte for each target)
    4. CRC32 checksum (4 bytes)

    :param tgts: List of targets, where each target is a tuple (x_tgt, y_tgt, ,z_tgt, range, dopp_tgt, time to arrive, class_byte)
    :param magic_cookie: The 8-byte magic cookie (default is b'\x01\x02\x04\x03\x06\x05\x08\x07')

    :return: Byte string representing the entire data payload.
    """
    # 1. Start with the magic cookie (8 bytes)
    data = magic_cookie

    # 2. Add data length (4 bytes) – length of the targets section in bytes
    # Each target consists of 6 floats (x, y, dopp) + 1 byte (class)
    target_size = 6 * 4 + 1  # 6 floats (4 bytes each) + 1 byte for class
    data_length = len(tgts) * target_size
    data += struct.pack('!I', data_length)  # Pack length as 4 bytes (big-endian)

    # 3. Add the targets (x_tgt, y_tgt, ,z_tgt, range, dopp_tgt, time to arrive, class_byte)
    for tgt in tgts:
        x_tgt, y_tgt, z_tgt, range_val, dopp_tgt, t2a, class_byte = tgt
        data += struct.pack('!6fc', x_tgt, y_tgt, z_tgt, range_val, dopp_tgt, t2a, class_byte)  # Pack 5 floats and 1 byte

    # 4. Calculate CRC32 checksum for the data (before appending the CRC32 itself)
    crc32_checksum = zlib.crc32(data) & 0xFFFFFFFF  # CRC32 as unsigned 32-bit integer
    data += struct.pack('!I', crc32_checksum)  # Append the CRC32 checksum

    return data


# Example usage:
tgts = [
    (1.23, 4.56, 0.21, 20.0, 7.89, 2.4, b'c'),  # (x_tgt, y_tgt, ,z_tgt, range, dopp_tgt, time to arrive, class_byte)
    (2.34, 5.67, 2.71, 15.3, 8.90, 1.2, b'h'),
    (3.45, 6.78, 1.31, 11.9, 9.01, 1.5, b'h')
]

# Create the data
data = create_data(tgts)

# Print the resulting data as a hex string for inspection
print("Generated Data (Hex):", data.hex())
