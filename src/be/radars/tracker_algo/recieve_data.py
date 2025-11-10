import struct
import zlib


def read_data(hex_data):
    """
    Read and parse the incoming data.

    The format is as follows:
    1. Magic cookie (8 bytes)
    2. Data length (4 bytes)
    3. Targets (x_tgt, y_tgt, ,z_tgt, range, dopp_tgt, time to arrive, class_byte for each target)
    4. CRC32 checksum (4 bytes)

    :param hex_data: Hexadecimal string representing the packed data
    :return: Parsed data, including magic_cookie, data_length, targets, and checksum
    """
    # Convert hex string to bytes
    data = bytes.fromhex(hex_data)

    # 1. Extract the magic cookie (8 bytes)
    magic_cookie = data[:8]

    # 2. Extract the data length (4 bytes)
    data_length = struct.unpack('!I', data[8:12])[0]  # Unpack as big-endian unsigned int

    # 3. Extract the targets (each target consists of 6 floats and 1 byte)
    targets = []
    target_size = 6 * 4 + 1  # 6 floats (4 bytes each) + 1 byte for class

    num_targets = data_length // target_size  # Since each target is 25 bytes (6 floats + 1 byte)
    target_data_start = 12  # Target data starts after magic cookie and data length

    for i in range(num_targets):
        # Extract target data (6 floats and 1 byte)
        target_data = data[target_data_start + i * target_size: target_data_start + (i + 1) * target_size]
        x_tgt, y_tgt, z_tgt, range_val,dopp_tgt, t2a, class_byte = struct.unpack('!6fc', target_data)  # Unpack target data
        targets.append((round(x_tgt,2), round(y_tgt,2), round(z_tgt,2), round(range_val,2), round(dopp_tgt,2), round(t2a,2), class_byte))

    # 4. Extract the CRC32 checksum (4 bytes)
    checksum = struct.unpack('!I', data[target_data_start + num_targets * target_size: target_data_start + num_targets * target_size + 4])[
        0]

    # 5. Verify the CRC32 checksum (for data integrity)
    calculated_checksum = zlib.crc32(data[:-4]) & 0xFFFFFFFF  # Exclude the last 4 bytes (checksum)

    if calculated_checksum != checksum:
        raise ValueError("CRC32 checksum mismatch: data is corrupted.")

    # Return the parsed data
    return {
        "magic_cookie": magic_cookie.decode('utf-8'),
        "data_length": data_length,
        "targets": targets,
        "checksum": checksum
    }


# Example usage:
hex_data = "01020403060508070000004b3f9d70a44091eb853e570a3d41a0000040fc7ae14019999a634015c28f40b570a4402d70a44174cccd410e66663f99999a68405ccccd40d8f5c33fa7ae14413e6666411028f63fc0000068ed8253e4"
parsed_data = read_data(hex_data)

# Print parsed data
print("Magic Cookie:", parsed_data["magic_cookie"])
print("Data Length:", parsed_data["data_length"])
print("Targets:")
for i, target in enumerate(parsed_data["targets"], 1):
    print(f" Target {i}: x_tgt={target[0]}, y_tgt={target[1]}, z_tgt={target[2]}, range={target[3]}, dopp_tgt={target[4]}, t2a={target[5]}, class={target[6]}")
print("Checksum:", parsed_data["checksum"])
