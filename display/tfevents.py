"""Read scalars from TensorBoard event files with only the standard library.

ROLL's TensorBoardTracker writes `events.out.tfevents.*` files with
`SummaryWriter.add_scalar` (simple_value). Newer writers store scalars as
tensors; both are handled. CRCs are not checked.
"""

import glob
import os
import struct


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) for one protobuf message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i:i + length], i + length
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:  # groups (3/4) are not used by TensorBoard
            return
        yield field, wire, value


def _tensor_value(buf):
    dtype, floats, doubles, content = 1, [], [], b""
    for field, wire, value in _fields(buf):
        if field == 1 and wire == 0:
            dtype = value
        elif field == 4 and wire == 2:
            content = value
        elif field == 5:
            if wire == 2:
                floats += list(struct.unpack(f"<{len(value) // 4}f", value))
            elif wire == 5:
                floats.append(struct.unpack("<f", value)[0])
        elif field == 6:
            if wire == 2:
                doubles += list(struct.unpack(f"<{len(value) // 8}d", value))
            elif wire == 1:
                doubles.append(struct.unpack("<d", value)[0])
    if floats:
        return floats[0]
    if doubles:
        return doubles[0]
    if content:
        if dtype == 2 and len(content) >= 8:
            return struct.unpack("<d", content[:8])[0]
        if len(content) >= 4:
            return struct.unpack("<f", content[:4])[0]
    return None


def _records(path):
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            f.read(4)
            if len(data) < length:
                return
            yield data


def read_scalars(path):
    """{tag: {step: value}} from one event file."""
    out = {}
    for record in _records(path):
        step, summary = 0, None
        for field, wire, value in _fields(record):
            if field == 2 and wire == 0:
                step = value
            elif field == 5 and wire == 2:
                summary = value
        if summary is None:
            continue
        for field, wire, value in _fields(summary):
            if field != 1 or wire != 2:
                continue
            tag, number = None, None
            for f2, w2, v2 in _fields(value):
                if f2 == 1 and w2 == 2:
                    tag = v2.decode("utf-8", errors="replace")
                elif f2 == 2 and w2 == 5:
                    number = struct.unpack("<f", v2)[0]
                elif f2 == 8 and w2 == 2:
                    number = _tensor_value(v2)
            if tag is not None and number is not None:
                out.setdefault(tag, {})[step] = number
    return out


def read_dir(directory):
    """Merge every event file under a directory (later files win on clashes)."""
    merged = {}
    files = sorted(glob.glob(os.path.join(directory, "**", "events.out.tfevents.*"), recursive=True),
                   key=os.path.getmtime)
    for path in files:
        for tag, points in read_scalars(path).items():
            merged.setdefault(tag, {}).update(points)
    return merged
