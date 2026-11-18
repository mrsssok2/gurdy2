"""Minimal RFC 6455 WebSocket helpers (server side): handshake key, frame encode / decode. Text, close, ping, pong; fragmented text is
reassembled by the caller via `read_message`. No extensions, no compression."""
from __future__ import annotations

import base64
import hashlib
import struct
from typing import BinaryIO

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 8, 9, 10
MAX_MESSAGE = 64 * 1024


class WsError(Exception):
    """Protocol violation; `code` is the close status to send."""
    def __init__(self, msg: str, code: int = 1002):
        super().__init__(msg)
        self.code = code


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


def encode_frame(opcode: int, payload: bytes = b"", mask: bytes | None = None) -> bytes:
    """Server frames are unmasked; `mask` exists so tests can build client frames."""
    n = len(payload)
    head = bytes([0x80 | opcode])
    mbit = 0x80 if mask else 0
    if n < 126:
        head += bytes([mbit | n])
    elif n < 1 << 16:
        head += bytes([mbit | 126]) + struct.pack(">H", n)
    else:
        head += bytes([mbit | 127]) + struct.pack(">Q", n)
    if mask:
        return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return head + payload


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("connection closed")
        buf += chunk
    return buf


def read_frame(f: BinaryIO, max_len: int = MAX_MESSAGE) -> tuple[bool, int, bytes]:
    """-> (fin, opcode, payload). Client frames must be masked (RFC 6455 5.1); oversized frames are refused BEFORE reading the payload."""
    b0, b1 = _read_exact(f, 2)
    if b0 & 0x70:
        raise WsError("reserved bits set")
    fin, op = bool(b0 & 0x80), b0 & 0x0F
    if not b1 & 0x80:
        raise WsError("client frame not masked")
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _read_exact(f, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _read_exact(f, 8))[0]
    if n > max_len:
        raise WsError("message too big", 1009)
    if op >= 8 and (n > 125 or not fin):
        raise WsError("bad control frame")
    mask = _read_exact(f, 4)
    data = _read_exact(f, n) if n else b""
    return fin, op, bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def read_message(f: BinaryIO, send, max_len: int = MAX_MESSAGE) -> tuple[int, bytes] | None:
    """Next data message (opcode, payload), answering pings with pongs through `send(bytes)`. None on a close frame."""
    opcode, parts, total = None, [], 0
    while True:
        fin, op, data = read_frame(f, max_len)
        if op == OP_CLOSE:
            return None
        if op == OP_PING:
            send(encode_frame(OP_PONG, data))
            continue
        if op == OP_PONG:
            continue
        if op in (OP_TEXT, OP_BIN):
            if opcode is not None:
                raise WsError("new message inside a fragmented one")
            opcode = op
        elif op == OP_CONT:
            if opcode is None:
                raise WsError("continuation without start")
        else:
            raise WsError("unknown opcode")
        parts.append(data)
        total += len(data)
        if total > max_len:
            raise WsError("message too big", 1009)
        if fin:
            return opcode, b"".join(parts)
