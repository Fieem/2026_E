"""树莓派与 Gimbal1 共用的无 CRC 二进制串口协议。"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, List, Optional


FRAME_HEADER = 0xA5
MAX_FLOATS = 8
MAX_PAYLOAD_BYTES = 4 + MAX_FLOATS * 4

VISION_START = 0x0201
VISION_RESULT = 0x0202
VISION_ERROR = 0x0203

ERROR_FRAGMENT_COUNT = 1
ERROR_NO_SOLUTION = 2
ERROR_IK_UNREACHABLE = 3
ERROR_INVALID_CONFIG = 4
ERROR_BUSY = 5
ERROR_INVALID_FRAME = 6


@dataclass(frozen=True)
class Frame:
    command: int
    flags: int
    values: List[float]


def _pack_frame(command: int, flags: int, values: Iterable[float]) -> bytes:
    values = list(values)
    if len(values) > MAX_FLOATS:
        raise ValueError("单帧最多只能携带 8 个 float32")
    payload = struct.pack("<HH", command, flags & 0xFFFF)
    if values:
        payload += struct.pack("<" + "f" * len(values), *values)
    return struct.pack("<BB", FRAME_HEADER, len(payload)) + payload


def encode_flags(sequence: int, piece_count: int = 0, piece_index: int = 0) -> int:
    """编码任务序号、碎片数量和碎片编号。"""

    if not 0 <= sequence <= 0x0FFF:
        raise ValueError("任务序号必须在 0～4095")
    if piece_count not in (0, 2, 3, 4):
        raise ValueError("碎片数量必须是 0、2、3 或 4")
    if not 0 <= piece_index <= 3:
        raise ValueError("碎片编号必须在 0～3")
    # 0 表示 START/ERROR 没有碎片数量，1/2/3 分别表示 2/3/4 块。
    count_code = 0 if piece_count == 0 else piece_count - 1
    return sequence | (count_code << 12) | (piece_index << 14)


def decode_flags(flags: int):
    count_code = (flags >> 12) & 0x3
    piece_count = 0 if count_code == 0 else count_code + 1
    return flags & 0x0FFF, piece_count, (flags >> 14) & 0x3


def pack_start(sequence: int) -> bytes:
    return _pack_frame(VISION_START, encode_flags(sequence), [])


def pack_result(
    sequence: int,
    piece_count: int,
    piece_index: int,
    pick_j1: float,
    place_j1: float,
    pick_j2: float,
    place_j2: float,
    pick_wrist: float,
    place_wrist: float,
) -> bytes:
    """打包一块碎片的动作角度。

    数据顺序固定为：拾取 J1、放置 J1、拾取 J2、放置 J2、
    拾取腕部角、放置腕部角。
    """
    return _pack_frame(
        VISION_RESULT,
        encode_flags(sequence, piece_count, piece_index),
        [pick_j1, place_j1, pick_j2, place_j2, pick_wrist, place_wrist],
    )


def pack_error(sequence: int, error_code: int) -> bytes:
    return _pack_frame(VISION_ERROR, encode_flags(sequence), [float(error_code)])


class FrameParser:
    """按 A5+长度格式从任意串口字节流中恢复完整帧。"""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[Frame]:
        self._buffer.extend(data)
        frames: List[Frame] = []
        while True:
            try:
                header_index = self._buffer.index(FRAME_HEADER)
            except ValueError:
                self._buffer.clear()
                break
            if header_index:
                del self._buffer[:header_index]
            if len(self._buffer) < 2:
                break
            payload_length = self._buffer[1]
            if payload_length < 4 or payload_length > MAX_PAYLOAD_BYTES:
                del self._buffer[0]
                continue
            total_length = 2 + payload_length
            if len(self._buffer) < total_length:
                break
            payload = bytes(self._buffer[2:total_length])
            del self._buffer[:total_length]
            command, flags = struct.unpack_from("<HH", payload, 0)
            float_bytes = payload[4:]
            if len(float_bytes) % 4:
                continue
            count = len(float_bytes) // 4
            values = list(struct.unpack("<" + "f" * count, float_bytes))
            frames.append(Frame(command, flags, values))
        return frames


def unpack_result(frame: Frame):
    if frame.command != VISION_RESULT or len(frame.values) != 6:
        raise ValueError("不是合法的 VISION_RESULT 帧")
    sequence, piece_count, piece_index = decode_flags(frame.flags)
    if piece_count not in (2, 3, 4) or piece_index >= piece_count:
        raise ValueError("VISION_RESULT 的碎片信息无效")
    return sequence, piece_count, piece_index, frame.values


def unpack_error(frame: Frame):
    if frame.command != VISION_ERROR or len(frame.values) != 1:
        raise ValueError("不是合法的 VISION_ERROR 帧")
    sequence, _, _ = decode_flags(frame.flags)
    return sequence, int(round(frame.values[0]))
