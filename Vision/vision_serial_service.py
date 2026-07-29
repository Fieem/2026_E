"""树莓派视觉串口服务：等待 Gimbal1 START，识别一次并返回动作位姿。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

try:
    import serial
except ImportError as error:  # pragma: no cover - 树莓派部署环境提供
    raise SystemExit("缺少 pyserial，请安装 python3-serial") from error

from algorithm import Pt
from camera_pipeline import DEFAULT_HEIGHT, DEFAULT_WIDTH, load_config, open_camera, process_frame
from scara_kinematics import KinematicsError, ScaraParameters
from serial_protocol import (
    ERROR_BUSY,
    ERROR_FRAGMENT_COUNT,
    ERROR_IK_UNREACHABLE,
    ERROR_INVALID_CONFIG,
    ERROR_NO_SOLUTION,
    FrameParser,
    VISION_START,
    decode_flags,
    pack_error,
    pack_result,
)


def make_result_frames(result: dict, sequence: int, parameters: ScaraParameters) -> List[bytes]:
    """把视觉 placements 转换成每块一帧的六浮点动作包。"""

    solution = result.get("solution")
    if not solution:
        raise KinematicsError("视觉没有返回有效矩形")
    placements = sorted(solution["placements"], key=lambda item: item["piece_index"])
    if not 2 <= len(placements) <= 4:
        raise ValueError("有效碎片数量不是 2～4 块")

    frames = []
    for piece_index, placement in enumerate(placements):
        pick_center = placement["source_center_mm"]
        place_center = placement["target_center_mm"]
        pick_j1, pick_j2 = parameters.solve(float(pick_center[0]), float(pick_center[1]))
        place_j1, place_j2 = parameters.solve(float(place_center[0]), float(place_center[1]))
        pick_wrist = float(placement["pick_wrist_rad"])
        place_wrist = float(placement["place_wrist_rad"])
        values = (pick_j1, place_j1, pick_j2, place_j2, pick_wrist, place_wrist)
        if not all(math.isfinite(value) for value in values):
            raise KinematicsError(f"碎片 {piece_index} 产生了非有限角度")
        frames.append(
            pack_result(
                sequence,
                len(placements),
                piece_index,
                pick_j1,
                place_j1,
                pick_j2,
                place_j2,
                pick_wrist,
                place_wrist,
            )
        )
    return frames


def error_code_for_result(result: dict) -> int:
    if result.get("status") == "fragment_count_error":
        return ERROR_FRAGMENT_COUNT
    return ERROR_NO_SOLUTION


def run_service(
    config_path: Path,
    serial_device: str,
    baudrate: int,
    source: str,
    camera_device: int,
    output_dir: Path,
) -> int:
    config = load_config(config_path)
    camera = open_camera(source, camera_device, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    parser = FrameParser()
    last_sequence = None
    last_response: List[bytes] = []

    print(f"视觉串口服务已启动：{serial_device} @ {baudrate}")
    print("等待 Gimbal1 的 VISION_START...")
    try:
        with serial.Serial(serial_device, baudrate=baudrate, timeout=0.05) as port:
            while True:
                received = port.read(128)
                if not received:
                    continue
                for frame in parser.feed(received):
                    if frame.command != VISION_START:
                        continue
                    sequence, piece_count, piece_index = decode_flags(frame.flags)
                    del piece_count, piece_index
                    if last_sequence == sequence and last_response:
                        for response in last_response:
                            port.write(response)
                        port.flush()
                        print(f"重复任务 {sequence}，已重发缓存结果")
                        continue

                    print(f"收到任务 {sequence}，开始采图和识别")
                    responses: List[bytes]
                    try:
                        try:
                            parameters = ScaraParameters.from_config(config)
                        except KinematicsError as error:
                            if "尚未配置" in str(error):
                                raise ValueError(str(error)) from error
                            raise
                        image = camera.read()
                        result, _ = process_frame(image, config, output_dir)
                        if result.get("status") != "ok":
                            responses = [pack_error(sequence, error_code_for_result(result))]
                            print(f"任务 {sequence} 失败：{result['message']}")
                        else:
                            responses = make_result_frames(result, sequence, parameters)
                            print(f"任务 {sequence} 成功，发送 {len(responses)} 个碎片结果帧")
                    except KinematicsError as error:
                        responses = [pack_error(sequence, ERROR_IK_UNREACHABLE)]
                        print(f"任务 {sequence} 逆运动学失败：{error}")
                    except (KeyError, ValueError, TypeError) as error:
                        responses = [pack_error(sequence, ERROR_INVALID_CONFIG)]
                        print(f"任务 {sequence} 配置或结果错误：{error}")
                    for response in responses:
                        port.write(response)
                    port.flush()
                    last_sequence = sequence
                    last_response = responses
    except KeyboardInterrupt:
        print("视觉串口服务已停止")
        return 0
    finally:
        camera.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("vision_config.json"))
    parser.add_argument("--serial", default="/dev/serial0", help="树莓派串口设备")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--source", choices=("auto", "picamera2", "usb"), default="picamera2")
    parser.add_argument("--device", type=int, default=0, help="USB 摄像头编号")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    args = parser.parse_args()
    try:
        return run_service(
            args.config,
            args.serial,
            args.baudrate,
            args.source,
            args.device,
            args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"视觉串口服务启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
