"""树莓派视觉串口服务：等待 Gimbal1 START，识别一次并返回动作位姿。"""

from __future__ import annotations

import argparse
import math
import time
import sys
from pathlib import Path
from typing import List

import cv2

try:
    import serial
except ImportError as error:  # pragma: no cover - 树莓派部署环境提供
    raise SystemExit("缺少 pyserial，请安装 python3-serial") from error

from algorithm import Pt
from camera_pipeline import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    add_text_banner,
    create_debug_view,
    load_config,
    open_camera,
    process_frame,
)
from scara_kinematics import KinematicsError, ScaraParameters
from serial_protocol import (
    ERROR_BUSY,
    ERROR_FRAGMENT_COUNT,
    ERROR_IK_UNREACHABLE,
    ERROR_INVALID_CONFIG,
    ERROR_NO_SOLUTION,
    FrameParser,
    VISION_NEXT,
    VISION_START,
    decode_flags,
    pack_error,
    pack_result,
)

RESULT_FRAME_GAP_S = 0.1
RESULT_FRAME_LIMIT = 0
PREVIEW_WINDOW_NAME = "Vision Serial Preview"
RESULT_WINDOW_NAME = "Vision Serial Result"


def show_preview(frame, config: dict, debug_view: bool) -> None:
    """在串口服务运行时显示实时预览或调试画面。"""

    if debug_view:
        threshold = int(config["segmentation"].get("threshold", 0))
        morphology_mm = float(config["segmentation"].get("morphology_mm", 0.8))
        preview = create_debug_view(frame, config, threshold, morphology_mm)
    else:
        preview = add_text_banner(frame, "SERIAL SERVICE RUNNING - waiting for VISION_START")
    cv2.imshow(PREVIEW_WINDOW_NAME, preview)
    cv2.waitKey(1)


def print_result_frame_debug(result: dict, responses: List[bytes], sequence: int) -> None:
    """把即将发送的每个结果帧详细打印到终端。"""

    solution = result.get("solution") or {}
    placements = sorted(solution.get("placements", []), key=lambda item: item["piece_index"])
    print(f"任务 {sequence} 共发送 {len(responses)} 帧 RESULT：")
    for placement, response in zip(placements, responses):
        piece_index = int(placement["piece_index"])
        pick_j1 = float(placement["pick_j1_rad"])
        place_j1 = float(placement["place_j1_rad"])
        pick_j2 = float(placement["pick_j2_rad"])
        place_j2 = float(placement["place_j2_rad"])
        pick_wrist = float(placement["pick_wrist_rad"])
        place_wrist = float(placement["place_wrist_rad"])
        pick_wrist_abs_cam = float(placement.get("pick_wrist_abs_rad", pick_wrist))
        place_wrist_abs_cam = float(placement.get("place_wrist_abs_rad", place_wrist))
        pick_wrist_abs_base = float(placement.get("pick_wrist_abs_base_rad", pick_wrist))
        place_wrist_abs_base = float(placement.get("place_wrist_abs_base_rad", place_wrist))
        pick_tool_abs_base = float(placement.get("pick_tool_abs_base_rad", pick_wrist))
        print(
            f"  帧 {piece_index}: piece={piece_index}, bytes={len(response)}, "
            f"pick_j1={pick_j1:.4f}, place_j1={place_j1:.4f}, "
            f"pick_j2={pick_j2:.4f}, place_j2={place_j2:.4f}, "
            f"pick_wrist={pick_wrist:.4f}, place_wrist={place_wrist:.4f}, "
            f"pick_abs_cam={pick_wrist_abs_cam:.4f}, place_abs_cam={place_wrist_abs_cam:.4f}, "
            f"pick_abs_base={pick_wrist_abs_base:.4f}, place_abs_base={place_wrist_abs_base:.4f}, "
            f"pick_tool_abs_base={pick_tool_abs_base:.4f}, pick_mode=fixed"
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
        placement["pick_j1_rad"] = pick_j1
        placement["place_j1_rad"] = place_j1
        placement["pick_j2_rad"] = pick_j2
        placement["place_j2_rad"] = place_j2
        pick_absolute_orientation = float(placement["pick_wrist_rad"])
        place_absolute_orientation = float(placement["place_wrist_rad"])
        pick_absolute_orientation_base = parameters.orientation_camera_to_base(
            pick_absolute_orientation
        )
        place_absolute_orientation_base = parameters.orientation_camera_to_base(
            place_absolute_orientation
        )
        pick_wrist = parameters.fixed_pick_wrist_command()
        place_wrist = parameters.place_wrist_from_pick_and_target(
            pick_absolute_orientation,
            place_absolute_orientation,
            pick_j1,
            pick_j2,
            place_j1,
            place_j2,
        )
        pick_tool_abs_base = parameters.fixed_pick_absolute_orientation_base(
            pick_j1,
            pick_j2,
        )
        placement["pick_wrist_abs_rad"] = pick_absolute_orientation
        placement["place_wrist_abs_rad"] = place_absolute_orientation
        placement["pick_wrist_abs_base_rad"] = pick_absolute_orientation_base
        placement["place_wrist_abs_base_rad"] = place_absolute_orientation_base
        placement["pick_tool_abs_base_rad"] = pick_tool_abs_base
        placement["pick_wrist_rad"] = pick_wrist
        placement["place_wrist_rad"] = place_wrist
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
    display: bool,
    debug_view: bool,
) -> int:
    config = load_config(config_path)
    camera = open_camera(source, camera_device, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    parser = FrameParser()
    last_sequence = None
    last_response: List[bytes] = []
    cached_sequence = None
    cached_responses: List[bytes] = []
    last_preview_time = 0.0

    print(f"视觉串口服务已启动：{serial_device} @ {baudrate}")
    print("等待 Gimbal1 的 VISION_START...")
    if display:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1000, 750)
    try:
        with serial.Serial(serial_device, baudrate=baudrate, timeout=0.05) as port:
            while True:
                if display and (time.monotonic() - last_preview_time) >= 0.08:
                    preview_frame = camera.read()
                    show_preview(preview_frame, config, debug_view)
                    last_preview_time = time.monotonic()

                received = port.read(128)
                if not received:
                    continue
                for frame in parser.feed(received):
                    sequence, piece_count, piece_index = decode_flags(frame.flags)

                    if frame.command == VISION_START:
                        del piece_count, piece_index
                        if cached_sequence == sequence and cached_responses:
                            port.write(cached_responses[0])
                            port.flush()
                            print(f"重复任务 {sequence}，已重发第 0 块结果")
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
                                responses = []
                                print(f"任务 {sequence} 失败：{result['message']}")
                            else:
                                responses = make_result_frames(result, sequence, parameters)
                                if RESULT_FRAME_LIMIT > 0:
                                    responses = responses[:RESULT_FRAME_LIMIT]
                                print(f"任务 {sequence} 成功，缓存 {len(responses)} 个碎片结果帧")
                                print_result_frame_debug(result, responses, sequence)
                        except KinematicsError as error:
                            responses = []
                            print(f"任务 {sequence} 逆运动学失败：{error}")
                        except (KeyError, ValueError, TypeError) as error:
                            responses = []
                            print(f"任务 {sequence} 配置或结果错误：{error}")

                        cached_sequence = sequence
                        cached_responses = responses
                        if responses:
                            port.write(responses[0])
                            port.flush()
                            print(f"任务 {sequence} 已发送第 0 块结果")
                        if display:
                            result_image = cv2.imread(str(output_dir / "latest_result.jpg"))
                            if result_image is not None:
                                cv2.namedWindow(RESULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(RESULT_WINDOW_NAME, 1200, 800)
                                cv2.imshow(RESULT_WINDOW_NAME, result_image)
                                cv2.waitKey(1)
                        last_sequence = sequence
                        last_response = responses
                        continue

                    if frame.command == VISION_NEXT:
                        del piece_count
                        if cached_sequence != sequence or not cached_responses:
                            print(f"收到 NEXT 但没有匹配缓存：seq={sequence}, index={piece_index}")
                            continue
                        if piece_index >= len(cached_responses):
                            print(f"收到越界 NEXT：seq={sequence}, index={piece_index}, total={len(cached_responses)}")
                            continue
                        port.write(cached_responses[piece_index])
                        port.flush()
                        print(f"任务 {sequence} 已发送第 {piece_index} 块结果")
    except KeyboardInterrupt:
        print("视觉串口服务已停止")
        return 0
    finally:
        camera.close()
        if display:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("vision_config.json"))
    parser.add_argument("--serial", default="/dev/serial0", help="树莓派串口设备")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--source", choices=("auto", "picamera2", "usb"), default="picamera2")
    parser.add_argument("--device", type=int, default=0, help="USB 摄像头编号")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--display", action="store_true", help="串口服务运行时显示实时预览")
    parser.add_argument("--debug-view", action="store_true", help="显示四宫格调试画面而不是普通预览")
    args = parser.parse_args()
    try:
        return run_service(
            args.config,
            args.serial,
            args.baudrate,
            args.source,
            args.device,
            args.output_dir,
            args.display,
            args.debug_view,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"视觉串口服务启动失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
