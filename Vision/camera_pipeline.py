"""树莓派 1600×1200 摄像头视觉程序。

功能流程：采集图像、四点透视矫正、碎片轮廓提取、毫米坐标转换、
矩形拼接求解，并输出带英文标注的结果图片和中文 JSON 数据。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from algorithm import Pt, find_rectangle_solution, polygon_area
from jqk_template_matcher import match_card_to_templates
from texture_scorer import TextureScorer, solve_with_texture


DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
COLORS = [
    (60, 76, 231),
    (219, 152, 52),
    (113, 204, 46),
    (18, 156, 243),
]
VALID_PIECE_MODES = {"plain", "playing_cards"}


@dataclass
class DetectedPiece:
    """一块完成多边形拟合的碎片。"""

    contour_px: np.ndarray
    points_mm: List[Pt]
    area_mm2: float
    piece_image: Optional[np.ndarray] = None
    polygon_in_image: Optional[List[Pt]] = None


def load_config(path: Path) -> Dict:
    """读取配置并检查关键参数。"""

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    camera = config["camera"]
    workspace = config["workspace"]
    if camera["width"] != DEFAULT_WIDTH or camera["height"] != DEFAULT_HEIGHT:
        raise ValueError("camera.width 和 camera.height 必须设置为 1600×1200")
    if len(workspace["image_points"]) != 4:
        raise ValueError("workspace.image_points 必须包含四个标定点")
    if workspace["pixels_per_mm"] <= 0:
        raise ValueError("workspace.pixels_per_mm 必须大于零")
    if float(workspace.get("rectify_margin_mm", 10.0)) < 0.0:
        raise ValueError("workspace.rectify_margin_mm 不能小于零")
    if float(config.get("solver", {}).get("overlap_tolerance_mm", 2.5)) < 0.0:
        raise ValueError("solver.overlap_tolerance_mm 不能小于零")
    if float(config.get("solver", {}).get("placement_spread_mm", 0.0)) < 0.0:
        raise ValueError("solver.placement_spread_mm 不能小于零")
    if float(config.get("solver", {}).get("local_overlap_clearance_mm", 2.0)) < 0.0:
        raise ValueError("solver.local_overlap_clearance_mm 不能小于零")
    if float(config.get("solver", {}).get("local_overlap_step_mm", 1.5)) <= 0.0:
        raise ValueError("solver.local_overlap_step_mm 必须大于零")
    if int(config.get("solver", {}).get("local_overlap_max_iterations", 8)) < 0:
        raise ValueError("solver.local_overlap_max_iterations 不能小于零")
    piece_mode = str(config.get("piece_mode", "plain"))
    if piece_mode not in VALID_PIECE_MODES:
        raise ValueError("piece_mode 只能是 plain 或 playing_cards")
    texture = config.get("texture", {})
    if float(texture.get("strip_width_mm", 5.0)) <= 0.0:
        raise ValueError("texture.strip_width_mm 必须大于零")
    if float(texture.get("lambda_texture", 0.5)) < 0.0:
        raise ValueError("texture.lambda_texture 不能小于零")
    jqk_template = config.get("jqk_template", {})
    if int(jqk_template.get("canvas_width_px", 256)) <= 0:
        raise ValueError("jqk_template.canvas_width_px 必须大于零")
    if int(jqk_template.get("canvas_height_px", 384)) <= 0:
        raise ValueError("jqk_template.canvas_height_px 必须大于零")
    if int(jqk_template.get("candidate_top_k", 7)) <= 0:
        raise ValueError("jqk_template.candidate_top_k 必须大于零")
    if float(jqk_template.get("min_confidence", 0.62)) < 0.0:
        raise ValueError("jqk_template.min_confidence 不能小于零")
    orientations = jqk_template.get("match_orientations_deg", [0, 180])
    if not orientations:
        raise ValueError("jqk_template.match_orientations_deg 不能为空")
    for orientation_deg in orientations:
        if int(orientation_deg) % 360 not in (0, 180):
            raise ValueError("jqk_template.match_orientations_deg 目前只支持 0 或 180")
    return config


def get_piece_mode(config: Dict) -> str:
    """返回当前碎片模式。"""

    piece_mode = str(config.get("piece_mode", "plain"))
    return piece_mode if piece_mode in VALID_PIECE_MODES else "plain"


def texture_pipeline_enabled(config: Dict) -> bool:
    """仅在扑克牌模式下启用纹理评分流程。"""

    if get_piece_mode(config) != "playing_cards":
        return False
    return bool(config.get("texture", {}).get("enabled", True))


def jqk_template_enabled(config: Dict) -> bool:
    """仅在扑克牌模式下启用 J/Q/K 模板重排。"""

    if get_piece_mode(config) != "playing_cards":
        return False
    return bool(config.get("jqk_template", {}).get("enabled", False))


def resolve_vision_relative_path(path_text: str) -> Path:
    """将配置中的相对路径解析到 Vision 目录。"""

    path = Path(path_text)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def save_config(path: Path, config: Dict) -> None:
    """以 UTF-8 中文格式保存配置。"""

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


class PiCameraSource:
    """树莓派 CSI 摄像头采集源。"""

    def __init__(self, width: int, height: int):
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("未安装 Picamera2，请安装 python3-picamera2") from error

        self.camera = Picamera2()
        camera_config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=3,
        )
        self.camera.configure(camera_config)
        self.camera.start()
        # 给自动曝光和自动白平衡留出稳定时间。
        time.sleep(1.5)

    def read(self) -> np.ndarray:
        rgb = self.camera.capture_array("main")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.camera.stop()
        self.camera.close()


class UsbCameraSource:
    """USB 摄像头采集源，主要用于无 CSI 摄像头时调试。"""

    def __init__(self, device: int, width: int, height: int):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(device)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开 USB 摄像头 {device}")

        # 丢弃启动阶段曝光不稳定的图像。
        for _ in range(8):
            self.capture.read()

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("USB 摄像头读取失败")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self.capture.release()


def open_camera(source: str, device: int, width: int, height: int):
    """按照指定类型打开摄像头，auto 模式优先使用 Picamera2。"""

    if source == "picamera2":
        return PiCameraSource(width, height)
    if source == "usb":
        return UsbCameraSource(device, width, height)

    try:
        return PiCameraSource(width, height)
    except (RuntimeError, ModuleNotFoundError) as error:
        print(f"Picamera2 不可用，尝试 USB 摄像头：{error}")
        return UsbCameraSource(device, width, height)


def calibration_mode(frame: np.ndarray, config: Dict, config_path: Path) -> bool:
    """通过鼠标依次点击工作区四角，保存透视标定点。"""

    window_name = "Calibration: TL -> TR -> BR -> BL"
    selected: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 750)
    cv2.setMouseCallback(window_name, on_mouse)
    print("请依次点击：左上角、右上角、右下角、左下角。")
    print("按 R 重新选择，四点完成后按 Enter 保存，按 Q 退出。")

    while True:
        view = frame.copy()
        for index, point in enumerate(selected):
            cv2.circle(view, point, 10, (0, 255, 0), -1)
            cv2.putText(
                view,
                str(index + 1),
                (point[0] + 12, point[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(selected) > 1:
            cv2.polylines(view, [np.array(selected, np.int32)], False, (0, 255, 0), 3)
        cv2.imshow(window_name, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return False
        if key in (ord("r"), ord("R")):
            selected.clear()
        if key in (10, 13) and len(selected) == 4:
            config["workspace"]["image_points"] = [list(point) for point in selected]
            save_config(config_path, config)
            cv2.destroyWindow(window_name)
            print(f"标定结果已保存到：{config_path}")
            return True


def get_rectify_margin_mm(config: Dict) -> float:
    """返回透视矫正留边，单位 mm。"""

    return max(0.0, float(config["workspace"].get("rectify_margin_mm", 10.0)))


def get_rectify_margin_px(config: Dict, pixels_per_mm: float) -> int:
    """返回透视矫正留边，单位 px。"""

    return round(get_rectify_margin_mm(config) * pixels_per_mm)


def perspective_transform(frame: np.ndarray, config: Dict) -> Tuple[np.ndarray, float]:
    """将摄像头画面矫正到俯视毫米平面。"""

    workspace = config["workspace"]
    pixels_per_mm = float(workspace["pixels_per_mm"])
    margin_px = get_rectify_margin_px(config, pixels_per_mm)
    workspace_width_px = round(workspace["width_mm"] * pixels_per_mm)
    workspace_height_px = round(workspace["height_mm"] * pixels_per_mm)
    output_width = workspace_width_px + margin_px * 2
    output_height = workspace_height_px + margin_px * 2
    source_points = np.array(workspace["image_points"], dtype=np.float32)
    target_points = np.array(
        [
            [margin_px, margin_px],
            [margin_px + workspace_width_px - 1, margin_px],
            [margin_px + workspace_width_px - 1, margin_px + workspace_height_px - 1],
            [margin_px, margin_px + workspace_height_px - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    corrected = cv2.warpPerspective(frame, matrix, (output_width, output_height))
    return corrected, pixels_per_mm


def _line_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> Optional[np.ndarray]:
    """求两条无限延长直线的交点，平行或近似平行时返回 None。"""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = float(
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    scale = max(float(np.linalg.norm(first_direction)), float(np.linalg.norm(second_direction)), 1.0)
    if abs(denominator) <= 1e-6 * scale * scale:
        return None
    offset = second_start - first_start
    ratio = float(
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / denominator
    return first_start + ratio * first_direction


def _merge_rounded_corner_vertices(
    points_mm: np.ndarray,
    min_turn_deg: float,
    max_corner_extension_mm: float,
) -> np.ndarray:
    """把圆角轮廓产生的连续小转角合并为一个理论角点。"""

    points = [np.asarray(point, dtype=np.float64) for point in points_mm]
    if len(points) < 5:
        return np.asarray(points, dtype=np.float64)

    # 先把环形列表旋转到一个非圆角位置，避免圆角刚好跨过数组首尾。
    turns: List[Tuple[float, float]] = []
    for index in range(len(points)):
        previous = points[(index - 1) % len(points)]
        current = points[index]
        following = points[(index + 1) % len(points)]
        incoming = current - previous
        outgoing = following - current
        if np.linalg.norm(incoming) <= 1e-9 or np.linalg.norm(outgoing) <= 1e-9:
            turns.append((0.0, 0.0))
            continue
        cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
        dot = float(np.dot(incoming, outgoing))
        turns.append((abs(math.degrees(math.atan2(cross, dot))), math.copysign(1.0, cross)))

    non_round = [
        index
        for index, (angle, _) in enumerate(turns)
        if angle < min_turn_deg or angle > 70.0
    ]
    if not non_round:
        return np.asarray(points, dtype=np.float64)
    start = non_round[0]
    points = points[start:] + points[:start]

    # 一次合并一个连续弧段，合并后重新计算，避免索引错位。
    for _ in range(3):
        if len(points) < 5:
            break
        run_start: Optional[int] = None
        run_sign = 0.0
        run_end = -1
        for index in range(1, len(points) - 1):
            previous = points[index - 1]
            current = points[index]
            following = points[index + 1]
            incoming = current - previous
            outgoing = following - current
            if np.linalg.norm(incoming) <= 1e-9 or np.linalg.norm(outgoing) <= 1e-9:
                angle = 0.0
                sign = 0.0
            else:
                cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
                dot = float(np.dot(incoming, outgoing))
                angle = abs(math.degrees(math.atan2(cross, dot)))
                sign = math.copysign(1.0, cross)
            is_round_turn = min_turn_deg <= angle <= 70.0
            if is_round_turn and (run_start is None or sign == run_sign):
                if run_start is None:
                    run_start = index
                    run_sign = sign
                run_end = index
                continue
            if run_start is not None and run_end - run_start + 1 >= 2:
                break
            run_start = None
            run_end = -1

        if run_start is None or run_end - run_start + 1 < 2:
            break
        corner = _line_intersection(
            points[run_start - 1],
            points[run_start],
            points[run_end],
            points[run_end + 1],
        )
        if corner is None:
            break
        if max(
            float(np.linalg.norm(corner - points[run_start])),
            float(np.linalg.norm(corner - points[run_end])),
        ) > max_corner_extension_mm:
            break
        points = points[:run_start] + [corner] + points[run_end + 1:]

    return np.asarray(points, dtype=np.float64)


def _regularize_polygon(
    points_mm: np.ndarray,
    short_edge_mm: float,
    merge_angle_deg: float,
    max_corner_extension_mm: float,
) -> np.ndarray:
    """删除短边伪顶点，并按夹角合并接近共线的相邻边。"""

    points = [np.asarray(point, dtype=np.float64) for point in points_mm]
    if len(points) < 3:
        return np.asarray(points, dtype=np.float64)

    # 一条边过短时，用短边两侧的两条边作无限延长线求真正的角点。
    # 每次只处理最短边，避免一次修改影响后续索引。
    while len(points) > 3:
        edge_lengths = [
            float(np.linalg.norm(points[(index + 1) % len(points)] - points[index]))
            for index in range(len(points))
        ]
        short_index = int(np.argmin(edge_lengths))
        if edge_lengths[short_index] > short_edge_mm:
            break

        point_index = short_index
        next_index = (point_index + 1) % len(points)
        previous_index = (point_index - 1) % len(points)
        after_index = (next_index + 1) % len(points)
        corner = _line_intersection(
            points[previous_index],
            points[point_index],
            points[next_index],
            points[after_index],
        )
        if corner is None:
            # 两侧几乎平行时没有可靠的延长交点，退化为直接删除短边终点。
            corner = points[point_index]
        elif max(
            float(np.linalg.norm(corner - points[point_index])),
            float(np.linalg.norm(corner - points[next_index])),
        ) > max_corner_extension_mm:
            # 交点太远通常意味着轮廓噪声，避免生成飞出去的伪角点。
            corner = points[point_index]

        points = [
            corner if index == point_index else point
            for index, point in enumerate(points)
            if index != next_index
        ]

    # 两条边的夹角接近 180°，说明这两段实际上属于同一条边。
    while len(points) > 3:
        removed = False
        for index in range(len(points)):
            previous = points[(index - 1) % len(points)]
            current = points[index]
            following = points[(index + 1) % len(points)]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
                continue
            dot = float(np.dot(incoming, outgoing))
            cross = float(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
            turn_deg = abs(math.degrees(math.atan2(cross, dot)))
            interior_angle_deg = 180.0 - turn_deg

            if interior_angle_deg >= merge_angle_deg:
                points.pop(index)
                removed = True
                break
        if not removed:
            break

    return np.asarray(points, dtype=np.float64)


def simplify_contour(
    contour: np.ndarray,
    pixels_per_mm: float,
    base_epsilon_mm: float,
    short_edge_mm: float = 2.0,
    merge_angle_deg: float = 170.0,
    max_corner_extension_mm: float = 20.0,
    max_epsilon_mm: float = 5.0,
    rounded_corner_enabled: bool = False,
    rounded_corner_min_turn_deg: float = 6.0,
) -> Optional[np.ndarray]:
    """将轮廓简化为 3～5 个角点，并修复短边和近共线伪顶点。"""

    epsilon_end_mm = max(float(base_epsilon_mm), float(max_epsilon_mm))
    for epsilon_mm in np.linspace(base_epsilon_mm, epsilon_end_mm, 15):
        epsilon_px = max(1.0, float(epsilon_mm) * pixels_per_mm)
        polygon = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)
        if len(polygon) < 3:
            break
        points_mm = polygon.astype(np.float64) / float(pixels_per_mm)
        if rounded_corner_enabled:
            points_mm = _merge_rounded_corner_vertices(
                points_mm,
                max(0.1, float(rounded_corner_min_turn_deg)),
                max(0.1, float(max_corner_extension_mm)),
            )
        regularized_mm = _regularize_polygon(
            points_mm,
            max(0.1, float(short_edge_mm)),
            float(np.clip(merge_angle_deg, 90.0, 179.9)),
            max(0.1, float(max_corner_extension_mm)),
        )
        if 3 <= len(regularized_mm) <= 5:
            return regularized_mm * float(pixels_per_mm)
    return None


def classify_playing_card_colors(
    corrected: np.ndarray,
    segmentation: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回扑克牌前景候选掩膜和深蓝背景模型掩膜。"""

    height = corrected.shape[0]
    row_profile = np.median(corrected.astype(np.float32), axis=1)
    row_profile = cv2.GaussianBlur(
        row_profile.reshape(height, 1, 3), (1, 41), 0
    ).reshape(height, 3)
    color_distance = np.linalg.norm(
        corrected.astype(np.float32) - row_profile[:, None, :], axis=2
    )
    background_threshold = float(
        segmentation.get("background_distance_threshold", 28.0)
    )

    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    bottom_start = int(height * 0.65)
    reference = hsv[bottom_start:]
    reference = reference[reference[:, :, 1] >= 45]
    if len(reference) >= 100:
        reference_hue = reference[:, 0].astype(np.float32)
        angles = reference_hue * (2.0 * math.pi / 180.0)
        hue_mean = math.atan2(
            float(np.sin(angles).mean()), float(np.cos(angles).mean())
        )
        if hue_mean < 0.0:
            hue_mean += 2.0 * math.pi
        background_hue = hue_mean * 180.0 / (2.0 * math.pi)
        background_saturation = float(np.median(reference[:, 1]))
        background_value = float(np.median(reference[:, 2]))
    else:
        background_hue = 105.0
        background_saturation = 120.0
        background_value = 130.0

    hue_tolerance = float(segmentation.get("blue_hue_tolerance_deg", 28.0))
    saturation_floor = float(
        segmentation.get(
            "blue_background_min_saturation",
            max(45.0, background_saturation * 0.35),
        )
    )
    value_margin = float(
        segmentation.get("blue_background_value_margin", 85.0)
    )
    hue_distance = np.abs(hsv[:, :, 0].astype(np.float32) - background_hue)
    hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)
    blue_background = (
        (hue_distance <= hue_tolerance)
        & (hsv[:, :, 1].astype(np.float32) >= saturation_floor)
        & (
            hsv[:, :, 2].astype(np.float32)
            <= background_value + value_margin
        )
    )
    foreground = (color_distance >= background_threshold) & (~blue_background)
    return foreground.astype(np.uint8) * 255, blue_background.astype(np.uint8) * 255


def extract_playing_card_pattern_masks(
    corrected: np.ndarray,
    foreground_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取扑克牌中的红色和黑色花纹掩膜。"""

    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    valid = foreground_mask > 0
    blue = corrected[:, :, 0].astype(np.int16)
    green = corrected[:, :, 1].astype(np.int16)
    red = corrected[:, :, 2].astype(np.int16)

    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.uint8)
    value = hsv[:, :, 2].astype(np.uint8)
    chroma_span = (
        np.max(corrected.astype(np.int16), axis=2)
        - np.min(corrected.astype(np.int16), axis=2)
    ).astype(np.uint8)
    red_dominance = (red - np.maximum(green, blue)).astype(np.int16)

    # 红色容易把皮肤色、黄底和浅色印刷一起吸进去，因此除了色相之外，
    # 还要求有足够的饱和度和红通道优势。
    red_mask = (
        (
            ((hue <= 16) | (hue >= 165))
            & (saturation >= 52)
            & (value >= 30)
            & (red_dominance >= 18)
            & valid
        )
        .astype(np.uint8)
        * 255
    )
    black_mask = (
        (
            (
                ((gray <= 112) & (value <= 138) & (saturation <= 72) & (chroma_span <= 72))
                | ((gray <= 90) & (value <= 108))
            )
            & (~(red_mask > 0))
            & valid
        ).astype(np.uint8)
        * 255
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    return red_mask, black_mask


def segment_image(
    corrected: np.ndarray,
    config: Dict,
    pixels_per_mm: float,
    threshold_override: Optional[int] = None,
    morphology_override: Optional[float] = None,
    card_open_override: Optional[float] = None,
    card_close_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成灰度图和二值图，调试窗口与正式检测共用这一套逻辑。"""

    segmentation = config["segmentation"]
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # 扑克牌碎片不能继续使用单一灰度阈值：红、蓝、黑色印刷会把一块
    # 完整碎片切成很多孔洞。扑克牌模式改为估计深蓝背景，再按颜色距离
    # 分割，这样白底、彩色牌面和黑色图案都会被视为同一块前景。
    if get_piece_mode(config) == "playing_cards":
        mask, _ = classify_playing_card_colors(corrected, segmentation)
    else:
        blur_size = int(segmentation.get("blur_size", 5))
        blur_size = max(3, blur_size | 1)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        mode = segmentation.get("mode", "light_on_dark")
        threshold_value = (
            int(segmentation.get("threshold", 0))
            if threshold_override is None
            else int(threshold_override)
        )
        threshold_type = cv2.THRESH_BINARY if mode == "light_on_dark" else cv2.THRESH_BINARY_INV
        if threshold_value <= 0:
            threshold_type |= cv2.THRESH_OTSU
        _, mask = cv2.threshold(blurred, threshold_value, 255, threshold_type)

    def build_kernel(size_mm: float) -> np.ndarray:
        kernel_size = max(3, round(float(size_mm) * pixels_per_mm))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    morphology_mm = (
        float(segmentation.get("morphology_mm", 0.8))
        if morphology_override is None
        else float(morphology_override)
    )

    if get_piece_mode(config) == "playing_cards":
        card_open_mm = (
            float(segmentation.get("playing_card_noise_open_mm", morphology_mm))
            if card_open_override is None
            else float(card_open_override)
        )
        card_close_mm = (
            float(segmentation.get("playing_card_hole_close_mm", morphology_mm))
            if card_close_override is None
            else float(card_close_override)
        )
        if card_open_mm > 0.0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, build_kernel(card_open_mm))
        if card_close_mm > 0.0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, build_kernel(card_close_mm))

        # 先去除小的孤立前景噪声，再交给面积和轮廓阶段处理，避免
        # 反光点、背景纹理或牌面残留形成额外候选轮廓。
        minimum_component_mm2 = float(
            segmentation.get("foreground_component_min_area_mm2", 30.0)
        )
        minimum_component_px = max(
            16, round(minimum_component_mm2 * pixels_per_mm * pixels_per_mm)
        )
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for label in range(1, labels_count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_component_px:
                cleaned[labels == label] = 255
        mask = cleaned
    else:
        kernel = build_kernel(morphology_mm)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return gray, mask


def restrict_mask_to_piece_search_area(
    mask: np.ndarray, config: Dict, pixels_per_mm: float
) -> np.ndarray:
    """只保留 A4 上半部分作为碎片搜索区域。"""

    restricted = mask.copy()
    margin_px = get_rectify_margin_px(config, pixels_per_mm)
    workspace_height_px = round(float(config["workspace"]["height_mm"]) * pixels_per_mm)
    split_y = margin_px + workspace_height_px // 2
    restricted[split_y:, :] = 0
    return restricted


def split_touching_card_contours(
    mask: np.ndarray,
    pixels_per_mm: float,
    polygon_epsilon_mm: float,
) -> List[np.ndarray]:
    """用距离变换和分水岭拆分相互接触的扑克牌碎片。

    扑克牌模式下两块碎片可能只在边角处接触，外部轮廓会变成一个
    6～8 边的凹多边形，直接拟合会把它整块丢弃。只有当轮廓明显
    超过 5 个角点时才尝试拆分，避免把正常的三角形或四边形过分切开。
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    split_contours: List[np.ndarray] = []
    epsilon_px = max(1.0, float(polygon_epsilon_mm) * pixels_per_mm)

    for contour in contours:
        approximation = cv2.approxPolyDP(contour, epsilon_px, True)
        if len(approximation) <= 5:
            split_contours.append(contour)
            continue

        x, y, width, height = cv2.boundingRect(contour)
        padding = 4
        local_width = width + padding * 2
        local_height = height + padding * 2
        local_contour = contour.reshape(-1, 2).astype(np.int32)
        local_contour -= np.array([[x - padding, y - padding]], dtype=np.int32)
        component = np.zeros((local_height, local_width), dtype=np.uint8)
        cv2.fillPoly(component, [local_contour], 255)

        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        peak_limit = max(3.0, float(distance.max()) * 0.35)
        peaks = np.where(distance >= peak_limit, 255, 0).astype(np.uint8)
        peaks = cv2.morphologyEx(
            peaks,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        marker_count, peak_markers = cv2.connectedComponents(peaks)
        if marker_count <= 2:
            split_contours.append(contour)
            continue

        # 标签 1 是外部背景，2、3……是每个距离峰对应的碎片。
        markers = np.zeros((local_height, local_width), dtype=np.int32)
        markers[component == 0] = 1
        markers[peak_markers > 0] = peak_markers[peak_markers > 0] + 1
        watershed_input = cv2.cvtColor(component, cv2.COLOR_GRAY2BGR)
        cv2.watershed(watershed_input, markers)

        separated: List[np.ndarray] = []
        for label in range(2, marker_count + 1):
            region = np.where(markers == label, 255, 0).astype(np.uint8)
            region_contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not region_contours:
                continue
            region_contour = max(region_contours, key=cv2.contourArea)
            if cv2.contourArea(region_contour) < (pixels_per_mm * pixels_per_mm * 10.0):
                continue
            region_contour = region_contour.reshape(-1, 2).astype(np.int32)
            region_contour += np.array([[x - padding, y - padding]], dtype=np.int32)
            separated.append(region_contour.reshape(-1, 1, 2))

        if len(separated) >= 2:
            split_contours.extend(separated)
        else:
            split_contours.append(contour)

    return split_contours


def extract_piece_texture(
    corrected: np.ndarray,
    polygon_px: np.ndarray,
) -> Tuple[np.ndarray, List[Pt]]:
    """从透视矫正图中裁出单块碎片的小图和其局部轮廓。"""

    polygon = polygon_px.astype(np.int32)
    x, y, width, height = cv2.boundingRect(polygon)
    image_height, image_width = corrected.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(image_width, int(x + width))
    y1 = min(image_height, int(y + height))
    if x1 <= x0 or y1 <= y0:
        empty_image = np.zeros((1, 1, 3), dtype=corrected.dtype)
        return empty_image, [Pt(0.0, 0.0)]
    crop = corrected[y0:y1, x0:x1].copy()
    relative_polygon = polygon - np.array([[x0, y0]], dtype=np.int32)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [relative_polygon], 255)
    piece_image = cv2.bitwise_and(crop, crop, mask=mask)
    polygon_in_image = [
        Pt(float(point[0]), float(point[1]))
        for point in relative_polygon.reshape(-1, 2)
    ]
    return piece_image, polygon_in_image


def detect_pieces(
    corrected: np.ndarray, config: Dict, pixels_per_mm: float
) -> Tuple[List[DetectedPiece], np.ndarray, List[str]]:
    """从透视矫正图中提取碎片，并转换为毫米多边形。"""

    _, mask = segment_image(corrected, config, pixels_per_mm)
    mask = restrict_mask_to_piece_search_area(mask, config, pixels_per_mm)
    segmentation = config["segmentation"]

    minimum_area = float(segmentation["min_piece_area_mm2"])
    maximum_area = float(segmentation["max_piece_area_mm2"])
    epsilon_mm = float(segmentation["polygon_epsilon_mm"])
    short_edge_mm = float(segmentation.get("short_edge_mm", 2.0))
    merge_angle_deg = float(
        segmentation.get(
            "merge_angle_deg",
            segmentation.get("collinear_angle_deg", 170.0),
        )
    )
    max_corner_extension_mm = float(
        segmentation.get("max_corner_extension_mm", 20.0)
    )
    max_epsilon_mm = float(
        segmentation.get(
            "playing_card_max_polygon_epsilon_mm"
            if get_piece_mode(config) == "playing_cards"
            else "max_polygon_epsilon_mm",
            12.0 if get_piece_mode(config) == "playing_cards" else 5.0,
        )
    )
    rounded_corner_enabled = bool(
        segmentation.get(
            "rounded_corner_enabled",
            get_piece_mode(config) == "playing_cards",
        )
    ) and get_piece_mode(config) == "playing_cards"
    rounded_corner_min_turn_deg = float(
        segmentation.get("rounded_corner_min_turn_deg", 6.0)
    )
    if get_piece_mode(config) == "playing_cards":
        contours = split_touching_card_contours(mask, pixels_per_mm, epsilon_mm)
    else:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    pieces: List[DetectedPiece] = []
    warnings: List[str] = []

    for contour in contours:
        area_mm2 = cv2.contourArea(contour) / (pixels_per_mm * pixels_per_mm)
        if not minimum_area <= area_mm2 <= maximum_area:
            continue
        polygon_px = simplify_contour(
            contour,
            pixels_per_mm,
            epsilon_mm,
            short_edge_mm,
            merge_angle_deg,
            max_corner_extension_mm,
            max_epsilon_mm,
            rounded_corner_enabled,
            rounded_corner_min_turn_deg,
        )
        if polygon_px is None:
            warnings.append(f"忽略一个无法简化为 3～5 边形的轮廓，面积 {area_mm2:.1f} mm²")
            continue
        margin_mm = get_rectify_margin_mm(config)
        points_mm = [
            Pt(
                float(point[0]) / pixels_per_mm - margin_mm,
                float(point[1]) / pixels_per_mm - margin_mm,
            )
            for point in polygon_px
        ]
        polygon_area_mm2 = polygon_area(points_mm)
        piece_image, polygon_in_image = extract_piece_texture(corrected, polygon_px)
        pieces.append(
            DetectedPiece(
                contour_px=polygon_px.astype(np.int32),
                points_mm=points_mm,
                area_mm2=polygon_area_mm2,
                piece_image=piece_image,
                polygon_in_image=polygon_in_image,
            )
        )

    # 固定编号顺序，避免同一场景中编号随 findContours 返回顺序跳变。
    pieces.sort(
        key=lambda piece: (
            sum(point.x for point in piece.points_mm) / len(piece.points_mm),
            sum(point.y for point in piece.points_mm) / len(piece.points_mm),
        )
    )
    return pieces, mask, warnings


def point_mm_to_px(point: Pt, pixels_per_mm: float, config: Dict) -> Tuple[int, int]:
    margin_px = get_rectify_margin_px(config, pixels_per_mm)
    return (
        round(point.x * pixels_per_mm) + margin_px,
        round(point.y * pixels_per_mm) + margin_px,
    )


def draw_polygon_with_alpha(
    image: np.ndarray, polygon: np.ndarray, color: Tuple[int, int, int], alpha: float
) -> None:
    """用半透明颜色填充多边形，并绘制清晰边界。"""

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    cv2.polylines(image, [polygon], True, (30, 30, 30), 3, cv2.LINE_AA)


@lru_cache(maxsize=16)
def create_text_banner(width: int, text: str) -> np.ndarray:
    """生成可重复使用的英文说明栏，避免预览时重复绘制。"""

    banner = np.full((58, width, 3), 245, dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (20, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return banner


def add_text_banner(image: np.ndarray, text: str) -> np.ndarray:
    """在图像顶部拼接缓存的英文说明栏。"""

    return np.vstack((create_text_banner(image.shape[1], text), image))


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    """给单个面板加英文标题栏。"""

    panel = image.copy()
    height, width = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (width - 1, min(42, height - 1)), (20, 20, 20), -1)
    cv2.putText(
        panel,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def create_fit_overlay(
    corrected: np.ndarray,
    solution: Optional[Dict],
    pixels_per_mm: float,
    config: Dict,
) -> np.ndarray:
    """生成仅显示矩形外框和拼接轮廓的误差观察图。"""

    overlay = np.full_like(corrected, 248)
    if solution is None:
        cv2.putText(
            overlay,
            "NO RECTANGLE SOLUTION",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )
        return overlay

    rectangle = np.array(
        [point_mm_to_px(point, pixels_per_mm, config) for point in solution["rectangle"]["corners"]],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [rectangle], True, (0, 0, 0), 5, cv2.LINE_AA)
    for corner_index, corner in enumerate(rectangle):
        point = tuple(int(value) for value in corner)
        cv2.circle(overlay, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(corner_index),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

    for placement in solution["placements"]:
        index = placement["piece_index"]
        color = COLORS[index % len(COLORS)]
        polygon = np.array(
            [point_mm_to_px(point, pixels_per_mm, config) for point in placement["target_pts"]],
            dtype=np.int32,
        )
        cv2.polylines(overlay, [polygon], True, color, 3, cv2.LINE_AA)
        for vertex in polygon:
            cv2.circle(overlay, tuple(int(value) for value in vertex), 4, color, -1, cv2.LINE_AA)
        center = point_mm_to_px(placement["target_center"], pixels_per_mm, config)
        cv2.circle(overlay, center, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(index),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )

    rectangle_info = solution["rectangle"]
    short_side, long_side = sorted((rectangle_info["width"], rectangle_info["height"]))
    info_lines = [
        f"Rect {long_side:.1f} x {short_side:.1f} mm",
        f"Area err {solution['area_error'] * 100:.2f}%",
        "Black = rectangle, Color = assembled pieces",
    ]
    for line_index, text in enumerate(info_lines):
        cv2.putText(
            overlay,
            text,
            (18, corrected.shape[0] - 56 + line_index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _choose_affine_triangle(points: Sequence[Pt]) -> Optional[Tuple[int, int, int]]:
    """从有序多边形中选出三个不共线顶点，用于纹理刚体变换。"""

    best_triangle: Optional[Tuple[int, int, int]] = None
    best_area = 0.0
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            for third in range(second + 1, len(points)):
                area = abs(
                    (points[second].x - points[first].x)
                    * (points[third].y - points[first].y)
                    - (points[second].y - points[first].y)
                    * (points[third].x - points[first].x)
                )
                if area > best_area:
                    best_area = area
                    best_triangle = (first, second, third)
    return best_triangle if best_area > 1e-6 else None


def _draw_textured_target_assembly(
    after: np.ndarray,
    pieces: Sequence[DetectedPiece],
    solution: Dict,
    pixels_per_mm: float,
    config: Dict,
    draw_annotations: bool = True,
) -> None:
    """把每块检测到的真实纹理按求解位姿贴到目标布局中。"""

    piece_by_index = {index: piece for index, piece in enumerate(pieces)}
    height, width = after.shape[:2]
    for placement in solution["placements"]:
        index = int(placement["piece_index"])
        piece = piece_by_index.get(index)
        if (
            piece is None
            or piece.piece_image is None
            or not piece.polygon_in_image
        ):
            continue

        source_points = piece.polygon_in_image
        # 求解器内部会将多边形顶点循环移位或反向规范化，因此不能
        # 直接把 placement["target_pts"] 与原始纹理顶点按下标配对。
        # 使用求解器输出的刚体角度、源中心和目标中心重新变换原始
        # 观测顶点，保证纹理与几何轮廓一一对应。
        source_center = placement["source_center"]
        target_center = placement["target_center"]
        angle = float(placement["angle"])
        target_points_mm = [
            point.sub(source_center).rotate(angle).add(target_center)
            for point in piece.points_mm
        ]
        target_points = [
            Pt(
                float(point_mm_to_px(point, pixels_per_mm, config)[0]),
                float(point_mm_to_px(point, pixels_per_mm, config)[1]),
            )
            for point in target_points_mm
        ]
        if len(source_points) != len(target_points):
            continue
        triangle = _choose_affine_triangle(source_points)
        if triangle is None:
            continue
        source_triangle = np.array(
            [[source_points[i].x, source_points[i].y] for i in triangle],
            dtype=np.float32,
        )
        target_triangle = np.array(
            [[target_points[i].x, target_points[i].y] for i in triangle],
            dtype=np.float32,
        )
        matrix = cv2.getAffineTransform(source_triangle, target_triangle)

        source_mask = np.zeros(piece.piece_image.shape[:2], dtype=np.uint8)
        source_polygon = np.array(
            [[round(point.x), round(point.y)] for point in source_points],
            dtype=np.int32,
        )
        cv2.fillPoly(source_mask, [source_polygon], 255)
        warped_image = cv2.warpAffine(
            piece.piece_image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        after[warped_mask > 0] = warped_image[warped_mask > 0]

        if draw_annotations:
            target_polygon = np.array(
                [[round(point.x), round(point.y)] for point in target_points],
                dtype=np.int32,
            )
            cv2.polylines(after, [target_polygon], True, COLORS[index % len(COLORS)], 2, cv2.LINE_AA)
            center = point_mm_to_px(placement["target_center"], pixels_per_mm, config)
            cv2.putText(
                after,
                str(index),
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )


def create_solution_texture_canvas(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    solution: Dict,
    pixels_per_mm: float,
    config: Dict,
    draw_annotations: bool = True,
) -> np.ndarray:
    """将求解后的真实碎片纹理贴到目标布局画布上。"""

    canvas = np.full_like(corrected, 245)
    _draw_textured_target_assembly(
        canvas,
        pieces,
        solution,
        pixels_per_mm,
        config,
        draw_annotations=draw_annotations,
    )
    return canvas


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """将四边形顶点整理为 TL、TR、BR、BL。"""

    if points.shape != (4, 2):
        raise ValueError("四边形必须包含 4 个顶点")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def render_solution_card_preview(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    solution: Dict,
    pixels_per_mm: float,
    config: Dict,
    canvas_width_px: int,
    canvas_height_px: int,
    draw_annotations: bool = False,
) -> Optional[np.ndarray]:
    """把候选拼法渲染为标准化整牌图。"""

    if solution is None:
        return None
    rectangle = solution.get("rectangle")
    if not rectangle:
        return None

    try:
        source_points = np.array(
            [
                point_mm_to_px(point, pixels_per_mm, config)
                for point in rectangle["corners"]
            ],
            dtype=np.float32,
        )
        source_points = order_quad_points(source_points)
    except (KeyError, ValueError, TypeError):
        return None

    assembly = create_solution_texture_canvas(
        corrected,
        pieces,
        solution,
        pixels_per_mm,
        config,
        draw_annotations=draw_annotations,
    )
    short_px = min(int(canvas_width_px), int(canvas_height_px))
    long_px = max(int(canvas_width_px), int(canvas_height_px))

    if float(rectangle.get("width", 0.0)) >= float(rectangle.get("height", 0.0)):
        destination = np.array(
            [
                [0.0, 0.0],
                [long_px - 1.0, 0.0],
                [long_px - 1.0, short_px - 1.0],
                [0.0, short_px - 1.0],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source_points, destination)
        warped = cv2.warpPerspective(
            assembly,
            transform,
            (long_px, short_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    destination = np.array(
        [
            [0.0, 0.0],
            [short_px - 1.0, 0.0],
            [short_px - 1.0, long_px - 1.0],
            [0.0, long_px - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_points, destination)
    return cv2.warpPerspective(
        assembly,
        transform,
        (short_px, long_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def create_result_panel(
    image: np.ndarray,
    title: str,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """生成统一尺寸的结果面板。"""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return add_panel_title(image, title)


def build_preview_signature(image: np.ndarray, size: Tuple[int, int] = (48, 72)) -> np.ndarray:
    """把候选整牌图压缩成稳定的小特征图，用于候选去重。"""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = float(np.mean(resized))
    std = float(np.std(resized))
    if std > 1e-6:
        resized = (resized - mean) / std
    else:
        resized = resized - mean
    return resized


def preview_signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    """返回两个候选预览图的平均绝对差，越小表示越像。"""

    return float(np.mean(np.abs(first - second)))


def create_candidate_gallery(
    candidate_previews: Sequence[Dict],
    columns: int = 3,
) -> Optional[np.ndarray]:
    """生成候选拼法总览图，便于人工确认正确拼法是否存在。"""

    if not candidate_previews:
        return None

    valid_items = [
        item
        for item in candidate_previews
        if isinstance(item, dict) and isinstance(item.get("image"), np.ndarray)
    ]
    if not valid_items:
        return None

    columns = max(1, int(columns))
    cell_width = max(int(item["image"].shape[1]) for item in valid_items)
    cell_height = max(int(item["image"].shape[0]) for item in valid_items)
    cell_size = (cell_width, cell_height)
    panels: List[np.ndarray] = []
    blank_panel = np.full((cell_height, cell_width, 3), 245, dtype=np.uint8)

    for item in valid_items:
        candidate_index = int(item.get("candidate_index", 0))
        score = float(item.get("score", 0.0))
        template_name = str(item.get("template_name", ""))
        source = str(item.get("candidate_source", ""))
        if source == "geometry_flip":
            source_tag = "F"
        elif source == "geometry":
            source_tag = "G"
        elif source == "perturb":
            source_tag = "P"
        else:
            source_tag = source[:1].upper() if source else "?"
        title = f"C{candidate_index}{source_tag} {score:.3f} {template_name}"
        panels.append(create_result_panel(item["image"], title, cell_size))

    while len(panels) % columns != 0:
        panels.append(create_result_panel(blank_panel, "EMPTY", cell_size))

    separator_v = np.full((panels[0].shape[0], 6, 3), 90, dtype=np.uint8)
    row_images: List[np.ndarray] = []
    for start in range(0, len(panels), columns):
        row = panels[start:start + columns]
        row_image = row[0]
        for panel in row[1:]:
            row_image = np.hstack((row_image, separator_v.copy(), panel))
        row_images.append(row_image)

    gallery = row_images[0]
    if len(row_images) > 1:
        separator_h = np.full((6, gallery.shape[1], 3), 90, dtype=np.uint8)
        for row_image in row_images[1:]:
            gallery = np.vstack((gallery, separator_h.copy(), row_image))
    return gallery


def create_result_image(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    solution: Optional[Dict],
    pixels_per_mm: float,
    message: str,
    config: Dict,
    template_visuals: Optional[Dict] = None,
) -> np.ndarray:
    """生成检测结果、目标布局、拟合轮廓及模板匹配对照图。"""

    before = corrected.copy()
    after = np.full_like(corrected, 245)
    fit_overlay = create_fit_overlay(corrected, solution, pixels_per_mm, config)
    for index, piece in enumerate(pieces):
        draw_polygon_with_alpha(before, piece.contour_px, COLORS[index % len(COLORS)], 0.22)
        center = np.mean(piece.contour_px, axis=0).astype(int)
        cv2.putText(before, str(index), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)

    if solution is not None:
        _draw_textured_target_assembly(
            after, pieces, solution, pixels_per_mm, config
        )
        for placement in solution["placements"]:
            index = placement["piece_index"]
            polygon = np.array(
                [point_mm_to_px(point, pixels_per_mm, config) for point in placement["target_pts"]],
                dtype=np.int32,
            )
            cv2.polylines(after, [polygon], True, COLORS[index % len(COLORS)], 2, cv2.LINE_AA)
        rectangle = np.array(
            [point_mm_to_px(point, pixels_per_mm, config) for point in solution["rectangle"]["corners"]],
            dtype=np.int32,
        )
        cv2.polylines(after, [rectangle], True, (0, 0, 0), 5, cv2.LINE_AA)

    panel_size = (corrected.shape[1], corrected.shape[0])
    before_panel = create_result_panel(before, "DETECTED PIECES", panel_size)
    after_panel = create_result_panel(after, "TARGET ASSEMBLY - TEXTURE", panel_size)
    fit_panel = create_result_panel(fit_overlay, "RECTANGLE FIT OVERLAY", panel_size)

    separator = np.full((before_panel.shape[0], 8, 3), 90, dtype=np.uint8)
    top_row = np.hstack((before_panel, separator, after_panel, separator.copy(), fit_panel))
    combined = top_row

    if template_visuals:
        jqk_template = config.get("jqk_template", {})
        final_card_preview = None
        if solution is not None:
            final_card_preview = render_solution_card_preview(
                corrected,
                pieces,
                solution,
                pixels_per_mm,
                config,
                int(jqk_template.get("canvas_width_px", 256)),
                int(jqk_template.get("canvas_height_px", 384)),
            )
        card_preview = (
            final_card_preview
            if isinstance(final_card_preview, np.ndarray)
            else template_visuals.get("card_preview")
        )
        template_preview = template_visuals.get("template_preview")
        mask_comparison = template_visuals.get("mask_comparison")
        candidate_gallery = template_visuals.get("candidate_gallery")
        if (
            isinstance(card_preview, np.ndarray)
            and isinstance(template_preview, np.ndarray)
            and isinstance(mask_comparison, np.ndarray)
        ):
            bottom_row = np.hstack(
                (
                    create_result_panel(card_preview, "FINAL ASSEMBLED CARD", panel_size),
                    separator.copy(),
                    create_result_panel(template_preview, "BEST JQK TEMPLATE", panel_size),
                    separator.copy(),
                    create_result_panel(mask_comparison, "RED / BLACK MASK COMPARE", panel_size),
                )
            )
            row_separator = np.full((8, top_row.shape[1], 3), 90, dtype=np.uint8)
            combined = np.vstack((top_row, row_separator, bottom_row))
            if isinstance(candidate_gallery, np.ndarray):
                gallery_panel = create_result_panel(
                    candidate_gallery,
                    "JQK CANDIDATE GALLERY",
                    (combined.shape[1], max(1, combined.shape[0] // 2)),
                )
                combined = np.vstack((combined, row_separator.copy(), gallery_panel))

    if solution is None:
        image_message = "DETECTION FAILED - see latest_result.json"
    else:
        rectangle = solution["rectangle"]
        short_side, long_side = sorted((rectangle["width"], rectangle["height"]))
        image_message = (
            f"SUCCESS: {long_side:.1f} x {short_side:.1f} mm, "
            f"area error {solution['area_error'] * 100:.2f}%"
        )
    return add_text_banner(combined, image_message)


def build_texture_pieces(pieces: Sequence[DetectedPiece]) -> List[Dict]:
    """转换为 texture_scorer 所需的输入格式。"""

    return [
        {
            "pts": piece.points_mm,
            "image": piece.piece_image,
            "polygon_in_image": piece.polygon_in_image,
        }
        for piece in pieces
    ]


def rerank_solution_with_jqk_templates(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    pixels_per_mm: float,
    config: Dict,
    solve_result: Dict,
    diagnostics: Dict,
) -> Optional[Tuple[Dict, Dict, Dict]]:
    """使用 J/Q/K 模板对候选拼法做二次排序。"""

    if not jqk_template_enabled(config):
        diagnostics["template_enabled"] = False
        return None

    jqk_template = config.get("jqk_template", {})
    template_dir = resolve_vision_relative_path(
        str(jqk_template.get("template_dir", "templates/jqk"))
    )
    candidates = list(solve_result.get("candidates", []))
    candidate_limit = max(1, int(jqk_template.get("candidate_top_k", 7)))
    raw_candidate_limit = max(
        candidate_limit,
        int(jqk_template.get("candidate_scan_limit", candidate_limit * 4)),
    )
    preview_min_diff = float(
        jqk_template.get("candidate_preview_min_diff", 0.05)
    )
    if not candidates:
        diagnostics["template_enabled"] = True
        diagnostics["template_error"] = "no_geometry_candidates"
        return None

    match_results: List[Dict] = []
    preview_signatures: List[np.ndarray] = []
    duplicate_preview_skips = 0
    ordered_candidates = sorted(
        list(enumerate(candidates[:raw_candidate_limit])),
        key=lambda item: (
            str(item[1].get("candidate_source", "geometry")) != "geometry",
            item[0],
        ),
    )
    for candidate_index, candidate in ordered_candidates:
        if len(match_results) >= candidate_limit:
            break
        solution = candidate.get("geometry")
        if not isinstance(solution, dict):
            continue
        card_preview = render_solution_card_preview(
            corrected,
            pieces,
            solution,
            pixels_per_mm,
            config,
            int(jqk_template.get("canvas_width_px", 256)),
            int(jqk_template.get("canvas_height_px", 384)),
        )
        if card_preview is None:
            continue
        preview_signature = build_preview_signature(card_preview)
        if any(
            preview_signature_distance(preview_signature, existing) < preview_min_diff
            for existing in preview_signatures
        ):
            duplicate_preview_skips += 1
            continue
        preview_signatures.append(preview_signature)
        match = match_card_to_templates(
            card_preview,
            template_dir,
            canvas_width_px=int(jqk_template.get("canvas_width_px", 256)),
            canvas_height_px=int(jqk_template.get("canvas_height_px", 384)),
            match_orientations_deg=tuple(
                int(value) for value in jqk_template.get("match_orientations_deg", [0, 180])
            ),
            center_weight=float(jqk_template.get("center_weight", 0.45)),
            corner_weight=float(jqk_template.get("corner_weight", 0.25)),
            red_weight=float(jqk_template.get("red_weight", 0.15)),
            black_weight=float(jqk_template.get("black_weight", 0.15)),
        )
        match["candidate_index"] = candidate_index
        match["geometry_score"] = float(candidate.get("j_shape", solution.get("score", 0.0)))
        match["texture_rank_score"] = float(candidate.get("j_total", solution.get("score", 0.0)))
        texture_info = candidate.get("texture", {}) if isinstance(candidate.get("texture"), dict) else {}
        match["texture_fallback_full_rect"] = bool(
            texture_info.get("fallback_full_rect", False)
        )
        match["texture_fallback_sparse_seam"] = bool(
            texture_info.get("fallback_sparse_seam", False)
        )
        match["texture_seam_count"] = len(texture_info.get("seam_scores", []))
        match["candidate_source"] = str(candidate.get("candidate_source", "geometry"))
        match["solution"] = solution
        match_results.append(match)

    diagnostics["template_enabled"] = True
    diagnostics["template_duplicate_preview_skips"] = int(duplicate_preview_skips)
    diagnostics["template_raw_candidate_scan_count"] = int(
        min(len(candidates), raw_candidate_limit)
    )
    diagnostics["template_unique_preview_count"] = int(len(match_results))
    if not match_results:
        diagnostics["template_error"] = "no_template_match_result"
        return None

    valid_results = [match for match in match_results if "best_score" in match]
    if not valid_results:
        first = match_results[0]
        diagnostics["template_error"] = first.get("error", "template_match_failed")
        diagnostics["template_loaded_count"] = int(first.get("loaded_template_count", 0))
        diagnostics["template_missing_files"] = first.get("missing_templates", [])
        return None

    valid_results.sort(key=lambda item: float(item.get("best_score", 0.0)), reverse=True)
    best_match = valid_results[0]
    second_score = float(valid_results[1].get("best_score", 0.0)) if len(valid_results) >= 2 else 0.0
    score_margin = float(best_match.get("best_score", 0.0)) - second_score
    min_confidence = float(jqk_template.get("min_confidence", 0.58))
    fallback_relaxed_confidence = float(
        jqk_template.get("fallback_relaxed_confidence", 0.56)
    )
    min_margin = float(jqk_template.get("min_margin", 0.0))
    fallback_to_texture = bool(jqk_template.get("fallback_to_texture", True))
    force_best_candidate = bool(
        jqk_template.get("force_best_candidate", False)
    )
    texture_is_weak = bool(best_match.get("texture_fallback_full_rect", False)) or (
        bool(best_match.get("texture_fallback_sparse_seam", False))
        and int(best_match.get("texture_seam_count", 0)) == 0
    )
    effective_min_confidence = (
        fallback_relaxed_confidence if texture_is_weak else min_confidence
    )
    fallback_triggered = (not force_best_candidate) and fallback_to_texture and (
        float(best_match.get("best_score", 0.0)) < effective_min_confidence
        or float(score_margin) < min_margin
    )
    applied = force_best_candidate or (not fallback_triggered)

    template_info = {
        "enabled": True,
        "applied": applied,
        "fallback_triggered": fallback_triggered,
        "force_best_candidate": force_best_candidate,
        "texture_is_weak": texture_is_weak,
        "effective_min_confidence": float(effective_min_confidence),
        "candidate_count": len(valid_results),
        "template_best_name": str(best_match.get("best_name", "")),
        "template_best_score": float(best_match.get("best_score", 0.0)),
        "template_second_score": float(second_score),
        "template_score_margin": float(score_margin),
        "template_best_orientation_deg": int(best_match.get("best_orientation_deg", 0)),
        "loaded_template_count": int(best_match.get("loaded_template_count", 0)),
        "missing_templates": list(best_match.get("missing_templates", [])),
        "template_candidate_scores": [
            {
                "candidate_index": int(match.get("candidate_index", 0)),
                "template_name": str(match.get("best_name", "")),
                "score": float(match.get("best_score", 0.0)),
                "candidate_source": str(match.get("candidate_source", "geometry")),
                "orientation_deg": int(match.get("best_orientation_deg", 0)),
                "geometry_score": float(match.get("geometry_score", 0.0)),
                "texture_rank_score": float(match.get("texture_rank_score", 0.0)),
                "center_score": float(match.get("center_score", 0.0)),
                "corner_score": float(match.get("corner_score", 0.0)),
                "red_score": float(match.get("red_score", 0.0)),
                "black_score": float(match.get("black_score", 0.0)),
                "portrait_ink_score": float(match.get("portrait_ink_score", 0.0)),
                "portrait_edge_score": float(match.get("portrait_edge_score", 0.0)),
                "portrait_block_ink_score": float(match.get("portrait_block_ink_score", 0.0)),
                "portrait_block_edge_score": float(match.get("portrait_block_edge_score", 0.0)),
                "top_band_ink_score": float(match.get("top_band_ink_score", 0.0)),
                "bottom_band_ink_score": float(match.get("bottom_band_ink_score", 0.0)),
                "rank_red_score": float(match.get("rank_red_score", 0.0)),
                "rank_black_score": float(match.get("rank_black_score", 0.0)),
                "rank_edge_score": float(match.get("rank_edge_score", 0.0)),
                "letter_ink_score": float(match.get("letter_ink_score", 0.0)),
                "letter_edge_score": float(match.get("letter_edge_score", 0.0)),
                "suit_ink_score": float(match.get("suit_ink_score", 0.0)),
                "suit_edge_score": float(match.get("suit_edge_score", 0.0)),
            }
            for match in valid_results
        ],
    }
    diagnostics["template_best_name"] = template_info["template_best_name"]
    diagnostics["template_best_score"] = template_info["template_best_score"]
    diagnostics["template_candidate_scores"] = template_info["template_candidate_scores"]
    diagnostics["template_loaded_count"] = template_info["loaded_template_count"]
    diagnostics["template_missing_files"] = template_info["missing_templates"]

    template_visuals = {
        "card_preview": best_match.get("card_preview"),
        "template_preview": best_match.get("template_preview"),
        "mask_comparison": best_match.get("mask_comparison"),
        "candidate_previews": [
            {
                "candidate_index": int(match.get("candidate_index", 0)),
                "score": float(match.get("best_score", 0.0)),
                "template_name": str(match.get("best_name", "")),
                "candidate_source": str(match.get("candidate_source", "geometry")),
                "image": match.get("card_preview"),
            }
            for match in valid_results
            if isinstance(match.get("card_preview"), np.ndarray)
        ],
    }
    template_visuals["candidate_gallery"] = create_candidate_gallery(
        template_visuals["candidate_previews"],
        columns=3,
    )
    return best_match["solution"], template_info, template_visuals


def score_solution_texture(
    solution: Dict,
    pieces: Sequence[DetectedPiece],
    pixels_per_mm: float,
    config: Dict,
) -> Optional[Dict]:
    """对当前几何解做一次纹理连续性评分。"""

    if not pieces:
        return None
    if not texture_pipeline_enabled(config):
        return None
    texture = config.get("texture", {})

    scorer = TextureScorer(
        mm_per_px=float(texture.get("mm_per_px_override", pixels_per_mm)),
        strip_width_mm=float(texture.get("strip_width_mm", 5.0)),
        lambda_texture=float(texture.get("lambda_texture", 0.5)),
        gw=float(texture.get("gradient_weight", 0.35)),
        nw=float(texture.get("ncc_weight", 0.30)),
        ow=float(texture.get("orb_weight", 0.35)),
        pw=float(texture.get("pattern_weight", 0.45)),
    )
    texture_result = scorer.score_solution(solution, build_texture_pieces(pieces))
    texture_result["j_texture"] = 1.0 - float(texture_result.get("texture_score", 0.5))
    texture_result["j_total"] = scorer.total_score(
        float(solution.get("score", 0.0)),
        texture_result,
    )
    return texture_result


def solve_texture_aware_solution(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    target_center: Pt,
    pixels_per_mm: float,
    config: Dict,
    diagnostics: Dict,
) -> Optional[Dict]:
    """在扑克牌模式下让纹理评分和 J/Q/K 模板参与候选解排序。"""

    texture_enabled = texture_pipeline_enabled(config)
    template_enabled = jqk_template_enabled(config)
    is_playing_cards = str(config.get("piece_mode", "")).lower() == "playing_cards"
    texture = config.get("texture", {})
    jqk_template = config.get("jqk_template", {})
    solver = config.get("solver", {})
    rerank_top_k = max(
        1,
        int(texture.get("candidate_top_k", 7)) if texture_enabled else 1,
        int(jqk_template.get("candidate_top_k", 7)) if template_enabled else 1,
    )
    if is_playing_cards:
        rerank_top_k = max(
            rerank_top_k,
            int(texture.get("playing_card_candidate_top_k", rerank_top_k)),
            int(jqk_template.get("playing_card_candidate_top_k", rerank_top_k)),
        )
    if not texture_enabled and not template_enabled:
        return None

    short_range = tuple(solver.get("rectangle_short_range_mm", [40.0, 100.0]))
    long_range = tuple(solver.get("rectangle_long_range_mm", [80.0, 130.0]))
    edge_tolerance_mm = float(solver.get("edge_tolerance_mm", 3.0))
    edge_relative_tolerance = float(solver.get("edge_relative_tolerance", 0.05))
    minimum_edge_contact_mm = float(solver.get("minimum_edge_contact_mm", 5.0))
    minimum_edge_contact_ratio = float(solver.get("minimum_edge_contact_ratio", 0.6))
    overlap_tolerance_mm = float(solver.get("overlap_tolerance_mm", 4.0))
    rectangle_area_tolerance = float(solver.get("rectangle_area_tolerance", 0.06))
    dimension_tolerance_mm = float(solver.get("dimension_tolerance_mm", 3.0))
    max_search_nodes = int(solver.get("max_search_nodes", 20_000))
    max_candidates_per_node = int(solver.get("max_candidates_per_node", 128))
    max_partial_candidates_per_node = int(
        solver.get("max_partial_candidates_per_node", 32)
    )
    max_complete_solutions = int(
        solver.get("max_complete_solutions", max(rerank_top_k, 12))
    )

    if is_playing_cards:
        minimum_edge_contact_ratio = float(
            solver.get(
                "playing_card_minimum_edge_contact_ratio",
                minimum_edge_contact_ratio,
            )
        )
        max_candidates_per_node = int(
            solver.get(
                "playing_card_max_candidates_per_node",
                max_candidates_per_node,
            )
        )
        max_partial_candidates_per_node = int(
            solver.get(
                "playing_card_max_partial_candidates_per_node",
                max_partial_candidates_per_node,
            )
        )
        max_complete_solutions = int(
            solver.get(
                "playing_card_max_complete_solutions",
                max(max_complete_solutions, rerank_top_k),
            )
        )
    anchor_count = int(solver.get("anchor_count", 1))
    partial_match_penalty = float(solver.get("partial_match_penalty", 0.18))
    geometry_candidate_min_angle_diff_rad = float(
        solver.get("geometry_candidate_min_angle_diff_rad", 0.0)
    )
    triangle_keep_per_piece = int(solver.get("triangle_keep_per_piece", 0))
    triangle_min_angle_diff_rad = float(
        solver.get("triangle_min_angle_diff_rad", 0.0)
    )
    triangle_symmetry_edge_tolerance_mm = float(
        solver.get("triangle_symmetry_edge_tolerance_mm", 0.0)
    )
    triangle_symmetry_edge_relative_tolerance = float(
        solver.get("triangle_symmetry_edge_relative_tolerance", 0.0)
    )
    triangle_symmetry_length_bonus_mm = float(
        solver.get("triangle_symmetry_length_bonus_mm", 0.0)
    )
    triangle_flip_side_enabled = bool(
        solver.get("triangle_flip_side_enabled", False)
    )
    triangle_flip_solution_quota = int(
        solver.get("triangle_flip_solution_quota", 0)
    )
    triangle_flip_requires_similar_edge = bool(
        solver.get("triangle_flip_requires_similar_edge", True)
    )
    triangle_flip_max_alignments_per_edge_pair = int(
        solver.get("triangle_flip_max_alignments_per_edge_pair", 2)
    )
    triangle_flip_candidate_quota = int(
        solver.get("triangle_flip_candidate_quota", 0)
    )
    triangle_flip_partial_diagonal_extra_mm = float(
        solver.get("triangle_flip_partial_diagonal_extra_mm", 0.0)
    )
    triangle_flip_partial_area_scale = float(
        solver.get("triangle_flip_partial_area_scale", 1.0)
    )
    triangle_flip_priority_bonus = float(
        solver.get("triangle_flip_priority_bonus", 0.0)
    )
    if is_playing_cards:
        anchor_count = int(
            solver.get("playing_card_anchor_count", anchor_count)
        )
        partial_match_penalty = float(
            solver.get(
                "playing_card_partial_match_penalty",
                partial_match_penalty,
            )
        )
        geometry_candidate_min_angle_diff_rad = float(
            solver.get(
                "playing_card_geometry_candidate_min_angle_diff_rad",
                geometry_candidate_min_angle_diff_rad,
            )
        )
        triangle_keep_per_piece = int(
            solver.get("playing_card_triangle_keep_per_piece", triangle_keep_per_piece)
        )
        triangle_min_angle_diff_rad = float(
            solver.get(
                "playing_card_triangle_min_angle_diff_rad",
                triangle_min_angle_diff_rad,
            )
        )
        triangle_symmetry_edge_tolerance_mm = float(
            solver.get(
                "playing_card_triangle_symmetry_edge_tolerance_mm",
                triangle_symmetry_edge_tolerance_mm,
            )
        )
        triangle_symmetry_edge_relative_tolerance = float(
            solver.get(
                "playing_card_triangle_symmetry_edge_relative_tolerance",
                triangle_symmetry_edge_relative_tolerance,
            )
        )
        triangle_symmetry_length_bonus_mm = float(
            solver.get(
                "playing_card_triangle_symmetry_length_bonus_mm",
                triangle_symmetry_length_bonus_mm,
            )
        )
        triangle_flip_side_enabled = bool(
            solver.get(
                "playing_card_triangle_flip_side_enabled",
                triangle_flip_side_enabled,
            )
        )
        triangle_flip_solution_quota = int(
            solver.get(
                "playing_card_triangle_flip_solution_quota",
                triangle_flip_solution_quota,
            )
        )
        triangle_flip_requires_similar_edge = bool(
            solver.get(
                "playing_card_triangle_flip_requires_similar_edge",
                triangle_flip_requires_similar_edge,
            )
        )
        triangle_flip_max_alignments_per_edge_pair = int(
            solver.get(
                "playing_card_triangle_flip_max_alignments_per_edge_pair",
                triangle_flip_max_alignments_per_edge_pair,
            )
        )
        triangle_flip_candidate_quota = int(
            solver.get(
                "playing_card_triangle_flip_candidate_quota",
                triangle_flip_candidate_quota,
            )
        )
        triangle_flip_partial_diagonal_extra_mm = float(
            solver.get(
                "playing_card_triangle_flip_partial_diagonal_extra_mm",
                triangle_flip_partial_diagonal_extra_mm,
            )
        )
        triangle_flip_partial_area_scale = float(
            solver.get(
                "playing_card_triangle_flip_partial_area_scale",
                triangle_flip_partial_area_scale,
            )
        )
        triangle_flip_priority_bonus = float(
            solver.get(
                "playing_card_triangle_flip_priority_bonus",
                triangle_flip_priority_bonus,
            )
        )

    result = solve_with_texture(
        [{"pts": piece.points_mm} for piece in pieces],
        build_texture_pieces(pieces),
        target_center=target_center,
        mm_per_px=float(texture.get("mm_per_px_override", pixels_per_mm)),
        strip_width_mm=float(texture.get("strip_width_mm", 5.0)),
        lambda_texture=float(texture.get("lambda_texture", 0.5)) if texture_enabled else 0.0,
        top_k=rerank_top_k,
        geometry_kwargs={
            "edge_tolerance_mm": edge_tolerance_mm,
            "edge_relative_tolerance": edge_relative_tolerance,
            "minimum_edge_contact_mm": minimum_edge_contact_mm,
            "minimum_edge_contact_ratio": minimum_edge_contact_ratio,
            "overlap_tolerance_mm": overlap_tolerance_mm,
            "rectangle_area_tolerance": rectangle_area_tolerance,
            "size_range_mm": (
                (float(short_range[0]), float(short_range[1])),
                (float(long_range[0]), float(long_range[1])),
            ),
            "dimension_tolerance_mm": dimension_tolerance_mm,
            "max_search_nodes": max_search_nodes,
            "max_candidates_per_node": max_candidates_per_node,
            "max_partial_candidates_per_node": max_partial_candidates_per_node,
            "max_complete_solutions": max_complete_solutions,
            "anchor_count": anchor_count,
            "partial_match_penalty": partial_match_penalty,
            "triangle_keep_per_piece": triangle_keep_per_piece,
            "triangle_min_angle_diff_rad": triangle_min_angle_diff_rad,
            "triangle_symmetry_edge_tolerance_mm": triangle_symmetry_edge_tolerance_mm,
            "triangle_symmetry_edge_relative_tolerance": triangle_symmetry_edge_relative_tolerance,
            "triangle_symmetry_length_bonus_mm": triangle_symmetry_length_bonus_mm,
            "triangle_flip_side_enabled": triangle_flip_side_enabled,
            "triangle_flip_solution_quota": triangle_flip_solution_quota,
            "triangle_flip_requires_similar_edge": triangle_flip_requires_similar_edge,
            "triangle_flip_max_alignments_per_edge_pair": triangle_flip_max_alignments_per_edge_pair,
            "triangle_flip_candidate_quota": triangle_flip_candidate_quota,
            "triangle_flip_partial_diagonal_extra_mm": triangle_flip_partial_diagonal_extra_mm,
            "triangle_flip_partial_area_scale": triangle_flip_partial_area_scale,
            "triangle_flip_priority_bonus": triangle_flip_priority_bonus,
            "diagnostics": diagnostics,
        },
        gw=float(texture.get("gradient_weight", 0.35)),
        nw=float(texture.get("ncc_weight", 0.30)),
        ow=float(texture.get("orb_weight", 0.35)),
        pw=float(texture.get("pattern_weight", 0.45)),
        fallback_full_rect_penalty=float(
            texture.get("fallback_full_rect_penalty", 0.12)
        ),
        fallback_sparse_seam_penalty=float(
            texture.get("fallback_sparse_seam_penalty", 0.08)
        ),
        geometry_candidate_min_angle_diff_rad=geometry_candidate_min_angle_diff_rad,
        angle_perturb_rad=float(texture.get("candidate_angle_perturb_rad", 0.035)),
        translate_perturb_mm=float(texture.get("candidate_translate_perturb_mm", 1.2)),
    )
    if not isinstance(result, dict) or result.get("error"):
        return None

    best_solution = result.get("best_solution")
    best_texture = result.get("texture") if texture_enabled else None
    if best_solution is None:
        return None

    template_info = None
    template_visuals = None
    template_rerank = rerank_solution_with_jqk_templates(
        corrected,
        pieces,
        pixels_per_mm,
        config,
        result,
        diagnostics,
    )
    if template_rerank is not None:
        reranked_solution, template_info, template_visuals = template_rerank
        if template_info.get("applied"):
            best_solution = reranked_solution

    if isinstance(best_texture, dict):
        best_texture = dict(best_texture)
        best_texture["candidate_count"] = len(result.get("candidates", []))
        best_texture["reranked"] = texture_enabled and rerank_top_k > 1
        best_texture["j_shape"] = float(result.get("j_shape", 0.0))
        best_texture["j_texture"] = float(result.get("j_texture", 0.0))
        best_texture["j_total"] = float(result.get("j_total", 0.0))
    return {
        "solution": best_solution,
        "texture": best_texture,
        "template": template_info,
        "template_visuals": template_visuals,
    }


def apply_solution_spread(solution: Dict, spread_mm: float) -> Dict:
    """把目标布局从矩形中心向外轻微推开，减少重叠。"""

    if spread_mm <= 1e-6:
        return solution

    rectangle = solution.get("rectangle")
    placements = solution.get("placements")
    if not rectangle or not placements:
        return solution

    center = rectangle["center"]
    for placement in placements:
        target_center = placement["target_center"]
        offset = target_center.sub(center)
        length = offset.length()
        if length <= 1e-6:
            continue
        shift = offset.scale(spread_mm / length)
        placement["target_center"] = target_center.add(shift)
        placement["offset"] = placement["target_center"].sub(placement["source_center"])
        placement["target_pts"] = [point.add(shift) for point in placement["target_pts"]]
    return solution


def estimate_polygon_overlap_mm2(
    first: Sequence[Pt],
    second: Sequence[Pt],
    pixels_per_mm: float,
) -> float:
    """用局部栅格近似估计两多边形重叠面积。"""

    min_x = min(point.x for point in first + second)
    max_x = max(point.x for point in first + second)
    min_y = min(point.y for point in first + second)
    max_y = max(point.y for point in first + second)
    padding_px = 4
    width = max(4, round((max_x - min_x) * pixels_per_mm) + 1 + padding_px * 2)
    height = max(4, round((max_y - min_y) * pixels_per_mm) + 1 + padding_px * 2)
    shift_x = -min_x * pixels_per_mm + padding_px
    shift_y = -min_y * pixels_per_mm + padding_px

    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros((height, width), dtype=np.uint8)
    first_polygon = np.array(
        [(round(point.x * pixels_per_mm + shift_x), round(point.y * pixels_per_mm + shift_y)) for point in first],
        dtype=np.int32,
    )
    second_polygon = np.array(
        [(round(point.x * pixels_per_mm + shift_x), round(point.y * pixels_per_mm + shift_y)) for point in second],
        dtype=np.int32,
    )
    cv2.fillPoly(first_mask, [first_polygon], 255)
    cv2.fillPoly(second_mask, [second_polygon], 255)
    overlap_pixels = cv2.countNonZero(cv2.bitwise_and(first_mask, second_mask))
    return overlap_pixels / max(pixels_per_mm * pixels_per_mm, 1e-6)


def apply_local_overlap_avoidance(
    solution: Dict,
    pixels_per_mm: float,
    config: Dict,
) -> Tuple[Dict, Dict]:
    """只对真正重叠的碎片对做局部避让。"""

    placements = solution.get("placements")
    if not placements:
        return solution, {"enabled": False, "iterations": 0, "adjusted_pairs": 0, "final_overlap_mm2": 0.0}

    solver = config.get("solver", {})
    if not bool(solver.get("local_overlap_avoidance", True)):
        return solution, {"enabled": False, "iterations": 0, "adjusted_pairs": 0, "final_overlap_mm2": 0.0}

    clearance_mm = float(solver.get("local_overlap_clearance_mm", 2.0))
    step_mm = float(solver.get("local_overlap_step_mm", 1.5))
    max_iterations = int(solver.get("local_overlap_max_iterations", 8))
    adjusted_pairs = 0
    final_overlap_mm2 = 0.0

    for iteration in range(max_iterations):
        shifts = [Pt(0.0, 0.0) for _ in placements]
        pair_count = 0
        total_overlap = 0.0

        for first_index in range(len(placements)):
            for second_index in range(first_index + 1, len(placements)):
                first_pts = placements[first_index]["target_pts"]
                second_pts = placements[second_index]["target_pts"]
                overlap_mm2 = estimate_polygon_overlap_mm2(first_pts, second_pts, pixels_per_mm)
                if overlap_mm2 <= 1e-3:
                    continue

                first_center = placements[first_index]["target_center"]
                second_center = placements[second_index]["target_center"]
                direction = second_center.sub(first_center)
                if direction.length() <= 1e-6:
                    direction = Pt(float(second_index - first_index), 0.0)
                unit = direction.norm()
                overlap_depth_mm = math.sqrt(overlap_mm2)
                move_each_mm = min(
                    step_mm,
                    max(0.5 * clearance_mm, 0.35 * overlap_depth_mm + 0.5 * clearance_mm),
                )
                delta = unit.scale(move_each_mm)
                shifts[first_index] = shifts[first_index].sub(delta)
                shifts[second_index] = shifts[second_index].add(delta)
                pair_count += 1
                total_overlap += overlap_mm2

        if pair_count == 0:
            return solution, {
                "enabled": True,
                "iterations": iteration,
                "adjusted_pairs": adjusted_pairs,
                "final_overlap_mm2": final_overlap_mm2,
            }

        adjusted_pairs += pair_count
        final_overlap_mm2 = total_overlap
        for placement, shift in zip(placements, shifts):
            if shift.length() <= 1e-6:
                continue
            placement["target_center"] = placement["target_center"].add(shift)
            placement["offset"] = placement["target_center"].sub(placement["source_center"])
            placement["target_pts"] = [point.add(shift) for point in placement["target_pts"]]

    return solution, {
        "enabled": True,
        "iterations": max_iterations,
        "adjusted_pairs": adjusted_pairs,
        "final_overlap_mm2": final_overlap_mm2,
    }


def solution_to_json(
    pieces: Sequence[DetectedPiece],
    solution: Optional[Dict],
    status: str,
    message: str,
    config: Dict,
    diagnostics: Optional[Dict] = None,
) -> Dict:
    """将求解结果转换为便于树莓派串口程序读取的 JSON。"""

    result = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "piece_mode": get_piece_mode(config),
        "coordinate_system": "透视矫正后的工作区坐标，原点左上，X 向右，Y 向下，单位 mm",
        "piece_count": len(pieces),
        "diagnostics": diagnostics or {},
        "observed_pieces": [
            {
                "piece_index": index,
                "area_mm2": piece.area_mm2,
                "points_mm": [[point.x, point.y] for point in piece.points_mm],
            }
            for index, piece in enumerate(pieces)
        ],
    }
    if solution is None:
        result["solution"] = None
        return result

    def placement_source_orientation(placement: Dict) -> float:
        if "source_orientation" in placement:
            return float(placement["source_orientation"])
        return 0.0

    def placement_target_orientation(placement: Dict) -> float:
        if "target_orientation" in placement:
            return float(placement["target_orientation"])
        return placement_source_orientation(placement) + float(placement.get("angle", 0.0))

    result["solution"] = {
        "area_error": solution["area_error"],
        "rectangle": {
            "center_mm": [solution["rectangle"]["center"].x, solution["rectangle"]["center"].y],
            "width_mm": solution["rectangle"]["width"],
            "height_mm": solution["rectangle"]["height"],
            "angle_rad": solution["rectangle"]["angle"],
            "corners_mm": [[point.x, point.y] for point in solution["rectangle"]["corners"]],
        },
        "placements": [
            {
                "piece_index": placement["piece_index"],
                "source_center_mm": [placement["source_center"].x, placement["source_center"].y],
                "target_center_mm": [placement["target_center"].x, placement["target_center"].y],
                "angle_rad": placement["angle"],
                "angle_deg": math.degrees(placement["angle"]),
                "pick_wrist_rad": placement_source_orientation(placement),
                "place_wrist_rad": placement_target_orientation(placement),
                "target_points_mm": [[point.x, point.y] for point in placement["target_pts"]],
            }
            for placement in solution["placements"]
        ],
    }
    return result


def diagnose_solver_failure(diagnostics: Dict) -> str:
    """根据搜索统计给出最可能的失败原因。"""

    if diagnostics.get("reason_code") == "fragment_count":
        return "碎片数量不是 2～4 块"
    if diagnostics.get("search_limit_reached"):
        return (
            f"搜索达到上限 {diagnostics.get('max_search_nodes', 0)} 个节点，"
            "通常表示边长容差过大、轮廓角点过多或存在大量错误接缝候选"
        )
    if diagnostics.get("root_candidate_count", 0) == 0:
        return (
            "没有找到第一条有效接缝；重点检查透视标定、像素到毫米比例、"
            "轮廓角点顺序，以及 edge_tolerance_mm 是否过小"
        )
    if diagnostics.get("complete_layouts", 0) == 0:
        return (
            "有接缝候选，但没有一种布局能放入全部碎片；"
            "可能是碎片轮廓误拟合、接缝方向错误或碎片之间发生重叠"
        )
    if diagnostics.get("dimension_rejections", 0) > 0:
        if diagnostics.get("area_rejections", 0) > 0:
            return "候选布局的矩形尺寸或面积覆盖率不满足约束"
        return "候选布局可以拼接，但矩形长短边不在设定范围内"
    if diagnostics.get("area_rejections", 0) > 0:
        return "候选布局的外接矩形未被碎片充分覆盖，面积误差超过阈值"
    return "所有候选布局都被几何重叠或边界约束排除"


def print_result_diagnostics(result: Dict) -> None:
    """将失败原因和关键统计打印到终端。"""

    print(f"[{result['timestamp']}] {result['message']}")
    texture = result.get("texture")
    template = result.get("template")
    if texture:
        print(
            "  纹理评分："
            f"texture={texture.get('texture_score', 0.0):.3f}，"
            f"gradient={texture.get('gradient_discontinuity', 0.0):.3f}，"
            f"ncc={texture.get('ncc_similarity', 0.0):.3f}，"
            f"orb={texture.get('orb_consistency', 0.0):.3f}，"
            f"pattern={texture.get('pattern_consistency', 0.0):.3f}，"
            f"j_total={texture.get('j_total', 0.0):.3f}"
        )
    if template:
        print(
            "  J/Q/K 模板："
            f"best={template.get('template_best_name', 'N/A')}，"
            f"score={template.get('template_best_score', 0.0):.3f}，"
            f"delta={template.get('template_score_margin', 0.0):.3f}，"
            f"candidates={template.get('candidate_count', 0)}，"
            f"fallback={'yes' if template.get('fallback_triggered') else 'no'}"
        )
    if result.get("status") == "ok":
        overlap_adjustment = result.get("overlap_adjustment")
        if overlap_adjustment and overlap_adjustment.get("enabled"):
            print(
                "  局部避让："
                f"iterations={overlap_adjustment.get('iterations', 0)}，"
                f"pairs={overlap_adjustment.get('adjusted_pairs', 0)}，"
                f"final_overlap={overlap_adjustment.get('final_overlap_mm2', 0.0):.2f} mm²"
            )
        return

    diagnostics = result.get("diagnostics", {})
    print(f"  失败原因：{diagnose_solver_failure(diagnostics)}")
    if diagnostics:
        print(
            "  搜索统计："
            f"节点 {diagnostics.get('search_nodes', 0)}/"
            f"{diagnostics.get('max_search_nodes', 0)}，"
            f"首层候选 {diagnostics.get('root_candidate_count', 0)}，"
            f"完整布局 {diagnostics.get('complete_layouts', 0)}"
        )
        print(
            "  候选排除："
            f"整边 {diagnostics.get('full_edge_candidates', 0)}，"
            f"局部边 {diagnostics.get('partial_edge_candidates', 0)}，"
            f"重叠 {diagnostics.get('overlap_rejections', 0)}，"
            f"尺寸/范围 {diagnostics.get('geometry_rejections', 0)}，"
            f"矩形尺寸 {diagnostics.get('dimension_rejections', 0)}，"
            f"面积 {diagnostics.get('area_rejections', 0)}"
        )

    print("  检测到的碎片：")
    for piece in result.get("observed_pieces", []):
        points = piece.get("points_mm", [])
        edge_lengths = []
        for index, first in enumerate(points):
            second = points[(index + 1) % len(points)]
            edge_lengths.append(
                math.hypot(second[0] - first[0], second[1] - first[1])
            )
        print(
            f"    碎片 {piece['piece_index']}：{len(points)} 边，"
            f"面积 {piece['area_mm2']:.1f} mm²，"
            f"边长 [{', '.join(f'{length:.1f}' for length in edge_lengths)}] mm"
        )


def process_frame(frame: np.ndarray, config: Dict, output_dir: Path) -> Tuple[Dict, np.ndarray]:
    """处理单帧图像并保存中间结果。"""

    corrected, pixels_per_mm = perspective_transform(frame, config)
    pieces, mask, warnings = detect_pieces(corrected, config, pixels_per_mm)
    solution = None
    texture_info = None
    texture_rerank_info = None
    template_info = None
    template_visuals = None
    overlap_adjustment = None
    diagnostics: Dict = {
        "template_enabled": jqk_template_enabled(config),
        "template_best_name": "",
        "template_best_score": 0.0,
        "template_candidate_scores": [],
    }

    if not 2 <= len(pieces) <= 4:
        status = "fragment_count_error"
        message = f"碎片数量错误：检测到 {len(pieces)} 块，应为 2～4 块"
        diagnostics["reason_code"] = "fragment_count"
    else:
        workspace = config["workspace"]
        target = workspace.get(
            "target_center_mm",
            [workspace["width_mm"] * 0.5, workspace["height_mm"] * 0.5],
        )
        target_center = Pt(float(target[0]), float(target[1]))
        texture_solved = solve_texture_aware_solution(
            corrected,
            pieces, target_center, pixels_per_mm, config, diagnostics
        )
        if texture_solved is not None:
            solution = texture_solved.get("solution")
            texture_info = texture_solved.get("texture")
            template_info = texture_solved.get("template")
            template_visuals = texture_solved.get("template_visuals")
            texture_rerank_info = texture_info
        else:
            solver_config = config.get("solver", {})
            solution = find_rectangle_solution(
                [{"pts": piece.points_mm} for piece in pieces],
                target_center=target_center,
                overlap_tolerance_mm=float(
                    solver_config.get("overlap_tolerance_mm", 2.5)
                ),
                diagnostics=diagnostics,
            )
        if solution is None:
            status = "no_solution"
            message = f"检测到碎片，但矩形求解失败：{diagnose_solver_failure(diagnostics)}"
        else:
            solution = apply_solution_spread(
                solution,
                float(config.get("solver", {}).get("placement_spread_mm", 0.0)),
            )
            solution, overlap_adjustment = apply_local_overlap_avoidance(
                solution,
                pixels_per_mm,
                config,
            )
            texture_info = score_solution_texture(solution, pieces, pixels_per_mm, config)
            if texture_info is not None and texture_rerank_info is not None:
                texture_info["candidate_count"] = texture_rerank_info.get("candidate_count", 0)
                texture_info["reranked"] = True
                texture_info["rerank_pre_j_total"] = texture_rerank_info.get("j_total", 0.0)
            status = "ok"
            rectangle = solution["rectangle"]
            short_side, long_side = sorted((rectangle["width"], rectangle["height"]))
            message = (
                f"拼接成功：{long_side:.1f}×{short_side:.1f} mm，"
                f"面积误差 {solution['area_error'] * 100:.2f}%"
            )
            if texture_info is not None:
                message += f"，纹理分 {texture_info['texture_score']:.3f}"
                if texture_info.get("reranked"):
                    message += f"，纹理重排 {texture_info.get('candidate_count', 0)} 候选"
            if template_info is not None:
                message += f"，模板分 {template_info.get('template_best_score', 0.0):.3f}"
                if template_info.get("applied"):
                    message += (
                        f"（{template_info.get('template_best_name', '')}，"
                        f"候选 {template_info.get('candidate_count', 0)}）"
                    )
                elif template_info.get("fallback_triggered"):
                    message += "（模板置信度不足，已回退）"
            spread_mm = float(config.get("solver", {}).get("placement_spread_mm", 0.0))
            if spread_mm > 0.0:
                message += f"，外扩 {spread_mm:.1f} mm"
            if overlap_adjustment and overlap_adjustment.get("enabled"):
                message += (
                    f"，局部避让 {overlap_adjustment.get('iterations', 0)} 轮/"
                    f"{overlap_adjustment.get('adjusted_pairs', 0)} 对"
                )

    if warnings:
        message += f"；另有 {len(warnings)} 个轮廓被忽略"
    result = solution_to_json(pieces, solution, status, message, config, diagnostics)
    result["camera_resolution"] = [int(frame.shape[1]), int(frame.shape[0])]
    result["workspace_mm"] = [
        float(config["workspace"]["width_mm"]),
        float(config["workspace"]["height_mm"]),
    ]
    result["texture"] = texture_info
    result["template"] = template_info
    result["overlap_adjustment"] = overlap_adjustment
    result["warnings"] = warnings
    result_image = create_result_image(
        corrected,
        pieces,
        solution,
        pixels_per_mm,
        message,
        config,
        template_visuals=template_visuals,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "latest_corrected.jpg"), corrected)
    cv2.imwrite(str(output_dir / "latest_mask.png"), mask)
    cv2.imwrite(str(output_dir / "latest_result.jpg"), result_image)
    if template_visuals:
        card_preview = template_visuals.get("card_preview")
        template_preview = template_visuals.get("template_preview")
        mask_comparison = template_visuals.get("mask_comparison")
        candidate_gallery = template_visuals.get("candidate_gallery")
        candidate_previews = template_visuals.get("candidate_previews", [])
        if isinstance(card_preview, np.ndarray):
            cv2.imwrite(str(output_dir / "latest_jqk_card_preview.png"), card_preview)
        if isinstance(template_preview, np.ndarray):
            cv2.imwrite(str(output_dir / "latest_jqk_best_template.png"), template_preview)
        if isinstance(mask_comparison, np.ndarray):
            cv2.imwrite(str(output_dir / "latest_jqk_mask_compare.png"), mask_comparison)
        if isinstance(candidate_gallery, np.ndarray):
            cv2.imwrite(str(output_dir / "latest_jqk_candidate_gallery.png"), candidate_gallery)
        if isinstance(candidate_previews, list):
            for rank, item in enumerate(candidate_previews):
                if not isinstance(item, dict):
                    continue
                image = item.get("image")
                if not isinstance(image, np.ndarray):
                    continue
                candidate_index = int(item.get("candidate_index", rank))
                cv2.imwrite(
                    str(output_dir / f"latest_jqk_candidate_{rank:02d}_idx_{candidate_index}.png"),
                    image,
                )
    for piece_index, piece in enumerate(pieces):
        if piece.piece_image is not None:
            cv2.imwrite(str(output_dir / f"latest_piece_{piece_index}.png"), piece.piece_image)
    with (output_dir / "latest_result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result, result_image


def create_debug_view(
    frame: np.ndarray,
    config: Dict,
    threshold: int,
    morphology_mm: float,
    background_distance: Optional[float] = None,
    blue_hue_tolerance: Optional[float] = None,
    blue_saturation_min: Optional[float] = None,
    card_noise_open_mm: Optional[float] = None,
    card_hole_close_mm: Optional[float] = None,
) -> np.ndarray:
    """生成调试图，不执行矩形求解。"""

    view_config = copy.deepcopy(config)
    if get_piece_mode(view_config) == "playing_cards":
        segmentation = view_config["segmentation"]
        if background_distance is not None:
            segmentation["background_distance_threshold"] = float(background_distance)
        if blue_hue_tolerance is not None:
            segmentation["blue_hue_tolerance_deg"] = float(blue_hue_tolerance)
        if blue_saturation_min is not None:
            segmentation["blue_background_min_saturation"] = float(blue_saturation_min)
        if card_noise_open_mm is not None:
            segmentation["playing_card_noise_open_mm"] = float(card_noise_open_mm)
        if card_hole_close_mm is not None:
            segmentation["playing_card_hole_close_mm"] = float(card_hole_close_mm)

    corrected, pixels_per_mm = perspective_transform(frame, view_config)
    gray, mask = segment_image(
        corrected,
        view_config,
        pixels_per_mm,
        threshold_override=threshold,
        morphology_override=morphology_mm,
        card_open_override=card_noise_open_mm,
        card_close_override=card_hole_close_mm,
    )
    mask = restrict_mask_to_piece_search_area(mask, view_config, pixels_per_mm)
    contour_view = corrected.copy()
    if get_piece_mode(view_config) == "playing_cards":
        contours = split_touching_card_contours(
            mask,
            pixels_per_mm,
            float(view_config["segmentation"].get("polygon_epsilon_mm", 1.2)),
        )
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_view, contours, -1, (0, 220, 0), 3)
    margin_px = get_rectify_margin_px(view_config, pixels_per_mm)
    workspace_height_px = round(float(view_config["workspace"]["height_mm"]) * pixels_per_mm)
    split_y = margin_px + workspace_height_px // 2
    cv2.line(contour_view, (0, split_y), (corrected.shape[1] - 1, split_y), (0, 0, 255), 2, cv2.LINE_AA)

    playing_cards = get_piece_mode(view_config) == "playing_cards"
    panel_columns = 3 if playing_cards else 2
    panel_width = corrected.shape[1] // panel_columns
    panel_height = corrected.shape[0] // 2

    def panel(image: np.ndarray, title: str) -> np.ndarray:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
        title_bar = np.full((42, panel_width, 3), 20, dtype=np.uint8)
        cv2.putText(
            title_bar,
            title,
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return np.vstack((title_bar, resized))

    if playing_cards:
        _, blue_background = classify_playing_card_colors(
            corrected, view_config["segmentation"]
        )
        blue_background = restrict_mask_to_piece_search_area(
            blue_background, view_config, pixels_per_mm
        )
        red_pattern_mask, black_pattern_mask = extract_playing_card_pattern_masks(
            corrected, mask
        )
        top = np.hstack(
            (
                panel(corrected, "CORRECTED"),
                panel(blue_background, "BLUE BACKGROUND MODEL"),
                panel(mask, "CARD FOREGROUND MASK"),
            )
        )
        bottom = np.hstack(
            (
                panel(red_pattern_mask, "RED PATTERN MASK"),
                panel(black_pattern_mask, "BLACK PATTERN MASK"),
                panel(contour_view, "CONTOURS"),
            )
        )
    else:
        top = np.hstack((panel(corrected, "CORRECTED"), panel(gray, "GRAY")))
        bottom = np.hstack((panel(mask, "BINARY MASK"), panel(contour_view, "CONTOURS")))
    return np.vstack((top, bottom))


def run_debug_triggered_mode(
    camera,
    initial_frame: np.ndarray,
    config: Dict,
    output_dir: Path,
    config_path: Path,
) -> int:
    """显示实时分割调试窗口，按空格使用当前参数求解一次。"""

    window_name = "Vision Debug"
    result_window = "SCARA Vision Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)
    cv2.createTrackbar("Threshold (0=Otsu)", window_name, int(config["segmentation"].get("threshold", 0)), 255, lambda _: None)
    morphology_initial = min(
        50,
        max(0, round(float(config["segmentation"].get("morphology_mm", 0.8)) * 10.0)),
    )
    cv2.createTrackbar("Morphology x0.1mm", window_name, morphology_initial, 50, lambda _: None)
    playing_cards = get_piece_mode(config) == "playing_cards"
    if playing_cards:
        segmentation = config["segmentation"]
        distance_initial = min(
            100,
            max(1, round(float(segmentation.get("background_distance_threshold", 28.0)))),
        )
        hue_initial = min(
            60,
            max(1, round(float(segmentation.get("blue_hue_tolerance_deg", 28.0)))),
        )
        saturation_initial = min(
            255,
            max(0, round(float(segmentation.get("blue_background_min_saturation", 70.0)))),
        )
        card_open_initial = min(
            50,
            max(
                0,
                round(float(segmentation.get("playing_card_noise_open_mm", 0.5)) * 10.0),
            ),
        )
        card_close_initial = min(
            50,
            max(
                0,
                round(float(segmentation.get("playing_card_hole_close_mm", 1.0)) * 10.0),
            ),
        )
        cv2.createTrackbar("Color distance", window_name, distance_initial, 100, lambda _: None)
        cv2.createTrackbar("Blue hue tolerance", window_name, hue_initial, 60, lambda _: None)
        cv2.createTrackbar("Blue saturation min", window_name, saturation_initial, 255, lambda _: None)
        cv2.createTrackbar("Card open x0.1mm", window_name, card_open_initial, 50, lambda _: None)
        cv2.createTrackbar("Card close x0.1mm", window_name, card_close_initial, 50, lambda _: None)
    print("调试窗口已启动：空格=按当前参数检测，D=保存当前分离图，S=保存参数，Q/Esc=退出。")
    if get_piece_mode(config) == "playing_cards":
        print("请观察 BLUE BACKGROUND MODEL：深蓝背景应为白色，扑克牌应为黑色。")
        print("请观察 CARD FOREGROUND MASK：整块扑克牌应为白色，背景应为黑色。")
    else:
        print("请观察 BINARY MASK：碎片应为白色，深蓝背景应为黑色。")

    frame = initial_frame
    while True:
        if camera is not None:
            frame = camera.read()
        threshold = cv2.getTrackbarPos("Threshold (0=Otsu)", window_name)
        morphology_mm = max(
            0.1,
            cv2.getTrackbarPos("Morphology x0.1mm", window_name) / 10.0,
        )
        if playing_cards:
            background_distance = max(
                1, cv2.getTrackbarPos("Color distance", window_name)
            )
            blue_hue_tolerance = max(
                1, cv2.getTrackbarPos("Blue hue tolerance", window_name)
            )
            blue_saturation_min = cv2.getTrackbarPos(
                "Blue saturation min", window_name
            )
            card_noise_open_mm = (
                cv2.getTrackbarPos("Card open x0.1mm", window_name) / 10.0
            )
            card_hole_close_mm = (
                cv2.getTrackbarPos("Card close x0.1mm", window_name) / 10.0
            )
        else:
            background_distance = None
            blue_hue_tolerance = None
            blue_saturation_min = None
            card_noise_open_mm = None
            card_hole_close_mm = None
        debug_view = create_debug_view(
            frame,
            config,
            threshold,
            morphology_mm,
            background_distance,
            blue_hue_tolerance,
            blue_saturation_min,
            card_noise_open_mm,
            card_hole_close_mm,
        )
        cv2.imshow(window_name, debug_view)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return 0
        if key in (ord("d"), ord("D")):
            output_dir.mkdir(parents=True, exist_ok=True)
            debug_path = output_dir / "latest_color_model_debug.png"
            cv2.imwrite(str(debug_path), debug_view)
            print(f"已保存颜色模型调试图：{debug_path}")
        if key in (ord("s"), ord("S")):
            config["segmentation"]["threshold"] = threshold
            config["segmentation"]["morphology_mm"] = morphology_mm
            if playing_cards:
                config["segmentation"]["background_distance_threshold"] = background_distance
                config["segmentation"]["blue_hue_tolerance_deg"] = blue_hue_tolerance
                config["segmentation"]["blue_background_min_saturation"] = blue_saturation_min
                config["segmentation"]["playing_card_noise_open_mm"] = card_noise_open_mm
                config["segmentation"]["playing_card_hole_close_mm"] = card_hole_close_mm
            save_config(config_path, config)
            print(
                f"分割参数已保存：threshold={threshold}, "
                f"morphology_mm={morphology_mm:.1f}"
            )
        if key == 32:
            debug_config = copy.deepcopy(config)
            debug_config["segmentation"]["threshold"] = threshold
            debug_config["segmentation"]["morphology_mm"] = morphology_mm
            if playing_cards:
                debug_config["segmentation"]["background_distance_threshold"] = background_distance
                debug_config["segmentation"]["blue_hue_tolerance_deg"] = blue_hue_tolerance
                debug_config["segmentation"]["blue_background_min_saturation"] = blue_saturation_min
                debug_config["segmentation"]["playing_card_noise_open_mm"] = card_noise_open_mm
                debug_config["segmentation"]["playing_card_hole_close_mm"] = card_hole_close_mm
            result, result_image = process_frame(frame.copy(), debug_config, output_dir)
            print_result_diagnostics(result)
            cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
            cv2.imshow(result_window, result_image)


def run_triggered_mode(
    camera,
    initial_frame: np.ndarray,
    config: Dict,
    output_dir: Path,
    display: bool,
) -> int:
    """待机时只预览画面，收到按键后才执行一次完整检测。"""

    if display:
        preview_window = "SCARA Vision Preview"
        result_window = "SCARA Vision Result"
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 800, 630)
        print("视觉程序已进入按键触发模式：按空格检测一次，按 Q 或 Esc 退出。")

        frame = initial_frame
        while True:
            if camera is not None:
                frame = camera.read()
            preview = add_text_banner(frame, "READY: press SPACE to detect, Q to quit")
            cv2.imshow(preview_window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return 0
            if key == 32:  # 空格键
                result, result_image = process_frame(frame.copy(), config, output_dir)
                print_result_diagnostics(result)
                cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
                cv2.imshow(result_window, result_image)

    print("视觉程序已进入按键触发模式：按回车检测一次，输入 Q 后回车退出。")
    frame = initial_frame
    while True:
        try:
            command = input("等待触发 > ").strip().lower()
        except EOFError:
            return 0
        if command == "q":
            return 0
        if camera is not None:
            frame = camera.read()
        result, _ = process_frame(frame.copy(), config, output_dir)
        print_result_diagnostics(result)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("vision_config.json"))
    parser.add_argument("--source", choices=("auto", "picamera2", "usb"), default="auto")
    parser.add_argument("--device", type=int, default=0, help="USB 摄像头编号")
    parser.add_argument("--image", type=Path, help="使用静态图片代替摄像头")
    parser.add_argument("--calibrate", action="store_true", help="进入四点透视标定")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--triggered", action="store_true", help="按键后才执行一次检测")
    run_mode.add_argument("--continuous", action="store_true", help="连续处理图像，仅用于调试")
    parser.add_argument("--display", action="store_true", help="显示实时结果窗口")
    parser.add_argument("--debug", action="store_true", help="显示灰度、二值图和轮廓调试窗口")
    parser.add_argument("--interval", type=float, default=0.25, help="连续模式处理间隔，单位秒")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    try:
        config = load_config(args.config)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"配置文件错误：{error}", file=sys.stderr)
        return 2

    camera = None
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise RuntimeError(f"无法读取图片：{args.image}")
            frame = cv2.resize(frame, (DEFAULT_WIDTH, DEFAULT_HEIGHT), interpolation=cv2.INTER_AREA)
        else:
            camera = open_camera(args.source, args.device, DEFAULT_WIDTH, DEFAULT_HEIGHT)
            frame = camera.read()

        if args.calibrate:
            return 0 if calibration_mode(frame, config, args.config) else 1

        if args.debug:
            if not args.display:
                print("--debug 需要同时添加 --display 才能显示调试窗口。", file=sys.stderr)
                return 2
            return run_debug_triggered_mode(
                camera, frame, config, args.output_dir, args.config
            )

        if args.triggered:
            return run_triggered_mode(
                camera,
                frame,
                config,
                args.output_dir,
                args.display,
            )

        while True:
            if camera is not None:
                frame = camera.read()
            result, result_image = process_frame(frame, config, args.output_dir)
            print_result_diagnostics(result)

            if args.display:
                cv2.namedWindow("SCARA Vision", cv2.WINDOW_NORMAL)
                cv2.imshow("SCARA Vision", result_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if not args.continuous or args.image:
                break
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        print("视觉程序已由用户停止。")
        return 0
    except (RuntimeError, cv2.error) as error:
        print(f"视觉程序运行失败：{error}", file=sys.stderr)
        return 3
    finally:
        if camera is not None:
            camera.close()
        if args.display or args.calibrate:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
