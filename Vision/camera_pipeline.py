"""树莓派 1600×1200 摄像头视觉程序。

功能流程：采集图像、四点透视矫正、碎片轮廓提取、毫米坐标转换、
矩形拼接求解，并输出带中文标注的结果图片和 JSON 数据。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from algorithm import Pt, find_rectangle_solution, polygon_area

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # 仅影响结果图中的中文文字，不影响几何求解
    Image = None
    ImageDraw = None
    ImageFont = None


DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
COLORS = [
    (60, 76, 231),
    (219, 152, 52),
    (113, 204, 46),
    (18, 156, 243),
]


@dataclass
class DetectedPiece:
    """一块完成多边形拟合的碎片。"""

    contour_px: np.ndarray
    points_mm: List[Pt]
    area_mm2: float


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
    return config


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


def perspective_transform(frame: np.ndarray, config: Dict) -> Tuple[np.ndarray, float]:
    """将摄像头画面矫正到俯视毫米平面。"""

    workspace = config["workspace"]
    pixels_per_mm = float(workspace["pixels_per_mm"])
    output_width = round(workspace["width_mm"] * pixels_per_mm)
    output_height = round(workspace["height_mm"] * pixels_per_mm)
    source_points = np.array(workspace["image_points"], dtype=np.float32)
    target_points = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source_points, target_points)
    corrected = cv2.warpPerspective(frame, matrix, (output_width, output_height))
    return corrected, pixels_per_mm


def simplify_contour(
    contour: np.ndarray, pixels_per_mm: float, base_epsilon_mm: float
) -> Optional[np.ndarray]:
    """将像素轮廓简化为具有 3～5 个真实角点的多边形。"""

    perimeter = cv2.arcLength(contour, True)
    for epsilon_mm in np.linspace(base_epsilon_mm, 5.0, 15):
        epsilon_px = max(1.0, float(epsilon_mm) * pixels_per_mm)
        polygon = cv2.approxPolyDP(contour, epsilon_px, True)
        vertex_count = len(polygon)
        if 3 <= vertex_count <= 5:
            return polygon.reshape(-1, 2)
        if vertex_count < 3:
            break
    return None


def detect_pieces(
    corrected: np.ndarray, config: Dict, pixels_per_mm: float
) -> Tuple[List[DetectedPiece], np.ndarray, List[str]]:
    """从透视矫正图中提取碎片，并转换为毫米多边形。"""

    segmentation = config["segmentation"]
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    blur_size = int(segmentation.get("blur_size", 5))
    blur_size = max(3, blur_size | 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    mode = segmentation.get("mode", "light_on_dark")
    threshold_value = int(segmentation.get("threshold", 0))
    threshold_type = cv2.THRESH_BINARY if mode == "light_on_dark" else cv2.THRESH_BINARY_INV
    if threshold_value <= 0:
        threshold_type |= cv2.THRESH_OTSU
    _, mask = cv2.threshold(blurred, threshold_value, 255, threshold_type)

    morphology_mm = float(segmentation.get("morphology_mm", 0.8))
    kernel_size = max(3, round(morphology_mm * pixels_per_mm))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = float(segmentation["min_piece_area_mm2"])
    maximum_area = float(segmentation["max_piece_area_mm2"])
    epsilon_mm = float(segmentation["polygon_epsilon_mm"])
    pieces: List[DetectedPiece] = []
    warnings: List[str] = []

    for contour in contours:
        area_mm2 = cv2.contourArea(contour) / (pixels_per_mm * pixels_per_mm)
        if not minimum_area <= area_mm2 <= maximum_area:
            continue
        polygon_px = simplify_contour(contour, pixels_per_mm, epsilon_mm)
        if polygon_px is None:
            warnings.append(f"忽略一个无法简化为 3～5 边形的轮廓，面积 {area_mm2:.1f} mm²")
            continue
        points_mm = [
            Pt(float(point[0]) / pixels_per_mm, float(point[1]) / pixels_per_mm)
            for point in polygon_px
        ]
        polygon_area_mm2 = polygon_area(points_mm)
        pieces.append(
            DetectedPiece(
                contour_px=polygon_px.astype(np.int32),
                points_mm=points_mm,
                area_mm2=polygon_area_mm2,
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


def point_mm_to_px(point: Pt, pixels_per_mm: float) -> Tuple[int, int]:
    return round(point.x * pixels_per_mm), round(point.y * pixels_per_mm)


def draw_polygon_with_alpha(
    image: np.ndarray, polygon: np.ndarray, color: Tuple[int, int, int], alpha: float
) -> None:
    """用半透明颜色填充多边形，并绘制清晰边界。"""

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)
    cv2.polylines(image, [polygon], True, (30, 30, 30), 3, cv2.LINE_AA)


def load_chinese_font(size: int):
    """加载树莓派或 Windows 中常见的中文字体。"""

    if ImageFont is None:
        return None
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_chinese_banner(image: np.ndarray, text: str) -> np.ndarray:
    """在图像顶部添加中文说明栏。"""

    result = cv2.copyMakeBorder(image, 58, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    if Image is None:
        cv2.putText(result, "Vision result", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
        return result
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    draw.text((20, 11), text, font=load_chinese_font(27), fill=(30, 30, 30))
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def create_result_image(
    corrected: np.ndarray,
    pieces: Sequence[DetectedPiece],
    solution: Optional[Dict],
    pixels_per_mm: float,
    message: str,
) -> np.ndarray:
    """生成左侧检测结果、右侧目标布局的直观对照图。"""

    before = corrected.copy()
    after = np.full_like(corrected, 245)
    for index, piece in enumerate(pieces):
        draw_polygon_with_alpha(before, piece.contour_px, COLORS[index % len(COLORS)], 0.22)
        center = np.mean(piece.contour_px, axis=0).astype(int)
        cv2.putText(before, str(index), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)

    if solution is not None:
        for placement in solution["placements"]:
            index = placement["piece_index"]
            polygon = np.array(
                [point_mm_to_px(point, pixels_per_mm) for point in placement["target_pts"]],
                dtype=np.int32,
            )
            draw_polygon_with_alpha(after, polygon, COLORS[index % len(COLORS)], 0.72)
            center = point_mm_to_px(placement["target_center"], pixels_per_mm)
            cv2.putText(after, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
        rectangle = np.array(
            [point_mm_to_px(point, pixels_per_mm) for point in solution["rectangle"]["corners"]],
            dtype=np.int32,
        )
        cv2.polylines(after, [rectangle], True, (0, 0, 0), 5, cv2.LINE_AA)

    separator = np.full((corrected.shape[0], 8, 3), 90, dtype=np.uint8)
    combined = np.hstack((before, separator, after))
    return add_chinese_banner(combined, message)


def solution_to_json(
    pieces: Sequence[DetectedPiece], solution: Optional[Dict], status: str, message: str
) -> Dict:
    """将求解结果转换为便于树莓派串口程序读取的 JSON。"""

    result = {
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "coordinate_system": "透视矫正后的工作区坐标，原点左上，X 向右，Y 向下，单位 mm",
        "piece_count": len(pieces),
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
                "target_points_mm": [[point.x, point.y] for point in placement["target_pts"]],
            }
            for placement in solution["placements"]
        ],
    }
    return result


def process_frame(frame: np.ndarray, config: Dict, output_dir: Path) -> Tuple[Dict, np.ndarray]:
    """处理单帧图像并保存中间结果。"""

    corrected, pixels_per_mm = perspective_transform(frame, config)
    pieces, mask, warnings = detect_pieces(corrected, config, pixels_per_mm)
    solution = None

    if not 2 <= len(pieces) <= 4:
        status = "fragment_count_error"
        message = f"碎片数量错误：检测到 {len(pieces)} 块，应为 2～4 块"
    else:
        workspace = config["workspace"]
        target = workspace.get(
            "target_center_mm",
            [workspace["width_mm"] * 0.5, workspace["height_mm"] * 0.5],
        )
        solution = find_rectangle_solution(
            [{"pts": piece.points_mm} for piece in pieces],
            target_center=Pt(float(target[0]), float(target[1])),
        )
        if solution is None:
            status = "no_solution"
            message = "检测到碎片，但未找到满足尺寸和面积约束的矩形"
        else:
            status = "ok"
            rectangle = solution["rectangle"]
            short_side, long_side = sorted((rectangle["width"], rectangle["height"]))
            message = (
                f"拼接成功：{long_side:.1f}×{short_side:.1f} mm，"
                f"面积误差 {solution['area_error'] * 100:.2f}%"
            )

    if warnings:
        message += f"；另有 {len(warnings)} 个轮廓被忽略"
    result = solution_to_json(pieces, solution, status, message)
    result["warnings"] = warnings
    result_image = create_result_image(corrected, pieces, solution, pixels_per_mm, message)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "latest_corrected.jpg"), corrected)
    cv2.imwrite(str(output_dir / "latest_mask.png"), mask)
    cv2.imwrite(str(output_dir / "latest_result.jpg"), result_image)
    with (output_dir / "latest_result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result, result_image


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("vision_config.json"))
    parser.add_argument("--source", choices=("auto", "picamera2", "usb"), default="auto")
    parser.add_argument("--device", type=int, default=0, help="USB 摄像头编号")
    parser.add_argument("--image", type=Path, help="使用静态图片代替摄像头")
    parser.add_argument("--calibrate", action="store_true", help="进入四点透视标定")
    parser.add_argument("--continuous", action="store_true", help="连续处理图像")
    parser.add_argument("--display", action="store_true", help="显示实时结果窗口")
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

        while True:
            if camera is not None:
                frame = camera.read()
            result, result_image = process_frame(frame, config, args.output_dir)
            print(f"[{result['timestamp']}] {result['message']}")

            if args.display:
                cv2.namedWindow("SCARA Vision", cv2.WINDOW_NORMAL)
                cv2.imshow("SCARA Vision", result_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if not args.continuous or args.image:
                break
            time.sleep(max(0.0, args.interval))
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
