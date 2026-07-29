"""生成矩形拼接算法的“拼接前/拼接后”直观对照图。

使用方法：
    python visual_demo.py --seed 29 --noise 1.0 --output rectangle_demo.png
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from algorithm import Pt, find_rectangle_solution, generate_puzzle, scatter_pieces


WIDTH = 1400
HEIGHT = 760
PANEL_TOP = 80
PANEL_BOTTOM = 575
LEFT_PANEL = (35, PANEL_TOP, 680, PANEL_BOTTOM)
RIGHT_PANEL = (720, PANEL_TOP, 1365, PANEL_BOTTOM)
COLORS = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (243, 156, 18),
]


def load_font(size: int, bold: bool = False):
    # 优先选择支持中文的 Windows 字体，最后再回退到通用字体。
    names = [
        "msyhbd.ttc" if bold else "msyh.ttc",
        "simhei.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


TITLE_FONT = load_font(27, True)
LABEL_FONT = load_font(19, True)
TEXT_FONT = load_font(17)
SMALL_FONT = load_font(14)


def all_points(polygons: Iterable[Sequence[Pt]]) -> List[Pt]:
    return [point for polygon in polygons for point in polygon]


def make_transform(points: Sequence[Pt], panel: Tuple[int, int, int, int]):
    left, top, right, bottom = panel
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min((right - left - 70) / span_x, (bottom - top - 70) / span_y)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    screen_cx = (left + right) * 0.5
    screen_cy = (top + bottom) * 0.5

    def transform(point: Pt) -> Tuple[int, int]:
        return (
            round(screen_cx + (point.x - center_x) * scale),
            round(screen_cy - (point.y - center_y) * scale),
        )

    return transform, scale


def draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    panel: Tuple[int, int, int, int],
    scale: float,
    length_mm: float = 20.0,
) -> None:
    left, _, _, bottom = panel
    x0 = left + 22
    y = bottom - 21
    x1 = round(x0 + length_mm * scale)
    draw.line((x0, y, x1, y), fill=(70, 70, 70), width=3)
    draw.line((x0, y - 5, x0, y + 5), fill=(70, 70, 70), width=2)
    draw.line((x1, y - 5, x1, y + 5), fill=(70, 70, 70), width=2)
    draw.text((x0, y - 22), f"{length_mm:g} mm", fill=(70, 70, 70), font=SMALL_FONT)


def draw_piece(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Pt],
    transform,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    screen_points = [transform(point) for point in points]
    draw.polygon(screen_points, fill=color, outline=(30, 30, 30), width=3)
    cx = sum(point[0] for point in screen_points) / len(screen_points)
    cy = sum(point[1] for point in screen_points) / len(screen_points)
    box = draw.textbbox((0, 0), label, font=LABEL_FONT)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.rounded_rectangle(
        (
            round(cx - text_width / 2 - 5),
            round(cy - text_height / 2 - 4),
            round(cx + text_width / 2 + 5),
            round(cy + text_height / 2 + 4),
        ),
        radius=5,
        fill=(255, 255, 255),
        outline=(30, 30, 30),
    )
    draw.text(
        (round(cx - text_width / 2), round(cy - text_height / 2 - 1)),
        label,
        fill=(20, 20, 20),
        font=LABEL_FONT,
    )


def render_solution(
    observed: Sequence[Dict], solution: Dict, output_path: Path, seed: int, noise: float
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (246, 247, 249))
    draw = ImageDraw.Draw(image)

    draw.text((35, 22), "Rectangle puzzle: geometry-only result", fill=(25, 25, 25), font=TITLE_FONT)
    draw.text((35, 53), f"seed={seed}, vertex noise=+/-{noise:g} mm", fill=(90, 90, 90), font=SMALL_FONT)
    draw.text((35, 84), "BEFORE: observed pieces", fill=(25, 25, 25), font=TEXT_FONT)
    draw.text((720, 84), "AFTER: computed placement", fill=(25, 25, 25), font=TEXT_FONT)

    for panel in (LEFT_PANEL, RIGHT_PANEL):
        draw.rounded_rectangle(panel, radius=12, fill=(255, 255, 255), outline=(205, 210, 218), width=2)

    source_polygons = [piece["pts"] for piece in observed]
    target_by_index = {
        placement["piece_index"]: placement for placement in solution["placements"]
    }
    target_polygons = [
        target_by_index[index]["target_pts"] for index in range(len(observed))
    ]
    source_transform, source_scale = make_transform(all_points(source_polygons), LEFT_PANEL)
    target_transform, target_scale = make_transform(all_points(target_polygons), RIGHT_PANEL)

    for index, polygon in enumerate(source_polygons):
        draw_piece(draw, polygon, source_transform, COLORS[index], str(index))
    for index, polygon in enumerate(target_polygons):
        draw_piece(draw, polygon, target_transform, COLORS[index], str(index))

    rectangle_points = [target_transform(point) for point in solution["rectangle"]["corners"]]
    draw.line(rectangle_points + [rectangle_points[0]], fill=(20, 20, 20), width=5)
    draw_scale_bar(draw, LEFT_PANEL, source_scale)
    draw_scale_bar(draw, RIGHT_PANEL, target_scale)

    short_side, long_side = sorted(
        (solution["rectangle"]["width"], solution["rectangle"]["height"])
    )
    status = (
        f"VALID RECTANGLE   size={long_side:.1f} x {short_side:.1f} mm   "
        f"uncovered={solution['area_error'] * 100:.2f}%"
    )
    draw.rounded_rectangle((35, 600, 1365, 642), radius=9, fill=(224, 247, 232), outline=(65, 160, 95), width=2)
    draw.text((54, 610), status, fill=(25, 105, 55), font=TEXT_FONT)

    x = 45
    for index in range(len(observed)):
        placement = target_by_index[index]
        angle_deg = math.degrees(placement["angle"])
        target = placement["target_center"]
        text = (
            f"Piece {index}: rotate {angle_deg:+.1f} deg  ->  "
            f"target ({target.x:.1f}, {target.y:.1f}) mm"
        )
        draw.text((x, 667 + (index % 2) * 30), text, fill=COLORS[index], font=TEXT_FONT)
        if index % 2 == 1:
            x = 720

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--noise", type=float, default=0.0, help="顶点噪声，单位为毫米")
    parser.add_argument("--output", type=Path, default=Path("rectangle_demo.png"))
    args = parser.parse_args()

    random.seed(args.seed)
    puzzle = generate_puzzle(0.0, 0.0, 100.0, 70.0)
    pieces = [
        {
            "pts": [
                Pt(
                    point.x + random.uniform(-args.noise, args.noise),
                    point.y + random.uniform(-args.noise, args.noise),
                )
                for point in piece["pts"]
            ]
        }
        for piece in puzzle["pieces"]
    ]
    observed = scatter_pieces(pieces, distance=130.0)
    solution = find_rectangle_solution(observed, target_center=Pt(0.0, 0.0))
    if solution is None:
        print("未找到有效矩形，请尝试降低 --noise 参数。")
        return 1

    output_path = args.output.resolve()
    render_solution(observed, solution, output_path, args.seed, args.noise)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
