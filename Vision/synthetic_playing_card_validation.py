"""生成合成扑克牌碎片图，并用现有视觉流程做一次离线验证。"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from algorithm import Pt, generate_puzzle, scatter_pieces
from camera_pipeline import process_frame


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "synthetic_card_validation"
PIXELS_PER_MM = 4.0
WORKSPACE_W_MM = 210.0
WORKSPACE_H_MM = 297.0
WORKSPACE_W_PX = round(WORKSPACE_W_MM * PIXELS_PER_MM)
WORKSPACE_H_PX = round(WORKSPACE_H_MM * PIXELS_PER_MM)
CARD_W_MM = 63.0
CARD_H_MM = 88.0
CARD_W_PX = round(CARD_W_MM * PIXELS_PER_MM)
CARD_H_PX = round(CARD_H_MM * PIXELS_PER_MM)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先加载本地常见字体，失败时回退。"""

    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\arialbd.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\simhei.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(28, True)
TEXT_FONT = load_font(18)
SMALL_FONT = load_font(15)
CARD_FONT = load_font(40, True)


def pt_list_to_np(points: Sequence[Pt], *, offset: Pt | None = None, scale: float = 1.0) -> np.ndarray:
    """把 Pt 列表转为 numpy 顶点数组。"""

    if offset is None:
        offset = Pt(0.0, 0.0)
    return np.array(
        [[(point.x + offset.x) * scale, (point.y + offset.y) * scale] for point in points],
        dtype=np.float32,
    )


def choose_affine_triangle(points: np.ndarray) -> np.ndarray:
    """从多边形顶点中选取面积足够的三个点，避免仿射退化。"""

    point_count = len(points)
    for first in range(point_count):
        for second in range(first + 1, point_count):
            for third in range(second + 1, point_count):
                triangle = np.array([points[first], points[second], points[third]], dtype=np.float32)
                area = abs(
                    (triangle[1, 0] - triangle[0, 0]) * (triangle[2, 1] - triangle[0, 1])
                    - (triangle[1, 1] - triangle[0, 1]) * (triangle[2, 0] - triangle[0, 0])
                )
                if area > 1e-3:
                    return triangle
    raise ValueError("无法从多边形顶点中找到有效仿射三角形")


def draw_diamond(draw: ImageDraw.ImageDraw, center: Tuple[float, float], size: float, fill: Tuple[int, int, int]) -> None:
    """绘制一个菱形花色。"""

    cx, cy = center
    draw.polygon(
        [
            (cx, cy - size),
            (cx + size * 0.8, cy),
            (cx, cy + size),
            (cx - size * 0.8, cy),
        ],
        fill=fill,
    )


def draw_spade(draw: ImageDraw.ImageDraw, center: Tuple[float, float], size: float, fill: Tuple[int, int, int]) -> None:
    """绘制一个简化黑桃，方便给扑克牌增加方向性纹理。"""

    cx, cy = center
    radius = size * 0.55
    draw.ellipse((cx - radius - size * 0.25, cy - radius, cx - size * 0.25 + radius, cy + radius), fill=fill)
    draw.ellipse((cx + size * 0.25 - radius, cy - radius, cx + size * 0.25 + radius, cy + radius), fill=fill)
    draw.polygon(
        [
            (cx - size * 0.95, cy + size * 0.05),
            (cx + size * 0.95, cy + size * 0.05),
            (cx, cy - size * 1.2),
        ],
        fill=fill,
    )
    draw.polygon(
        [
            (cx - size * 0.18, cy + size * 0.05),
            (cx + size * 0.18, cy + size * 0.05),
            (cx + size * 0.33, cy + size * 1.1),
            (cx - size * 0.33, cy + size * 1.1),
        ],
        fill=fill,
    )


def make_card_texture() -> Image.Image:
    """生成一张带明显方向纹理的合成扑克牌。"""

    image = Image.new("RGB", (CARD_W_PX, CARD_H_PX), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    safe_margin = 22

    # 边框与轻微底纹。这里刻意保留较宽白边，避免深色纹理碰到外轮廓。
    draw.rounded_rectangle((4, 4, CARD_W_PX - 5, CARD_H_PX - 5), radius=16, outline=(180, 180, 180), width=2)
    for offset in range(-CARD_H_PX, CARD_W_PX, 18):
        draw.line((offset, 0, offset + CARD_H_PX, CARD_H_PX), fill=(236, 236, 236), width=2)
    for y in range(34, CARD_H_PX - 34, 46):
        draw.line((safe_margin, y, CARD_W_PX - safe_margin, y), fill=(238, 241, 244), width=2)

    # 中央方向纹理：斜向色带 + 大花色。
    draw.polygon(
        [
            (safe_margin + 8, CARD_H_PX * 0.70),
            (CARD_W_PX * 0.57, CARD_H_PX * 0.37),
            (CARD_W_PX - safe_margin - 8, CARD_H_PX * 0.47),
            (CARD_W_PX * 0.43, CARD_H_PX * 0.80),
        ],
        fill=(214, 48, 49),
    )
    draw.polygon(
        [
            (safe_margin + 4, CARD_H_PX * 0.28),
            (CARD_W_PX * 0.42, CARD_H_PX * 0.19),
            (CARD_W_PX * 0.66, CARD_H_PX * 0.36),
            (CARD_W_PX * 0.25, CARD_H_PX * 0.48),
        ],
        fill=(52, 152, 219),
    )
    draw_spade(draw, (CARD_W_PX * 0.52, CARD_H_PX * 0.52), 38, (20, 20, 20))

    # 角标和边角花色。
    draw.text((safe_margin, 10), "A", font=CARD_FONT, fill=(20, 20, 20))
    draw_spade(draw, (safe_margin + 20, 72), 11, (20, 20, 20))
    draw.text((CARD_W_PX - safe_margin - 28, CARD_H_PX - 58), "A", font=CARD_FONT, fill=(20, 20, 20))
    draw_spade(draw, (CARD_W_PX - safe_margin - 10, CARD_H_PX - 22), 11, (20, 20, 20))

    # 再放几个小图形，给 ORB / NCC 更多可用纹理。
    draw_diamond(draw, (CARD_W_PX * 0.24, CARD_H_PX * 0.65), 12, (180, 40, 45))
    draw_diamond(draw, (CARD_W_PX * 0.74, CARD_H_PX * 0.23), 12, (180, 40, 45))
    draw_spade(draw, (CARD_W_PX * 0.28, CARD_H_PX * 0.28), 11, (20, 20, 20))
    draw_spade(draw, (CARD_W_PX * 0.76, CARD_H_PX * 0.73), 11, (20, 20, 20))
    return image


def create_background() -> np.ndarray:
    """创建深蓝背景，和你现在真实环境更接近。"""

    x = np.linspace(0.0, 1.0, WORKSPACE_W_PX, dtype=np.float32)
    y = np.linspace(0.0, 1.0, WORKSPACE_H_PX, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    blue = 95 + 45 * yy + 12 * np.sin(xx * math.pi * 2.3)
    green = 58 + 24 * yy + 8 * np.cos(xx * math.pi * 1.7)
    red = 8 + 10 * yy
    image = np.stack([blue, green, red], axis=2).astype(np.uint8)
    return image


def shift_polygons_to_region(pieces: Sequence[Dict], x_min: float, x_max: float, y_min: float, y_max: float) -> List[Dict]:
    """把散落碎片整体平移到工作区上半部分，避免出界。"""

    min_x = min(point.x for piece in pieces for point in piece["pts"])
    max_x = max(point.x for piece in pieces for point in piece["pts"])
    min_y = min(point.y for piece in pieces for point in piece["pts"])
    max_y = max(point.y for piece in pieces for point in piece["pts"])

    target_cx = 0.5 * (x_min + x_max)
    target_cy = 0.5 * (y_min + y_max)
    current_cx = 0.5 * (min_x + max_x)
    current_cy = 0.5 * (min_y + max_y)
    dx = target_cx - current_cx
    dy = target_cy - current_cy

    shifted = []
    for piece in pieces:
        shifted.append(
            {
                **piece,
                "pts": [Pt(point.x + dx, point.y + dy) for point in piece["pts"]],
            }
        )
    return shifted


def make_four_piece_puzzle(seed: int) -> Dict:
    """固定随机种子，直到生成四块碎片。"""

    random.seed(seed)
    while True:
        puzzle = generate_puzzle(0.0, 0.0, CARD_W_MM, CARD_H_MM)
        if len(puzzle["pieces"]) == 4:
            return puzzle


def render_scattered_card_scene(puzzle: Dict, seed: int) -> Tuple[np.ndarray, Image.Image, List[Dict]]:
    """把扑克牌纹理切成碎片并散布到深蓝背景上。"""

    random.seed(seed + 1)
    card_texture = make_card_texture()
    card_np = cv2.cvtColor(np.array(card_texture), cv2.COLOR_RGB2BGR)
    background = create_background()
    observed = scatter_pieces(puzzle["pieces"], distance=48.0)
    observed = shift_polygons_to_region(observed, 28.0, 182.0, 28.0, 138.0)

    card_offset = Pt(CARD_W_MM * 0.5, CARD_H_MM * 0.5)
    output_size = (WORKSPACE_W_PX, WORKSPACE_H_PX)

    for piece_observed in observed:
        piece_target = puzzle["pieces"][int(piece_observed["source_index"])]
        src_polygon = pt_list_to_np(piece_target["pts"], offset=card_offset, scale=PIXELS_PER_MM)
        dst_polygon = pt_list_to_np(piece_observed["pts"], scale=PIXELS_PER_MM)

        src_int = np.round(src_polygon).astype(np.int32)
        piece_mask = np.zeros((CARD_H_PX, CARD_W_PX), dtype=np.uint8)
        cv2.fillPoly(piece_mask, [src_int], 255)
        piece_texture = cv2.bitwise_and(card_np, card_np, mask=piece_mask)

        src_tri = choose_affine_triangle(src_polygon)
        dst_tri = choose_affine_triangle(dst_polygon)
        matrix = cv2.getAffineTransform(src_tri, dst_tri)

        warped_texture = cv2.warpAffine(
            piece_texture,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv2.warpAffine(
            piece_mask,
            matrix,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        shadow_mask = cv2.GaussianBlur(warped_mask, (0, 0), 3)
        shadow_layer = np.zeros_like(background)
        shadow_layer[:] = (0, 0, 0)
        shadow_strength = (shadow_mask.astype(np.float32) / 255.0 * 0.08)[..., None]
        background = np.clip(background.astype(np.float32) * (1.0 - shadow_strength), 0, 255).astype(np.uint8)

        foreground = cv2.bitwise_and(warped_texture, warped_texture, mask=warped_mask)
        mask_3 = (warped_mask > 0)[..., None]
        background = np.where(mask_3, foreground, background)

    return background, card_texture, observed


def prepare_config(piece_mode: str, texture_enabled: bool) -> Dict:
    """构造离线验证所需配置，不改用户当前正式配置。"""

    with (PROJECT_DIR / "vision_config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)

    config = copy.deepcopy(config)
    config["piece_mode"] = piece_mode
    config["texture"]["enabled"] = texture_enabled
    config["workspace"]["pixels_per_mm"] = PIXELS_PER_MM
    config["workspace"]["image_points"] = [
        [0, 0],
        [WORKSPACE_W_PX - 1, 0],
        [WORKSPACE_W_PX - 1, WORKSPACE_H_PX - 1],
        [0, WORKSPACE_H_PX - 1],
    ]
    config["workspace"]["rectify_margin_mm"] = 0.0
    config["segmentation"]["threshold"] = 70
    config["segmentation"]["morphology_mm"] = 0.6
    config["segmentation"]["polygon_epsilon_mm"] = 1.0
    config["segmentation"]["short_edge_mm"] = 6.0
    config["segmentation"]["max_piece_area_mm2"] = 9000.0
    config["solver"]["placement_spread_mm"] = 0.0
    return config


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    """保存 BGR 图像。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)


def build_overview(
    reference_card: Image.Image,
    synthetic_scene_bgr: np.ndarray,
    plain_result_path: Path,
    card_result_path: Path,
    summary: Dict,
    output_path: Path,
) -> None:
    """生成一张直观总览图。"""

    reference = reference_card.convert("RGB")
    scene = Image.fromarray(cv2.cvtColor(synthetic_scene_bgr, cv2.COLOR_BGR2RGB))
    plain_result = Image.open(plain_result_path).convert("RGB")
    card_result = Image.open(card_result_path).convert("RGB")

    canvas = Image.new("RGB", (1760, 1420), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((26, 18), "Synthetic Playing Card Validation", font=TITLE_FONT, fill=(28, 32, 38))
    draw.text((28, 54), "Generate poker-like fragments locally, then run the current vision pipeline.", font=SMALL_FONT, fill=(92, 98, 108))

    panels = [
        ("REFERENCE CARD", reference, (26, 96, 530, 458)),
        ("SYNTHETIC INPUT", scene, (566, 96, 1718, 698)),
        ("PLAIN MODE RESULT", plain_result, (26, 492, 852, 1274)),
        ("PLAYING_CARD MODE RESULT", card_result, (892, 732, 1718, 1274)),
    ]

    for title, image, rect in panels:
        left, top, right, bottom = rect
        draw.rounded_rectangle(rect, radius=14, fill=(255, 255, 255), outline=(210, 215, 223), width=2)
        draw.text((left + 14, top + 8), title, font=TEXT_FONT, fill=(35, 40, 46))
        fit = image.copy()
        fit.thumbnail((right - left - 20, bottom - top - 50))
        canvas.paste(fit, (left + (right - left - fit.width) // 2, top + 38 + (bottom - top - 48 - fit.height) // 2))

    stat_left, stat_top, stat_right, stat_bottom = 566, 732, 852, 1274
    draw.rounded_rectangle((stat_left, stat_top, stat_right, stat_bottom), radius=14, fill=(255, 255, 255), outline=(210, 215, 223), width=2)
    draw.text((stat_left + 14, stat_top + 8), "SUMMARY", font=TEXT_FONT, fill=(35, 40, 46))

    lines = [
        f"seed = {summary['seed']}",
        f"piece_count = {summary['piece_count']}",
        "",
        f"plain.status = {summary['plain']['status']}",
        f"plain.message = {summary['plain']['message']}",
        "",
        f"playing_cards.status = {summary['playing_cards']['status']}",
        f"playing_cards.message = {summary['playing_cards']['message']}",
        f"texture_score = {summary['playing_cards'].get('texture_score', 'N/A')}",
        "",
        "Note:",
        "Current pipeline in playing_cards mode",
        "already computes texture score,",
        "but the main camera flow still solves",
        "the rectangle primarily by geometry.",
    ]
    y = stat_top + 44
    for line in lines:
        draw.text((stat_left + 14, y), line, font=SMALL_FONT, fill=(48, 54, 60))
        y += 26

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_validation(seed: int = 20260730) -> Dict:
    """生成扑克牌碎片图，并分别用 plain / playing_cards 模式验证。"""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    puzzle = make_four_piece_puzzle(seed)
    synthetic_scene, reference_card, observed = render_scattered_card_scene(puzzle, seed)

    save_image(OUTPUT_ROOT / "synthetic_playing_card_input.png", synthetic_scene)
    reference_card.save(OUTPUT_ROOT / "synthetic_playing_card_reference.png")

    plain_output = OUTPUT_ROOT / "plain_mode"
    card_output = OUTPUT_ROOT / "playing_cards_mode"
    plain_result, _ = process_frame(synthetic_scene.copy(), prepare_config("plain", False), plain_output)
    card_result, _ = process_frame(synthetic_scene.copy(), prepare_config("playing_cards", True), card_output)

    summary = {
        "seed": seed,
        "piece_count": len(observed),
        "plain": {
            "status": plain_result["status"],
            "message": plain_result["message"],
        },
        "playing_cards": {
            "status": card_result["status"],
            "message": card_result["message"],
        },
    }
    texture_info = card_result.get("texture") or {}
    if texture_info:
        summary["playing_cards"]["texture_score"] = round(float(texture_info.get("texture_score", 0.0)), 4)
        summary["playing_cards"]["j_total"] = round(float(texture_info.get("j_total", 0.0)), 4)

    with (OUTPUT_ROOT / "validation_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    build_overview(
        reference_card,
        synthetic_scene,
        plain_output / "latest_result.jpg",
        card_output / "latest_result.jpg",
        summary,
        OUTPUT_ROOT / "validation_overview.png",
    )
    return summary


def main() -> int:
    summary = run_validation()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出目录：{OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
