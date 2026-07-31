"""扑克牌模板匹配工具，支持 A~10 / J / Q / K。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


VALID_SUITS: Tuple[str, ...] = ("spade", "heart", "club", "diamond")
RANK_SORT_ORDER: Dict[str, int] = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
}
SUPPORTED_TEMPLATE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class TemplateFeature:
    name: str
    rank: str
    rank_group: str
    suit_color: str
    image: np.ndarray
    red_mask: np.ndarray
    black_mask: np.ndarray
    ink_mask: np.ndarray
    white_mask: np.ndarray
    edge_mask: np.ndarray


@dataclass(frozen=True)
class CardFeature:
    image: np.ndarray
    red_mask: np.ndarray
    black_mask: np.ndarray
    ink_mask: np.ndarray
    white_mask: np.ndarray
    edge_mask: np.ndarray


def _template_suit_color_from_name(template_name: str) -> str:
    suit = template_name.split("_", 1)[0].lower()
    if suit in ("heart", "diamond"):
        return "red"
    if suit in ("spade", "club"):
        return "black"
    return "unknown"


def _template_rank_from_name(template_name: str) -> str:
    parts = template_name.split("_", 1)
    return parts[1].upper() if len(parts) == 2 else ""


def _rank_group(rank: str) -> str:
    normalized = str(rank).upper()
    if normalized in ("J", "Q", "K"):
        return "face"
    if normalized in RANK_SORT_ORDER:
        return "number"
    return "unknown"


def expected_template_files() -> List[str]:
    return [
        f"{suit}_{rank}.jpg"
        for suit in VALID_SUITS
        for rank in ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
    ]


def _parse_template_stem(stem: str) -> Tuple[str, str] | None:
    if "_" not in stem:
        return None
    suit, rank = stem.split("_", 1)
    suit = suit.strip().lower()
    rank = rank.strip().upper()
    if suit not in VALID_SUITS:
        return None
    if rank not in RANK_SORT_ORDER:
        return None
    return suit, rank


def _discover_template_names(template_dir: Path) -> List[str]:
    names = set()
    for path in template_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_TEMPLATE_SUFFIXES:
            continue
        parsed = _parse_template_stem(path.stem)
        if parsed is None:
            continue
        suit, rank = parsed
        names.add(f"{suit}_{rank}")
    return sorted(
        names,
        key=lambda name: (
            VALID_SUITS.index(name.split("_", 1)[0]),
            RANK_SORT_ORDER.get(name.split("_", 1)[1], 999),
            name,
        ),
    )


def _find_template_path(template_dir: Path, template_name: str) -> Path | None:
    for suffix in SUPPORTED_TEMPLATE_SUFFIXES:
        path = template_dir / f"{template_name}{suffix}"
        if path.exists():
            return path
    return None


def _load_image_with_white_background(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255.0)
        merged = bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])
        return np.clip(merged, 0, 255).astype(np.uint8)
    return image[:, :, :3].copy()


def _resize_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 3) -> np.ndarray:
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _extract_card_feature(
    image: np.ndarray,
    *,
    merge_chromatic_into_black: bool = False,
) -> CardFeature:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)

    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.uint8)
    value = hsv[:, :, 2].astype(np.uint8)
    max_delta_from_white = np.max(255 - image.astype(np.int16), axis=2).astype(np.uint8)
    chroma_span = (
        np.max(image.astype(np.int16), axis=2) - np.min(image.astype(np.int16), axis=2)
    ).astype(np.uint8)
    red_dominance = (red - np.maximum(green, blue)).astype(np.int16)

    red_mask = (
        (
            ((hue <= 16) | (hue >= 165))
            & (saturation >= 52)
            & (value >= 30)
            & (red_dominance >= 18)
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
        )
        .astype(np.uint8)
        * 255
    )
    chromatic_mask = (
        (
            (saturation >= 56)
            & (value >= 42)
            & (value <= 242)
            & (chroma_span >= 28)
            & (~(red_mask > 0))
            & (~(black_mask > 0))
        ).astype(np.uint8)
        * 255
    )
    if merge_chromatic_into_black:
        black_mask = cv2.bitwise_or(black_mask, chromatic_mask)
    # 牌面底色可能因灯光变成绿色/青色。不能把整块底色当作 ink，
    # 否则数字牌的黑色点数会被大面积彩色背景淹没。红色已经由
    # red_mask 单独提取，其余彩色细节只在满足黑色阈值时进入 ink。
    # 模板中的彩色细节如果需要保留，会通过 merge_chromatic_into_black
    # 合并进模板 black_mask，再进入模板 ink_mask。
    ink_mask = (
        (
            (red_mask > 0)
            | (black_mask > 0)
            | ((max_delta_from_white >= 40) & (gray <= 205) & (saturation <= 72))
            | (gray <= 125)
        ).astype(np.uint8)
        * 255
    )

    red_mask = _clean_mask(red_mask, 3, 3)
    black_mask = _clean_mask(black_mask, 3, 3)
    ink_mask = _clean_mask(ink_mask, 3, 3)
    white_mask = (
        (
            (ink_mask == 0)
            & (value >= 118)
            & (gray >= 122)
            & (max_delta_from_white <= 165)
        ).astype(np.uint8)
        * 255
    )
    white_mask = _clean_mask(white_mask, 3, 5)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    edges = cv2.bitwise_and(edges, ink_mask)

    return CardFeature(
        image=image,
        red_mask=red_mask,
        black_mask=black_mask,
        ink_mask=ink_mask,
        white_mask=white_mask,
        edge_mask=edges,
    )


def _dominant_card_color_hint(
    card: CardFeature,
    rank_roi: np.ndarray,
    corner_roi: np.ndarray,
) -> str:
    """根据角标区域估计当前牌更像红牌还是黑牌。"""

    focus_roi = cv2.bitwise_or(rank_roi, corner_roi) > 0
    if not np.any(focus_roi):
        return "unknown"

    red_count = int(np.count_nonzero((card.red_mask > 0) & focus_roi))
    black_count = int(np.count_nonzero((card.black_mask > 0) & focus_roi))
    total = red_count + black_count
    if total < 12:
        return "unknown"
    if red_count >= black_count * 1.35:
        return "red"
    if black_count >= red_count * 1.35:
        return "black"
    return "unknown"


@lru_cache(maxsize=8)
def _load_template_bank_cached(
    template_dir_text: str,
    canvas_width_px: int,
    canvas_height_px: int,
) -> Tuple[TemplateFeature, ...]:
    template_dir = Path(template_dir_text)
    features: List[TemplateFeature] = []
    for template_name in _discover_template_names(template_dir):
        path = _find_template_path(template_dir, template_name)
        if path is None:
            continue
        image = _load_image_with_white_background(path)
        if image is None:
            continue
        card = _extract_card_feature(
            _resize_to_canvas(image, canvas_width_px, canvas_height_px),
            merge_chromatic_into_black=True,
        )
        features.append(
            TemplateFeature(
                name=template_name,
                rank=_template_rank_from_name(template_name),
                rank_group=_rank_group(_template_rank_from_name(template_name)),
                suit_color=_template_suit_color_from_name(template_name),
                image=card.image,
                red_mask=card.red_mask,
                black_mask=card.black_mask,
                ink_mask=card.ink_mask,
                white_mask=card.white_mask,
                edge_mask=card.edge_mask,
            )
        )
    return tuple(features)


def load_template_bank(
    template_dir: Path,
    canvas_width_px: int,
    canvas_height_px: int,
) -> Dict:
    templates = list(
        _load_template_bank_cached(
            str(template_dir.resolve()),
            int(canvas_width_px),
            int(canvas_height_px),
        )
    )
    return {
        "templates": templates,
        "missing_templates": [],
        "loaded_template_count": len(templates),
    }


@lru_cache(maxsize=16)
def _roi_masks(
    width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_mask = np.zeros((height, width), dtype=np.uint8)
    corner_mask = np.zeros((height, width), dtype=np.uint8)
    portrait_mask = np.zeros((height, width), dtype=np.uint8)
    rank_mask = np.zeros((height, width), dtype=np.uint8)

    center_width = int(round(width * 0.60))
    center_height = int(round(height * 0.65))
    center_x0 = (width - center_width) // 2
    center_y0 = (height - center_height) // 2
    center_mask[
        center_y0:center_y0 + center_height,
        center_x0:center_x0 + center_width,
    ] = 255

    corner_width = int(round(width * 0.26))
    corner_height = int(round(height * 0.22))
    corner_margin_x = int(round(width * 0.06))
    corner_margin_y = int(round(height * 0.05))

    corner_mask[
        corner_margin_y:corner_margin_y + corner_height,
        corner_margin_x:corner_margin_x + corner_width,
    ] = 255
    corner_mask[
        height - corner_margin_y - corner_height:height - corner_margin_y,
        width - corner_margin_x - corner_width:width - corner_margin_x,
    ] = 255

    portrait_width = int(round(width * 0.34))
    portrait_height = int(round(height * 0.44))
    portrait_x0 = (width - portrait_width) // 2
    portrait_y0 = int(round(height * 0.27))
    portrait_mask[
        portrait_y0:portrait_y0 + portrait_height,
        portrait_x0:portrait_x0 + portrait_width,
    ] = 255

    rank_width = int(round(width * 0.18))
    rank_height = int(round(height * 0.18))
    rank_margin_x = int(round(width * 0.04))
    rank_margin_y = int(round(height * 0.03))
    rank_mask[
        rank_margin_y:rank_margin_y + rank_height,
        rank_margin_x:rank_margin_x + rank_width,
    ] = 255
    rank_mask[
        height - rank_margin_y - rank_height:height - rank_margin_y,
        width - rank_margin_x - rank_width:width - rank_margin_x,
    ] = 255
    return center_mask, corner_mask, portrait_mask, rank_mask


def _make_rect_mask(
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


@lru_cache(maxsize=16)
def _detail_roi_masks(
    width: int, height: int
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    portrait_width = int(round(width * 0.34))
    portrait_height = int(round(height * 0.44))
    portrait_x0 = (width - portrait_width) // 2
    portrait_y0 = int(round(height * 0.27))
    portrait_x1 = portrait_x0 + portrait_width
    portrait_y1 = portrait_y0 + portrait_height

    portrait_blocks_list = []
    x_cuts = [portrait_x0]
    y_cuts = [portrait_y0]
    for index in range(1, 3):
        x_cuts.append(portrait_x0 + (portrait_width * index) // 3)
        y_cuts.append(portrait_y0 + (portrait_height * index) // 3)
    x_cuts.append(portrait_x1)
    y_cuts.append(portrait_y1)

    for row in range(3):
        for col in range(3):
            portrait_blocks_list.append(
                _make_rect_mask(
                    width,
                    height,
                    x_cuts[col],
                    y_cuts[row],
                    x_cuts[col + 1],
                    y_cuts[row + 1],
                )
            )
    portrait_blocks = tuple(portrait_blocks_list)

    rank_width = int(round(width * 0.18))
    rank_height = int(round(height * 0.18))
    rank_margin_x = int(round(width * 0.04))
    rank_margin_y = int(round(height * 0.03))
    top_left_x0 = rank_margin_x
    top_left_y0 = rank_margin_y
    top_left_x1 = top_left_x0 + rank_width
    top_left_y1 = top_left_y0 + rank_height
    bottom_right_x0 = width - rank_margin_x - rank_width
    bottom_right_y0 = height - rank_margin_y - rank_height
    bottom_right_x1 = bottom_right_x0 + rank_width
    bottom_right_y1 = bottom_right_y0 + rank_height

    top_letter = _make_rect_mask(
        width,
        height,
        top_left_x0,
        top_left_y0,
        top_left_x0 + int(round(rank_width * 0.58)),
        top_left_y0 + int(round(rank_height * 0.56)),
    )
    top_suit = _make_rect_mask(
        width,
        height,
        top_left_x0,
        top_left_y0 + int(round(rank_height * 0.50)),
        top_left_x0 + int(round(rank_width * 0.58)),
        top_left_y1,
    )
    bottom_letter = _make_rect_mask(
        width,
        height,
        bottom_right_x0 + int(round(rank_width * 0.42)),
        bottom_right_y0 + int(round(rank_height * 0.44)),
        bottom_right_x1,
        bottom_right_y1,
    )
    bottom_suit = _make_rect_mask(
        width,
        height,
        bottom_right_x0 + int(round(rank_width * 0.42)),
        bottom_right_y0,
        bottom_right_x1,
        bottom_right_y0 + int(round(rank_height * 0.50)),
    )
    letter_mask = cv2.bitwise_or(top_letter, bottom_letter)
    suit_mask = cv2.bitwise_or(top_suit, bottom_suit)

    top_band = _make_rect_mask(
        width,
        height,
        int(round(width * 0.22)),
        int(round(height * 0.05)),
        int(round(width * 0.78)),
        int(round(height * 0.17)),
    )
    bottom_band = _make_rect_mask(
        width,
        height,
        int(round(width * 0.22)),
        int(round(height * 0.83)),
        int(round(width * 0.78)),
        int(round(height * 0.95)),
    )
    outer_margin_x = max(1, int(round(width * 0.015)))
    outer_margin_y = max(1, int(round(height * 0.015)))
    inner_margin_x = max(outer_margin_x + 2, 43)
    inner_margin_y = max(outer_margin_y + 2, 35)
    outer_band = _make_rect_mask(
        width,
        height,
        outer_margin_x,
        outer_margin_y,
        width - outer_margin_x,
        height - outer_margin_y,
    )
    inner_cut = _make_rect_mask(
        width,
        height,
        inner_margin_x,
        inner_margin_y,
        width - inner_margin_x,
        height - inner_margin_y,
    )
    border_band = cv2.bitwise_and(outer_band, cv2.bitwise_not(inner_cut))
    return portrait_blocks, letter_mask, suit_mask, top_band, bottom_band, border_band


def _coarse_rank_group_hint(
    card: CardFeature,
    templates: Sequence[TemplateFeature],
    portrait_roi: np.ndarray,
    portrait_blocks: Sequence[np.ndarray],
    top_band_roi: np.ndarray,
    bottom_band_roi: np.ndarray,
    margin: float,
) -> Dict[str, float | str]:
    """先粗判当前整牌更像人物牌还是数字牌，减少跨类型误匹配。"""

    best_scores: Dict[str, float] = {"face": 0.0, "number": 0.0}
    best_names: Dict[str, str] = {"face": "", "number": ""}

    for template in templates:
        if template.rank_group not in ("face", "number"):
            continue
        full_ink = _mask_f1_score(card.ink_mask, template.ink_mask)
        full_edge = _edge_distance_score(card.edge_mask, template.edge_mask)
        portrait_ink = _mask_f1_score(card.ink_mask, template.ink_mask, portrait_roi)
        portrait_edge = _edge_distance_score(
            card.edge_mask,
            template.edge_mask,
            portrait_roi,
        )
        portrait_block_edge = _mean_edge_distance_score(
            card.edge_mask,
            template.edge_mask,
            portrait_blocks,
        )
        top_band_ink = _mask_f1_score(card.ink_mask, template.ink_mask, top_band_roi)
        bottom_band_ink = _mask_f1_score(
            card.ink_mask,
            template.ink_mask,
            bottom_band_roi,
        )

        # 粗分类阶段必须使用同一套权重比较两类模板。
        # 原来 face 和 number 使用不同权重，比较出来的分数不在同一
        # 个尺度上，数字牌的复杂黑色花纹很容易被误判成人头牌。
        coarse_score = (
            0.22 * full_ink
            + 0.16 * full_edge
            + 0.12 * portrait_ink
            + 0.12 * portrait_edge
            + 0.08 * portrait_block_edge
            + 0.15 * top_band_ink
            + 0.15 * bottom_band_ink
        )

        if coarse_score > best_scores[template.rank_group]:
            best_scores[template.rank_group] = float(coarse_score)
            best_names[template.rank_group] = template.name

    face_score = float(best_scores["face"])
    number_score = float(best_scores["number"])
    if face_score - number_score >= float(margin):
        hint = "face"
    elif number_score - face_score >= float(margin):
        hint = "number"
    else:
        hint = "unknown"

    return {
        "hint": hint,
        "face_score": face_score,
        "number_score": number_score,
        "confidence": max(face_score, number_score),
        "margin": abs(face_score - number_score),
        "face_template": best_names["face"],
        "number_template": best_names["number"],
    }


def _mean_mask_f1_score(
    first: np.ndarray,
    second: np.ndarray,
    rois: Sequence[np.ndarray],
) -> float:
    if not rois:
        return 0.0
    return float(np.mean([_mask_f1_score(first, second, roi) for roi in rois]))


def _mean_edge_distance_score(
    first_edge: np.ndarray,
    second_edge: np.ndarray,
    rois: Sequence[np.ndarray],
) -> float:
    if not rois:
        return 0.0
    return float(np.mean([_edge_distance_score(first_edge, second_edge, roi) for roi in rois]))


def _mask_f1_score(first: np.ndarray, second: np.ndarray, roi: np.ndarray | None = None) -> float:
    a = first > 0
    b = second > 0
    if roi is not None:
        valid = roi > 0
        a &= valid
        b &= valid
    a_sum = int(np.count_nonzero(a))
    b_sum = int(np.count_nonzero(b))
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    overlap = int(np.count_nonzero(a & b))
    return float((2.0 * overlap) / max(a_sum + b_sum, 1))


def _edge_distance_score(
    first_edge: np.ndarray,
    second_edge: np.ndarray,
    roi: np.ndarray | None = None,
) -> float:
    a = first_edge > 0
    b = second_edge > 0
    if roi is not None:
        valid = roi > 0
        a &= valid
        b &= valid
    a_count = int(np.count_nonzero(a))
    b_count = int(np.count_nonzero(b))
    if a_count < 6 and b_count < 6:
        return 0.6
    if a_count == 0 or b_count == 0:
        return 0.0

    b_inv = np.where(b, 0, 255).astype(np.uint8)
    a_inv = np.where(a, 0, 255).astype(np.uint8)
    dist_to_b = cv2.distanceTransform(b_inv, cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform(a_inv, cv2.DIST_L2, 3)

    mean_ab = float(np.mean(dist_to_b[a])) if a_count else 12.0
    mean_ba = float(np.mean(dist_to_a[b])) if b_count else 12.0
    mean_dist = 0.5 * (mean_ab + mean_ba)
    return float(math.exp(-mean_dist / 4.0))


def _rotate_card_feature(card: CardFeature, orientation_deg: int) -> CardFeature:
    normalized = orientation_deg % 360
    if normalized == 0:
        return card
    if normalized == 180:
        return CardFeature(
            image=cv2.rotate(card.image, cv2.ROTATE_180),
            red_mask=cv2.rotate(card.red_mask, cv2.ROTATE_180),
            black_mask=cv2.rotate(card.black_mask, cv2.ROTATE_180),
            ink_mask=cv2.rotate(card.ink_mask, cv2.ROTATE_180),
            white_mask=cv2.rotate(card.white_mask, cv2.ROTATE_180),
            edge_mask=cv2.rotate(card.edge_mask, cv2.ROTATE_180),
        )
    raise ValueError("第一阶段只支持 0° 和 180° 模板方向")


def create_mask_comparison_image(card: CardFeature, template: TemplateFeature) -> np.ndarray:
    def to_bgr(mask: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack((to_bgr(card.red_mask), to_bgr(template.red_mask)))
    bottom = np.hstack((to_bgr(card.black_mask), to_bgr(template.black_mask)))
    return np.vstack((top, bottom))


def match_card_to_templates(
    card_image: np.ndarray,
    template_dir: Path,
    *,
    canvas_width_px: int,
    canvas_height_px: int,
    match_orientations_deg: Sequence[int],
    center_weight: float,
    corner_weight: float,
    red_weight: float,
    black_weight: float,
    rank_group_prefilter_enabled: bool = True,
    rank_group_prefilter_margin: float = 0.025,
    rank_group_prefilter_min_score: float = 0.56,
) -> Dict:
    template_bank = load_template_bank(
        template_dir,
        canvas_width_px,
        canvas_height_px,
    )
    templates: Sequence[TemplateFeature] = template_bank["templates"]
    if not templates:
        return {
            "error": "no_templates_loaded",
            "loaded_template_count": 0,
            "missing_templates": template_bank["missing_templates"],
        }

    card = _extract_card_feature(
        _resize_to_canvas(card_image, canvas_width_px, canvas_height_px)
    )
    center_roi, corner_roi, portrait_roi, rank_roi = _roi_masks(
        canvas_width_px, canvas_height_px
    )
    portrait_blocks, letter_roi, suit_roi, top_band_roi, bottom_band_roi, border_band_roi = _detail_roi_masks(
        canvas_width_px, canvas_height_px
    )
    border_eval_roi = cv2.bitwise_and(border_band_roi, cv2.bitwise_not(corner_roi))
    color_hint = _dominant_card_color_hint(card, rank_roi, corner_roi)
    rank_group_info = _coarse_rank_group_hint(
        card,
        templates,
        portrait_roi,
        portrait_blocks,
        top_band_roi,
        bottom_band_roi,
        rank_group_prefilter_margin,
    )
    # 粗分类只在“分数足够高且类别差距明显”时用于预筛选。
    # 低质量拼接图、碎片缺失或花纹被遮挡时保留两类模板，避免数字牌
    # 因一次不可靠的 face 判断而永远失去数字模板。
    rank_group_hint = "unknown"
    if (
        rank_group_prefilter_enabled
        and str(rank_group_info["hint"]) in ("face", "number")
        and float(rank_group_info["confidence"]) >= float(rank_group_prefilter_min_score)
        and float(rank_group_info["margin"]) >= float(rank_group_prefilter_margin)
    ):
        rank_group_hint = str(rank_group_info["hint"])
    filtered_templates = [
        template
        for template in templates
        if (color_hint == "unknown" or template.suit_color == color_hint)
        and (rank_group_hint == "unknown" or template.rank_group == rank_group_hint)
    ]
    if not filtered_templates:
        filtered_templates = [
            template
            for template in templates
            if color_hint == "unknown" or template.suit_color == color_hint
        ]
    if not filtered_templates:
        filtered_templates = list(templates)

    best_payload: Dict | None = None
    second_best_payload: Dict | None = None
    best_oriented_card: CardFeature | None = None
    best_template: TemplateFeature | None = None

    for orientation_deg in match_orientations_deg:
        oriented_card = _rotate_card_feature(card, int(orientation_deg))
        for template in filtered_templates:
            full_ink = _mask_f1_score(oriented_card.ink_mask, template.ink_mask)
            full_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
            )
            center_ink = _mask_f1_score(oriented_card.ink_mask, template.ink_mask, center_roi)
            center_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                center_roi,
            )
            portrait_ink = _mask_f1_score(
                oriented_card.ink_mask, template.ink_mask, portrait_roi
            )
            portrait_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                portrait_roi,
            )
            portrait_block_ink = _mean_mask_f1_score(
                oriented_card.ink_mask,
                template.ink_mask,
                portrait_blocks,
            )
            portrait_block_edge = _mean_edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                portrait_blocks,
            )
            top_band_ink = _mask_f1_score(
                oriented_card.ink_mask,
                template.ink_mask,
                top_band_roi,
            )
            bottom_band_ink = _mask_f1_score(
                oriented_card.ink_mask,
                template.ink_mask,
                bottom_band_roi,
            )
            border_black = _mask_f1_score(
                oriented_card.black_mask,
                template.black_mask,
                border_eval_roi,
            )
            corner_ink = _mask_f1_score(oriented_card.ink_mask, template.ink_mask, corner_roi)
            corner_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                corner_roi,
            )
            rank_red = _mask_f1_score(
                oriented_card.red_mask, template.red_mask, rank_roi
            )
            rank_black = _mask_f1_score(
                oriented_card.black_mask, template.black_mask, rank_roi
            )
            rank_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                rank_roi,
            )
            letter_ink = _mask_f1_score(
                oriented_card.ink_mask,
                template.ink_mask,
                letter_roi,
            )
            suit_ink = _mask_f1_score(
                oriented_card.ink_mask,
                template.ink_mask,
                suit_roi,
            )
            letter_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                letter_roi,
            )
            suit_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                suit_roi,
            )
            if template.rank_group == "face":
                center_score = (
                    0.06 * center_ink
                    + 0.08 * center_edge
                    + 0.08 * portrait_ink
                    + 0.18 * portrait_edge
                    + 0.12 * portrait_block_ink
                    + 0.26 * portrait_block_edge
                )
                center_score += 0.02 * top_band_ink + 0.02 * bottom_band_ink
                corner_score = (
                    0.14 * corner_ink
                    + 0.11 * corner_edge
                    + 0.13 * rank_red
                    + 0.13 * rank_black
                    + 0.10 * rank_edge
                    + 0.10 * letter_ink
                    + 0.10 * letter_edge
                    + 0.05 * suit_ink
                    + 0.05 * suit_edge
                    + 0.50 * border_black
                )
                # face 分支的权重和原先分别为 0.82、1.31，直接参与总分
                # 会系统性抬高人头牌。归一化后才能和 number 公平比较。
                center_score /= 0.82
                corner_score /= 1.31
            else:
                center_score = (
                    0.22 * full_ink
                    + 0.20 * full_edge
                    + 0.18 * center_ink
                    + 0.14 * center_edge
                    + 0.10 * top_band_ink
                    + 0.10 * bottom_band_ink
                    + 0.03 * portrait_block_ink
                    + 0.03 * portrait_block_edge
                )
                corner_score = (
                    0.10 * corner_ink
                    + 0.08 * corner_edge
                    + 0.16 * rank_red
                    + 0.16 * rank_black
                    + 0.16 * rank_edge
                    + 0.12 * letter_ink
                    + 0.10 * letter_edge
                    + 0.06 * suit_ink
                    + 0.06 * suit_edge
                    + 0.12 * border_black
                )
                # number 分支权重和分别为 1.00、0.96，同样显式归一化。
                corner_score /= 0.96

            red_score = _mask_f1_score(oriented_card.red_mask, template.red_mask)
            black_score = _mask_f1_score(oriented_card.black_mask, template.black_mask)
            total_score = (
                center_weight * center_score
                + corner_weight * corner_score
                + red_weight * red_score
                + black_weight * black_score
            )

            payload = {
                "template_name": template.name,
                "template_rank": template.rank,
                "template_rank_group": template.rank_group,
                "orientation_deg": int(orientation_deg),
                "score": float(total_score),
                "full_ink_score": float(full_ink),
                "full_edge_score": float(full_edge),
                "center_score": float(center_score),
                "corner_score": float(corner_score),
                "portrait_ink_score": float(portrait_ink),
                "portrait_edge_score": float(portrait_edge),
                "portrait_block_ink_score": float(portrait_block_ink),
                "portrait_block_edge_score": float(portrait_block_edge),
                "top_band_ink_score": float(top_band_ink),
                "bottom_band_ink_score": float(bottom_band_ink),
                "border_white_score": float(border_black),
                "border_black_score": float(border_black),
                "rank_red_score": float(rank_red),
                "rank_black_score": float(rank_black),
                "rank_edge_score": float(rank_edge),
                "letter_ink_score": float(letter_ink),
                "letter_edge_score": float(letter_edge),
                "suit_ink_score": float(suit_ink),
                "suit_edge_score": float(suit_edge),
                "red_score": float(red_score),
                "black_score": float(black_score),
            }
            if best_payload is None or payload["score"] > best_payload["score"]:
                second_best_payload = best_payload
                best_payload = payload
                best_oriented_card = oriented_card
                best_template = template
            elif (
                second_best_payload is None
                or payload["score"] > second_best_payload["score"]
            ):
                second_best_payload = payload

    if best_payload is None or best_oriented_card is None or best_template is None:
        return {
            "error": "template_match_failed",
            "loaded_template_count": len(templates),
            "missing_templates": template_bank["missing_templates"],
        }
    second_best_score = (
        float(second_best_payload["score"])
        if isinstance(second_best_payload, dict)
        else 0.0
    )
    return {
        "best_name": best_payload["template_name"],
        "best_rank": best_payload["template_rank"],
        "best_rank_group": best_payload["template_rank_group"],
        "best_score": float(best_payload["score"]),
        "second_best_score": second_best_score,
        "best_orientation_deg": int(best_payload["orientation_deg"]),
        "full_ink_score": float(best_payload["full_ink_score"]),
        "full_edge_score": float(best_payload["full_edge_score"]),
        "center_score": float(best_payload["center_score"]),
        "corner_score": float(best_payload["corner_score"]),
        "portrait_ink_score": float(best_payload["portrait_ink_score"]),
        "portrait_edge_score": float(best_payload["portrait_edge_score"]),
        "portrait_block_ink_score": float(best_payload["portrait_block_ink_score"]),
        "portrait_block_edge_score": float(best_payload["portrait_block_edge_score"]),
        "top_band_ink_score": float(best_payload["top_band_ink_score"]),
        "bottom_band_ink_score": float(best_payload["bottom_band_ink_score"]),
        "border_white_score": float(best_payload["border_white_score"]),
        "rank_red_score": float(best_payload["rank_red_score"]),
        "rank_black_score": float(best_payload["rank_black_score"]),
        "rank_edge_score": float(best_payload["rank_edge_score"]),
        "letter_ink_score": float(best_payload["letter_ink_score"]),
        "letter_edge_score": float(best_payload["letter_edge_score"]),
        "suit_ink_score": float(best_payload["suit_ink_score"]),
        "suit_edge_score": float(best_payload["suit_edge_score"]),
        "red_score": float(best_payload["red_score"]),
        "black_score": float(best_payload["black_score"]),
        "loaded_template_count": len(templates),
        "compared_template_count": len(filtered_templates),
        "card_color_hint": color_hint,
        "card_rank_group_hint": rank_group_hint,
        "coarse_face_score": float(rank_group_info["face_score"]),
        "coarse_number_score": float(rank_group_info["number_score"]),
        "coarse_rank_group_confidence": float(rank_group_info["confidence"]),
        "coarse_rank_group_margin": float(rank_group_info["margin"]),
        "coarse_face_template": str(rank_group_info["face_template"]),
        "coarse_number_template": str(rank_group_info["number_template"]),
        "missing_templates": template_bank["missing_templates"],
        "card_preview": best_oriented_card.image,
        "template_preview": best_template.image,
        "mask_comparison": create_mask_comparison_image(best_oriented_card, best_template),
    }
