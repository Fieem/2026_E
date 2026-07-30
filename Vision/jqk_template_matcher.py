"""J/Q/K 扑克牌模板匹配工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


EXPECTED_TEMPLATE_NAMES: Tuple[str, ...] = tuple(
    f"{suit}_{rank}"
    for suit in ("spade", "heart", "club", "diamond")
    for rank in ("J", "Q", "K")
)


@dataclass(frozen=True)
class TemplateFeature:
    name: str
    image: np.ndarray
    red_mask: np.ndarray
    black_mask: np.ndarray
    ink_mask: np.ndarray
    edge_mask: np.ndarray


@dataclass(frozen=True)
class CardFeature:
    image: np.ndarray
    red_mask: np.ndarray
    black_mask: np.ndarray
    ink_mask: np.ndarray
    edge_mask: np.ndarray


def expected_template_files() -> List[str]:
    return [f"{name}.png" for name in EXPECTED_TEMPLATE_NAMES]


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


def _extract_card_feature(image: np.ndarray) -> CardFeature:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.uint8)
    value = hsv[:, :, 2].astype(np.uint8)
    max_delta_from_white = np.max(255 - image.astype(np.int16), axis=2).astype(np.uint8)

    red_mask = (
        (((hue <= 18) | (hue >= 160)) & (saturation >= 35) & (value >= 20))
        .astype(np.uint8)
        * 255
    )
    black_mask = (
        ((gray <= 138) & (value <= 168) & (~(red_mask > 0)))
        .astype(np.uint8)
        * 255
    )
    ink_mask = (
        (
            (max_delta_from_white >= 26)
            | (saturation >= 42)
            | (gray <= 205)
        ).astype(np.uint8)
        * 255
    )

    red_mask = _clean_mask(red_mask, 3, 3)
    black_mask = _clean_mask(black_mask, 3, 1)
    ink_mask = _clean_mask(ink_mask, 3, 3)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    edges = cv2.bitwise_and(edges, ink_mask)

    return CardFeature(
        image=image,
        red_mask=red_mask,
        black_mask=black_mask,
        ink_mask=ink_mask,
        edge_mask=edges,
    )


@lru_cache(maxsize=8)
def _load_template_bank_cached(
    template_dir_text: str,
    canvas_width_px: int,
    canvas_height_px: int,
) -> Tuple[TemplateFeature, ...]:
    template_dir = Path(template_dir_text)
    features: List[TemplateFeature] = []
    for template_name in EXPECTED_TEMPLATE_NAMES:
        path = template_dir / f"{template_name}.png"
        image = _load_image_with_white_background(path)
        if image is None:
            continue
        card = _extract_card_feature(
            _resize_to_canvas(image, canvas_width_px, canvas_height_px)
        )
        features.append(
            TemplateFeature(
                name=template_name,
                image=card.image,
                red_mask=card.red_mask,
                black_mask=card.black_mask,
                ink_mask=card.ink_mask,
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
    available_names = {template.name for template in templates}
    missing_names = [
        f"{template_name}.png"
        for template_name in EXPECTED_TEMPLATE_NAMES
        if template_name not in available_names
    ]
    return {
        "templates": templates,
        "missing_templates": missing_names,
        "loaded_template_count": len(templates),
    }


@lru_cache(maxsize=16)
def _roi_masks(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    center_mask = np.zeros((height, width), dtype=np.uint8)
    corner_mask = np.zeros((height, width), dtype=np.uint8)

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
    return center_mask, corner_mask


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
            "candidate_scores": [],
        }

    card = _extract_card_feature(
        _resize_to_canvas(card_image, canvas_width_px, canvas_height_px)
    )
    center_roi, corner_roi = _roi_masks(canvas_width_px, canvas_height_px)

    all_scores: List[Dict] = []
    best_payload: Dict | None = None
    best_oriented_card: CardFeature | None = None
    best_template: TemplateFeature | None = None

    for orientation_deg in match_orientations_deg:
        oriented_card = _rotate_card_feature(card, int(orientation_deg))
        for template in templates:
            center_ink = _mask_f1_score(oriented_card.ink_mask, template.ink_mask, center_roi)
            center_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                center_roi,
            )
            center_score = 0.55 * center_ink + 0.45 * center_edge

            corner_ink = _mask_f1_score(oriented_card.ink_mask, template.ink_mask, corner_roi)
            corner_edge = _edge_distance_score(
                oriented_card.edge_mask,
                template.edge_mask,
                corner_roi,
            )
            corner_score = 0.60 * corner_ink + 0.40 * corner_edge

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
                "orientation_deg": int(orientation_deg),
                "score": float(total_score),
                "center_score": float(center_score),
                "corner_score": float(corner_score),
                "red_score": float(red_score),
                "black_score": float(black_score),
            }
            all_scores.append(payload)
            if best_payload is None or payload["score"] > best_payload["score"]:
                best_payload = payload
                best_oriented_card = oriented_card
                best_template = template

    all_scores.sort(key=lambda item: item["score"], reverse=True)
    if best_payload is None or best_oriented_card is None or best_template is None:
        return {
            "error": "template_match_failed",
            "loaded_template_count": len(templates),
            "missing_templates": template_bank["missing_templates"],
            "candidate_scores": [],
        }

    top_scores = all_scores[: min(6, len(all_scores))]
    return {
        "best_name": best_payload["template_name"],
        "best_score": float(best_payload["score"]),
        "best_orientation_deg": int(best_payload["orientation_deg"]),
        "center_score": float(best_payload["center_score"]),
        "corner_score": float(best_payload["corner_score"]),
        "red_score": float(best_payload["red_score"]),
        "black_score": float(best_payload["black_score"]),
        "candidate_scores": top_scores,
        "loaded_template_count": len(templates),
        "missing_templates": template_bank["missing_templates"],
        "card_preview": best_oriented_card.image,
        "template_preview": best_template.image,
        "mask_comparison": create_mask_comparison_image(best_oriented_card, best_template),
    }
