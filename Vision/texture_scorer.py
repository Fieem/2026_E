"""
texture_scorer.py - 扑克牌碎片纹理评分与联合求解

用法：
    from texture_scorer import TextureScorer, solve_with_texture
    result = solve_with_texture(pieces_geometry, pieces_texture, target_center)
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np



# ===== GEOMETRY SOLVER (merged from algorithm.py) =====
EPS = 1e-8


class Pt:
    """二维点或向量。"""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Pt({self.x:.3f}, {self.y:.3f})"

    def add(self, p: "Pt") -> "Pt":
        return Pt(self.x + p.x, self.y + p.y)

    def sub(self, p: "Pt") -> "Pt":
        return Pt(self.x - p.x, self.y - p.y)

    def scale(self, scale: float) -> "Pt":
        return Pt(self.x * scale, self.y * scale)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    # 保留该方法以兼容原始实现。
    def len(self) -> float:
        return self.length()

    def norm(self) -> "Pt":
        length = self.length()
        return self.scale(1.0 / length) if length > EPS else Pt(0.0, 0.0)

    def dot(self, p: "Pt") -> float:
        return self.x * p.x + self.y * p.y

    def cross(self, p: "Pt") -> float:
        return self.x * p.y - self.y * p.x

    def rotate(self, angle_rad: float) -> "Pt":
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        return Pt(
            self.x * cosine - self.y * sine,
            self.x * sine + self.y * cosine,
        )

    def dist(self, p: "Pt") -> float:
        return self.sub(p).length()

    def lerp(self, p: "Pt", ratio: float) -> "Pt":
        return Pt(
            self.x + (p.x - self.x) * ratio,
            self.y + (p.y - self.y) * ratio,
        )

    def eq(self, p: "Pt", tolerance: float = 0.001) -> bool:
        return self.dist(p) <= tolerance


def signed_polygon_area(points: Sequence[Pt]) -> float:
    area_twice = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        area_twice += point.cross(following)
    return area_twice * 0.5


def polygon_area(points: Sequence[Pt]) -> float:
    return abs(signed_polygon_area(points))


def polygon_centroid(points: Sequence[Pt]) -> Pt:
    """返回简单多边形的面积重心。"""

    area_twice = 0.0
    cx = 0.0
    cy = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        cross = point.cross(following)
        area_twice += cross
        cx += (point.x + following.x) * cross
        cy += (point.y + following.y) * cross

    if abs(area_twice) <= EPS:
        return Pt(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )
    return Pt(cx / (3.0 * area_twice), cy / (3.0 * area_twice))


def polygon_center(points: Sequence[Pt]) -> Pt:
    return Pt(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def clean_polygon(points: Sequence[Pt], tolerance: float = 1e-6) -> List[Pt]:
    """保持轮廓顺序不变，并删除相邻的重复顶点。"""

    if len(points) < 3:
        raise ValueError("a puzzle piece needs at least three vertices")

    cleaned: List[Pt] = []
    for point in points:
        copied = Pt(point.x, point.y)
        if not cleaned or not copied.eq(cleaned[-1], tolerance):
            cleaned.append(copied)
    if len(cleaned) > 2 and cleaned[0].eq(cleaned[-1], tolerance):
        cleaned.pop()
    if len(cleaned) < 3 or polygon_area(cleaned) <= EPS:
        raise ValueError("invalid or zero-area polygon")
    return cleaned


def ensure_ccw(points: Sequence[Pt]) -> List[Pt]:
    cleaned = clean_polygon(points)
    return cleaned if signed_polygon_area(cleaned) > 0.0 else list(reversed(cleaned))


def order_cw(points: Sequence[Pt]) -> List[Pt]:
    """兼容性接口：将轮廓统一为逆时针方向。"""

    return ensure_ccw(points)


def point_on_segment(
    point: Pt, start: Pt, end: Pt, tolerance: float = 1e-6
) -> bool:
    segment = end.sub(start)
    length_squared = segment.dot(segment)
    if length_squared <= EPS:
        return point.dist(start) <= tolerance
    ratio = point.sub(start).dot(segment) / length_squared
    if ratio < -tolerance or ratio > 1.0 + tolerance:
        return False
    projection = start.add(segment.scale(max(0.0, min(1.0, ratio))))
    return point.dist(projection) <= tolerance


def point_in_polygon(point: Pt, points: Sequence[Pt]) -> bool:
    """用射线法判断点是否位于多边形中，边界点视为内部点。"""

    for index, start in enumerate(points):
        if point_on_segment(point, start, points[(index + 1) % len(points)]):
            return True

    inside = False
    previous = points[-1]
    for current in points:
        intersects = (current.y > point.y) != (previous.y > point.y)
        if intersects:
            crossing_x = (
                (previous.x - current.x)
                * (point.y - current.y)
                / (previous.y - current.y)
                + current.x
            )
            if point.x < crossing_x:
                inside = not inside
        previous = current
    return inside


def point_strictly_inside(point: Pt, points: Sequence[Pt], tolerance: float) -> bool:
    for index, start in enumerate(points):
        if point_on_segment(
            point, start, points[(index + 1) % len(points)], tolerance
        ):
            return False
    return point_in_polygon(point, points)


def _point_segment_distance(point: Pt, start: Pt, end: Pt) -> float:
    segment = end.sub(start)
    length_squared = segment.dot(segment)
    if length_squared <= EPS:
        return point.dist(start)
    ratio = point.sub(start).dot(segment) / length_squared
    projection = start.add(segment.scale(max(0.0, min(1.0, ratio))))
    return point.dist(projection)


def proper_segment_intersection(
    a: Pt, b: Pt, c: Pt, d: Pt, tolerance: float = 0.0
) -> bool:
    """仅在线段内部真正交叉时返回 True，端点接触不算交叉。

    两条观测边经过刚体对齐后会产生微小浮点误差。如果不检查交点到端点的
    距离，原本共用的角点可能被误判为具有正面积的重叠。
    """

    ab = b.sub(a)
    cd = d.sub(c)
    side_c = ab.cross(c.sub(a))
    side_d = ab.cross(d.sub(a))
    side_a = cd.cross(a.sub(c))
    side_b = cd.cross(b.sub(c))
    crosses = side_c * side_d < -EPS and side_a * side_b < -EPS
    if not crosses or tolerance <= 0.0:
        return crosses
    return not (
        _point_segment_distance(a, c, d) <= tolerance
        or _point_segment_distance(b, c, d) <= tolerance
        or _point_segment_distance(c, a, b) <= tolerance
        or _point_segment_distance(d, a, b) <= tolerance
    )


def polygons_overlap(
    first: Sequence[Pt], second: Sequence[Pt], tolerance: float
) -> bool:
    """检测具有正面积的重叠，同时允许碎片共边或共用顶点。"""

    for i, a in enumerate(first):
        b = first[(i + 1) % len(first)]
        for j, c in enumerate(second):
            d = second[(j + 1) % len(second)]
            if proper_segment_intersection(a, b, c, d, tolerance):
                return True

    if any(point_strictly_inside(point, second, tolerance) for point in first):
        return True
    if any(point_strictly_inside(point, first, tolerance) for point in second):
        return True

    # 处理所有顶点都落在另一多边形边界上的完全重合情况。
    first_centroid = polygon_centroid(first)
    second_centroid = polygon_centroid(second)
    return point_strictly_inside(first_centroid, second, tolerance) or point_strictly_inside(
        second_centroid, first, tolerance
    )


def transform_polygon(points: Sequence[Pt], angle: float, center: Pt) -> List[Pt]:
    return [point.rotate(angle).add(center) for point in points]


def convex_hull(points: Iterable[Pt]) -> List[Pt]:
    unique = sorted({(round(point.x, 9), round(point.y, 9)) for point in points})
    if len(unique) <= 1:
        return [Pt(*point) for point in unique]

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= EPS:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= EPS:
            upper.pop()
        upper.append(point)

    return [Pt(*point) for point in lower[:-1] + upper[:-1]]


def minimum_area_rectangle(points: Sequence[Pt]) -> Optional[Dict]:
    hull = convex_hull(points)
    if len(hull) < 3:
        return None

    best: Optional[Dict] = None
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        angle = math.atan2(end.y - start.y, end.x - start.x)
        rotated = [point.rotate(-angle) for point in hull]
        min_x = min(point.x for point in rotated)
        max_x = max(point.x for point in rotated)
        min_y = min(point.y for point in rotated)
        max_y = max(point.y for point in rotated)
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        if best is not None and area >= best["area"]:
            continue

        center_rotated = Pt((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
        corners_rotated = [
            Pt(min_x, min_y),
            Pt(max_x, min_y),
            Pt(max_x, max_y),
            Pt(min_x, max_y),
        ]
        best = {
            "center": center_rotated.rotate(angle),
            "width": width,
            "height": height,
            "angle": angle,
            "area": area,
            "corners": [corner.rotate(angle) for corner in corners_rotated],
        }
    return best


def _edge(points: Sequence[Pt], index: int) -> Tuple[Pt, Pt]:
    return points[index], points[(index + 1) % len(points)]


def _angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _solution_angle_signature(solution: Dict) -> Tuple[Tuple[int, float], ...]:
    placements = solution.get("placements", [])
    signature: List[Tuple[int, float]] = []
    for placement in placements:
        if not isinstance(placement, dict):
            continue
        signature.append(
            (
                int(placement.get("piece_index", 0)),
                _angle_wrap(float(placement.get("angle", 0.0))),
            )
        )
    signature.sort(key=lambda item: item[0])
    return tuple(signature)


def _solution_angle_distance(first: Dict, second: Dict) -> float:
    first_signature = dict(_solution_angle_signature(first))
    second_signature = dict(_solution_angle_signature(second))
    common_keys = sorted(set(first_signature.keys()) & set(second_signature.keys()))
    if not common_keys:
        return 0.0
    distances = [
        abs(_angle_wrap(first_signature[key] - second_signature[key]))
        for key in common_keys
    ]
    return max(distances) if distances else 0.0


def _select_diverse_geometry_candidates(
    candidates: Sequence[Dict],
    limit: int,
    min_angle_diff_rad: float,
) -> List[Dict]:
    if limit <= 0:
        return []
    ordered = sorted(candidates, key=lambda item: float(item.get("score", 0.0)))
    if not ordered:
        return []
    if min_angle_diff_rad <= 1e-6:
        return ordered[:limit]

    selected: List[Dict] = [ordered[0]]
    remaining = ordered[1:]
    for candidate in remaining:
        if len(selected) >= limit:
            break
        if all(
            _solution_angle_distance(candidate, existing) >= min_angle_diff_rad
            for existing in selected
        ):
            selected.append(candidate)

    if len(selected) < limit:
        for candidate in remaining:
            if len(selected) >= limit:
                break
            if candidate in selected:
                continue
            selected.append(candidate)
    return selected


def _edge_lengths_from_points(points: Sequence[Pt]) -> List[float]:
    return [
        points[index].dist(points[(index + 1) % len(points)])
        for index in range(len(points))
    ]


def _is_near_equilateral_triangle(
    points: Sequence[Pt],
    tolerance_mm: float,
    relative_tolerance: float,
) -> bool:
    if len(points) != 3:
        return False
    edge_lengths = _edge_lengths_from_points(points)
    if not edge_lengths:
        return False
    min_length = min(edge_lengths)
    max_length = max(edge_lengths)
    if min_length <= EPS:
        return False
    allowed_error = max(float(tolerance_mm), float(relative_tolerance) * max_length)
    return (max_length - min_length) <= allowed_error


def _line_distance(point: Pt, start: Pt, end: Pt) -> float:
    direction = end.sub(start)
    length = direction.length()
    if length <= EPS:
        return point.dist(start)
    return abs(direction.cross(point.sub(start))) / length


def _edge_contact_metrics(
    first_start: Pt,
    first_end: Pt,
    second_start: Pt,
    second_end: Pt,
    *,
    angle_tolerance_rad: float,
    distance_tolerance_mm: float,
) -> Optional[Tuple[float, float]]:
    first_direction = first_end.sub(first_start)
    second_direction = second_end.sub(second_start)
    first_length = first_direction.length()
    second_length = second_direction.length()
    if first_length <= EPS or second_length <= EPS:
        return None

    first_angle = math.atan2(first_direction.y, first_direction.x)
    second_angle = math.atan2(second_direction.y, second_direction.x)
    angle_diff = abs(_angle_wrap(first_angle - second_angle))
    angle_diff = min(angle_diff, abs(math.pi - angle_diff))
    if angle_diff > angle_tolerance_rad:
        return None

    gap = min(
        0.5
        * (
            _line_distance(second_start, first_start, first_end)
            + _line_distance(second_end, first_start, first_end)
        ),
        0.5
        * (
            _line_distance(first_start, second_start, second_end)
            + _line_distance(first_end, second_start, second_end)
        ),
    )
    if gap > distance_tolerance_mm:
        return None

    axis = first_direction.scale(1.0 / first_length)
    second_interval = sorted(
        (
            second_start.sub(first_start).dot(axis),
            second_end.sub(first_start).dot(axis),
        )
    )
    overlap = max(
        0.0,
        min(first_length, second_interval[1]) - max(0.0, second_interval[0]),
    )
    if overlap <= EPS:
        return None
    return overlap, gap


def _find_piece_contact_pairs(
    world_polygons: Dict[int, List[Pt]],
    moving_index: int,
    *,
    angle_tolerance_rad: float = 0.35,
    distance_tolerance_mm: float = 2.5,
    minimum_overlap_mm: float = 5.0,
    max_pairs: int = 4,
) -> List[Dict]:
    moving_polygon = world_polygons.get(moving_index)
    if not moving_polygon:
        return []

    pairs: List[Dict] = []
    for moving_edge_index in range(len(moving_polygon)):
        moving_start, moving_end = _edge(moving_polygon, moving_edge_index)
        for fixed_index, fixed_polygon in world_polygons.items():
            if fixed_index == moving_index:
                continue
            for fixed_edge_index in range(len(fixed_polygon)):
                fixed_start, fixed_end = _edge(fixed_polygon, fixed_edge_index)
                metrics = _edge_contact_metrics(
                    fixed_start,
                    fixed_end,
                    moving_start,
                    moving_end,
                    angle_tolerance_rad=angle_tolerance_rad,
                    distance_tolerance_mm=distance_tolerance_mm,
                )
                if metrics is None:
                    continue
                overlap, gap = metrics
                if overlap + EPS < minimum_overlap_mm:
                    continue
                pairs.append(
                    {
                        "moving_edge_index": moving_edge_index,
                        "fixed_index": fixed_index,
                        "fixed_edge_index": fixed_edge_index,
                        "overlap": overlap,
                        "gap": gap,
                    }
                )
    pairs.sort(key=lambda item: (-float(item["overlap"]), float(item["gap"])))
    unique_pairs: List[Dict] = []
    seen = set()
    for pair in pairs:
        key = (
            int(pair["moving_edge_index"]),
            int(pair["fixed_index"]),
            int(pair["fixed_edge_index"]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(pair)
        if len(unique_pairs) >= max_pairs:
            break
    return unique_pairs


def _edge_alignment_translations_for_fixed_angle(
    fixed_start: Pt,
    fixed_end: Pt,
    moving_start: Pt,
    moving_end: Pt,
    minimum_contact: float,
    reverse_direction: bool = True,
) -> List[Tuple[Pt, float, float]]:
    fixed_direction = fixed_end.sub(fixed_start)
    moving_direction = moving_end.sub(moving_start)
    fixed_length = fixed_direction.length()
    moving_length = moving_direction.length()
    if fixed_length <= EPS or moving_length <= EPS:
        return []

    fixed_midpoint = fixed_start.lerp(fixed_end, 0.5)
    moving_midpoint = moving_start.lerp(moving_end, 0.5)
    if reverse_direction:
        centers = [
            fixed_end.sub(moving_start),
            fixed_start.sub(moving_end),
            fixed_start.sub(moving_start),
            fixed_end.sub(moving_end),
            fixed_midpoint.sub(moving_midpoint),
        ]
    else:
        centers = [
            fixed_start.sub(moving_start),
            fixed_end.sub(moving_end),
            fixed_start.sub(moving_end),
            fixed_end.sub(moving_start),
            fixed_midpoint.sub(moving_midpoint),
        ]

    axis = fixed_direction.scale(1.0 / fixed_length)
    results: List[Tuple[Pt, float, float]] = []
    seen = set()
    for center in centers:
        key = (round(center.x, 6), round(center.y, 6))
        if key in seen:
            continue
        seen.add(key)

        aligned_start = moving_start.add(center)
        aligned_end = moving_end.add(center)
        moving_interval = sorted(
            (
                aligned_start.sub(fixed_start).dot(axis),
                aligned_end.sub(fixed_start).dot(axis),
            )
        )
        overlap = max(
            0.0,
            min(fixed_length, moving_interval[1]) - max(0.0, moving_interval[0]),
        )
        if overlap + EPS < minimum_contact:
            continue
        contact_ratio = overlap / min(fixed_length, moving_length)
        results.append((center, contact_ratio, overlap))
    return results


def polygon_orientation(points: Sequence[Pt]) -> float:
    """用最小面积外接矩形的主边估计碎片方向，结果单位为弧度。"""

    rectangle = minimum_area_rectangle(points)
    return 0.0 if rectangle is None else _angle_wrap(rectangle["angle"])


def _rotate_about(point: Pt, center: Pt, angle: float) -> Pt:
    return point.sub(center).rotate(angle).add(center)


def _normalize_rectangle_pose(
    rectangle: Dict,
    poses: Dict[int, Tuple[float, Pt]],
    world_polygons: Dict[int, List[Pt]],
) -> Tuple[Dict, Dict[int, Tuple[float, Pt]], Dict[int, List[Pt]]]:
    """把最终矩形摆正到：长边平行 X，短边平行 Y。"""

    width = float(rectangle["width"])
    height = float(rectangle["height"])
    center = rectangle["center"]

    long_side_angle = rectangle["angle"] if width >= height else rectangle["angle"] + math.pi * 0.5
    normalize_angle = -_angle_wrap(long_side_angle)

    normalized_rectangle = dict(rectangle)
    long_side = max(width, height)
    short_side = min(width, height)
    normalized_rectangle["width"] = long_side
    normalized_rectangle["height"] = short_side
    normalized_rectangle["angle"] = 0.0
    half_long = long_side * 0.5
    half_short = short_side * 0.5
    normalized_rectangle["corners"] = [
        center.add(Pt(-half_long, -half_short)),
        center.add(Pt(half_long, -half_short)),
        center.add(Pt(half_long, half_short)),
        center.add(Pt(-half_long, half_short)),
    ]

    normalized_poses: Dict[int, Tuple[float, Pt]] = {}
    normalized_polygons: Dict[int, List[Pt]] = {}
    for piece_index, (angle, piece_center) in poses.items():
        normalized_poses[piece_index] = (
            _angle_wrap(angle + normalize_angle),
            _rotate_about(piece_center, center, normalize_angle),
        )
        normalized_polygons[piece_index] = [
            _rotate_about(point, center, normalize_angle)
            for point in world_polygons[piece_index]
        ]

    return normalized_rectangle, normalized_poses, normalized_polygons


def _align_opposite_edges(
    fixed_start: Pt,
    fixed_end: Pt,
    moving_start: Pt,
    moving_end: Pt,
) -> Tuple[float, Pt]:
    """通过刚体变换，将移动边的起点→终点映射到固定边的终点→起点。"""

    source_direction = moving_end.sub(moving_start)
    target_direction = fixed_start.sub(fixed_end)
    angle = math.atan2(target_direction.y, target_direction.x) - math.atan2(
        source_direction.y, source_direction.x
    )
    source_midpoint = moving_start.lerp(moving_end, 0.5).rotate(angle)
    target_midpoint = fixed_start.lerp(fixed_end, 0.5)
    return _angle_wrap(angle), target_midpoint.sub(source_midpoint)


def _edge_alignment_candidates(
    fixed_start: Pt,
    fixed_end: Pt,
    moving_start: Pt,
    moving_end: Pt,
    minimum_contact: float,
    reverse_direction: bool = True,
) -> List[Tuple[float, Pt, float, float]]:
    """生成整边匹配以及兼容 T 形接缝的候选位姿。

    后续切割可能只在接缝的一侧增加分点，因此一条长边可能对应两条短边。
    通过端点对齐可以保留这种局部接触，再由重叠检测和最终矩形检测排除
    只有单点接触的错误候选。
    """

    source_direction = moving_end.sub(moving_start)
    fixed_direction = fixed_end.sub(fixed_start)
    fixed_length = fixed_direction.length()
    moving_length = source_direction.length()
    if fixed_length <= EPS or moving_length <= EPS:
        return []

    target_direction = (
        fixed_start.sub(fixed_end)
        if reverse_direction
        else fixed_end.sub(fixed_start)
    )
    angle = _angle_wrap(
        math.atan2(target_direction.y, target_direction.x)
        - math.atan2(source_direction.y, source_direction.x)
    )
    rotated_start = moving_start.rotate(angle)
    rotated_end = moving_end.rotate(angle)
    fixed_midpoint = fixed_start.lerp(fixed_end, 0.5)
    moving_midpoint = rotated_start.lerp(rotated_end, 0.5)

    if reverse_direction:
        centers = [
            fixed_end.sub(rotated_start),
            fixed_start.sub(rotated_end),
            fixed_start.sub(rotated_start),
            fixed_end.sub(rotated_end),
            fixed_midpoint.sub(moving_midpoint),
        ]
    else:
        centers = [
            fixed_start.sub(rotated_start),
            fixed_end.sub(rotated_end),
            fixed_start.sub(rotated_end),
            fixed_end.sub(rotated_start),
            fixed_midpoint.sub(moving_midpoint),
        ]
    axis = fixed_direction.scale(1.0 / fixed_length)
    results: List[Tuple[float, Pt, float, float]] = []
    seen = set()
    for center in centers:
        key = (round(center.x, 6), round(center.y, 6))
        if key in seen:
            continue
        seen.add(key)

        aligned_start = rotated_start.add(center)
        aligned_end = rotated_end.add(center)
        moving_interval = sorted(
            (
                aligned_start.sub(fixed_start).dot(axis),
                aligned_end.sub(fixed_start).dot(axis),
            )
        )
        overlap = max(
            0.0,
            min(fixed_length, moving_interval[1])
            - max(0.0, moving_interval[0]),
        )
        if overlap + EPS < minimum_contact:
            continue

        contact_ratio = overlap / min(fixed_length, moving_length)
        results.append((angle, center, contact_ratio, overlap))
    return results


def _prepare_pieces(pieces: Sequence[Dict]) -> List[Dict]:
    if not 2 <= len(pieces) <= 4:
        raise ValueError("the solver supports 2 to 4 pieces")

    prepared = []
    for index, piece in enumerate(pieces):
        if "pts" not in piece:
            raise ValueError(f"piece {index} has no 'pts' contour")
        observed = ensure_ccw(piece["pts"])
        if len(observed) > 8:
            raise ValueError(f"piece {index} has more than eight edges")
        source_center = polygon_centroid(observed)
        local_points = [point.sub(source_center) for point in observed]
        edge_lengths = [
            local_points[edge_index].dist(
                local_points[(edge_index + 1) % len(local_points)]
            )
            for edge_index in range(len(local_points))
        ]
        prepared.append(
            {
                "index": index,
                "source_center": source_center,
                "source_orientation": polygon_orientation(observed),
                "local_points": local_points,
                "edge_count": len(local_points),
                "edge_lengths": edge_lengths,
                "area": polygon_area(local_points),
            }
        )
    return prepared


def _triangle_similar_edge_indices(
    piece: Dict,
    edge_index: int,
    tolerance_mm: float,
    relative_tolerance: float,
) -> List[int]:
    if int(piece.get("edge_count", 0)) != 3:
        return []
    edge_lengths = piece.get("edge_lengths", [])
    if not isinstance(edge_lengths, list) or edge_index >= len(edge_lengths):
        return []
    base_length = float(edge_lengths[edge_index])
    similar: List[int] = []
    for other_index, other_length_raw in enumerate(edge_lengths):
        if other_index == edge_index:
            continue
        other_length = float(other_length_raw)
        allowed = max(float(tolerance_mm), float(relative_tolerance) * max(base_length, other_length))
        if abs(base_length - other_length) <= allowed:
            similar.append(other_index)
    return similar


def _dimensions_valid(
    rectangle: Dict,
    size_range_mm: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    tolerance_mm: float,
) -> bool:
    if size_range_mm is None:
        return True
    short_side, long_side = sorted((rectangle["width"], rectangle["height"]))
    short_range, long_range = size_range_mm
    return (
        short_range[0] - tolerance_mm
        <= short_side
        <= short_range[1] + tolerance_mm
        and long_range[0] - tolerance_mm
        <= long_side
        <= long_range[1] + tolerance_mm
    )


def find_rectangle_solution(
    pieces: Sequence[Dict],
    target_center: Optional[Pt] = None,
    *,
    edge_tolerance_mm: float = 3.0,
    edge_relative_tolerance: float = 0.05,
    minimum_edge_contact_mm: float = 5.0,
    minimum_edge_contact_ratio: float = 0.6,
    overlap_tolerance_mm: float = 4.0,
    rectangle_area_tolerance: float = 0.06,
    size_range_mm: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
        (40.0, 100.0),
        (80.0, 130.0),
    ),
    dimension_tolerance_mm: float = 3.0,
    max_search_nodes: int = 20_000,
    max_candidates_per_node: int = 128,
    max_partial_candidates_per_node: int = 32,
    max_complete_solutions: int = 12,
    anchor_count: int = 1,
    partial_match_penalty: float = 0.18,
    triangle_keep_per_piece: int = 0,
    triangle_min_angle_diff_rad: float = 0.0,
    triangle_symmetry_edge_tolerance_mm: float = 0.0,
    triangle_symmetry_edge_relative_tolerance: float = 0.0,
    triangle_symmetry_length_bonus_mm: float = 0.0,
    triangle_flip_side_enabled: bool = False,
    triangle_flip_solution_quota: int = 0,
    triangle_flip_requires_similar_edge: bool = True,
    triangle_flip_max_alignments_per_edge_pair: int = 2,
    triangle_flip_candidate_quota: int = 0,
    triangle_flip_overlap_tolerance_extra_mm: float = 0.0,
    triangle_flip_partial_diagonal_extra_mm: float = 0.0,
    triangle_flip_partial_area_scale: float = 1.0,
    triangle_flip_priority_bonus: float = 0.0,
    diagnostics: Optional[Dict] = None,
) -> Optional[Dict]:
    """寻找一个互不重叠且组合后能够填满矩形的布局。

    每个输入项必须包含 ``{"pts": [Pt, ...]}``。坐标应当已经完成透视矫正，
    单位为毫米。返回的放置角表示从观测轮廓方向旋转到目标位姿的角度。
    """

    prepared = _prepare_pieces(pieces)
    total_piece_area = sum(piece["area"] for piece in prepared)

    # 如果最大碎片的接缝都被后续切割分段，从它开始搜索会产生大量局部匹配。
    # 因此优先选择与其他碎片具有最多等长边联系的碎片作为锚点；完整接缝对
    # 位姿的约束远强于 T 形接缝处的一小段局部接触。
    def anchor_score(piece: Dict) -> Tuple[int, float, float]:
        match_count = 0
        match_quality = 0.0
        for edge_index in range(len(piece["local_points"])):
            start, end = _edge(piece["local_points"], edge_index)
            edge_length = start.dist(end)
            best_quality = 0.0
            for other in prepared:
                if other["index"] == piece["index"]:
                    continue
                for other_edge_index in range(len(other["local_points"])):
                    other_start, other_end = _edge(
                        other["local_points"], other_edge_index
                    )
                    other_length = other_start.dist(other_end)
                    allowed_error = max(
                        edge_tolerance_mm,
                        edge_relative_tolerance * max(edge_length, other_length),
                    )
                    length_error = abs(edge_length - other_length)
                    if length_error <= allowed_error:
                        best_quality = max(
                            best_quality,
                            1.0 - length_error / max(allowed_error, EPS),
                        )
            if best_quality > 0.0:
                match_count += 1
                match_quality += best_quality
        return match_count, match_quality, piece["area"]

    anchor_order = [
        piece["index"]
        for piece in sorted(prepared, key=anchor_score, reverse=True)
    ]
    anchor_limit = len(anchor_order) if int(anchor_count) <= 0 else min(len(anchor_order), int(anchor_count))
    poses: Dict[int, Tuple[float, Pt]] = {}
    world_polygons: Dict[int, List[Pt]] = {}
    best_solution: Optional[Dict] = None
    complete_solutions: List[Dict] = []
    complete_solution_keys: Set[Tuple] = set()
    search_nodes = 0
    stats = {
        "search_nodes": 0,
        "max_search_nodes": max_search_nodes,
        "search_limit_reached": False,
        "root_candidate_count": 0,
        "candidate_states": 0,
        "full_edge_candidates": 0,
        "partial_edge_candidates": 0,
        "overlap_rejections": 0,
        "geometry_rejections": 0,
        "complete_layouts": 0,
        "invalid_rectangle_rejections": 0,
        "dimension_rejections": 0,
        "area_rejections": 0,
        "no_candidate_nodes": 0,
        "candidate_pruned_by_limit": 0,
        "triangle_diverse_kept": 0,
        "triangle_flip_alignment_candidates": 0,
        "triangle_flip_solution_kept": 0,
        "triangle_flip_candidates_kept": 0,
        "triangle_flip_overlap_rejections": 0,
        "triangle_flip_geometry_rejections": 0,
        "candidate_counts_by_depth": [],
        "anchor_index": anchor_order[0] if anchor_order else -1,
        "anchor_indices_tried": [],
        "piece_count": len(prepared),
        "total_piece_area_mm2": total_piece_area,
    }

    def partial_geometry_rectangle(
        candidate_polygons: Iterable[Sequence[Pt]],
        *,
        diagonal_extra_mm: float = 0.0,
        area_scale: float = 1.0,
    ) -> Optional[Dict]:
        safe_area_scale = max(1.0, float(area_scale))
        maximum_final_area = float("inf")
        if size_range_mm is not None:
            maximum_final_area = safe_area_scale * total_piece_area / max(
                1.0 - rectangle_area_tolerance, EPS
            )
        all_points = [point for polygon in candidate_polygons for point in polygon]
        rectangle = minimum_area_rectangle(all_points)
        if rectangle is None:
            return None
        if rectangle["area"] > maximum_final_area + EPS:
            return None
        if size_range_mm is None:
            return rectangle
        short_range, long_range = size_range_mm
        maximum_area = safe_area_scale * (
            (short_range[1] + dimension_tolerance_mm)
            * (long_range[1] + dimension_tolerance_mm)
        )
        if rectangle["area"] > maximum_area + EPS:
            return None

        maximum_diagonal = math.hypot(
            short_range[1] + dimension_tolerance_mm,
            long_range[1] + dimension_tolerance_mm,
        ) + max(0.0, float(diagonal_extra_mm))
        for first_index, first in enumerate(all_points):
            for second in all_points[first_index + 1 :]:
                if first.dist(second) > maximum_diagonal:
                    return None
        return rectangle

    def evaluate(edge_error: float, uses_triangle_flip: bool) -> bool:
        nonlocal best_solution, complete_solutions

        all_points = [
            point for polygon in world_polygons.values() for point in polygon
        ]
        rectangle = minimum_area_rectangle(all_points)
        if rectangle is None or rectangle["area"] <= EPS:
            stats["invalid_rectangle_rejections"] += 1
            return False
        stats["complete_layouts"] += 1
        if not _dimensions_valid(rectangle, size_range_mm, dimension_tolerance_mm):
            stats["dimension_rejections"] += 1
            return False

        area_error = abs(rectangle["area"] - total_piece_area) / rectangle["area"]
        if area_error > rectangle_area_tolerance:
            stats["area_rejections"] += 1
            return False

        score = area_error * 1.0 + edge_error * 0.05 / max(1, len(prepared) - 1)
        solution_key = tuple(
            sorted(
                (
                    int(index),
                    round(angle, 6),
                    round(center.x, 3),
                    round(center.y, 3),
                )
                for index, (angle, center) in poses.items()
            )
        )
        if solution_key in complete_solution_keys:
            return False
        complete_solution_keys.add(solution_key)

        solution_record = {
            "score": score,
            "area_error": area_error,
            "rectangle": rectangle,
            "poses": dict(poses),
            "uses_triangle_flip": bool(uses_triangle_flip),
            "world_polygons": {
                index: list(points) for index, points in world_polygons.items()
            },
        }
        complete_solutions.append(solution_record)
        complete_solutions.sort(key=lambda solution: solution["score"])
        if len(complete_solutions) > max_complete_solutions:
            flip_quota = max(
                0,
                min(int(triangle_flip_solution_quota), int(max_complete_solutions)),
            )
            kept_solutions: List[Dict] = []
            kept_ids: Set[int] = set()
            if flip_quota > 0:
                for solution in complete_solutions:
                    if len(kept_solutions) >= flip_quota:
                        break
                    if not bool(solution.get("uses_triangle_flip", False)):
                        continue
                    kept_solutions.append(solution)
                    kept_ids.add(id(solution))
            for solution in complete_solutions:
                if len(kept_solutions) >= max_complete_solutions:
                    break
                if id(solution) in kept_ids:
                    continue
                kept_solutions.append(solution)
                kept_ids.add(id(solution))
            removed_solutions = [
                solution
                for solution in complete_solutions
                if id(solution) not in kept_ids
            ]
            complete_solutions = kept_solutions
            for removed in removed_solutions:
                removed_key = tuple(
                    sorted(
                        (
                            int(index),
                            round(angle, 6),
                            round(center.x, 3),
                            round(center.y, 3),
                        )
                        for index, (angle, center) in removed["poses"].items()
                    )
                )
                if removed_key != solution_key:
                    complete_solution_keys.discard(removed_key)
        stats["triangle_flip_solution_kept"] = sum(
            1 for solution in complete_solutions if bool(solution.get("uses_triangle_flip", False))
        )
        best_solution = complete_solutions[0]
        return True

    def recurse(edge_error: float, uses_triangle_flip: bool) -> bool:
        nonlocal search_nodes
        search_nodes += 1
        stats["search_nodes"] = search_nodes
        if search_nodes > max_search_nodes:
            stats["search_limit_reached"] = True
            return True
        if len(poses) == len(prepared):
            evaluate(edge_error, uses_triangle_flip)
            return False

        candidate_keys: Set[Tuple] = set()
        candidates = []
        placed_polygons = list(world_polygons.values())
        current_placed_area = sum(prepared[index]["area"] for index in poses)
        unplaced = [piece["index"] for piece in prepared if piece["index"] not in poses]

        for moving_index in unplaced:
            moving_points = prepared[moving_index]["local_points"]
            moving_area = prepared[moving_index]["area"]
            for fixed_index, fixed_polygon in list(world_polygons.items()):
                for fixed_edge_index in range(len(fixed_polygon)):
                    fixed_start, fixed_end = _edge(fixed_polygon, fixed_edge_index)
                    fixed_length = fixed_start.dist(fixed_end)

                    for moving_edge_index in range(len(moving_points)):
                        edge_variants = [(moving_edge_index, False, 0.0)]
                        similar_edge_indices = _triangle_similar_edge_indices(
                            prepared[moving_index],
                            moving_edge_index,
                            triangle_symmetry_edge_tolerance_mm,
                            triangle_symmetry_edge_relative_tolerance,
                        )
                        if similar_edge_indices:
                            similar_lengths = [
                                float(prepared[moving_index]["edge_lengths"][idx])
                                for idx in [moving_edge_index] + similar_edge_indices
                            ]
                            shared_length = float(sum(similar_lengths) / len(similar_lengths))
                            for alt_edge_index in similar_edge_indices:
                                edge_variants.append(
                                    (
                                        alt_edge_index,
                                        True,
                                        max(
                                            0.0,
                                            triangle_symmetry_length_bonus_mm,
                                        ),
                                    )
                                )
                        else:
                            shared_length = 0.0

                        for candidate_edge_index, symmetry_generated, length_bonus_mm in edge_variants:
                            moving_start, moving_end = _edge(
                                moving_points, candidate_edge_index
                            )
                            moving_length = float(prepared[moving_index]["edge_lengths"][candidate_edge_index])
                            effective_moving_length = (
                                shared_length if symmetry_generated and shared_length > EPS else moving_length
                            )
                            effective_moving_length = max(
                                EPS,
                                effective_moving_length + length_bonus_mm,
                            )
                            allowed_error = max(
                                edge_tolerance_mm,
                                edge_relative_tolerance * max(fixed_length, effective_moving_length),
                            )
                            length_error = abs(fixed_length - effective_moving_length)
                            minimum_contact = max(
                                minimum_edge_contact_mm,
                                minimum_edge_contact_ratio
                                * min(fixed_length, effective_moving_length),
                            )
                            alignment_variants = [
                                (
                                    angle,
                                    center,
                                    contact_ratio,
                                    contact_length,
                                    False,
                                )
                                for angle, center, contact_ratio, contact_length in _edge_alignment_candidates(
                                    fixed_start,
                                    fixed_end,
                                    moving_start,
                                    moving_end,
                                    minimum_contact,
                                    True,
                                )
                            ]
                            alignments = alignment_variants
                            flip_allowed = (
                                triangle_flip_side_enabled
                                and int(prepared[moving_index].get("edge_count", 0)) == 3
                                and (
                                    not triangle_flip_requires_similar_edge
                                    or bool(similar_edge_indices)
                                    or bool(symmetry_generated)
                                )
                            )
                            if flip_allowed:
                                flip_alignments = [
                                    (
                                        angle,
                                        center,
                                        contact_ratio,
                                        contact_length,
                                        True,
                                    )
                                    for angle, center, contact_ratio, contact_length in _edge_alignment_candidates(
                                        fixed_start,
                                        fixed_end,
                                        moving_start,
                                        moving_end,
                                        minimum_contact,
                                        False,
                                    )
                                ]
                                if triangle_flip_max_alignments_per_edge_pair > 0 and len(flip_alignments) > triangle_flip_max_alignments_per_edge_pair:
                                    flip_alignments.sort(
                                        key=lambda item: (-item[2], -item[3])
                                    )
                                    flip_alignments = flip_alignments[
                                        :triangle_flip_max_alignments_per_edge_pair
                                    ]
                                stats["triangle_flip_alignment_candidates"] += len(flip_alignments)
                                alignments.extend(flip_alignments)
                            for angle, center, contact_ratio, contact_length, flip_generated in alignments:
                                key = (
                                    moving_index,
                                    round(angle, 6),
                                    round(center.x, 4),
                                    round(center.y, 4),
                                )
                                if key in candidate_keys:
                                    continue
                                candidate_keys.add(key)

                                candidate_polygon = transform_polygon(
                                    moving_points, angle, center
                                )
                                candidate_overlap_tolerance = (
                                    overlap_tolerance_mm
                                    + max(0.0, float(triangle_flip_overlap_tolerance_extra_mm))
                                    if flip_generated
                                    else overlap_tolerance_mm
                                )
                                if any(
                                    polygons_overlap(
                                        candidate_polygon,
                                        placed_polygon,
                                        candidate_overlap_tolerance,
                                    )
                                    for placed_polygon in placed_polygons
                                ):
                                    stats["overlap_rejections"] += 1
                                    if flip_generated:
                                        stats["triangle_flip_overlap_rejections"] += 1
                                    continue
                                partial_rectangle = partial_geometry_rectangle(
                                    placed_polygons + [candidate_polygon],
                                    diagonal_extra_mm=(
                                        triangle_flip_partial_diagonal_extra_mm
                                        if flip_generated
                                        else 0.0
                                    ),
                                    area_scale=(
                                        triangle_flip_partial_area_scale
                                        if flip_generated
                                        else 1.0
                                    ),
                                )
                                if partial_rectangle is None:
                                    stats["geometry_rejections"] += 1
                                    if flip_generated:
                                        stats["triangle_flip_geometry_rejections"] += 1
                                    continue

                                placed_area = current_placed_area + moving_area
                                empty_area_ratio = 0.0
                                empty_area_ratio = max(
                                    0.0,
                                    partial_rectangle["area"] - placed_area,
                                ) / total_piece_area

                                # 对近似等边 / 等腰三角形，允许近似等长边共享一个更稳的
                                # 参考边长，避免替代接边方式因为毫米级误差被过早剪掉。
                                seam_error = (
                                    1.0
                                    - contact_ratio
                                    + length_error / max(fixed_length, effective_moving_length)
                                )
                                full_edge_match = length_error <= allowed_error
                                if full_edge_match:
                                    stats["full_edge_candidates"] += 1
                                else:
                                    stats["partial_edge_candidates"] += 1
                                priority = seam_error + 0.5 * empty_area_ratio - (
                                    triangle_flip_priority_bonus if flip_generated else 0.0
                                )
                                candidates.append(
                                    (
                                        moving_index,
                                        fixed_index,
                                        fixed_edge_index,
                                        candidate_edge_index,
                                        angle,
                                        center,
                                        candidate_polygon,
                                        seam_error,
                                        contact_length,
                                        full_edge_match,
                                        priority,
                                        flip_generated,
                                    )
                                )

        stats["candidate_states"] += len(candidates)
        if not candidates:
            stats["no_candidate_nodes"] += 1
        if len(poses) == 1:
            stats["root_candidate_count"] = len(candidates)
        stats["candidate_counts_by_depth"].append(
            {"pieces_already_placed": len(poses), "candidate_count": len(candidates)}
        )

        # 精确整边接缝的可信度远高于局部接触。该排序会优先恢复普通切割产生
        # 的碎片对，只在确有需要时才回退到 T 形接缝布局。
        candidates.sort(
            key=lambda candidate: (
                candidate[10] + (0.0 if candidate[9] else partial_match_penalty),
                -candidate[8],
            )
        )
        if len(candidates) > max_candidates_per_node:
            kept_candidates = []
            kept_keys: Set[Tuple] = set()
            kept_partial = 0
            kept_flip = 0

            if triangle_flip_candidate_quota > 0:
                for candidate in candidates:
                    if kept_flip >= triangle_flip_candidate_quota:
                        break
                    if not bool(candidate[11]):
                        continue
                    candidate_key = (
                        int(candidate[0]),
                        round(float(candidate[4]), 6),
                        round(float(candidate[5].x), 4),
                        round(float(candidate[5].y), 4),
                    )
                    if candidate_key in kept_keys:
                        continue
                    kept_candidates.append(candidate)
                    kept_keys.add(candidate_key)
                    kept_flip += 1
                stats["triangle_flip_candidates_kept"] += kept_flip

            if triangle_keep_per_piece > 0 and triangle_min_angle_diff_rad > 1e-6:
                triangle_angles: Dict[int, List[float]] = {}
                for candidate in candidates:
                    moving_index = int(candidate[0])
                    angle = float(candidate[4])
                    if int(prepared[moving_index].get("edge_count", 0)) != 3:
                        continue
                    kept_for_piece = triangle_angles.setdefault(moving_index, [])
                    if len(kept_for_piece) >= triangle_keep_per_piece:
                        continue
                    if any(
                        abs(_angle_wrap(angle - existing_angle)) < triangle_min_angle_diff_rad
                        for existing_angle in kept_for_piece
                    ):
                        continue
                    candidate_key = (
                        int(candidate[0]),
                        round(float(candidate[4]), 6),
                        round(float(candidate[5].x), 4),
                        round(float(candidate[5].y), 4),
                    )
                    if candidate_key in kept_keys:
                        continue
                    kept_candidates.append(candidate)
                    kept_keys.add(candidate_key)
                    kept_for_piece.append(angle)
                    if not bool(candidate[9]):
                        kept_partial += 1
                    stats["triangle_diverse_kept"] += 1
                    if len(kept_candidates) >= max_candidates_per_node:
                        break
            for candidate in candidates:
                if len(kept_candidates) >= max_candidates_per_node:
                    break
                candidate_key = (
                    int(candidate[0]),
                    round(float(candidate[4]), 6),
                    round(float(candidate[5].x), 4),
                    round(float(candidate[5].y), 4),
                )
                if candidate_key in kept_keys:
                    continue
                if bool(candidate[9]):
                    kept_candidates.append(candidate)
                    kept_keys.add(candidate_key)
                    continue
                if kept_partial < max_partial_candidates_per_node:
                    kept_candidates.append(candidate)
                    kept_keys.add(candidate_key)
                    kept_partial += 1
            stats["candidate_pruned_by_limit"] += max(
                0, len(candidates) - len(kept_candidates)
            )
            candidates = kept_candidates
        for candidate in candidates:
            (
                moving_index,
                fixed_index,
                fixed_edge_index,
                moving_edge_index,
                angle,
                center,
                candidate_polygon,
                normalized_edge_error,
                contact_length,
                full_edge_match,
                priority,
                flip_generated,
            ) = candidate
            del contact_length, full_edge_match, priority

            poses[moving_index] = (angle, center)
            world_polygons[moving_index] = candidate_polygon

            stop_search = recurse(
                edge_error + normalized_edge_error,
                uses_triangle_flip or bool(flip_generated),
            )

            del world_polygons[moving_index]
            del poses[moving_index]
            if stop_search:
                return True
        return False

    for anchor_index in anchor_order[:anchor_limit]:
        stats["anchor_indices_tried"].append(int(anchor_index))
        poses = {anchor_index: (0.0, Pt(0.0, 0.0))}
        world_polygons = {
            anchor_index: transform_polygon(
                prepared[anchor_index]["local_points"], 0.0, Pt(0.0, 0.0)
            )
        }
        stop_search = recurse(0.0, False)
        if stop_search:
            break
    stats["returned_complete_solutions"] = len(complete_solutions)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stats)
    if best_solution is None:
        return None

    def finalize_solution(solution_record: Dict) -> Dict:
        rectangle = solution_record["rectangle"]
        rectangle, normalized_poses, normalized_world_polygons = _normalize_rectangle_pose(
            rectangle,
            solution_record["poses"],
            solution_record["world_polygons"],
        )
        desired_center = (
            target_center if target_center is not None else rectangle["center"]
        )
        shift = desired_center.sub(rectangle["center"])

        placements = []
        for piece in prepared:
            angle, center = normalized_poses[piece["index"]]
            shifted_center = center.add(shift)
            placements.append(
                {
                    "piece_index": piece["index"],
                    "source_center": piece["source_center"],
                    "target_center": shifted_center,
                    "offset": shifted_center.sub(piece["source_center"]),
                    "angle": _angle_wrap(angle),
                    "source_orientation": piece["source_orientation"],
                    "target_orientation": _angle_wrap(
                        piece["source_orientation"] + angle
                    ),
                    "target_pts": [
                        point.add(shift)
                        for point in normalized_world_polygons[piece["index"]]
                    ],
                }
            )

        rectangle = dict(rectangle)
        rectangle["center"] = desired_center
        rectangle["corners"] = [corner.add(shift) for corner in rectangle["corners"]]
        return {
            "placements": placements,
            "rectangle": rectangle,
            "score": solution_record["score"],
            "area_error": solution_record["area_error"],
            "uses_triangle_flip": bool(solution_record.get("uses_triangle_flip", False)),
            "poses": normalized_poses,
            "world_polygons": {
                index: [point.add(shift) for point in polygon]
                for index, polygon in normalized_world_polygons.items()
            },
        }

    finalized_candidates = [finalize_solution(solution) for solution in complete_solutions]
    result = finalize_solution(best_solution)
    result["geometry_candidates"] = finalized_candidates
    return result


class TextureScorer:
    """纹理连续性评分器。"""

    def __init__(self, mm_per_px=10.0, strip_width_mm=5.0,
                 lambda_texture=0.5, gw=0.35, nw=0.30, ow=0.35, pw=0.45,
                 fallback_full_rect_penalty=0.12,
                 fallback_sparse_seam_penalty=0.08):
        self.mm_per_px = mm_per_px
        self.strip_r = max(2, int(strip_width_mm * mm_per_px * 0.5))
        self.lambda_t = lambda_texture
        self.w_g = max(0.01,gw); self.w_n = max(0.01,nw); self.w_o = max(0.01,ow); self.w_p = max(0.01,pw)
        s = self.w_g + self.w_n + self.w_o + self.w_p
        self.w_g /= s; self.w_n /= s; self.w_o /= s; self.w_p /= s
        self.full_rect_penalty = max(0.0, float(fallback_full_rect_penalty))
        self.sparse_seam_penalty = max(0.0, float(fallback_sparse_seam_penalty))

    def score_solution(self, solution: Dict, pieces: Sequence[Dict]) -> Dict:
        r = self._render(solution, pieces)
        if r is None: return self._def("no images")
        canvas, masks, pattern_layers = r
        seams = self._find_seams(masks)
        if not seams:
            fb = self._full_rect(canvas, masks, pattern_layers)
            return self._asm(fb, [], True, False)
        ss = []; gs = ns = os_ = ps = is_ = 0.0; fb_flag = False; total_weight = 0.0
        for s in seams:
            sc = self._score_seam(canvas, pattern_layers, s)
            ss.append(sc)
            if sc.get("fb"): fb_flag = True
            weight = max(1.0, float(sc.get("weight", 1.0)))
            total_weight += weight
            gs += sc["g"] * weight
            ns += sc["n"] * weight
            os_ += sc["o"] * weight
            ps += sc["p"] * weight
            is_ += sc.get("i", 0.5) * weight
        if total_weight <= 1e-6:
            total_weight = float(max(1, len(ss)))
        return self._asm(
            {
                "g": gs/total_weight,
                "n": ns/total_weight,
                "o": os_/total_weight,
                "p": ps/total_weight,
                "i": is_/total_weight,
            },
            ss, False, fb_flag)

    def total_score(self, shape_score: float, tex: Dict) -> float:
        jt = 1.0 - tex.get("texture_score", 0.5)
        return shape_score + self.lambda_t * jt

    def _render(self, solution, pieces):
        """Mask-weighted rendering to avoid transparency artifacts."""
        poses = solution.get("poses")
        if not poses:
            placements = solution.get("placements",[])
            if placements:
                poses = {}
                for pl in placements:
                    idx = pl.get("piece_index", len(poses))
                    tc = pl["target_center"]
                    pt = Pt(tc.x, tc.y) if hasattr(tc, "x") and hasattr(tc, "y") else Pt(tc[0], tc[1])
                    poses[idx] = (pl.get("angle",0.0), pt)
            else:
                poses = {i: (0.0, Pt(0.0,0.0)) for i in range(len(pieces))}

        world = solution.get("world_polygons")
        if not world:
            world = {}
            for pi, p in enumerate(pieces):
                ct = polygon_centroid(p["pts"])
                lp = [pt.sub(ct) for pt in p["pts"]]
                a, c = poses.get(pi, (0.0, Pt(0.0,0.0)))
                world[pi] = [pt.rotate(a).add(c) for pt in lp]

        all_pts = [pt for poly in world.values() for pt in poly]
        min_x = min(p.x for p in all_pts); max_x = max(p.x for p in all_pts)
        min_y = min(p.y for p in all_pts); max_y = max(p.y for p in all_pts)
        m = self.strip_r * 4
        cw = int((max_x-min_x)*self.mm_per_px) + 2*m + 1
        ch = int((max_y-min_y)*self.mm_per_px) + 2*m + 1
        ox = -min_x*self.mm_per_px + m; oy = -min_y*self.mm_per_px + m

        canvas_acc = np.zeros((ch,cw,3), dtype=np.float64)
        weight_acc = np.zeros((ch,cw), dtype=np.float64)
        red_pattern_acc = np.zeros((ch,cw), dtype=np.uint8)
        black_pattern_acc = np.zeros((ch,cw), dtype=np.uint8)
        masks = {}

        for pi, p in enumerate(pieces):
            a, c = poses.get(pi, (0.0, Pt(0.0,0.0)))
            if "image" in p and p["image"] is not None:
                img = p["image"]
                if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h2,w2 = img.shape[:2]
                sp = p.get("polygon_in_image")
                if sp is None: sp = [Pt(0,0), Pt(w2,0), Pt(w2,h2), Pt(0,h2)]
                sc = polygon_centroid(sp)
                ca=math.cos(a); sa=math.sin(a)
                tx = c.x*self.mm_per_px + ox; ty = c.y*self.mm_per_px + oy
                M = np.array([[ca, -sa, tx-(ca*sc.x-sa*sc.y)],
                              [sa,  ca, ty-(sa*sc.x+ca*sc.y)]], dtype=np.float64)
                mask_src = np.zeros((h2,w2), dtype=np.uint8)
                cv2.fillPoly(mask_src, [np.array([(pt.x,pt.y) for pt in sp], dtype=np.int32)], 255)
                red_src, black_src = self._extract_card_pattern_masks(img, mask_src)
                md = np.zeros((ch,cw), dtype=np.uint8)
                cv2.warpAffine(mask_src, M, (cw,ch), dst=md,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST)
                masks[pi] = md
                img_f = img.astype(np.float64)
                iw = np.zeros((ch,cw,3), dtype=np.float64)
                cv2.warpAffine(img_f, M, (cw,ch), dst=iw,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_LINEAR)
                red_dst = np.zeros((ch,cw), dtype=np.uint8)
                black_dst = np.zeros((ch,cw), dtype=np.uint8)
                cv2.warpAffine(red_src, M, (cw,ch), dst=red_dst,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST)
                cv2.warpAffine(black_src, M, (cw,ch), dst=black_dst,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0, flags=cv2.INTER_NEAREST)
                red_pattern_acc = cv2.bitwise_or(red_pattern_acc, red_dst)
                black_pattern_acc = cv2.bitwise_or(black_pattern_acc, black_dst)
                mf = md.astype(np.float64)/255.0
                for cc in range(3): canvas_acc[:,:,cc] += iw[:,:,cc]*mf
                weight_acc += mf
            else:
                masks[pi] = self._shape_mask(world[pi], cw, ch, ox, oy)

        canvas = np.zeros((ch,cw,3), dtype=np.uint8)
        valid = weight_acc > 0
        for cc in range(3):
            canvas[:,:,cc] = np.where(valid,
                np.clip(canvas_acc[:,:,cc]/np.maximum(weight_acc, 1e-10), 0, 255), 0).astype(np.uint8)
        return canvas, masks, {"red": red_pattern_acc, "black": black_pattern_acc}

    def _extract_card_pattern_masks(self, img, polygon_mask):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        valid = polygon_mask > 0
        h = hsv[:, :, 0].astype(np.int16)
        s = hsv[:, :, 1].astype(np.uint8)
        v = hsv[:, :, 2].astype(np.uint8)

        red = (((h <= 14) | (h >= 166)) & (s >= 55) & (v >= 35) & valid)
        black = ((gray <= 118) & (v <= 150) & (~red) & valid)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_u8 = (red.astype(np.uint8) * 255)
        black_u8 = (black.astype(np.uint8) * 255)
        red_u8 = cv2.morphologyEx(red_u8, cv2.MORPH_OPEN, kernel)
        black_u8 = cv2.morphologyEx(black_u8, cv2.MORPH_OPEN, kernel)
        return red_u8, black_u8

    def _shape_mask(self, poly, cw, ch, ox, oy):
        mask = np.zeros((ch,cw), dtype=np.uint8)
        pts = np.array([(int(p.x*self.mm_per_px+ox),int(p.y*self.mm_per_px+oy)) for p in poly], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def _find_seams(self, masks):
        idxs = list(masks.keys()); seams = []
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                pi,pj = idxs[i],idxs[j]; mi,mj = masks[pi],masks[pj]
                if cv2.countNonZero(cv2.bitwise_and(mi,mj)) > 0: continue
                di = cv2.dilate(mi, k, iterations=self.strip_r)
                dj = cv2.dilate(mj, k, iterations=self.strip_r)
                reg = cv2.bitwise_and(di,dj)
                if cv2.countNonZero(reg) < 5: continue
                ei = cv2.morphologyEx(mi, cv2.MORPH_GRADIENT, k)
                ej = cv2.morphologyEx(mj, cv2.MORPH_GRADIENT, k)
                gap_band = cv2.bitwise_and(
                    reg,
                    cv2.bitwise_not(cv2.bitwise_or(mi, mj)),
                )
                if cv2.countNonZero(gap_band) < 5:
                    gap_band = reg
                edge_support = cv2.bitwise_and(
                    cv2.dilate(ei, k, iterations=max(1, self.strip_r // 2)),
                    cv2.dilate(ej, k, iterations=max(1, self.strip_r // 2)),
                )
                sample_band = cv2.bitwise_or(gap_band, edge_support)
                seams.append(
                    {
                        "i":pi,
                        "j":pj,
                        "mi":mi,
                        "mj":mj,
                        "r":reg,
                        "probe":gap_band,
                        "sample":sample_band,
                    }
                )
        return seams

    def _extract_ink_mask_from_canvas(self, canvas):
        if canvas.ndim == 2:
            gray = canvas
            saturation = np.zeros_like(gray, dtype=np.uint8)
            max_delta_from_white = (255 - gray).astype(np.uint8)
        else:
            hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            saturation = hsv[:, :, 1].astype(np.uint8)
            max_delta_from_white = np.max(
                255 - canvas.astype(np.int16),
                axis=2,
            ).astype(np.uint8)

        ink_mask = (
            (
                (max_delta_from_white >= 24)
                | (saturation >= 30)
                | (gray <= 205)
            ).astype(np.uint8)
            * 255
        )
        return cv2.morphologyEx(
            ink_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

    def _build_informative_sample_mask(self, canvas, pattern_layers, si, sj, base_mask):
        if cv2.countNonZero(base_mask) < 5:
            return base_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        informative = np.zeros_like(base_mask, dtype=np.uint8)

        if pattern_layers:
            red = pattern_layers.get("red")
            black = pattern_layers.get("black")
            if red is not None and black is not None:
                pattern_union = cv2.bitwise_or(red, black)
                pattern_near = cv2.dilate(
                    cv2.bitwise_and(pattern_union, cv2.bitwise_or(si, sj)),
                    kernel,
                    iterations=1,
                )
                informative = cv2.bitwise_or(
                    informative,
                    cv2.bitwise_and(base_mask, pattern_near),
                )

        ink_mask = self._extract_ink_mask_from_canvas(canvas)
        ink_near = cv2.dilate(
            cv2.bitwise_and(ink_mask, cv2.bitwise_or(si, sj)),
            kernel,
            iterations=1,
        )
        informative = cv2.bitwise_or(
            informative,
            cv2.bitwise_and(base_mask, ink_near),
        )

        if cv2.countNonZero(informative) < max(8, cv2.countNonZero(base_mask) // 12):
            return base_mask
        return informative

    def _color_gradient_mag(self, canvas):
        if canvas.ndim==2 or canvas.shape[2]==1:
            gx=cv2.Sobel(canvas,cv2.CV_32F,1,0,ksize=3)
            gy=cv2.Sobel(canvas,cv2.CV_32F,0,1,ksize=3)
            return np.sqrt(gx**2+gy**2)
        gx = np.zeros_like(canvas, dtype=np.float32)
        gy = np.zeros_like(canvas, dtype=np.float32)
        for c in range(canvas.shape[2]):
            gx[:,:,c]=cv2.Sobel(canvas[:,:,c],cv2.CV_32F,1,0,ksize=3)
            gy[:,:,c]=cv2.Sobel(canvas[:,:,c],cv2.CV_32F,0,1,ksize=3)
        return np.max(np.sqrt(gx**2+gy**2), axis=2)

    def _score_seam(self, canvas, pattern_layers, s):
        r = self.strip_r; mi,mj = s["mi"],s["mj"]; reg = s["r"]
        probe = s.get("probe", reg)
        sample_band = s.get("sample", probe)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        di = cv2.dilate(mi,k,iterations=r); dj = cv2.dilate(mj,k,iterations=r)
        si = cv2.bitwise_and(cv2.bitwise_and(di,cv2.bitwise_not(mj)), probe)
        sj = cv2.bitwise_and(cv2.bitwise_and(dj,cv2.bitwise_not(mi)), probe)
        if np.sum(si>0)<10 or np.sum(sj>0)<10:
            si = cv2.bitwise_and(cv2.bitwise_and(di,cv2.bitwise_not(mj)), reg)
            sj = cv2.bitwise_and(cv2.bitwise_and(dj,cv2.bitwise_not(mi)), reg)
        if np.sum(si>0)<10 or np.sum(sj>0)<10:
            fb = self._full_rect(canvas, {0: cv2.bitwise_or(mi,mj)}, pattern_layers)
            return {"g":fb["g"],"n":fb["n"],"o":fb["o"],"p":fb["p"],"i":fb.get("i", 0.5),"fb":True,"weight":1.0}
        sample_mask = self._build_informative_sample_mask(
            canvas,
            pattern_layers,
            si,
            sj,
            sample_band,
        )
        # Gradient
        gm = self._color_gradient_mag(canvas)
        sg = np.median(gm[sample_mask>0]) if np.sum(sample_mask>0)>0 else 0.0
        er = cv2.erode(mi,k,iterations=r*2)
        inner = cv2.bitwise_and(mi,cv2.bitwise_not(er))
        ig_ = np.median(gm[inner>0]) if np.sum(inner>0)>0 else 1.0
        grad_s = min(sg/max(ig_,1.0),2.0)/2.0
        ncc_s = self._ncc_along_edge(canvas, si, sj, sample_mask, r)
        orb_s = self._orb(canvas, si, sj)
        pattern_s, pattern_strength = self._pattern_consistency(pattern_layers, si, sj, sample_mask, r)
        ink_s, ink_strength = self._ink_consistency(canvas, si, sj, sample_mask, r)
        info_strength = 0.65 * pattern_strength + 0.35 * ink_strength
        info_factor = 0.25 + 0.75 * info_strength
        ncc_s = 0.5 + (ncc_s - 0.5) * info_factor
        orb_s = 0.5 + (orb_s - 0.5) * info_factor
        pattern_s = 0.80 * pattern_s + 0.20 * ink_s
        seam_pixels = float(max(cv2.countNonZero(sample_mask), cv2.countNonZero(probe)))
        texture_energy = float(np.median(gm[sample_mask > 0])) if np.any(sample_mask > 0) else 0.0
        weight = max(1.0, seam_pixels) * (0.15 + min(texture_energy / 28.0, 1.0) + 1.0 * info_strength)
        return {
            "g":float(grad_s),"n":float(ncc_s),"o":float(orb_s),"p":float(pattern_s),
            "i":float(ink_s),
            "fb":False,"weight":float(weight),
            "pattern_strength":float(pattern_strength),
            "ink_strength":float(ink_strength),
            "sample_pixels": float(cv2.countNonZero(sample_mask)),
        }

    def _pattern_consistency(self, pattern_layers, si, sj, edge, r):
        if not pattern_layers:
            return 0.5, 0.0
        red = pattern_layers.get("red")
        black = pattern_layers.get("black")
        if red is None or black is None:
            return 0.5, 0.0

        ys, xs = np.where(edge > 0)
        if len(xs) < 5:
            return 0.5, 0.0

        count = min(20, len(xs))
        idx = np.linspace(0, len(xs) - 1, count, dtype=int)
        radius = min(r, 5)
        scores = []
        weights = []
        for sample_index in idx:
            cx, cy = int(xs[sample_index]), int(ys[sample_index])
            y0 = max(0, cy - radius); y1 = cy + radius + 1
            x0 = max(0, cx - radius); x1 = cx + radius + 1
            local_i = si[y0:y1, x0:x1] > 0
            local_j = sj[y0:y1, x0:x1] > 0
            if local_i.sum() < 3 or local_j.sum() < 3:
                continue

            red_i = float(np.mean(red[y0:y1, x0:x1][local_i] > 0))
            red_j = float(np.mean(red[y0:y1, x0:x1][local_j] > 0))
            black_i = float(np.mean(black[y0:y1, x0:x1][local_i] > 0))
            black_j = float(np.mean(black[y0:y1, x0:x1][local_j] > 0))

            red_info = max(red_i, red_j)
            black_info = max(black_i, black_j)
            local_weight = red_info + black_info
            if local_weight < 0.03:
                scores.append(0.5)
                weights.append(0.1)
                continue

            red_similarity = 1.0 - abs(red_i - red_j)
            black_similarity = 1.0 - abs(black_i - black_j)
            local_score = (
                red_similarity * red_info + black_similarity * black_info
            ) / max(local_weight, 1e-6)
            scores.append(float(local_score))
            weights.append(float(local_weight))

        if not scores:
            return 0.5, 0.0
        weights_np = np.asarray(weights, dtype=np.float32)
        scores_np = np.asarray(scores, dtype=np.float32)
        return float(np.average(scores_np, weights=weights_np)), float(np.mean(weights_np))

    def _ink_consistency(self, canvas, si, sj, edge, r):
        ink_mask = self._extract_ink_mask_from_canvas(canvas)

        ys, xs = np.where(edge > 0)
        if len(xs) < 5:
            return 0.5, 0.0

        count = min(20, len(xs))
        idx = np.linspace(0, len(xs) - 1, count, dtype=int)
        radius = min(r, 5)
        scores = []
        weights = []
        for sample_index in idx:
            cx, cy = int(xs[sample_index]), int(ys[sample_index])
            y0 = max(0, cy - radius); y1 = cy + radius + 1
            x0 = max(0, cx - radius); x1 = cx + radius + 1
            local_i = si[y0:y1, x0:x1] > 0
            local_j = sj[y0:y1, x0:x1] > 0
            if local_i.sum() < 3 or local_j.sum() < 3:
                continue

            local_ink = ink_mask[y0:y1, x0:x1] > 0
            ink_i_mask = np.logical_and(local_ink, local_i)
            ink_j_mask = np.logical_and(local_ink, local_j)
            ink_i = float(np.mean(ink_i_mask)) if local_i.any() else 0.0
            ink_j = float(np.mean(ink_j_mask)) if local_j.any() else 0.0
            combined_info = max(ink_i, ink_j)
            paired_info = min(ink_i, ink_j)

            if combined_info < 0.04:
                scores.append(0.5)
                weights.append(0.05)
                continue

            if paired_info < 0.015:
                scores.append(0.12)
                weights.append(float(combined_info))
                continue

            ink_i_u8 = (ink_i_mask.astype(np.uint8) * 255)
            ink_j_u8 = (ink_j_mask.astype(np.uint8) * 255)
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            contact_overlap = cv2.bitwise_and(
                cv2.dilate(ink_i_u8, bridge_kernel, iterations=1),
                cv2.dilate(ink_j_u8, bridge_kernel, iterations=1),
            )
            contact_score = min(
                1.0,
                cv2.countNonZero(contact_overlap) / max(4.0, 0.5 * (cv2.countNonZero(ink_i_u8) + cv2.countNonZero(ink_j_u8))),
            )

            j_inv = np.where(ink_j_mask, 0, 255).astype(np.uint8)
            i_inv = np.where(ink_i_mask, 0, 255).astype(np.uint8)
            dist_to_j = cv2.distanceTransform(j_inv, cv2.DIST_L2, 3)
            dist_to_i = cv2.distanceTransform(i_inv, cv2.DIST_L2, 3)
            mean_ij = float(np.mean(dist_to_j[ink_i_mask])) if np.any(ink_i_mask) else 4.0
            mean_ji = float(np.mean(dist_to_i[ink_j_mask])) if np.any(ink_j_mask) else 4.0
            distance_score = math.exp(-(mean_ij + mean_ji) * 0.5 / 1.6)

            density_balance = max(0.0, 1.0 - abs(ink_i - ink_j) / max(combined_info, 1e-6))
            local_score = 0.45 * contact_score + 0.40 * distance_score + 0.15 * density_balance
            scores.append(float(local_score))
            weights.append(float(0.5 * combined_info + 0.5 * paired_info))

        if not scores:
            return 0.5, 0.0
        weights_np = np.asarray(weights, dtype=np.float32)
        scores_np = np.asarray(scores, dtype=np.float32)
        return float(np.average(scores_np, weights=weights_np)), float(np.mean(weights_np))

    def _ncc_along_edge(self, canvas, si, sj, edge, r):
        """沿接缝逐点计算 NCC 相似度。"""
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        ys, xs = np.where(edge > 0)
        if len(xs) < 5: return 0.5
        n = min(20, len(xs))
        idx = np.linspace(0, len(xs)-1, n, dtype=int)
        p = min(r, 5); vals = []
        for k in idx:
            cx, cy = int(xs[k]), int(ys[k])
            y0 = max(0, cy-p); y1 = cy+p+1; x0 = max(0, cx-p); x1 = cx+p+1
            vi = gray[y0:y1,x0:x1][si[y0:y1,x0:x1] > 0]
            vj = gray[y0:y1,x0:x1][sj[y0:y1,x0:x1] > 0]
            if len(vi)<3 or len(vj)<3: continue
            mi,mj = np.mean(vi), np.mean(vj); si_,sj_ = np.std(vi), np.std(vj)
            if si_<0.5 or sj_<0.5: vals.append(0.5); continue
            n = min(len(vi), len(vj)); ncc = np.mean((vi[:n]-mi)*(vj[:n]-mj))/(si_*sj_) if n >= 3 else 0.5
            vals.append(max(0.0, min(1.0, (ncc+1.0)/2.0)))
        return float(np.mean(vals)) if vals else 0.5

    def _orb(self, canvas, si, sj):
        gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        ri = cv2.bitwise_and(gray,gray,mask=si); rj = cv2.bitwise_and(gray,gray,mask=sj)
        orb = cv2.ORB_create(nfeatures=80)
        kpi,di = orb.detectAndCompute(ri,None); kpj,dj = orb.detectAndCompute(rj,None)
        if di is None or dj is None or len(kpi)<2 or len(kpj)<2: return 0.5
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        mm = bf.match(di,dj)
        if len(mm)<2: return 0.5
        return float(max(0.0, 1.0-np.mean([m.distance for m in mm])/100.0))

    def _full_rect(self, canvas, masks, pattern_layers=None):
        gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY) if canvas.ndim==3 else canvas
        c = np.zeros_like(gray, dtype=np.uint8)
        for m in masks.values(): c = cv2.bitwise_or(c,m)
        if np.sum(c>0) < 50: return {"g":0.5,"n":0.5,"o":0.5,"p":0.5,"i":0.5}
        gm = self._color_gradient_mag(canvas)
        rg = gm[c>0]
        if len(rg)>0:
            grad_ref = np.percentile(rg, 90)
            grad = min(np.median(rg)/max(grad_ref,1.0), 1.0)
        else: grad = 0.5
        roi = cv2.bitwise_and(gray,gray,mask=c)
        orb = cv2.ORB_create(nfeatures=150)
        kp,de = orb.detectAndCompute(roi,None)
        orb_s = min(len(kp)/50.0,1.0) if (de is not None and len(kp)>=4) else 0.3
        pattern_score = 0.5
        ink_score = 0.5
        if pattern_layers:
            red = pattern_layers.get("red")
            black = pattern_layers.get("black")
            if red is not None and black is not None and np.sum(c > 0) > 0:
                red_density = float(np.mean(red[c > 0] > 0))
                black_density = float(np.mean(black[c > 0] > 0))
                pattern_score = 0.5 + min(0.25, 0.8 * max(red_density, black_density))
        if canvas.ndim == 3 and np.sum(c > 0) > 0:
            saturation = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)[:, :, 1]
            max_delta_from_white = np.max(255 - canvas.astype(np.int16), axis=2)
            ink_density = float(
                np.mean(
                    (
                        (max_delta_from_white[c > 0] >= 24)
                        | (saturation[c > 0] >= 30)
                        | (gray[c > 0] <= 205)
                    )
                )
            )
            ink_score = 0.5 + min(0.25, 0.7 * ink_density)
        return {"g":float(grad),"n":0.5,"o":float(orb_s),"p":float(pattern_score),"i":float(ink_score)}

    def _asm(self, sc, ss, ff, fs):
        g,n,o,p = sc["g"],sc["n"],sc["o"],sc.get("p", 0.5)
        ink = sc.get("i", 0.5)
        texture_core = self.w_g*(1.0-g) + self.w_n*n + self.w_o*o + self.w_p*p
        tex = 0.85 * texture_core + 0.15 * ink
        if ff:
            tex -= self.full_rect_penalty
        if fs:
            tex -= self.sparse_seam_penalty
        return {"texture_score":float(max(0.0,min(1.0,tex))),
                "gradient_discontinuity":float(g),"ncc_similarity":float(n),
                "orb_consistency":float(o),"pattern_consistency":float(p),"ink_consistency":float(ink),"seam_scores":ss,
                "fallback_full_rect":ff,"fallback_sparse_seam":fs}

    def _def(self, reason):
        return {"texture_score":0.5,"gradient_discontinuity":0.0,
                "ncc_similarity":0.0,"orb_consistency":0.0,"pattern_consistency":0.0,"ink_consistency":0.0,
                "seam_scores":[],"fallback_full_rect":False,
                "fallback_sparse_seam":False,"error":reason}

# ---- Public API: joint solver ----

def solve_with_texture(pieces_geometry, pieces_texture, target_center=None,
                       *, mm_per_px=10.0, strip_width_mm=5.0,
                       lambda_texture=0.5, top_k=1, geometry_kwargs=None,
                       gw=0.35, nw=0.30, ow=0.35, pw=0.45,
                       fallback_full_rect_penalty=0.12,
                       fallback_sparse_seam_penalty=0.08,
                       geometry_candidate_min_angle_diff_rad=0.0,
                       base_geometry_candidate_limit=0,
                       candidate_rerank_enabled=True,
                       perturb_candidate_enabled=True,
                       equilateral_triangle_rotation_enabled=False,
                       equilateral_triangle_tolerance_mm=0.0,
                       equilateral_triangle_relative_tolerance=0.0,
                       equilateral_triangle_rotation_angles_deg=None,
                       equilateral_triangle_shape_penalty=0.0,
                       angle_perturb_rad=0.035,
                       translate_perturb_mm=1.2):
    """Geometry candidate solver with optional texture re-ranking.

    1. Run geometric solver to find candidate pose(s)
    2. Generate perturbed variants for mirror/symmetry resolution
    3. Optionally score each with texture continuity
    4. Optionally re-rank by J_total = J_shape + lambda * J_texture

    Returns dict with best_solution, texture scores, and candidates list.
    """
    if geometry_kwargs is None:
        geometry_kwargs = {}
    total_piece_area = sum(abs(polygon_area(piece["pts"])) for piece in pieces_geometry)

    def _ensure_Pt(p):
        if isinstance(p, Pt): return p
        return Pt(p[0], p[1])

    def _pose_signature(solution):
        poses = solution.get("poses", {})
        return tuple(
            sorted(
                (
                    int(index),
                    round(float(angle), 6),
                    round(float(center.x), 4),
                    round(float(center.y), 4),
                )
                for index, (angle, center) in poses.items()
            )
        )

    # Step 1: Find geometric solution(s)
    sol = find_rectangle_solution(
        pieces_geometry, target_center, **geometry_kwargs)
    if sol is None:
        return {"error": "no geometric solution"}

    candidates = list(sol.get("geometry_candidates", []))
    if not candidates:
        candidates = [sol]
    candidates = _select_diverse_geometry_candidates(
        candidates,
        max(1, top_k),
        float(geometry_candidate_min_angle_diff_rad),
    )
    for candidate in candidates:
        candidate["candidate_source"] = (
            "geometry_flip"
            if bool(candidate.get("uses_triangle_flip", False))
            else "geometry"
        )
    if int(base_geometry_candidate_limit) > 0:
        geometry_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_source", "")).startswith("geometry")
        ]
        other_candidates = [
            candidate
            for candidate in candidates
            if not str(candidate.get("candidate_source", "")).startswith("geometry")
        ]
        candidates = geometry_candidates[: int(base_geometry_candidate_limit)] + other_candidates

    local_pts = {}
    equilateral_triangle_indices = []
    for pi, p in enumerate(pieces_geometry):
        pts = p.get("pts", [])
        ct = polygon_centroid(pts)
        local_pts[pi] = [pt.sub(ct) for pt in pts]
        if equilateral_triangle_rotation_enabled and _is_near_equilateral_triangle(
            pts,
            equilateral_triangle_tolerance_mm,
            equilateral_triangle_relative_tolerance,
        ):
            equilateral_triangle_indices.append(pi)

    candidate_signatures = {_pose_signature(candidate) for candidate in candidates}

    def _build_rotated_triangle_candidate(
        current_candidate: Dict,
        triangle_index: int,
        angle_deg: float,
    ) -> Optional[Dict]:
        current_poses = current_candidate.get("poses", {})
        current_world = current_candidate.get("world_polygons", {})
        current_placements = current_candidate.get("placements", [])
        if (
            triangle_index not in current_poses
            or not current_world
            or not current_placements
        ):
            return None

        current_angle, current_center = current_poses[triangle_index]
        triangle_local_points = local_pts.get(triangle_index)
        if not triangle_local_points:
            return None

        contact_pairs = _find_piece_contact_pairs(
            current_world,
            triangle_index,
            angle_tolerance_rad=0.40,
            distance_tolerance_mm=max(
                2.5,
                float(geometry_kwargs.get("edge_tolerance_mm", 3.0)),
            ),
            minimum_overlap_mm=max(
                5.0,
                0.25 * min(_edge_lengths_from_points(triangle_local_points)),
            ),
            max_pairs=3,
        )
        if not contact_pairs:
            return None

        desired_angle = _angle_wrap(current_angle + math.radians(angle_deg))
        best_alt = None
        best_alt_score = float("inf")

        for pair in contact_pairs:
            fixed_polygon = current_world.get(int(pair["fixed_index"]))
            if not fixed_polygon:
                continue
            fixed_start, fixed_end = _edge(
                fixed_polygon,
                int(pair["fixed_edge_index"]),
            )
            for moving_edge_index in range(len(triangle_local_points)):
                moving_start = triangle_local_points[moving_edge_index].rotate(
                    desired_angle
                )
                moving_end = triangle_local_points[
                    (moving_edge_index + 1) % len(triangle_local_points)
                ].rotate(desired_angle)
                minimum_contact = max(
                    5.0,
                    0.55
                    * min(
                        fixed_start.dist(fixed_end),
                        moving_start.dist(moving_end),
                    ),
                )
                centers = _edge_alignment_translations_for_fixed_angle(
                    fixed_start,
                    fixed_end,
                    moving_start,
                    moving_end,
                    minimum_contact,
                    True,
                )
                if not centers:
                    continue
                for center, _contact_ratio, _contact_length in centers:
                    rotated_world = {
                        idx: [Pt(point.x, point.y) for point in polygon]
                        for idx, polygon in current_world.items()
                    }
                    rotated_triangle = transform_polygon(
                        triangle_local_points,
                        desired_angle,
                        _ensure_Pt(center),
                    )
                    if any(
                        polygons_overlap(
                            rotated_triangle,
                            polygon,
                            float(geometry_kwargs.get("overlap_tolerance_mm", 1.5)),
                        )
                        for idx, polygon in current_world.items()
                        if idx != triangle_index
                    ):
                        continue
                    rotated_world[triangle_index] = rotated_triangle
                    support_pairs = _find_piece_contact_pairs(
                        rotated_world,
                        triangle_index,
                        angle_tolerance_rad=0.28,
                        distance_tolerance_mm=min(
                            1.8,
                            float(geometry_kwargs.get("edge_tolerance_mm", 3.0)),
                        ),
                        minimum_overlap_mm=4.0,
                        max_pairs=4,
                    )
                    contact_count = len(support_pairs)
                    contact_overlap = sum(
                        float(item.get("overlap", 0.0))
                        for item in support_pairs[:2]
                    )
                    avg_gap = (
                        sum(float(item.get("gap", 0.0)) for item in support_pairs[:2])
                        / max(1, min(2, contact_count))
                    )
                    rectangle = minimum_area_rectangle(
                        [
                            point
                            for polygon in rotated_world.values()
                            for point in polygon
                        ]
                    )
                    if rectangle is None or rectangle["area"] <= EPS:
                        continue
                    area_error = abs(rectangle["area"] - total_piece_area) / rectangle["area"]
                    local_score = (
                        area_error
                        + 0.22 * max(0, 2 - min(2, contact_count))
                        + 0.04 * avg_gap
                        + 0.0015 * _ensure_Pt(center).dist(_ensure_Pt(current_center))
                        - 0.002 * min(contact_overlap, 25.0)
                    )
                    if local_score >= best_alt_score:
                        continue

                    rotated_poses = dict(current_poses)
                    rotated_poses[triangle_index] = (
                        desired_angle,
                        _ensure_Pt(center),
                    )
                    alt = dict(current_candidate)
                    alt["poses"] = rotated_poses
                    alt["world_polygons"] = rotated_world
                    alt["score"] = (
                        float(current_candidate.get("score", 0.0))
                        + float(equilateral_triangle_shape_penalty)
                        + float(local_score)
                    )
                    rotation_steps = list(
                        current_candidate.get("triangle_rotation_steps", [])
                    )
                    rotation_steps.append(
                        {
                            "piece_index": int(triangle_index),
                            "angle_deg": int(round(angle_deg)),
                        }
                    )
                    alt["triangle_rotation_steps"] = rotation_steps
                    alt["candidate_source"] = "triangle_rot_" + "_".join(
                        f"p{int(step['piece_index'])}_{int(step['angle_deg'])}"
                        for step in rotation_steps
                    )
                    alt_placements = copy.deepcopy(current_placements)
                    for pl in alt_placements:
                        pi = int(pl.get("piece_index", 0))
                        if pi != triangle_index:
                            continue
                        a, c = rotated_poses[pi]
                        pl["target_center"] = _ensure_Pt(c)
                        pl["angle"] = _angle_wrap(a)
                        source_orientation = float(pl.get("source_orientation", 0.0))
                        pl["target_orientation"] = _angle_wrap(source_orientation + a)
                        pl["target_pts"] = rotated_world[pi]
                        if "source_center" in pl:
                            pl["offset"] = pl["target_center"].sub(pl["source_center"])
                    alt["placements"] = alt_placements
                    best_alt = alt
                    best_alt_score = local_score
        return best_alt

    # Step 2: For near-equilateral triangles, additionally test 60° / 120°
    # rotations on top of the geometric candidate and let template/texture
    # decide whether the pattern becomes more consistent.
    if (
        equilateral_triangle_rotation_enabled
        and equilateral_triangle_indices
        and equilateral_triangle_rotation_angles_deg
    ):
        base_rotation_candidates = list(candidates)
        rotation_angles_deg = [
            float(angle_deg)
            for angle_deg in equilateral_triangle_rotation_angles_deg
            if abs(float(angle_deg)) > 1e-6
        ]
        for base_candidate in base_rotation_candidates:
            base_poses = base_candidate.get("poses", {})
            if not base_poses:
                continue
            active_triangle_indices = [
                triangle_index
                for triangle_index in equilateral_triangle_indices
                if triangle_index in base_poses
            ]
            if not active_triangle_indices:
                continue

            def _enumerate_triangle_rotation_combinations(
                triangle_pos: int,
                current_candidate: Dict,
                changed: bool,
            ) -> None:
                if triangle_pos >= len(active_triangle_indices):
                    if not changed:
                        return
                    signature = _pose_signature(current_candidate)
                    if signature in candidate_signatures:
                        return
                    candidate_signatures.add(signature)
                    candidates.append(current_candidate)
                    return

                triangle_index = active_triangle_indices[triangle_pos]

                # 当前三角形不旋转，直接进入下一个。
                _enumerate_triangle_rotation_combinations(
                    triangle_pos + 1,
                    current_candidate,
                    changed,
                )

                # 当前三角形分别尝试 60° / 120° 等候选；如果有两个等边三角形，
                # 递归会自然把它们的组合情况全部枚举出来。
                for angle_deg in rotation_angles_deg:
                    rotated_candidate = _build_rotated_triangle_candidate(
                        current_candidate,
                        triangle_index,
                        angle_deg,
                    )
                    if rotated_candidate is None:
                        continue
                    _enumerate_triangle_rotation_combinations(
                        triangle_pos + 1,
                        rotated_candidate,
                        True,
                    )

            _enumerate_triangle_rotation_combinations(0, base_candidate, False)

    # Step 3: Generate perturbed alternatives to fill the remaining slots
    base_solution = candidates[0]
    if (
        perturb_candidate_enabled
        and top_k > len(candidates)
        and 'poses' in base_solution
        and 'world_polygons' in base_solution
    ):
        base_poses = base_solution['poses']
        perturb_patterns = [
            (0.0, 0.0, 0.0),
            (+angle_perturb_rad, 0.0, 0.0),
            (-angle_perturb_rad, 0.0, 0.0),
            (0.0, +translate_perturb_mm, 0.0),
            (0.0, -translate_perturb_mm, 0.0),
            (0.0, 0.0, +translate_perturb_mm),
            (0.0, 0.0, -translate_perturb_mm),
            (+angle_perturb_rad * 0.6, +translate_perturb_mm * 0.6, -translate_perturb_mm * 0.6),
            (-angle_perturb_rad * 0.6, -translate_perturb_mm * 0.6, +translate_perturb_mm * 0.6),
        ]
        for attempt in range(max(0, top_k - len(candidates))):
            da, dx, dy = perturb_patterns[attempt % len(perturb_patterns)]
            perturbed_poses = {}
            perturbed_world = {}
            for pi, (angle, center) in base_poses.items():
                # 用确定性的小扰动生成多个邻域候选，避免随机性导致评分忽高忽低。
                scale = 1.0 + 0.12 * (pi % 3)
                new_center = _ensure_Pt(center).add(Pt(dx * scale, dy * scale))
                perturbed_poses[pi] = (angle + da * scale, new_center)
                # Rebuild world_polygon for this piece
                lp = local_pts.get(pi)
                if lp is None:
                    ct = polygon_centroid(pieces_geometry[pi]['pts'])
                    lp = [pt.sub(ct) for pt in pieces_geometry[pi]['pts']]
                    local_pts[pi] = lp
                perturbed_world[pi] = [pt.rotate(angle + da * scale).add(new_center) for pt in lp]
            alt = dict(base_solution)
            alt['poses'] = perturbed_poses
            alt['world_polygons'] = perturbed_world
            alt['score'] = base_solution['score'] + 0.05
            alt['candidate_source'] = "perturb"
            # Copy placements with updated positions
            alt_placements = copy.deepcopy(base_solution.get('placements', []))
            for pl in alt_placements:
                pi = pl.get('piece_index', 0)
                if pi in perturbed_poses:
                    a, c = perturbed_poses[pi]
                    pl['target_center'] = _ensure_Pt(c)
                    pl['angle'] = _angle_wrap(a)
                    source_orientation = float(pl.get('source_orientation', 0.0))
                    pl['target_orientation'] = _angle_wrap(source_orientation + a)
                    if pi in perturbed_world:
                        pl['target_pts'] = perturbed_world[pi]
                    if 'source_center' in pl:
                        pl['offset'] = pl['target_center'].sub(pl['source_center'])
            alt['placements'] = alt_placements
            candidates.append(alt)

    # Step 4: Optionally texture-score and re-rank each candidate.
    results = []
    if candidate_rerank_enabled:
        scorer = TextureScorer(
            mm_per_px=mm_per_px,
            strip_width_mm=strip_width_mm,
            lambda_texture=lambda_texture,
            gw=gw,
            nw=nw,
            ow=ow,
            pw=pw,
            fallback_full_rect_penalty=fallback_full_rect_penalty,
            fallback_sparse_seam_penalty=fallback_sparse_seam_penalty,
        )
        for cs in candidates:
            tex = scorer.score_solution(cs, pieces_texture)
            j_shape = cs.get("score", 0.0)
            j_total = scorer.total_score(j_shape, tex)
            results.append({
                "geometry": cs,
                "texture": tex,
                "j_shape": j_shape,
                "j_texture": 1.0 - tex["texture_score"],
                "j_total": j_total,
            })
        results.sort(key=lambda c: c["j_total"])
    else:
        for cs in candidates:
            results.append({
                "geometry": cs,
                "texture": {},
                "j_shape": float(cs.get("score", 0.0)),
                "j_texture": 0.0,
                "j_total": float(cs.get("score", 0.0)),
            })
    best = results[0]
    return {
        "best_solution": best["geometry"],
        "texture": best["texture"],
        "j_shape": best["j_shape"],
        "j_texture": best["j_texture"],
        "j_total": best["j_total"],
        "candidates": results,
    }


def rank_candidates_with_texture(geometry_solutions, pieces_texture,
                               *, mm_per_px=10.0, strip_width_mm=5.0,
                               lambda_texture=0.5, gw=0.35, nw=0.30, ow=0.35, pw=0.45):
    """Rank multiple geometry solutions by texture continuity.
    Returns candidates sorted by J_total (ascending).
    """
    if not geometry_solutions:
        return []
    scorer = TextureScorer(mm_per_px=mm_per_px, strip_width_mm=strip_width_mm,
                          lambda_texture=lambda_texture, gw=gw, nw=nw, ow=ow, pw=pw)
    results = []
    for sol in geometry_solutions:
        tex = scorer.score_solution(sol, pieces_texture)
        j_shape = sol.get("score", 0.0)
        j_total = scorer.total_score(j_shape, tex)
        results.append({"geometry": sol, "texture": tex,
                        "j_shape": j_shape,
                        "j_texture": 1.0 - tex["texture_score"],
                        "j_total": j_total})
    results.sort(key=lambda c: c['j_total'])
    return results




__all__ = ["TextureScorer", "solve_with_texture", "rank_candidates_with_texture"]
