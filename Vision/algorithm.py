"""E 题碎片拼矩形的纯几何求解器。

求解器接收 2～4 个处于同一毫米坐标系中的多边形轮廓，仅通过刚体变换
重建矩形，不使用生成过程的隐藏信息，也不使用图像纹理。
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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

    # 包围盒完全分离时不可能重叠，避免进入逐边相交和点包含判断。
    first_min_x = min(point.x for point in first)
    first_max_x = max(point.x for point in first)
    first_min_y = min(point.y for point in first)
    first_max_y = max(point.y for point in first)
    second_min_x = min(point.x for point in second)
    second_max_x = max(point.x for point in second)
    second_min_y = min(point.y for point in second)
    second_max_y = max(point.y for point in second)
    if (
        first_max_x < second_min_x - tolerance
        or second_max_x < first_min_x - tolerance
        or first_max_y < second_min_y - tolerance
        or second_max_y < first_min_y - tolerance
    ):
        return False

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


def polygon_orientation(points: Sequence[Pt]) -> float:
    """用最小面积外接矩形的主边估计碎片方向，结果单位为弧度。"""

    rectangle = minimum_area_rectangle(points)
    return 0.0 if rectangle is None else _angle_wrap(rectangle["angle"])


def _edge(points: Sequence[Pt], index: int) -> Tuple[Pt, Pt]:
    return points[index], points[(index + 1) % len(points)]


def _angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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

    target_direction = fixed_start.sub(fixed_end)
    angle = _angle_wrap(
        math.atan2(target_direction.y, target_direction.x)
        - math.atan2(source_direction.y, source_direction.x)
    )
    rotated_start = moving_start.rotate(angle)
    rotated_end = moving_end.rotate(angle)
    fixed_midpoint = fixed_start.lerp(fixed_end, 0.5)
    moving_midpoint = rotated_start.lerp(rotated_end, 0.5)

    centers = [
        fixed_end.sub(rotated_start),
        fixed_start.sub(rotated_end),
        fixed_start.sub(rotated_start),
        fixed_end.sub(rotated_end),
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
        if len(observed) > 5:
            raise ValueError(f"piece {index} has more than five edges")
        source_center = polygon_centroid(observed)
        local_points = [point.sub(source_center) for point in observed]
        prepared.append(
            {
                "index": index,
                "source_center": source_center,
                "source_orientation": polygon_orientation(observed),
                "local_points": local_points,
                "edge_lengths": [
                    local_points[edge_index].dist(
                        local_points[(edge_index + 1) % len(local_points)]
                    )
                    for edge_index in range(len(local_points))
                ],
                "area": polygon_area(local_points),
            }
        )
    return prepared


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
    # 轮廓拟合和边长误差会让相邻碎片出现少量数值穿插，使用毫米级
    # 几何容差排除这种测量误差，但仍会拒绝明显的面积重叠。
    overlap_tolerance_mm: float = 4.0,
    rectangle_area_tolerance: float = 0.06,
    size_range_mm: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
        (40.0, 100.0),
        (80.0, 130.0),
    ),
    dimension_tolerance_mm: float = 3.0,
    max_search_nodes: int = 20_000,
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
            edge_length = piece["edge_lengths"][edge_index]
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

    anchor_index = max(prepared, key=anchor_score)["index"]
    poses: Dict[int, Tuple[float, Pt]] = {
        anchor_index: (0.0, Pt(0.0, 0.0))
    }
    world_polygons: Dict[int, List[Pt]] = {
        anchor_index: transform_polygon(
            prepared[anchor_index]["local_points"], 0.0, Pt(0.0, 0.0)
        )
    }
    best_solution: Optional[Dict] = None
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
        "candidate_counts_by_depth": [],
        "anchor_index": anchor_index,
        "piece_count": len(prepared),
        "total_piece_area_mm2": total_piece_area,
    }

    def partial_geometry_valid(candidate_polygons: Iterable[Sequence[Pt]]) -> bool:
        if size_range_mm is None:
            maximum_final_area = float("inf")
        else:
            maximum_final_area = (
                total_piece_area / max(1.0 - rectangle_area_tolerance, EPS)
            )
        all_points = [point for polygon in candidate_polygons for point in polygon]
        rectangle = minimum_area_rectangle(all_points)
        if rectangle is None:
            return False
        # 当前外接矩形已经超过最终允许面积时，后续只会继续变大，直接剪枝。
        if rectangle["area"] > maximum_final_area + EPS:
            return False
        if size_range_mm is None:
            return True
        short_range, long_range = size_range_mm
        maximum_area = (
            (short_range[1] + dimension_tolerance_mm)
            * (long_range[1] + dimension_tolerance_mm)
        )
        if rectangle["area"] > maximum_area + EPS:
            return False

        maximum_diagonal = math.hypot(
            short_range[1] + dimension_tolerance_mm,
            long_range[1] + dimension_tolerance_mm,
        )
        for first_index, first in enumerate(all_points):
            for second in all_points[first_index + 1 :]:
                if first.dist(second) > maximum_diagonal:
                    return False
        return True

    def evaluate(edge_error: float) -> bool:
        nonlocal best_solution

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

        score = area_error * 10.0 + edge_error / max(1, len(prepared) - 1)
        if best_solution is not None and score >= best_solution["score"]:
            return False

        best_solution = {
            "score": score,
            "area_error": area_error,
            "rectangle": rectangle,
            "poses": dict(poses),
            "world_polygons": {
                index: list(points) for index, points in world_polygons.items()
            },
        }
        return True

    def recurse(edge_error: float) -> bool:
        nonlocal search_nodes
        search_nodes += 1
        stats["search_nodes"] = search_nodes
        if search_nodes > max_search_nodes:
            stats["search_limit_reached"] = True
            return False
        if len(poses) == len(prepared):
            return evaluate(edge_error)

        candidate_keys: Set[Tuple] = set()
        candidates = []
        unplaced = [piece["index"] for piece in prepared if piece["index"] not in poses]

        for moving_index in unplaced:
            moving_points = prepared[moving_index]["local_points"]
            for fixed_index, fixed_polygon in list(world_polygons.items()):
                for fixed_edge_index in range(len(fixed_polygon)):
                    fixed_start, fixed_end = _edge(fixed_polygon, fixed_edge_index)
                    fixed_length = fixed_start.dist(fixed_end)

                    for moving_edge_index in range(len(moving_points)):
                        moving_start, moving_end = _edge(
                            moving_points, moving_edge_index
                        )
                        moving_length = prepared[moving_index]["edge_lengths"][
                            moving_edge_index
                        ]
                        allowed_error = max(
                            edge_tolerance_mm,
                            edge_relative_tolerance * max(fixed_length, moving_length),
                        )
                        length_error = abs(fixed_length - moving_length)
                        minimum_contact = max(
                            minimum_edge_contact_mm,
                            minimum_edge_contact_ratio
                            * min(fixed_length, moving_length),
                        )
                        alignments = _edge_alignment_candidates(
                            fixed_start,
                            fixed_end,
                            moving_start,
                            moving_end,
                            minimum_contact,
                        )
                        for angle, center, contact_ratio, contact_length in alignments:
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
                            if any(
                                polygons_overlap(
                                    candidate_polygon,
                                    placed_polygon,
                                    overlap_tolerance_mm,
                                )
                                for placed_polygon in world_polygons.values()
                            ):
                                stats["overlap_rejections"] += 1
                                continue
                            if not partial_geometry_valid(
                                list(world_polygons.values()) + [candidate_polygon]
                            ):
                                stats["geometry_rejections"] += 1
                                continue

                            partial_rectangle = minimum_area_rectangle(
                                [
                                    point
                                    for polygon in list(world_polygons.values())
                                    + [candidate_polygon]
                                    for point in polygon
                                ]
                            )
                            placed_area = prepared[moving_index]["area"] + sum(
                                prepared[index]["area"] for index in poses
                            )
                            empty_area_ratio = 0.0
                            if partial_rectangle is not None:
                                empty_area_ratio = max(
                                    0.0,
                                    partial_rectangle["area"] - placed_area,
                                ) / total_piece_area

                            # 等长接缝比任意局部接触包含更多约束信息。局部接触
                            # 仍保留给 T 形接缝，但在真正的整边匹配之后再尝试。
                            seam_error = (
                                1.0
                                - contact_ratio
                                + length_error / max(fixed_length, moving_length)
                            )
                            full_edge_match = length_error <= allowed_error
                            if full_edge_match:
                                stats["full_edge_candidates"] += 1
                            else:
                                stats["partial_edge_candidates"] += 1
                            priority = seam_error + 0.5 * empty_area_ratio
                            candidates.append(
                                (
                                    moving_index,
                                    fixed_index,
                                    fixed_edge_index,
                                    moving_edge_index,
                                    angle,
                                    center,
                                    candidate_polygon,
                                    seam_error,
                                    contact_length,
                                    full_edge_match,
                                    priority,
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
                not candidate[-2],
                candidate[-1],
                -candidate[-3],
            )
        )
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
            ) = candidate
            del contact_length, full_edge_match, priority

            poses[moving_index] = (angle, center)
            world_polygons[moving_index] = candidate_polygon

            solved = recurse(edge_error + normalized_edge_error)

            del world_polygons[moving_index]
            del poses[moving_index]
            if solved:
                return True
        return False

    recurse(0.0)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(stats)
    if best_solution is None:
        return None

    rectangle = best_solution["rectangle"]
    rectangle, normalized_poses, normalized_world_polygons = _normalize_rectangle_pose(
        rectangle,
        best_solution["poses"],
        best_solution["world_polygons"],
    )
    desired_center = target_center if target_center is not None else rectangle["center"]
    shift = desired_center.sub(rectangle["center"])

    placements = []
    for piece in prepared:
        angle, center = normalized_poses[piece["index"]]
        shifted_center = center.add(shift)
        source_orientation = piece["source_orientation"]
        placements.append(
            {
                "piece_index": piece["index"],
                "source_center": piece["source_center"],
                "target_center": shifted_center,
                "offset": shifted_center.sub(piece["source_center"]),
                "angle": _angle_wrap(angle),
                "source_orientation": source_orientation,
                "target_orientation": _angle_wrap(source_orientation + angle),
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
        "score": best_solution["score"],
        "area_error": best_solution["area_error"],
    }


def solve_puzzle(
    pieces: Sequence[Dict], target_center: Optional[Pt] = None, **kwargs
) -> Optional[List[Dict]]:
    """兼容性封装：仅返回每块碎片的目标位姿。"""

    solution = find_rectangle_solution(pieces, target_center, **kwargs)
    return None if solution is None else solution["placements"]


# ---------------------------------------------------------------------------
# 用于可重复测试的合成拼图生成器。
# ---------------------------------------------------------------------------

COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#16a085",
]


def random_point_on_edge(start: Pt, end: Pt, margin: float = 0.18) -> Pt:
    ratio = margin + random.random() * (1.0 - 2.0 * margin)
    return start.lerp(end, ratio)


def cut_polygon(
    points: Sequence[Pt], first_cut: Pt, second_cut: Pt
) -> Optional[Tuple[List[Pt], List[Pt]]]:
    """沿两个边界点之间的线段切割凸多边形。"""

    edge_count = len(points)
    first_edge = -1
    second_edge = -1
    for index in range(edge_count):
        start, end = _edge(points, index)
        if point_on_segment(first_cut, start, end, 3.0):
            first_edge = index
        if point_on_segment(second_cut, start, end, 3.0):
            second_edge = index
    if first_edge < 0 or second_edge < 0 or first_edge == second_edge:
        return None

    first_polygon = [first_cut]
    index = (first_edge + 1) % edge_count
    while True:
        first_polygon.append(points[index])
        if index == second_edge:
            break
        index = (index + 1) % edge_count
    first_polygon.append(second_cut)

    second_polygon = [second_cut]
    index = (second_edge + 1) % edge_count
    while True:
        second_polygon.append(points[index])
        if index == first_edge:
            break
        index = (index + 1) % edge_count
    second_polygon.append(first_cut)

    try:
        first_polygon = ensure_ccw(first_polygon)
        second_polygon = ensure_ccw(second_polygon)
    except ValueError:
        return None
    if len(first_polygon) > 5 or len(second_polygon) > 5:
        return None
    return first_polygon, second_polygon


def generate_cut(piece: Dict, target_area: float) -> Optional[Dict]:
    points = piece["pts"]
    edge_count = len(points)
    allow_adjacent = edge_count <= 4
    attempts = []
    for first_edge in range(edge_count):
        for second_edge in range(edge_count):
            if first_edge == second_edge:
                continue
            if not allow_adjacent and (
                (first_edge + 1) % edge_count == second_edge
                or (second_edge + 1) % edge_count == first_edge
            ):
                continue
            for _ in range(4):
                attempts.append(
                    (
                        random_point_on_edge(*_edge(points, first_edge)),
                        random_point_on_edge(*_edge(points, second_edge)),
                    )
                )
    random.shuffle(attempts)
    for first_cut, second_cut in attempts:
        result = cut_polygon(points, first_cut, second_cut)
        if result is None:
            continue
        first_area = polygon_area(result[0])
        second_area = polygon_area(result[1])
        if first_area > target_area * 0.1 and second_area > target_area * 0.1:
            return {"result": result, "p1": first_cut, "p2": second_cut}
    return None


def generate_puzzle(cx: float, cy: float, width: float, height: float) -> Dict:
    """生成由 2～4 块碎片组成的矩形拼图，供求解器测试使用。"""

    piece_count = random.randint(2, 4)
    half_width = width * 0.5
    half_height = height * 0.5
    rectangle = ensure_ccw(
        [
            Pt(cx - half_width, cy - half_height),
            Pt(cx + half_width, cy - half_height),
            Pt(cx + half_width, cy + half_height),
            Pt(cx - half_width, cy + half_height),
        ]
    )
    target_area = width * height
    pieces = [{"pts": list(rectangle)}]
    cut_lines = []

    for _ in range(piece_count - 1):
        candidates = sorted(
            range(len(pieces)),
            key=lambda index: polygon_area(pieces[index]["pts"]),
            reverse=True,
        )
        cut = None
        cut_index = -1
        for candidate_index in candidates:
            cut = generate_cut(pieces[candidate_index], target_area)
            if cut is not None:
                cut_index = candidate_index
                break
        if cut is None:
            break
        pieces[cut_index : cut_index + 1] = [
            {"pts": cut["result"][0]},
            {"pts": cut["result"][1]},
        ]
        cut_lines.append({"p1": cut["p1"], "p2": cut["p2"]})

    for index, piece in enumerate(pieces):
        piece["c"] = COLORS[index % len(COLORS)]

    return {
        "targetPts": rectangle,
        "pieces": pieces,
        "targetCenter": Pt(cx, cy),
        "cutLines": cut_lines,
    }


def scatter_pieces(pieces: Sequence[Dict], distance: float = 180.0) -> List[Dict]:
    """为测试生成各自经过平移和旋转的碎片观测结果。"""

    scattered = []
    for index, piece in enumerate(pieces):
        points = ensure_ccw(piece["pts"])
        center = polygon_centroid(points)
        local_points = [point.sub(center) for point in points]
        angle = random.uniform(-math.pi, math.pi)
        direction = 2.0 * math.pi * index / len(pieces) + random.uniform(-0.2, 0.2)
        observed_center = Pt(
            math.cos(direction) * distance,
            math.sin(direction) * distance,
        )
        scattered.append(
            {
                "pts": transform_polygon(local_points, angle, observed_center),
                "source_index": index,
            }
        )
    random.shuffle(scattered)
    return scattered
