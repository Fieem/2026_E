"""
texture_scorer.py - æ‰‘å…‹ç‰Œç¢Žç‰‡çº¹ç†è¿žç»­æ€§è¯„åˆ† + è”åˆæ±‚è§£ (merged)

ç”¨æ³•:
    from texture_scorer import TextureScorer, solve_with_texture
    result = solve_with_texture(pieces_geometry, pieces_texture, target_center)
"""

import sys, math, copy, numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:
    _p = r"D:\.codex\visualizations\2026\07\29\019fab5a-1829-7a40-8497-9086f35e901b\opencv_pkg"
    if _p not in sys.path: sys.path.insert(0, _p)
    import cv2



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
        if len(observed) > 8:
            raise ValueError(f"piece {index} has more than eight edges")
        source_center = polygon_centroid(observed)
        local_points = [point.sub(source_center) for point in observed]
        prepared.append(
            {
                "index": index,
                "source_center": source_center,
                "source_orientation": polygon_orientation(observed),
                "local_points": local_points,
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
    edge_tolerance_mm: float = 15.0,
    edge_relative_tolerance: float = 0.30,
    minimum_edge_contact_mm: float = 0.5,
    minimum_edge_contact_ratio: float = 0.05,
    overlap_tolerance_mm: float = 0.0,
    rectangle_area_tolerance: float = 0.50,
    size_range_mm: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
        (30.0, 130.0),
        (40.0, 200.0),
    ),
    dimension_tolerance_mm: float = 15.0,
    max_search_nodes: int = 100_000,
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
            return True
        all_points = [point for polygon in candidate_polygons for point in polygon]
        rectangle = minimum_area_rectangle(all_points)
        if rectangle is None:
            return False
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

        score = area_error * 1.0 + edge_error * 0.05 / max(1, len(prepared) - 1)
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
                        moving_length = moving_start.dist(moving_end)
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
                                max(0.0, 1.0 - contact_ratio * 10.0)
                                + length_error / max(fixed_length, moving_length) * 0.05
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
        placements.append(
            {
                "piece_index": piece["index"],
                "source_center": piece["source_center"],
                "target_center": shifted_center,
                "offset": shifted_center.sub(piece["source_center"]),
                "angle": _angle_wrap(angle),
                "source_orientation": piece["source_orientation"],
                "target_orientation": _angle_wrap(piece["source_orientation"] + angle),
                "target_pts": [
                    point.add(shift)
                    for point in normalized_world_polygons[piece["index"]]
                ],
            }
        )

    rectangle = dict(rectangle)
    rectangle["center"] = desired_center
    rectangle["corners"] = [corner.add(shift) for corner in rectangle["corners"]]
    # Refine solution: slightly optimize piece positions
    refined_poses = dict(normalized_poses)
    refined_world = dict(normalized_world_polygons)
    
    for iteration in range(5):
        placed_indices = list(refined_poses.keys())
        for idx in placed_indices:
            piece = prepared[idx]
            local_pts = piece["local_points"]
            old_angle, old_center = refined_poses[idx]
            best_adj_score = float("inf")
            best_adj = (old_angle, old_center)
            # Try small perturbations
            for da in [-0.06, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.06]:
                for dx in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
                    for dy in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
                        new_angle = old_angle + da
                        new_center = Pt(old_center.x + dx, old_center.y + dy)
                        new_poly = transform_polygon(local_pts, new_angle, new_center)
                        # Check overlap with other pieces
                        overlap = False
                        for oi, opoly in refined_world.items():
                            if oi == idx: continue
                            if polygons_overlap(new_poly, opoly, overlap_tolerance_mm):
                                overlap = True
                                break
                        if overlap: continue
                        all_polys = [refined_world[oi] for oi in refined_world if oi != idx] + [new_poly]
                        all_pts = [pt for poly in all_polys for pt in poly]
                        rect = minimum_area_rectangle(all_pts)
                        if rect is None: continue
                        if not _dimensions_valid(rect, size_range_mm, dimension_tolerance_mm): continue
                        placed_area = sum(prepared[oi]["area"] for oi in refined_world if oi != idx) + piece["area"]
                        a_err = abs(rect["area"] - placed_area) / rect["area"]
                        if a_err > rectangle_area_tolerance: continue
                        # Score = area fit + edge continuity
                        adj_score = a_err * 3.0
                        if adj_score < best_adj_score:
                            best_adj_score = adj_score
                            best_adj = (new_angle, new_center)
            refined_poses[idx] = best_adj
            refined_world[idx] = transform_polygon(local_pts, best_adj[0], best_adj[1])
    
    # Recalculate placements with refined positions (apply shift to world coords)
    refined_placements = []
    for piece in prepared:
        angle, center = refined_poses[piece["index"]]
        shifted_center = center.add(shift)
        refined_placements.append({
            "piece_index": piece["index"],
            "source_center": piece["source_center"],
            "target_center": shifted_center,
            "offset": shifted_center.sub(piece["source_center"]),
            "angle": _angle_wrap(angle),
            "source_orientation": piece["source_orientation"],
            "target_orientation": _angle_wrap(piece["source_orientation"] + angle),
            "target_pts": [pt.add(shift) for pt in refined_world[piece["index"]]],
        })
    
    return {
        "placements": refined_placements,
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


class TextureScorer:
    """çº¹ç†è¿žç»­æ€§è¯„åˆ†å™¨ (v3)."""

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
        ss = []; gs = ns = os_ = ps = 0.0; fb_flag = False; total_weight = 0.0
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
        if total_weight <= 1e-6:
            total_weight = float(max(1, len(ss)))
        return self._asm(
            {"g": gs/total_weight, "n": ns/total_weight, "o": os_/total_weight, "p": ps/total_weight},
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
                seams.append({"i":pi,"j":pj,"mi":mi,"mj":mj,"r":reg,"e":cv2.bitwise_and(ei,ej)})
        return seams

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
        r = self.strip_r; mi,mj = s["mi"],s["mj"]; reg,edg = s["r"],s["e"]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        di = cv2.dilate(mi,k,iterations=r); dj = cv2.dilate(mj,k,iterations=r)
        si = cv2.bitwise_and(cv2.bitwise_and(di,cv2.bitwise_not(mj)), reg)
        sj = cv2.bitwise_and(cv2.bitwise_and(dj,cv2.bitwise_not(mi)), reg)
        if np.sum(si>0)<10 or np.sum(sj>0)<10:
            fb = self._full_rect(canvas, {0: cv2.bitwise_or(mi,mj)}, pattern_layers)
            return {"g":fb["g"],"n":fb["n"],"o":fb["o"],"p":fb["p"],"fb":True,"weight":1.0}
        # Gradient
        gm = self._color_gradient_mag(canvas)
        sg = np.median(gm[reg>0]) if np.sum(reg>0)>0 else 0.0
        er = cv2.erode(mi,k,iterations=r*2)
        inner = cv2.bitwise_and(mi,cv2.bitwise_not(er))
        ig_ = np.median(gm[inner>0]) if np.sum(inner>0)>0 else 1.0
        grad_s = min(sg/max(ig_,1.0),2.0)/2.0
        ncc_s = self._ncc_along_edge(canvas, si, sj, edg, r)
        orb_s = self._orb(canvas, si, sj)
        pattern_s, pattern_strength = self._pattern_consistency(pattern_layers, si, sj, edg, r)
        seam_pixels = float(max(cv2.countNonZero(edg), cv2.countNonZero(reg)))
        texture_energy = float(np.median(gm[cv2.bitwise_or(si, sj) > 0])) if np.any(cv2.bitwise_or(si, sj) > 0) else 0.0
        weight = max(1.0, seam_pixels) * (0.25 + min(texture_energy / 24.0, 1.0) + 0.8 * pattern_strength)
        return {
            "g":float(grad_s),"n":float(ncc_s),"o":float(orb_s),"p":float(pattern_s),
            "fb":False,"weight":float(weight),
            "pattern_strength":float(pattern_strength),
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

    def _ncc_along_edge(self, canvas, si, sj, edge, r):
        """æ²¿æŽ¥ç¼çº¿é€ç‚¹ NCCï¼ˆå‘é‡åŒ–ï¼Œæ—  Python å¾ªçŽ¯ï¼‰ã€‚"""
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
        if np.sum(c>0) < 50: return {"g":0.5,"n":0.5,"o":0.5,"p":0.5}
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
        if pattern_layers:
            red = pattern_layers.get("red")
            black = pattern_layers.get("black")
            if red is not None and black is not None and np.sum(c > 0) > 0:
                red_density = float(np.mean(red[c > 0] > 0))
                black_density = float(np.mean(black[c > 0] > 0))
                pattern_score = 0.5 + min(0.25, 0.8 * max(red_density, black_density))
        return {"g":float(grad),"n":0.5,"o":float(orb_s),"p":float(pattern_score)}

    def _asm(self, sc, ss, ff, fs):
        g,n,o,p = sc["g"],sc["n"],sc["o"],sc.get("p", 0.5)
        tex = self.w_g*(1.0-g) + self.w_n*n + self.w_o*o + self.w_p*p
        if ff:
            tex -= self.full_rect_penalty
        if fs:
            tex -= self.sparse_seam_penalty
        return {"texture_score":float(max(0.0,min(1.0,tex))),
                "gradient_discontinuity":float(g),"ncc_similarity":float(n),
                "orb_consistency":float(o),"pattern_consistency":float(p),"seam_scores":ss,
                "fallback_full_rect":ff,"fallback_sparse_seam":fs}

    def _def(self, reason):
        return {"texture_score":0.5,"gradient_discontinuity":0.0,
                "ncc_similarity":0.0,"orb_consistency":0.0,"pattern_consistency":0.0,
                "seam_scores":[],"fallback_full_rect":False,
                "fallback_sparse_seam":False,"error":reason}


if __name__ == "__main__":
    print("TextureScorer v3 loaded. OpenCV", cv2.__version__)


# ---- Public API: joint solver ----

def solve_with_texture(pieces_geometry, pieces_texture, target_center=None,
                       *, mm_per_px=10.0, strip_width_mm=5.0,
                       lambda_texture=0.5, top_k=1, geometry_kwargs=None,
                       gw=0.35, nw=0.30, ow=0.35, pw=0.45,
                       fallback_full_rect_penalty=0.12,
                       fallback_sparse_seam_penalty=0.08,
                       angle_perturb_rad=0.035,
                       translate_perturb_mm=1.2):
    """Geometry + texture joint solver with top-K re-ranking.

    1. Run geometric solver to find candidate pose(s)
    2. Generate perturbed variants for mirror/symmetry resolution
    3. Score each with texture continuity
    4. Re-rank by J_total = J_shape + lambda * J_texture

    Returns dict with best_solution, texture scores, and candidates list.
    """
    if geometry_kwargs is None:
        geometry_kwargs = {}

    def _ensure_Pt(p):
        if isinstance(p, Pt): return p
        return Pt(p[0], p[1])

    # Step 1: Find geometric solution
    sol = find_rectangle_solution(
        pieces_geometry, target_center, **geometry_kwargs)
    if sol is None:
        return {"error": "no geometric solution"}

    candidates = [sol]

    # Step 2: Generate perturbed alternatives for top-K
    if top_k > 1 and 'poses' in sol and 'world_polygons' in sol:
        base_poses = sol['poses']
        # Build local_points for each piece
        local_pts = {}
        for pi, p in enumerate(pieces_geometry):
            ct = polygon_centroid(p['pts'])
            local_pts[pi] = [pt.sub(ct) for pt in p['pts']]
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
        for attempt in range(max(0, top_k - 1)):
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
            alt = dict(sol)
            alt['poses'] = perturbed_poses
            alt['world_polygons'] = perturbed_world
            alt['score'] = sol['score'] + 0.05
            # Copy placements with updated positions
            alt_placements = copy.deepcopy(sol.get('placements', []))
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

    # Step 3: Texture score each candidate
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
    results = []
    for i, cs in enumerate(candidates):
        tex = scorer.score_solution(cs, pieces_texture)
        j_shape = cs.get("score", 0.0)
        j_total = scorer.total_score(j_shape, tex)
        results.append({
            "geometry": cs, "texture": tex,
            "j_shape": j_shape,
            "j_texture": 1.0 - tex["texture_score"],
            'j_total': j_total,
        })

    results.sort(key=lambda c: c['j_total'])
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
