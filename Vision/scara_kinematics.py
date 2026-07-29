"""SCARA 二连杆逆运动学和相机坐标转换。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


class KinematicsError(ValueError):
    """目标点不可达或运动学参数不完整。"""


@dataclass(frozen=True)
class ScaraParameters:
    link1_mm: float
    link2_mm: float
    camera_rotation_rad: float
    camera_tx_mm: float
    camera_ty_mm: float
    j1_zero_offset_rad: float
    j2_zero_offset_rad: float
    j1_direction: int
    j2_direction: int
    j1_min_rad: float
    j1_max_rad: float
    j2_min_rad: float
    j2_max_rad: float
    elbow_branch: int

    @classmethod
    def from_config(cls, config: Dict) -> "ScaraParameters":
        data = config.get("scara", {})
        required = ("link1_mm", "link2_mm")
        if any(float(data.get(key, 0.0)) <= 0.0 for key in required):
            raise KinematicsError("SCARA 连杆长度尚未配置")
        return cls(
            link1_mm=float(data["link1_mm"]),
            link2_mm=float(data["link2_mm"]),
            camera_rotation_rad=float(data.get("camera_to_base", {}).get("rotation_rad", 0.0)),
            camera_tx_mm=float(data.get("camera_to_base", {}).get("tx_mm", 0.0)),
            camera_ty_mm=float(data.get("camera_to_base", {}).get("ty_mm", 0.0)),
            j1_zero_offset_rad=float(data.get("j1_zero_offset_rad", 0.0)),
            j2_zero_offset_rad=float(data.get("j2_zero_offset_rad", 0.0)),
            j1_direction=1 if int(data.get("j1_direction", 1)) >= 0 else -1,
            j2_direction=1 if int(data.get("j2_direction", 1)) >= 0 else -1,
            j1_min_rad=float(data.get("j1_min_rad", -2.8)),
            j1_max_rad=float(data.get("j1_max_rad", 2.8)),
            j2_min_rad=float(data.get("j2_min_rad", -2.8)),
            j2_max_rad=float(data.get("j2_max_rad", 2.8)),
            elbow_branch=1 if int(data.get("elbow_branch", 1)) >= 0 else -1,
        )

    def camera_to_base(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        cosine = math.cos(self.camera_rotation_rad)
        sine = math.sin(self.camera_rotation_rad)
        return (
            cosine * x_mm - sine * y_mm + self.camera_tx_mm,
            sine * x_mm + cosine * y_mm + self.camera_ty_mm,
        )

    def solve(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """返回可直接交给关节控制器的 J1/J2 弧度。J2 默认是相对 J1 的角度。"""

        x, y = self.camera_to_base(x_mm, y_mm)
        radius_squared = x * x + y * y
        cosine_q2 = (
            radius_squared - self.link1_mm**2 - self.link2_mm**2
        ) / (2.0 * self.link1_mm * self.link2_mm)
        if cosine_q2 < -1.0 - 1e-6 or cosine_q2 > 1.0 + 1e-6:
            raise KinematicsError(f"目标点不可达：({x_mm:.2f}, {y_mm:.2f}) mm")
        cosine_q2 = max(-1.0, min(1.0, cosine_q2))
        q2 = self.elbow_branch * math.acos(cosine_q2)
        q1 = math.atan2(y, x) - math.atan2(
            self.link2_mm * math.sin(q2),
            self.link1_mm + self.link2_mm * math.cos(q2),
        )

        command_q1 = self.j1_direction * (q1 - self.j1_zero_offset_rad)
        command_q2 = self.j2_direction * (q2 - self.j2_zero_offset_rad)
        if not self.j1_min_rad <= command_q1 <= self.j1_max_rad:
            raise KinematicsError(f"J1 超出限位：{command_q1:.4f} rad")
        if not self.j2_min_rad <= command_q2 <= self.j2_max_rad:
            raise KinematicsError(f"J2 超出限位：{command_q2:.4f} rad")
        return command_q1, command_q2
