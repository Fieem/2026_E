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
    wrist_zero_offset_rad: float
    wrist_direction: int
    pick_wrist_joint_rad: float
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
            wrist_zero_offset_rad=float(data.get("wrist_zero_offset_rad", 0.0)),
            wrist_direction=1 if int(data.get("wrist_direction", 1)) >= 0 else -1,
            pick_wrist_joint_rad=float(data.get("pick_wrist_joint_rad", 0.0)),
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

    @staticmethod
    def wrap_to_pi(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi

    def orientation_camera_to_base(self, angle_rad: float) -> float:
        """把相机/工作区坐标中的绝对朝向转换到基座坐标。"""

        return self.wrap_to_pi(angle_rad + self.camera_rotation_rad)

    def j1_command_to_geometric(self, command_q1_rad: float) -> float:
        """把 J1 命令角还原为几何建模中的关节角。"""

        return self.wrap_to_pi(
            self.j1_zero_offset_rad + self.j1_direction * command_q1_rad
        )

    def j2_command_to_geometric(self, command_q2_rad: float) -> float:
        """把 J2 命令角还原为几何建模中的相对关节角。"""

        return self.wrap_to_pi(
            self.j2_zero_offset_rad + self.j2_direction * command_q2_rad
        )

    def wrist_from_absolute(
        self,
        absolute_orientation_rad: float,
        j1_command_rad: float,
        j2_command_rad: float,
    ) -> float:
        """把末端绝对朝向转换成腕关节角。

        当前默认机构定义为：末端绝对朝向 = J1 + J2 + Wrist，
        其中 J2 为相对 J1 的关节角。
        """

        absolute_orientation_base_rad = self.orientation_camera_to_base(
            absolute_orientation_rad
        )
        geometric_q1 = self.j1_command_to_geometric(j1_command_rad)
        geometric_q2 = self.j2_command_to_geometric(j2_command_rad)
        wrist_joint_rad = self.wrap_to_pi(
            absolute_orientation_base_rad - geometric_q1 - geometric_q2
        )
        return self.wrap_to_pi(
            self.wrist_direction * (wrist_joint_rad - self.wrist_zero_offset_rad)
        )

    def wrist_command_from_joint(self, wrist_joint_rad: float) -> float:
        """把腕关节几何角转换为腕部执行命令角。"""

        return self.wrap_to_pi(
            self.wrist_direction * (wrist_joint_rad - self.wrist_zero_offset_rad)
        )

    def fixed_pick_wrist_command(self) -> float:
        """抓取阶段使用固定腕角。"""

        return self.wrist_command_from_joint(self.pick_wrist_joint_rad)

    def fixed_pick_absolute_orientation_base(
        self,
        pick_j1_command_rad: float,
        pick_j2_command_rad: float,
    ) -> float:
        """固定抓取腕角时，末端在抓取瞬间的绝对朝向。"""

        geometric_q1 = self.j1_command_to_geometric(pick_j1_command_rad)
        geometric_q2 = self.j2_command_to_geometric(pick_j2_command_rad)
        return self.wrap_to_pi(
            geometric_q1 + geometric_q2 + self.pick_wrist_joint_rad
        )

    def place_wrist_from_pick_and_target(
        self,
        source_absolute_orientation_rad: float,
        target_absolute_orientation_rad: float,
        pick_j1_command_rad: float,
        pick_j2_command_rad: float,
        place_j1_command_rad: float,
        place_j2_command_rad: float,
    ) -> float:
        """固定抓取腕角时，根据抓取前碎片方向和目标方向计算放置腕角。

        碎片被吸起后，相对末端保留抓取瞬间的相对角度，因此放置时需要满足：
        target_piece_abs = place_tool_abs + (source_piece_abs - pick_tool_abs)
        """

        source_abs_base = self.orientation_camera_to_base(
            source_absolute_orientation_rad
        )
        target_abs_base = self.orientation_camera_to_base(
            target_absolute_orientation_rad
        )
        pick_tool_abs_base = self.fixed_pick_absolute_orientation_base(
            pick_j1_command_rad,
            pick_j2_command_rad,
        )
        place_tool_abs_base = self.wrap_to_pi(
            target_abs_base - source_abs_base + pick_tool_abs_base
        )
        geometric_q1_place = self.j1_command_to_geometric(place_j1_command_rad)
        geometric_q2_place = self.j2_command_to_geometric(place_j2_command_rad)
        place_wrist_joint_rad = self.wrap_to_pi(
            place_tool_abs_base - geometric_q1_place - geometric_q2_place
        )
        return self.wrist_command_from_joint(place_wrist_joint_rad)
