import math
import unittest

from scara_kinematics import KinematicsError, ScaraParameters
from serial_protocol import (
    FrameParser,
    VISION_RESULT,
    decode_flags,
    pack_result,
    pack_start,
    unpack_result,
)


class SerialProtocolTests(unittest.TestCase):
    def test_start_frame_round_trip(self):
        parser = FrameParser()
        frames = parser.feed(b"noise" + pack_start(37))
        self.assertEqual(len(frames), 1)
        sequence, count, index = decode_flags(frames[0].flags)
        self.assertEqual(frames[0].command, 0x0201)
        self.assertEqual((sequence, count, index), (37, 0, 0))

    def test_result_frames_support_two_to_four_pieces(self):
        parser = FrameParser()
        for piece_count in (2, 3, 4):
            for piece_index in range(piece_count):
                frame = pack_result(
                    9,
                    piece_count,
                    piece_index,
                    0.1,
                    0.3,
                    0.2,
                    0.4,
                    0.5,
                    0.6,
                )
                # 模拟串口分成多个包到达。
                parsed = parser.feed(frame[:3]) + parser.feed(frame[3:])
                self.assertEqual(len(parsed), 1)
                sequence, count, index, values = unpack_result(parsed[0])
                self.assertEqual((sequence, count, index), (9, piece_count, piece_index))
                self.assertEqual(len(values), 6)
                for actual, expected in zip(values, [0.1, 0.3, 0.2, 0.4, 0.5, 0.6]):
                    self.assertAlmostEqual(actual, expected, places=6)
                self.assertTrue(all(math.isfinite(value) for value in values))


class KinematicsTests(unittest.TestCase):
    def make_parameters(self):
        return ScaraParameters(
            link1_mm=100.0,
            link2_mm=100.0,
            camera_rotation_rad=0.0,
            camera_tx_mm=0.0,
            camera_ty_mm=0.0,
            j1_zero_offset_rad=0.0,
            j2_zero_offset_rad=0.0,
            j1_direction=1,
            j2_direction=1,
            j1_min_rad=-math.pi,
            j1_max_rad=math.pi,
            j2_min_rad=-math.pi,
            j2_max_rad=math.pi,
            elbow_branch=1,
        )

    def test_reachable_target(self):
        parameters = self.make_parameters()
        j1, j2 = parameters.solve(100.0, 100.0)
        self.assertAlmostEqual(j1, 0.0, places=5)
        self.assertAlmostEqual(j2, math.pi / 2.0, places=5)

    def test_unreachable_target(self):
        with self.assertRaises(KinematicsError):
            self.make_parameters().solve(250.0, 0.0)


if __name__ == "__main__":
    unittest.main()
