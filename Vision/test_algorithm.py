import math
import random
import unittest

from algorithm import (
    Pt,
    find_rectangle_solution,
    generate_puzzle,
    polygon_area,
    scatter_pieces,
)


class RectangleSolverTests(unittest.TestCase):
    def test_random_exact_puzzles(self):
        for seed in range(30):
            with self.subTest(seed=seed):
                random.seed(seed)
                puzzle = generate_puzzle(0.0, 0.0, 100.0, 70.0)
                observed = scatter_pieces(puzzle["pieces"])
                solution = find_rectangle_solution(
                    observed,
                    Pt(0.0, 0.0),
                    edge_tolerance_mm=0.01,
                    edge_relative_tolerance=0.001,
                    overlap_tolerance_mm=0.001,
                    rectangle_area_tolerance=0.001,
                    dimension_tolerance_mm=0.01,
                )
                self.assertIsNotNone(solution)
                self.assertLess(solution["area_error"], 1e-5)
                self.assertEqual(len(solution["placements"]), len(observed))

                rectangle = solution["rectangle"]
                sides = sorted((rectangle["width"], rectangle["height"]))
                self.assertAlmostEqual(sides[0], 70.0, places=4)
                self.assertAlmostEqual(sides[1], 100.0, places=4)

                target_area = sum(
                    polygon_area(placement["target_pts"])
                    for placement in solution["placements"]
                )
                self.assertAlmostEqual(target_area, 7000.0, places=4)

    def test_one_millimetre_vertex_noise(self):
        """摄像头提取的角点不需要将接缝恢复得完全一致。"""

        for seed in range(10):
            with self.subTest(seed=seed):
                random.seed(seed)
                puzzle = generate_puzzle(0.0, 0.0, 100.0, 70.0)
                noisy_pieces = [
                    {
                        "pts": [
                            Pt(
                                point.x + random.uniform(-1.0, 1.0),
                                point.y + random.uniform(-1.0, 1.0),
                            )
                            for point in piece["pts"]
                        ]
                    }
                    for piece in puzzle["pieces"]
                ]
                observed = scatter_pieces(noisy_pieces)
                solution = find_rectangle_solution(observed, Pt(0.0, 0.0))

                self.assertIsNotNone(solution)
                self.assertLessEqual(solution["area_error"], 0.06)
                self.assertEqual(len(solution["placements"]), len(observed))

    def test_rejects_non_rectangular_set(self):
        triangles = [
            {"pts": [Pt(0, 0), Pt(30, 0), Pt(0, 20)]},
            {"pts": [Pt(100, 100), Pt(120, 100), Pt(120, 110)]},
        ]
        solution = find_rectangle_solution(
            triangles,
            size_range_mm=None,
            edge_tolerance_mm=0.01,
            rectangle_area_tolerance=0.001,
        )
        self.assertIsNone(solution)


if __name__ == "__main__":
    unittest.main()
