import unittest

from desalination.domain import (
    classify_energy,
    production_progress,
    quality_achievement,
)


class EnergyStatusTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(classify_energy(3.49).level, "normal")
        self.assertEqual(classify_energy(3.5).level, "warning")
        self.assertEqual(classify_energy(3.7).level, "warning")
        self.assertEqual(classify_energy(3.71).level, "danger")


class QualityAchievementTests(unittest.TestCase):
    def test_uses_reduction_against_required_reduction(self):
        self.assertAlmostEqual(quality_achievement(5, 3, 1), 0.75)
        self.assertAlmostEqual(quality_achievement(1.37, 0.37, 1), 1.0)

    def test_clamps_result_and_handles_already_compliant_water(self):
        self.assertEqual(quality_achievement(0.5, 0, 1), 1.0)
        self.assertEqual(quality_achievement(5, 10, 1), 1.0)
        self.assertEqual(quality_achievement(5, -1, 1), 0.0)


class ProductionProgressTests(unittest.TestCase):
    def test_progress_is_bounded(self):
        self.assertEqual(production_progress(0, 0), 0.0)
        self.assertLessEqual(production_progress(23, 59), 100.0)

    def test_rejects_invalid_time(self):
        with self.assertRaises(ValueError):
            production_progress(24, 0)


if __name__ == "__main__":
    unittest.main()
