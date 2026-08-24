import unittest

import numpy as np

from desalination.analytics import enrich_history
from desalination.resources import load_history, load_models


class IntegrationTests(unittest.TestCase):
    def test_real_resources_produce_finite_chained_predictions(self):
        history, profile = load_history()
        models = load_models()
        sample = enrich_history(history.head(24), models.pressure, models.sec)

        self.assertGreater(profile.usable_rows, 5_000)
        self.assertFalse(sample["timestamp"].duplicated().any())
        self.assertTrue(np.isfinite(sample["model_pressure_bar"]).all())
        self.assertTrue(np.isfinite(sample["model_sec_kwh_m3"]).all())


if __name__ == "__main__":
    unittest.main()
