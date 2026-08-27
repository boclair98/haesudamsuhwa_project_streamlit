import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from desalination.analytics import (
    anomaly_watchlist,
    model_explainability,
    monthly_summary,
    percentile_rank,
    regression_metrics,
)


class AnalyticsTests(unittest.TestCase):
    def test_model_explainability_exposes_signed_and_relative_signals(self):
        pressure = SimpleNamespace(
            feature_names_in_=np.array(["수온", "수소이온농도"]),
            coef_=np.array([0.25, -0.08]),
        )
        sec = SimpleNamespace(
            feature_names_in_=np.array(["총인", "탁도"]),
            feature_importances_=np.array([0.7, 0.3]),
        )

        signals = model_explainability(pressure, sec)

        self.assertListEqual(signals["pressure"]["feature"].tolist(), ["수온", "수소이온농도"])
        self.assertAlmostEqual(float(signals["pressure"].loc[1, "value"]), -0.08)
        self.assertAlmostEqual(float(signals["sec"]["value"].sum()), 1.0)

    def test_regression_metrics_are_exact_for_perfect_fit(self):
        metrics = regression_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(metrics.mae, 0)
        self.assertEqual(metrics.rmse, 0)
        self.assertEqual(metrics.r2, 1)

    def test_percentile_rank_is_bounded(self):
        self.assertEqual(percentile_rank([1, 2, 3, 4], 0), 0)
        self.assertEqual(percentile_rank([1, 2, 3, 4], 4), 100)

    def test_monthly_summary_counts_only_exact_hours(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-01-01 00:00", "2021-01-01 00:30"]),
                "pressure_stage1_bar": [50.0, 52.0],
                "pressure_stage2_bar": [20.0, 21.0],
                "model_pressure_bar": [50.5, 51.5],
                "sec_kwh_m3": [3.0, 3.1],
                "model_sec_kwh_m3": [3.0, 3.0],
                "tds_stage1_mg_l": [80.0, 82.0],
                "tds_stage2_mg_l": [1.0, 1.1],
                "turbidity_ntu": [1.0, 2.0],
                "cod_mg_l": [1.0, 2.0],
                "total_n_mg_l": [0.1, 0.2],
                "total_p_mg_l": [0.01, 0.02],
            }
        )
        summary = monthly_summary(frame)
        self.assertEqual(int(summary.loc[0, "samples"]), 2)
        self.assertEqual(int(summary.loc[0, "exact_hour_samples"]), 1)
        self.assertTrue(np.isfinite(summary.loc[0, "exact_hour_density_pct"]))

    def test_anomaly_watchlist_returns_data_derived_review_queue(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2021-01-01", periods=20, freq="h"),
                "pressure_error_bar": [0.1] * 19 + [4.0],
                "sec_error_kwh_m3": [0.01] * 19 + [0.8],
            }
        )
        watchlist, thresholds = anomaly_watchlist(frame)
        self.assertEqual(len(watchlist), 1)
        self.assertEqual(watchlist.loc[0, "watch_reason"], "압력 차이 · SEC 차이")
        self.assertGreater(float(thresholds["pressure_cutoff"]), 0.1)
        self.assertLess(float(thresholds["pressure_cutoff"]), 4.0)
        self.assertGreater(float(thresholds["sec_cutoff"]), 0.01)
        self.assertLess(float(thresholds["sec_cutoff"]), 0.8)

    def test_anomaly_watchlist_rejects_invalid_quantile(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2021-01-01"]),
                "pressure_error_bar": [1.0],
                "sec_error_kwh_m3": [0.1],
            }
        )
        with self.assertRaises(ValueError):
            anomaly_watchlist(frame, quantile=1.0)


if __name__ == "__main__":
    unittest.main()

