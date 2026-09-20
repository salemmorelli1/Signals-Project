"""Regression tests for repeated-measures analysis integrity."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from scripts.analyze_joint_results import holm_adjust, multivariate_within_seed
from scripts.build_site import SITE_METRICS, simulation_summaries


class AnalysisIntegrityTests(unittest.TestCase):
    def test_holm_adjustment_matches_known_example(self) -> None:
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.06]))

    def test_holm_adjustment_rejects_invalid_probabilities(self) -> None:
        for values in (np.array([]), np.array([float("nan")]), np.array([-0.1, 0.2])):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    holm_adjust(values)

    def test_multivariate_analysis_rejects_an_incomplete_design(self) -> None:
        rows = []
        architectures = ("Analytical joint score", "Amortized joint surrogate")
        for seed in range(5):
            for architecture in architectures:
                for snr in (-5.0, 5.0, 15.0):
                    for channel in ("LOS", "MP", "NLOS"):
                        rows.append(
                            {
                                "seed": seed,
                                "architecture": architecture,
                                "snr_db": snr,
                                "channel": channel,
                                "mse_frequency": 1.0 + seed,
                            }
                        )
        frame = pd.DataFrame(rows).iloc[:-1]
        with self.assertRaisesRegex(ValueError, "Incomplete factorial design"):
            multivariate_within_seed(frame, "mse_frequency", "log")

    def test_dashboard_all_channel_interval_uses_independent_seed_blocks(self) -> None:
        rows = []
        for seed in range(2):
            for architecture in ("Analytical joint score", "Amortized joint surrogate"):
                for channel, base in (("LOS", 0.0), ("MP", 100.0)):
                    value = base + 10.0 * seed
                    row = {
                        "seed": seed,
                        "architecture": architecture,
                        "snr_db": 5.0,
                        "channel": channel,
                    }
                    row.update({metric: value for metric in SITE_METRICS})
                    rows.append(row)
        summaries = simulation_summaries(pd.DataFrame(rows))
        overall = next(
            row
            for row in summaries
            if row["architecture"] == "Analytical joint score"
            and row["snr_db"] == 5.0
            and row["channel"] == "All"
        )
        expected_half_width = float(student_t.ppf(0.975, 1) * 5.0)
        self.assertEqual(overall["n_blocks"], 2)
        self.assertAlmostEqual(float(overall["mse_frequency_mean"]), 55.0)
        self.assertAlmostEqual(float(overall["mse_frequency_ci95"]), expected_half_width)


if __name__ == "__main__":
    unittest.main()
