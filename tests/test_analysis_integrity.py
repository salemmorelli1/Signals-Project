"""Regression tests for repeated-measures analysis integrity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image as PillowImage
from PIL.PngImagePlugin import PngInfo
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from scripts.analyze_joint_results import holm_adjust, multivariate_within_seed
from scripts.build_report import _write_png_if_pixels_changed
from scripts.build_site import SITE_METRICS, simulation_conclusion, simulation_summaries
from src.signals_project.joint_ssm import (
    canonical_artifact_float,
    paired_contrasts,
    write_csv_artifact,
    write_text_artifact,
)


class AnalysisIntegrityTests(unittest.TestCase):
    def test_published_conclusion_recomputes_numbers_and_significance(self) -> None:
        frame = self.nonsingular_frame()
        frame["latency_ms_per_sample"] = frame["mse_frequency"] / 100
        effects = multivariate_within_seed(frame, "mse_frequency", "log")
        pairs = paired_contrasts(frame, ["mse_frequency"])
        html = simulation_conclusion(frame, effects, pairs)
        mean = frame.groupby("architecture")["mse_frequency"].mean()
        self.assertIn(f'{mean["Analytical joint score"]:.2f} Hz²', html)
        effects.loc[effects.effect == "Architecture", "p_holm_within_response"] = 0.8
        self.assertIn("No clear primary architecture effect", simulation_conclusion(frame, effects, pairs))
        effects.loc[effects.effect == "Architecture", "p_holm_within_response"] = 0.001
        self.assertIn("Primary architecture effect detected", simulation_conclusion(frame, effects, pairs))

    @staticmethod
    def nonsingular_frame() -> pd.DataFrame:
        rng = np.random.default_rng(882)
        rows = []
        for seed in range(30):
            for architecture in ("Analytical joint score", "Amortized joint surrogate"):
                for snr in (-5.0, 5.0, 15.0):
                    for channel in ("LOS", "MP", "NLOS"):
                        rows.append(dict(
                            seed=seed, architecture=architecture, snr_db=snr,
                            channel=channel, mse_frequency=20.0 + rng.normal(),
                        ))
        return pd.DataFrame(rows)

    def test_hotelling_one_coordinate_matches_paired_seed_t_test(self) -> None:
        frame = self.nonsingular_frame()
        output = multivariate_within_seed(frame, "mse_frequency", "raw")
        blocks = frame.groupby(["seed", "architecture"])["mse_frequency"].mean().unstack()
        differences = blocks["Amortized joint surrogate"] - blocks["Analytical joint score"]
        expected = ttest_1samp(differences.to_numpy(), 0.0)
        effect = output[output.effect == "Architecture"].iloc[0]
        self.assertAlmostEqual(float(effect.F), float(expected.statistic**2), places=11)
        self.assertAlmostEqual(float(effect.p_value), float(expected.pvalue), places=11)

    def test_hotelling_inference_is_invariant_to_response_units(self) -> None:
        frame = self.nonsingular_frame()
        original = multivariate_within_seed(frame, "mse_frequency", "raw")
        frame["mse_frequency"] *= 1e-9
        rescaled = multivariate_within_seed(frame, "mse_frequency", "raw")
        np.testing.assert_allclose(original.F, rescaled.F, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(original.p_value, rescaled.p_value, rtol=1e-10)

    def test_zero_variance_contrast_cannot_produce_an_ordinary_f_test(self) -> None:
        frame = self.nonsingular_frame()
        frame["mse_frequency"] = np.where(
            frame.architecture == "Analytical joint score", 2.0, 3.0
        )
        with self.assertRaisesRegex(ValueError, "positive contrast variance"):
            multivariate_within_seed(frame, "mse_frequency", "raw")

    def test_rank_deficient_covariance_cannot_produce_an_ordinary_f_test(self) -> None:
        frame = self.nonsingular_frame()
        cell = frame.groupby("seed").cumcount().to_numpy()
        frame["mse_frequency"] = (frame.seed.to_numpy() + 1) * (cell + 1.0)
        with self.assertRaisesRegex(ValueError, "full-rank contrast covariance"):
            multivariate_within_seed(frame, "mse_frequency", "raw")

    def test_identical_text_artifacts_are_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.txt"
            path.write_bytes(b"stable\n")
            with patch.object(Path, "write_text", side_effect=AssertionError("rewritten")):
                write_text_artifact(path, "stable\n")

            write_text_artifact(path, "changed\n")
            self.assertEqual(path.read_bytes(), b"changed\n")

    def test_csv_artifacts_use_lf_and_skip_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.csv"
            frame = pd.DataFrame({"value": [1.25, 2.5]})
            write_csv_artifact(frame, path, float_format="%.12g")
            original = path.read_bytes()
            self.assertNotIn(b"\r\n", original)
            with patch.object(Path, "write_text", side_effect=AssertionError("rewritten")):
                write_csv_artifact(frame, path, float_format="%.12g")

    def test_identical_png_pixels_preserve_committed_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figure.png"
            image = PillowImage.new("RGBA", (3, 2), (12, 34, 56, 255))
            metadata = PngInfo()
            metadata.add_text("fixture", "original encoding")
            image.save(path, format="PNG", pnginfo=metadata, compress_level=0)
            original_bytes = path.read_bytes()

            _write_png_if_pixels_changed(image.copy(), path)
            self.assertEqual(path.read_bytes(), original_bytes)

            changed = image.copy()
            changed.putpixel((1, 1), (99, 34, 56, 255))
            _write_png_if_pixels_changed(changed, path)
            self.assertNotEqual(path.read_bytes(), original_bytes)
            with PillowImage.open(path) as rebuilt:
                self.assertEqual(rebuilt.convert("RGBA").getpixel((1, 1)), (99, 34, 56, 255))

    def test_artifact_float_is_finite_and_canonical(self) -> None:
        self.assertEqual(canonical_artifact_float(0.7835929866351484), 0.783592986635)
        self.assertEqual(canonical_artifact_float(1.2345678901234e-106), 1.23456789012e-106)
        with self.assertRaisesRegex(ValueError, "finite"):
            canonical_artifact_float(float("nan"))

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
