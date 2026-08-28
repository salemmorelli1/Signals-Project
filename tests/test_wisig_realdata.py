"""Structural tests for the WiSig real-data adapter.

The miniature arrays below are generated fixtures. They test schema and
pipeline invariants only; they are never represented as WiSig evidence.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.signals_project.wisig_realdata import (
    build_overlap_manifest,
    materialize_overlaps,
    split_domains,
    validate_compact_dataset,
    write_manifest,
)


def fixture_dataset() -> dict:
    rng = np.random.default_rng(17)
    n_tx, n_rx, n_day, n_sig = 4, 5, 3, 7
    data = []
    for _tx in range(n_tx):
        tx_axis = []
        for _rx in range(n_rx):
            rx_axis = []
            for _day in range(n_day):
                non_equalized = rng.normal(size=(n_sig, 256, 2)).astype(np.float32)
                equalized = rng.normal(size=(n_sig, 256, 2)).astype(np.float32)
                rx_axis.append([non_equalized, equalized])
            tx_axis.append(rx_axis)
        data.append(tx_axis)
    return {
        "tx_list": [f"tx-{i}" for i in range(n_tx)],
        "rx_list": [f"rx-{i}" for i in range(n_rx)],
        "capture_date_list": [f"day-{i}" for i in range(n_day)],
        "equalized_list": [0, 1],
        "data": data,
    }


class WiSigAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = fixture_dataset()

    def test_compact_schema(self) -> None:
        summary = validate_compact_dataset(self.dataset)
        self.assertEqual(summary["transmitters"], 4)
        self.assertEqual(summary["receivers"], 5)
        self.assertEqual(summary["samples_per_signal"], 256)

    def test_receiver_day_splits_are_disjoint(self) -> None:
        plan = split_domains(self.dataset, seed=2026)
        test_receivers = {rx for (rx, _day), split in plan.items() if split == "test"}
        training_receivers = {rx for (rx, _day), split in plan.items() if split == "train"}
        validation_days = {day for (_rx, day), split in plan.items() if split == "validation"}
        training_days = {day for (_rx, day), split in plan.items() if split == "train"}
        self.assertTrue(test_receivers)
        self.assertTrue(validation_days)
        self.assertTrue(test_receivers.isdisjoint(training_receivers))
        self.assertTrue(validation_days.isdisjoint(training_days))

    def test_manifest_is_deterministic_and_uses_distinct_transmitters(self) -> None:
        first = build_overlap_manifest(self.dataset, mixtures_per_domain=3, seed=99)
        second = build_overlap_manifest(self.dataset, mixtures_per_domain=3, seed=99)
        self.assertEqual(first, second)
        self.assertTrue(all(row["tx_a_index"] != row["tx_b_index"] for row in first))
        self.assertTrue(all(row["equalized"] == 0 for row in first))
        self.assertEqual(len(first), 5 * 3 * 3)

    def test_materialized_overlap_has_exact_component_truth(self) -> None:
        manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=2, seed=7)
        source_a, source_b, mixture = materialize_overlaps(self.dataset, manifest)
        self.assertEqual(mixture.shape[1:], (256, 2))
        np.testing.assert_array_equal(mixture, source_a + source_b)
        for index, row in enumerate(manifest):
            rms_a = np.sqrt(np.mean(np.sum(source_a[index] ** 2, axis=-1)))
            rms_b = np.sqrt(np.mean(np.sum(source_b[index] ** 2, axis=-1)))
            measured_sir = 20.0 * np.log10(rms_a / rms_b)
            self.assertAlmostEqual(measured_sir, row["sir_db"], places=5)

    def test_manifest_csv_round_trip_shape(self) -> None:
        manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=1, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            write_manifest(path, manifest)
            with path.open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), len(manifest) + 1)


if __name__ == "__main__":
    unittest.main()
