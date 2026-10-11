"""Structural tests for the WiSig real-data adapter.

The miniature arrays below are generated fixtures. They test schema and
pipeline invariants only; they are never represented as WiSig evidence.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.signals_project.wisig_realdata import (
    build_overlap_manifest,
    load_compact_dataset,
    materialize_overlaps,
    prepare_dataset,
    sha256_file,
    split_domains,
    validate_compact_dataset,
    verify_file_sha256,
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
    def test_invalid_power_cannot_replace_a_prior_preparation(self) -> None:
        for tx in self.dataset["data"]:
            for rx in tx:
                for day in rx:
                    day[0][:] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pkl"
            path.write_bytes(pickle.dumps(self.dataset, protocol=pickle.HIGHEST_PROTOCOL))
            output = Path(tmp) / "derived"
            output.mkdir()
            names = ("wisig_overlap_manifest.csv", "wisig_digital_overlaps.npz", "wisig_preparation_metadata.json")
            for name in names:
                (output / name).write_bytes(b"previous verified preparation")
            with self.assertRaisesRegex(ValueError, "Invalid signal power"):
                prepare_dataset(
                    path, output, mixtures_per_domain=1, trust_official_pickle=True,
                    expected_sha256=sha256_file(path),
                )
            for name in names:
                self.assertEqual((output / name).read_bytes(), b"previous verified preparation")

    def test_manifest_only_cannot_pair_new_metadata_with_an_old_overlap_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pkl"
            path.write_bytes(pickle.dumps(self.dataset, protocol=pickle.HIGHEST_PROTOCOL))
            output = Path(tmp) / "derived"
            output.mkdir()
            (output / "wisig_digital_overlaps.npz").write_bytes(b"previous archive")
            with self.assertRaisesRegex(ValueError, "existing overlap archive"):
                prepare_dataset(
                    path, output, mixtures_per_domain=1, materialize=False,
                    trust_official_pickle=True, expected_sha256=sha256_file(path),
                )
            self.assertFalse((output / "wisig_overlap_manifest.csv").exists())

    def test_recording_indices_cannot_wrap_or_truncate(self) -> None:
        for value in (-1, 0.5, True, "-1", 99999):
            with self.subTest(value=value):
                manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=1, seed=7)
                manifest[0]["signal_a_index"] = value
                with self.assertRaisesRegex(ValueError, "recording"):
                    materialize_overlaps(self.dataset, manifest)

    def test_manifest_rejects_lineage_and_split_tampering(self) -> None:
        for field, value in (("tx_a", "wrong-tx"), ("receiver", "wrong-rx"), ("split", "unknown")):
            with self.subTest(field=field):
                manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=1, seed=7)
                manifest[0][field] = value
                with self.assertRaises(ValueError):
                    materialize_overlaps(self.dataset, manifest)
        manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=1, seed=7)
        manifest[1]["mixture_id"] = manifest[0]["mixture_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            materialize_overlaps(self.dataset, manifest)

    def test_materialization_enforces_receiver_and_day_holdouts(self) -> None:
        for split, replacement, error in (
            ("test", "train", "Test receivers"),
            ("validation", "train", "Validation capture days"),
        ):
            with self.subTest(split=split):
                manifest = build_overlap_manifest(self.dataset, mixtures_per_domain=1, seed=7)
                next(row for row in manifest if row["split"] == split)["split"] = replacement
                with self.assertRaisesRegex(ValueError, error):
                    materialize_overlaps(self.dataset, manifest)

    def test_duplicate_acquisition_labels_are_rejected(self) -> None:
        self.dataset["rx_list"][1] = self.dataset["rx_list"][0]
        with self.assertRaisesRegex(ValueError, "unique acquisition identifiers"):
            validate_compact_dataset(self.dataset)

    def test_pickle_parser_uses_the_verified_snapshot_after_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pkl"
            path.write_bytes(pickle.dumps(self.dataset, protocol=pickle.HIGHEST_PROTOCOL))
            digest = sha256_file(path)

            def replace_after_verification(actual, expected):
                self.assertEqual(actual, expected)
                path.write_bytes(pickle.dumps({"unverified": True}))
                return True

            with patch(
                "src.signals_project.wisig_realdata.hmac.compare_digest",
                side_effect=replace_after_verification,
            ):
                loaded = load_compact_dataset(
                    path, trust_official_pickle=True, expected_sha256=digest
                )
            self.assertEqual(loaded["tx_list"], self.dataset["tx_list"])

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

    def test_invalid_manifest_parameters_fail_closed(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(mixtures_per_domain=value):
                with self.assertRaises(ValueError):
                    build_overlap_manifest(self.dataset, mixtures_per_domain=value)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_overlap_manifest(self.dataset, sir_levels_db=(0.0, float("nan")))
        with self.assertRaises(ValueError):
            materialize_overlaps(self.dataset, [])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_manifest(Path(tmp) / "empty.csv", [])

    def test_pickle_requires_acknowledgement_and_pre_unpickle_hash_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pkl"
            with path.open("wb") as handle:
                pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            digest = sha256_file(path)

            with self.assertRaisesRegex(ValueError, "trust_official_pickle"):
                load_compact_dataset(path, expected_sha256=digest)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_compact_dataset(
                    path,
                    trust_official_pickle=True,
                    expected_sha256="0" * 64,
                )
            loaded = load_compact_dataset(
                path,
                trust_official_pickle=True,
                expected_sha256=digest,
            )
            self.assertEqual(loaded["tx_list"], self.dataset["tx_list"])

    def test_prepare_dataset_cannot_bypass_pickle_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pkl"
            output = Path(tmp) / "derived"
            with path.open("wb") as handle:
                pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            digest = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "trust_official_pickle"):
                prepare_dataset(
                    path,
                    output,
                    materialize=False,
                    expected_sha256=digest,
                )
            metadata = prepare_dataset(
                path,
                output,
                mixtures_per_domain=1,
                materialize=False,
                trust_official_pickle=True,
                expected_sha256=digest,
            )
            self.assertEqual(metadata["dataset_sha256"], digest)
            self.assertTrue(metadata["integrity_verified_before_unpickle"])

    def test_hash_validation_rejects_invalid_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                sha256_file(path, block_size=0)
            with self.assertRaises(ValueError):
                verify_file_sha256(path, "not-a-sha256")


if __name__ == "__main__":
    unittest.main()
