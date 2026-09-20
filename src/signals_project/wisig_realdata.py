"""Leakage-safe WiSig compact-data adapter for real-I/Q external evaluation.

The WiSig compact files are Python pickle objects.  Only load a file obtained
from the official UCLA WiSig distribution and verified locally.  This module
does not download data and never commits raw captures to the repository.

WiSig transmitters were recorded separately.  We therefore create controlled
*digital* overlaps of two hardware-captured, non-equalized preambles.  The
result is useful for source-reconstruction and transfer testing, but it is not
equivalent to a simultaneous over-the-air co-channel capture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OFFICIAL_DATASET_PAGE = "https://cores.ee.ucla.edu/downloads/datasets/wisig/"
OFFICIAL_EXAMPLES_REPO = "https://github.com/WiSig-dataset/wisig-examples"
REQUIRED_KEYS = ("tx_list", "rx_list", "capture_date_list", "equalized_list", "data")
EXPECTED_MANYRX_SHA256 = "f634d90585167437d196c89b7c5a344903180bf4f5a55d02d175b8074e009d9a"
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class CaptureRef:
    tx_index: int
    rx_index: int
    day_index: int
    signal_index: int


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> str:
    """Verify a file before any parser capable of code execution sees its bytes."""
    if not isinstance(expected_sha256, str) or SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, expected_sha256.lower()):
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256.lower()}, found {actual}"
        )
    return actual


def _load_verified_compact_dataset(
    path: Path,
    *,
    trust_official_pickle: bool,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not trust_official_pickle:
        raise ValueError(
            "WiSig compact files use pickle. Pass trust_official_pickle=True only "
            "for a file obtained from the official UCLA WiSig distribution."
        )
    actual_sha256 = verify_file_sha256(path, expected_sha256)
    with path.open("rb") as handle:
        dataset = pickle.load(handle)  # nosec B301 -- gated to an acknowledged official file
    validate_compact_dataset(dataset)
    return dataset, actual_sha256


def load_compact_dataset(
    path: Path,
    *,
    trust_official_pickle: bool = False,
    expected_sha256: str = EXPECTED_MANYRX_SHA256,
) -> dict[str, Any]:
    """Load an acknowledged official WiSig pickle only after hash verification."""
    dataset, _ = _load_verified_compact_dataset(
        path,
        trust_official_pickle=trust_official_pickle,
        expected_sha256=expected_sha256,
    )
    return dataset


def validate_compact_dataset(dataset: dict[str, Any]) -> dict[str, int]:
    """Validate the public compact schema and return structural counts."""
    if not isinstance(dataset, dict):
        raise TypeError("WiSig compact dataset must be a dictionary")
    missing = [key for key in REQUIRED_KEYS if key not in dataset]
    if missing:
        raise ValueError(f"Missing WiSig keys: {', '.join(missing)}")

    n_tx = len(dataset["tx_list"])
    n_rx = len(dataset["rx_list"])
    n_day = len(dataset["capture_date_list"])
    eq_values = list(dataset["equalized_list"])
    if n_tx < 2 or n_rx < 2 or n_day < 2:
        raise ValueError("External evaluation requires at least 2 Tx, 2 Rx, and 2 days")
    if eq_values.count(0) != 1:
        raise ValueError("Exactly one non-equalized WiSig stratum (equalized=0) is required")
    if len(dataset["data"]) != n_tx:
        raise ValueError("Top-level data axis does not match tx_list")

    eq_index = eq_values.index(0)
    nonempty = 0
    signals = 0
    for tx_i in range(n_tx):
        if len(dataset["data"][tx_i]) != n_rx:
            raise ValueError(f"Receiver axis mismatch for transmitter {tx_i}")
        for rx_i in range(n_rx):
            if len(dataset["data"][tx_i][rx_i]) != n_day:
                raise ValueError(f"Day axis mismatch for Tx {tx_i}, Rx {rx_i}")
            for day_i in range(n_day):
                eq_axis = dataset["data"][tx_i][rx_i][day_i]
                if len(eq_axis) != len(eq_values):
                    raise ValueError("Equalization axis is inconsistent with equalized_list")
                array = np.asarray(eq_axis[eq_index])
                if array.size == 0:
                    continue
                if array.ndim != 3 or array.shape[1:] != (256, 2):
                    raise ValueError(
                        f"Expected (signals, 256, 2) I/Q at Tx {tx_i}, Rx {rx_i}, day {day_i}; "
                        f"found {array.shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError("Non-finite I/Q values found")
                nonempty += 1
                signals += int(array.shape[0])
    if nonempty == 0:
        raise ValueError("No non-equalized I/Q captures were found")
    return {
        "transmitters": n_tx,
        "receivers": n_rx,
        "capture_days": n_day,
        "nonempty_tx_rx_day_cells": nonempty,
        "non_equalized_signals": signals,
        "samples_per_signal": 256,
    }


def split_domains(dataset: dict[str, Any], seed: int = 2026) -> dict[tuple[int, int], str]:
    """Create disjoint receiver/day domains for training, validation, and test.

    Test uses receivers absent from training and validation. Validation uses a
    day absent from training for the remaining receivers. This produces an
    unseen-receiver test and an unseen-day validation without signal-level
    leakage.
    """
    validate_compact_dataset(dataset)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    rx_order = rng.permutation(len(dataset["rx_list"]))
    day_order = rng.permutation(len(dataset["capture_date_list"]))
    n_test_rx = max(1, math.ceil(0.20 * len(rx_order)))
    test_rx = {int(x) for x in rx_order[-n_test_rx:]}
    validation_day = day_order[-1]
    plan: dict[tuple[int, int], str] = {}
    for rx_i in range(len(dataset["rx_list"])):
        for day_i in range(len(dataset["capture_date_list"])):
            if rx_i in test_rx:
                split = "test"
            elif day_i == validation_day:
                split = "validation"
            else:
                split = "train"
            plan[(rx_i, day_i)] = split
    return plan


def _non_equalized_array(dataset: dict[str, Any], tx_i: int, rx_i: int, day_i: int) -> np.ndarray:
    eq_index = list(dataset["equalized_list"]).index(0)
    return np.asarray(dataset["data"][tx_i][rx_i][day_i][eq_index], dtype=np.float32)


def build_overlap_manifest(
    dataset: dict[str, Any],
    *,
    mixtures_per_domain: int = 25,
    seed: int = 2026,
    sir_levels_db: Iterable[float] = (-5.0, 0.0, 5.0),
) -> list[dict[str, Any]]:
    """Pair distinct transmitters within a receiver/day domain deterministically."""
    validate_compact_dataset(dataset)
    if (
        isinstance(mixtures_per_domain, bool)
        or not isinstance(mixtures_per_domain, (int, np.integer))
        or mixtures_per_domain < 1
    ):
        raise ValueError("mixtures_per_domain must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    sir_levels = tuple(float(x) for x in sir_levels_db)
    if not sir_levels or not np.isfinite(sir_levels).all():
        raise ValueError("At least one finite SIR level is required")

    rng = np.random.default_rng(seed)
    plan = split_domains(dataset, seed)
    rows: list[dict[str, Any]] = []
    mixture_number = 0
    for rx_i in range(len(dataset["rx_list"])):
        for day_i in range(len(dataset["capture_date_list"])):
            available = [
                tx_i
                for tx_i in range(len(dataset["tx_list"]))
                if len(_non_equalized_array(dataset, tx_i, rx_i, day_i)) > 0
            ]
            if len(available) < 2:
                continue
            for local_i in range(mixtures_per_domain):
                tx_a, tx_b = (int(x) for x in rng.choice(available, size=2, replace=False))
                arr_a = _non_equalized_array(dataset, tx_a, rx_i, day_i)
                arr_b = _non_equalized_array(dataset, tx_b, rx_i, day_i)
                signal_a = int(rng.integers(len(arr_a)))
                signal_b = int(rng.integers(len(arr_b)))
                sir_db = sir_levels[local_i % len(sir_levels)]
                rows.append(
                    {
                        "mixture_id": f"wisig-{mixture_number:07d}",
                        "split": plan[(rx_i, day_i)],
                        "tx_a_index": tx_a,
                        "tx_a": str(dataset["tx_list"][tx_a]),
                        "signal_a_index": signal_a,
                        "tx_b_index": tx_b,
                        "tx_b": str(dataset["tx_list"][tx_b]),
                        "signal_b_index": signal_b,
                        "rx_index": rx_i,
                        "receiver": str(dataset["rx_list"][rx_i]),
                        "day_index": day_i,
                        "capture_date": str(dataset["capture_date_list"][day_i]),
                        "equalized": 0,
                        "sir_db": sir_db,
                        "samples": 256,
                        "source_kind": "hardware-captured WiSig I/Q",
                        "overlap_kind": "digital",
                    }
                )
                mixture_number += 1
    if not rows:
        raise ValueError("No receiver/day domain contains two usable transmitters")
    return rows


def _rms(iq: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(iq, dtype=np.float64), axis=-1))))


def materialize_overlaps(
    dataset: dict[str, Any], manifest: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source A, SIR-scaled source B, and their exact digital sum."""
    validate_compact_dataset(dataset)
    if not manifest:
        raise ValueError("manifest cannot be empty")
    source_a, source_b, mixture = [], [], []
    for row in manifest:
        required = {
            "mixture_id",
            "tx_a_index",
            "tx_b_index",
            "rx_index",
            "day_index",
            "signal_a_index",
            "signal_b_index",
            "sir_db",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Manifest row is missing fields: {', '.join(missing)}")
        if int(row["tx_a_index"]) == int(row["tx_b_index"]):
            raise ValueError(f"Manifest row {row['mixture_id']} reuses one transmitter")
        a = _non_equalized_array(
            dataset, int(row["tx_a_index"]), int(row["rx_index"]), int(row["day_index"])
        )[int(row["signal_a_index"])].astype(np.float32, copy=True)
        b = _non_equalized_array(
            dataset, int(row["tx_b_index"]), int(row["rx_index"]), int(row["day_index"])
        )[int(row["signal_b_index"])].astype(np.float32, copy=True)
        rms_a, rms_b = _rms(a), _rms(b)
        if not math.isfinite(rms_a) or not math.isfinite(rms_b) or rms_a <= 0.0 or rms_b <= 0.0:
            raise ValueError(f"Invalid signal power in {row['mixture_id']}")
        sir_db = float(row["sir_db"])
        if not math.isfinite(sir_db):
            raise ValueError(f"Non-finite SIR in {row['mixture_id']}")
        scale_b = rms_a / (rms_b * 10.0 ** (sir_db / 20.0))
        if not math.isfinite(scale_b) or scale_b <= 0.0:
            raise ValueError(f"Invalid SIR scale in {row['mixture_id']}")
        b *= np.float32(scale_b)
        y = a + b
        if not np.isfinite(b).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite overlap in {row['mixture_id']}")
        source_a.append(a)
        source_b.append(b)
        mixture.append(y)
    a_out = np.stack(source_a).astype(np.float32, copy=False)
    b_out = np.stack(source_b).astype(np.float32, copy=False)
    y_out = np.stack(mixture).astype(np.float32, copy=False)
    if not np.array_equal(y_out, a_out + b_out):
        raise RuntimeError("Digital overlap identity failed")
    return a_out, b_out, y_out


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows cannot be empty")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("All manifest rows must use the same fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    dataset_path: Path,
    output: Path,
    *,
    mixtures_per_domain: int = 25,
    seed: int = 2026,
    materialize: bool = True,
    trust_official_pickle: bool = False,
    expected_sha256: str = EXPECTED_MANYRX_SHA256,
) -> dict[str, Any]:
    dataset, dataset_sha256 = _load_verified_compact_dataset(
        dataset_path,
        trust_official_pickle=trust_official_pickle,
        expected_sha256=expected_sha256,
    )
    structure = validate_compact_dataset(dataset)
    manifest = build_overlap_manifest(dataset, mixtures_per_domain=mixtures_per_domain, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output / "wisig_overlap_manifest.csv", manifest)
    if materialize:
        source_a, source_b, mixture = materialize_overlaps(dataset, manifest)
        np.savez_compressed(
            output / "wisig_digital_overlaps.npz",
            source_a=source_a,
            source_b=source_b,
            mixture=mixture,
        )

    split_counts = {
        split: sum(row["split"] == split for row in manifest)
        for split in ("train", "validation", "test")
    }
    metadata = {
        "dataset": "WiSig ManyRx compact subset",
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "expected_dataset_sha256": expected_sha256.lower(),
        "integrity_verified_before_unpickle": True,
        "source_page": OFFICIAL_DATASET_PAGE,
        "examples_repository": OFFICIAL_EXAMPLES_REPO,
        "license": "CC BY-NC-SA 4.0 (dataset); BSD-3-Clause (wisig-examples code)",
        "structure": structure,
        "seed": seed,
        "equalized": 0,
        "mixtures_per_receiver_day": mixtures_per_domain,
        "mixtures": len(manifest),
        "split_counts": split_counts,
        "real_source_iq": True,
        "simultaneous_rf_capture": False,
        "digital_overlap": True,
        "truth_definition": "The two component recordings and deterministic SIR scale used in each digital sum.",
        "claim_scope": (
            "External evaluation on digitally overlapped hardware-captured WiSig I/Q; "
            "not simultaneous over-the-air co-channel or operational SIGINT validation."
        ),
    }
    (output / "wisig_preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Official ManyRx.pkl path")
    parser.add_argument("--output", type=Path, default=Path("external_data/derived/wisig_manyrx"))
    parser.add_argument("--mixtures-per-domain", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--expected-sha256",
        default=EXPECTED_MANYRX_SHA256,
        help="Expected SHA-256; defaults to the audited ManyRx compact file hash",
    )
    parser.add_argument(
        "--trust-official-pickle",
        action="store_true",
        help="Required acknowledgement that the pickle came from the official UCLA distribution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trust_official_pickle:
        raise SystemExit(
            "Refusing to unpickle without --trust-official-pickle. Verify that the file came "
            f"from {OFFICIAL_DATASET_PAGE}"
        )
    metadata = prepare_dataset(
        args.dataset,
        args.output,
        mixtures_per_domain=args.mixtures_per_domain,
        seed=args.seed,
        materialize=not args.manifest_only,
        trust_official_pickle=args.trust_official_pickle,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
