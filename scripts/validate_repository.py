"""Validate committed scientific, provenance, publication, and CI contracts."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

from scripts.build_site import simulation_summaries
from src.signals_project.joint_ssm import (
    ARCHITECTURES,
    FACTORIAL_METRICS,
    paired_contrasts,
    summarize_cells,
    validate_factorial_frame,
)
from src.signals_project.wisig_realdata import EXPECTED_MANYRX_SHA256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAIR_METRICS = (
    "mse_frequency",
    "phase_circular_rmse",
    "channel_nmse",
    "particle_ess_mean",
    "latency_ms_per_sample",
)
RESPONSES = set(PAIR_METRICS)
EFFECTS = {
    "Architecture",
    "SNR",
    "Channel",
    "Architecture × SNR",
    "Architecture × Channel",
    "SNR × Channel",
    "Architecture × SNR × Channel",
}


class ValidationError(RuntimeError):
    """Raised when a committed repository contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path.name} must contain a JSON object")
    return value


def assert_frame_close(actual: pd.DataFrame, expected: pd.DataFrame, name: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-11,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValidationError(f"{name} is inconsistent with row-level results: {exc}") from exc


def validate_simulation_artifacts() -> str:
    metadata = read_json(DATA / "joint_experiment_metadata.json")
    config = metadata["config"]
    repetitions = int(metadata["repetitions_per_cell"])
    results = pd.read_csv(DATA / "joint_factorial_results.csv")
    validate_factorial_frame(
        results,
        FACTORIAL_METRICS,
        expected_repetitions=repetitions,
        expected_snrs=config["snr_levels_db"],
        expected_channels=config["channel_modes"],
    )
    expected_rows = (
        repetitions
        * len(ARCHITECTURES)
        * len(config["snr_levels_db"])
        * len(config["channel_modes"])
    )
    require(
        len(results) == expected_rows == int(metadata["experimental_rows"]),
        "Result row count mismatch",
    )
    require(
        int(metadata["unique_physical_trajectories"]) == expected_rows // len(ARCHITECTURES),
        "Physical-trajectory count mismatch",
    )
    require(
        metadata.get("architecture_execution_order") == list(ARCHITECTURES),
        "Architecture execution order is missing or inconsistent",
    )
    require(
        "fixed, nonrandomized method order" in metadata.get("latency_interpretation", ""),
        "Latency interpretation does not disclose the fixed execution order",
    )

    particles = int(config["particles"])
    samples = int(config["n_samples"])
    require(
        results["particle_ess_mean"].between(0.0, particles, inclusive="right").all(),
        "ESS out of range",
    )
    require(
        results["unique_ancestor_mean"].between(0.0, particles, inclusive="right").all(),
        "Ancestor count out of range",
    )
    require(results["mala_acceptance"].between(0.0, 1.0).all(), "MALA acceptance out of range")
    require(
        results["resampling_count"].between(0.0, samples).all(), "Resampling count out of range"
    )
    require(
        np.equal(results["resampling_count"], np.floor(results["resampling_count"])).all(),
        "Resampling count must be integral",
    )
    require((results["latency_ms_per_sample"] > 0.0).all(), "Latency must be positive")
    require((results["phase_circular_rmse"] <= np.pi).all(), "Circular RMSE exceeds pi")

    committed_cells = pd.read_csv(DATA / "joint_cell_summary.csv")
    regenerated_cells = summarize_cells(results)
    assert_frame_close(committed_cells, regenerated_cells, "joint_cell_summary.csv")
    require((committed_cells["n"] == repetitions).all(), "Cell counts do not match repetitions")

    committed_pairs = pd.read_csv(DATA / "joint_paired_effects.csv")
    regenerated_pairs = paired_contrasts(results, PAIR_METRICS)
    assert_frame_close(committed_pairs, regenerated_pairs, "joint_paired_effects.csv")
    require(len(committed_pairs) == len(PAIR_METRICS) * 7, "Unexpected paired-effect row count")
    require((committed_pairs["n_blocks"] == repetitions).all(), "Paired block counts mismatch")
    require(
        set(committed_pairs.loc[committed_pairs["group"] == "Overall", "n_pairs"])
        == {repetitions * len(config["snr_levels_db"]) * len(config["channel_modes"])},
        "Overall paired count mismatch",
    )
    require(
        set(committed_pairs.loc[committed_pairs["group"] != "Overall", "n_pairs"])
        == {repetitions * 3},
        "Subgroup paired counts mismatch",
    )
    require(
        (committed_pairs["ci95_low"] <= committed_pairs["ci95_high"]).all(),
        "Invalid paired intervals",
    )

    effects = pd.read_csv(DATA / "joint_factorial_effects.csv")
    require(len(effects) == len(RESPONSES) * len(EFFECTS), "Unexpected factorial-effect row count")
    require(set(effects["response"]) == RESPONSES, "Factorial responses mismatch")
    require(set(effects["effect"]) == EFFECTS, "Factorial effects mismatch")
    effect_numbers = effects[
        ["df1", "df2", "F", "p_value", "partial_eta_squared", "p_holm_within_response"]
    ]
    require(np.isfinite(effect_numbers.to_numpy(dtype=float)).all(), "Non-finite factorial effect")
    require(effects["p_value"].between(0.0, 1.0).all(), "Raw p-value out of range")
    require(effects["p_holm_within_response"].between(0.0, 1.0).all(), "Holm p-value out of range")
    require(effects["partial_eta_squared"].between(0.0, 1.0).all(), "Effect size out of range")

    trace = pd.read_csv(DATA / "joint_example_trace.csv")
    require(len(trace) == samples * len(ARCHITECTURES), "Example-trace row count mismatch")
    require(
        set(trace["architecture"]) == set(ARCHITECTURES), "Example-trace architectures mismatch"
    )
    trace_numeric = trace.select_dtypes(include=[np.number])
    require(
        np.isfinite(trace_numeric.to_numpy(dtype=float)).all(),
        "Example trace contains non-finite values",
    )
    return f"{len(results)} complete paired simulation rows"


def validate_wisig_status() -> str:
    status = read_json(DATA / "wisig_integration_status.json")
    require(status["status"] == "dataset_prepared", "WiSig status must remain dataset_prepared")
    require(status["dataset_preparation_executed"] is True, "WiSig preparation flag mismatch")
    require(status["dataset_sha256"] == EXPECTED_MANYRX_SHA256, "WiSig SHA-256 mismatch")
    require(
        re.fullmatch(r"[0-9a-f]{64}", status["dataset_sha256"]) is not None, "Invalid WiSig SHA-256"
    )
    require(
        status["mixtures"] == sum(status["split_counts"].values()), "WiSig split counts mismatch"
    )
    require(
        status["split_counts"] == {"train": 1875, "validation": 625, "test": 700},
        "Unexpected WiSig split",
    )
    require(status["performance_results_available"] is False, "Unsubstantiated WiSig results flag")
    require(status["simultaneous_rf_capture"] is False, "Unsubstantiated simultaneous-RF flag")
    require(status["hardware_in_the_loop"] is False, "Unsubstantiated HIL flag")
    return f"{status['mixtures']} prepared mixtures with pinned provenance"


def validate_publication() -> str:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    require(EXPECTED_MANYRX_SHA256 in html, "Dashboard omits the audited WiSig hash")
    require(
        "x.group==='Channel'&&x.level===channel" in html,
        "Dashboard channel contrast is not selected",
    )
    summary_match = re.search(r"const status=.*?,cells=(\[.*?\]),pairs=", html, flags=re.DOTALL)
    require(summary_match is not None, "Dashboard simulation summaries cannot be parsed")
    assert summary_match is not None
    published_summaries = pd.DataFrame(json.loads(summary_match.group(1)))
    expected_summaries = pd.DataFrame(
        simulation_summaries(pd.read_csv(DATA / "joint_factorial_results.csv"))
    )
    assert_frame_close(published_summaries, expected_summaries, "dashboard seed-block summaries")
    require("rows.map(x=>x.ci)" in html, "Dashboard does not use seed-block intervals")
    require(
        "timing contrast is descriptive, not causal" in html.lower(),
        "Dashboard omits the nonrandomized latency limitation",
    )
    require("not simultaneous" in html.lower(), "Dashboard claim boundary is missing")
    require(
        re.search(r"__[A-Z0-9_]+__", html) is None, "Dashboard contains unresolved template tokens"
    )
    reader = PdfReader(ROOT / "report" / "Signals_Project_APA_Report.pdf")
    pages = len(reader.pages)
    require(pages == 27, f"Expected a 27-page report, found {pages}")
    embedded_dejavu = False
    for page in reader.pages:
        resources = page.get("/Resources", {}).get_object()
        fonts = resources.get("/Font", {}).get_object()
        for font_reference in fonts.values():
            font = font_reference.get_object()
            descriptor_reference = font.get("/FontDescriptor")
            if descriptor_reference is None:
                continue
            descriptor = descriptor_reference.get_object()
            if "DejaVuSerif" in str(font.get("/BaseFont", "")) and "/FontFile2" in descriptor:
                embedded_dejavu = True
    require(embedded_dejavu, "Report must embed the portable DejaVu serif font")
    metadata = reader.metadata
    creation_date = "" if metadata is None else str(metadata.get("/CreationDate") or "")
    require(creation_date.startswith("D:20000101"), "Report metadata is not deterministic")
    return f"dashboard contracts and {pages}-page report"


def validate_ci_contract() -> str:
    lock_lines = [
        line.strip()
        for line in (ROOT / "requirements-ci-lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    require(
        bool(lock_lines) and all("==" in line for line in lock_lines),
        "CI dependencies must be exact pins",
    )
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )
    action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow_text)
    require(
        bool(action_refs)
        and all(re.fullmatch(r"[0-9a-f]{40}", ref) is not None for ref in action_refs),
        "Actions must use commit SHAs",
    )
    validate_workflow = (ROOT / ".github" / "workflows" / "validate.yml").read_text(
        encoding="utf-8"
    )
    require('python-version: "3.12"' in validate_workflow, "CI must use the locked CPython 3.12")
    report_builder = (ROOT / "scripts" / "build_report.py").read_text(encoding="utf-8")
    require(
        "CODEX_PRIMARY_RUNTIME_ROOT" not in report_builder
        and "get_data_path" in report_builder
        and "DejaVuSerif.ttf" in report_builder,
        "Report generation must not depend on environment-specific fonts",
    )
    require("invariant=1" in report_builder, "Report generation must suppress volatile metadata")
    tracked_pickles = subprocess.run(
        ["git", "ls-files", "*.pkl"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not tracked_pickles, "Pickle inputs must not be tracked")
    return f"{len(lock_lines)} dependency pins and {len(action_refs)} pinned Actions"


def validate_repository() -> list[tuple[str, str]]:
    return [
        ("simulation", validate_simulation_artifacts()),
        ("wisig", validate_wisig_status()),
        ("publication", validate_publication()),
        ("ci", validate_ci_contract()),
    ]


def main() -> None:
    checks = validate_repository()
    for name, detail in checks:
        print(f"PASS {name}: {detail}")
    print(f"PASS repository: {len(checks)} contract groups")


if __name__ == "__main__":
    main()
