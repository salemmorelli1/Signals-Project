"""Blocked multivariate repeated-measures analysis for the joint-SSM factorial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def orthonormal_basis(levels: int) -> np.ndarray:
    if levels == 2:
        return np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    if levels == 3:
        return np.column_stack(
            [
                np.ones(3) / np.sqrt(3.0),
                np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0),
                np.array([1.0, -2.0, 1.0]) / np.sqrt(6.0),
            ]
        )
    raise ValueError(levels)


def design_basis() -> tuple[np.ndarray, dict[str, list[int]], list[tuple[str, float, str]]]:
    architectures = ["Analytical joint score", "Amortized joint surrogate"]
    snrs = [-5.0, 5.0, 15.0]
    channels = ["LOS", "MP", "NLOS"]
    a, b, c = orthonormal_basis(2), orthonormal_basis(3), orthonormal_basis(3)
    rows: list[np.ndarray] = []
    order: list[tuple[str, float, str]] = []
    column_terms: list[tuple[int, int, int]] = []
    for ia in range(2):
        for ib in range(3):
            for ic in range(3):
                rows.append(np.kron(np.kron(a[ia], b[ib]), c[ic]))
                order.append((architectures[ia], snrs[ib], channels[ic]))
    for ja in range(2):
        for jb in range(3):
            for jc in range(3):
                column_terms.append((ja, jb, jc))
    effects = {
        "Architecture": [],
        "SNR": [],
        "Channel": [],
        "Architecture × SNR": [],
        "Architecture × Channel": [],
        "SNR × Channel": [],
        "Architecture × SNR × Channel": [],
    }
    for column, (ja, jb, jc) in enumerate(column_terms):
        if ja == jb == jc == 0:
            continue
        active = tuple(x > 0 for x in (ja, jb, jc))
        label = {
            (True, False, False): "Architecture",
            (False, True, False): "SNR",
            (False, False, True): "Channel",
            (True, True, False): "Architecture × SNR",
            (True, False, True): "Architecture × Channel",
            (False, True, True): "SNR × Channel",
            (True, True, True): "Architecture × SNR × Channel",
        }[active]
        effects[label].append(column)
    design = np.asarray(rows)
    if not np.allclose(design.T @ design, np.eye(18), atol=1e-12):
        raise RuntimeError("Factorial contrast basis is not orthonormal")
    return design, effects, order


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def multivariate_within_seed(frame: pd.DataFrame, metric: str, transform: str) -> pd.DataFrame:
    design, effects, order = design_basis()
    subject_scores = []
    for _, block in frame.groupby("seed", sort=True):
        keyed = block.set_index(["architecture", "snr_db", "channel"])[metric]
        values = np.array([keyed.loc[key] for key in order], dtype=float)
        if transform == "log":
            values = np.log(values)
        subject_scores.append(design.T @ values)
    scores = np.asarray(subject_scores)
    n = len(scores)
    rows: list[dict[str, object]] = []
    for effect, columns in effects.items():
        values = scores[:, columns]
        if values.ndim == 1:
            values = values[:, None]
        q = values.shape[1]
        mean = values.mean(axis=0)
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
        covariance += np.eye(q) * np.finfo(float).eps
        t_squared = float(n * mean @ np.linalg.pinv(covariance) @ mean)
        f_value = float((n - q) * t_squared / (q * (n - 1)))
        df1, df2 = q, n - q
        p_value = float(f_distribution.sf(f_value, df1, df2))
        rows.append(
            {
                "response": metric,
                "transform": transform,
                "effect": effect,
                "df1": df1,
                "df2": df2,
                "F": f_value,
                "p_value": p_value,
                "partial_eta_squared": (f_value * df1) / (f_value * df1 + df2),
            }
        )
    output = pd.DataFrame(rows)
    output["p_holm_within_response"] = holm_adjust(output.p_value.to_numpy())
    return output


def main() -> None:
    frame = pd.read_csv(DATA / "joint_factorial_results.csv")
    responses = {
        "mse_frequency": "log",
        "phase_circular_rmse": "raw",
        "channel_nmse": "log",
        "particle_ess_mean": "raw",
        "latency_ms_per_sample": "log",
    }
    effects = pd.concat(
        [multivariate_within_seed(frame, metric, transform) for metric, transform in responses.items()],
        ignore_index=True,
    )
    effects.to_csv(DATA / "joint_factorial_effects.csv", index=False)

    architecture_means = frame.groupby("architecture")[list(responses)].mean().to_dict(orient="index")
    primary = effects[effects.response == "mse_frequency"].copy()
    summary = {
        "analysis": "Orthonormal-contrast multivariate repeated-measures analysis with trajectory seed as the block",
        "primary_response": "log frequency MSE",
        "multiplicity": "Holm adjustment across the seven primary-response omnibus effects",
        "architecture_means": architecture_means,
        "primary_effects": primary.to_dict(orient="records"),
        "warning": "This is simulation evidence and is not field or operational SIGINT validation.",
    }
    (DATA / "joint_inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
