"""Reproducible reference benchmark for blind non-stationary signal de-mixing.

This module implements the experiment reported by the Signals Project website.
It is deliberately narrower than the full torus-valued state-space model in the
paper: a spectral pseudo-posterior is used so that every reported result can be
reproduced on an ordinary CPU. Both proposal operators are corrected against
the same exact, fixed pseudo-posterior with a Metropolis-Hastings step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp
from scipy.stats import t as student_t


@dataclass(frozen=True)
class ExperimentConfig:
    sample_rate_hz: float = 256.0
    window_size: int = 64
    windows_per_trajectory: int = 4
    n_fft: int = 2048
    snr_levels_db: tuple[float, ...] = (15.0, 5.0, -5.0)
    channel_modes: tuple[str, ...] = ("LOS", "MP", "NLOS")
    source1_channels_hz: tuple[float, ...] = (18.0, 26.0, 34.0, 42.0)
    source2_channels_hz: tuple[float, ...] = (50.0, 58.0, 66.0, 74.0)
    jump_probability: float = 0.42
    mala_iterations: int = 110
    mala_burn: int = 30
    mala_tau: float = 1.25
    spectral_weight: float = 1.8
    surrogate_hidden: int = 48
    surrogate_ridge: float = 1e-2
    training_trajectories: int = 120
    training_candidates_per_source: int = 3

    @property
    def trajectory_length(self) -> int:
        return self.window_size * self.windows_per_trajectory


def _seed(*parts: object) -> int:
    """Stable integer seed independent of Python's randomized hash."""
    value = 2166136261
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
    return value


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros_like(x)
    if samples <= 0:
        out[:] = x
    elif samples < len(x):
        out[samples:] = x[:-samples]
    return out


def _rayleigh_fading(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Correlated Rayleigh envelope from a stationary complex AR(1) path."""
    scale = math.sqrt((1.0 - rho * rho) / 2.0)
    z = np.empty(n, dtype=np.complex128)
    z[0] = (rng.normal() + 1j * rng.normal()) / math.sqrt(2.0)
    for i in range(1, n):
        innovation = scale * (rng.normal() + 1j * rng.normal())
        z[i] = rho * z[i - 1] + innovation
    envelope = np.abs(z)
    return envelope / math.sqrt(np.mean(envelope**2))


def _frequency_path(seed: int, cfg: ExperimentConfig) -> np.ndarray:
    rng = np.random.default_rng(_seed("latent", seed))
    channels = (cfg.source1_channels_hz, cfg.source2_channels_hz)
    frequencies = np.empty((cfg.windows_per_trajectory, 2), dtype=float)
    frequencies[0] = [rng.choice(channels[0]), rng.choice(channels[1])]
    for w in range(1, cfg.windows_per_trajectory):
        for emitter in range(2):
            previous = frequencies[w - 1, emitter]
            if rng.random() < cfg.jump_probability:
                alternatives = [x for x in channels[emitter] if x != previous]
                frequencies[w, emitter] = rng.choice(alternatives)
            else:
                frequencies[w, emitter] = previous
    return frequencies


def _truncated_cauchy(rng: np.random.Generator, n: int, bound: float = 5.0) -> np.ndarray:
    angle = math.atan(bound)
    return np.tan((2.0 * rng.random(n) - 1.0) * angle)


def simulate_trajectory(
    seed: int,
    snr_db: float,
    channel_mode: str,
    cfg: ExperimentConfig,
) -> dict[str, np.ndarray | float]:
    """Generate one paired physical trajectory for both inference methods."""
    n = cfg.trajectory_length
    t = np.arange(n) / cfg.sample_rate_hz
    frequency_windows = _frequency_path(seed, cfg)
    frequencies = np.repeat(frequency_windows, cfg.window_size, axis=0)

    latent_rng = np.random.default_rng(_seed("amplitude", seed))
    a1 = 1.35 + 0.13 * np.sin(2.0 * np.pi * 1.7 * t + latent_rng.uniform(0, 2 * np.pi))
    a2 = 1.00 + 0.10 * np.cos(2.0 * np.pi * 2.2 * t + latent_rng.uniform(0, 2 * np.pi))
    packet = 0.72 + 0.28 * np.sin(np.pi * ((np.arange(n) % cfg.window_size) + 0.5) / cfg.window_size) ** 2
    a1 *= packet
    a2 *= packet

    phase1 = np.cumsum(2.0 * np.pi * frequencies[:, 0] / cfg.sample_rate_hz)
    phase2 = np.cumsum(2.0 * np.pi * frequencies[:, 1] / cfg.sample_rate_hz)
    s1 = a1 * np.cos(phase1)
    s2 = a2 * np.cos(phase2)

    channel_rng = np.random.default_rng(_seed("channel", seed, channel_mode))
    if channel_mode == "LOS":
        clean = s1 + s2
    elif channel_mode == "MP":
        h1 = _rayleigh_fading(channel_rng, n, rho=0.985)
        h2 = _rayleigh_fading(channel_rng, n, rho=0.982)
        clean = s1 + s2 + 0.30 * h1 * _delay(s1, 2) + 0.22 * h2 * _delay(s2, 3)
    elif channel_mode == "NLOS":
        h11 = _rayleigh_fading(channel_rng, n, rho=0.975)
        h12 = _rayleigh_fading(channel_rng, n, rho=0.968)
        h21 = _rayleigh_fading(channel_rng, n, rho=0.972)
        h22 = _rayleigh_fading(channel_rng, n, rho=0.965)
        clean = (
            0.58 * h11 * _delay(s1, 2)
            + 0.34 * h12 * _delay(s1, 6)
            + 0.52 * h21 * _delay(s2, 3)
            + 0.31 * h22 * _delay(s2, 7)
        )
    else:
        raise ValueError(f"Unknown channel mode: {channel_mode}")

    signal_power = float(np.mean(clean**2))
    target_noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    noise_rng = np.random.default_rng(_seed("noise", seed, snr_db, channel_mode))

    gaussian = noise_rng.normal(0.0, math.sqrt(0.90 * target_noise_variance), n)
    cauchy_base = _truncated_cauchy(noise_rng, n, bound=5.0)
    a = 5.0
    base_variance = (a - math.atan(a)) / math.atan(a)
    cauchy_scale = math.sqrt(0.10 * target_noise_variance / base_variance)
    noise = gaussian + cauchy_scale * cauchy_base
    observation = clean + noise
    achieved_snr = 10.0 * math.log10(signal_power / float(np.mean(noise**2)))

    return {
        "observation": observation,
        "clean": clean,
        "true_frequencies": frequency_windows,
        "achieved_snr_db": achieved_snr,
    }


def spectral_surface(window: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, CubicSpline]:
    """Robust standardized log-periodogram and differentiable interpolant."""
    centered = window - np.median(window)
    mad = np.median(np.abs(centered)) + 1e-9
    centered = np.clip(centered, -7.5 * mad, 7.5 * mad)
    tapered = centered * np.hanning(len(centered))
    spectrum = np.fft.rfft(tapered, n=cfg.n_fft)
    power = np.abs(spectrum) ** 2 + 1e-12
    frequency = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate_hz)
    mask = (frequency >= 10.0) & (frequency <= 82.0)
    frequency = frequency[mask]
    log_power = np.log(power[mask])
    log_power = (log_power - np.median(log_power)) / (np.std(log_power) + 1e-9)
    return frequency, log_power, CubicSpline(frequency, log_power, bc_type="natural")


def _peak_start(frequency: np.ndarray, log_power: np.ndarray, bounds: tuple[float, float]) -> float:
    mask = (frequency >= bounds[0]) & (frequency <= bounds[1])
    idx = np.argmax(log_power[mask])
    return float(frequency[mask][idx])


def _score_features(f: float, spline: CubicSpline, bounds: tuple[float, float]) -> np.ndarray:
    span = bounds[1] - bounds[0]
    x = (f - bounds[0]) / span
    offsets = (-1.5, 0.0, 1.5)
    values = [float(spline(np.clip(f + offset, bounds[0], bounds[1]))) for offset in offsets]
    return np.array([x, values[0], values[1], values[2], values[2] - values[0]], dtype=float)


class SmoothScoreSurrogate:
    """Single-hidden-layer SiLU random-feature network with ridge-trained output."""

    def __init__(self, hidden: int, ridge: float, seed: int = 9173):
        self.hidden = hidden
        self.ridge = ridge
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.75, (5, hidden))
        self.b = rng.normal(0.0, 0.30, hidden)
        self.x_mean = np.zeros(5)
        self.x_scale = np.ones(5)
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.beta = np.zeros(hidden + 1)

    @staticmethod
    def _silu(z: np.ndarray) -> np.ndarray:
        clipped = np.clip(z, -35.0, 35.0)
        return clipped / (1.0 + np.exp(-clipped))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SmoothScoreSurrogate":
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std() + 1e-8)
        xs = (x - self.x_mean) / self.x_scale
        ys = (y - self.y_mean) / self.y_scale
        hidden = self._silu(xs @ self.w + self.b)
        design = np.column_stack([np.ones(len(hidden)), hidden])
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.beta = np.linalg.solve(design.T @ design + penalty, design.T @ ys)
        return self

    def predict_one(self, features: np.ndarray) -> float:
        xs = (features - self.x_mean) / self.x_scale
        hidden = self._silu(xs @ self.w + self.b)
        prediction = np.r_[1.0, hidden] @ self.beta
        return float(self.y_mean + self.y_scale * prediction)

    def save(self, path: Path) -> None:
        np.savez(
            path,
            w=self.w,
            b=self.b,
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale,
            beta=self.beta,
            hidden=self.hidden,
            ridge=self.ridge,
        )


def train_surrogate(cfg: ExperimentConfig) -> tuple[SmoothScoreSurrogate, dict[str, float]]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    bounds = ((12.0, 46.0), (46.0, 80.0))
    rng = np.random.default_rng(24491)
    modes = cfg.channel_modes
    snrs = cfg.snr_levels_db

    for training_id in range(cfg.training_trajectories):
        seed = 100_000 + training_id
        sim = simulate_trajectory(seed, float(snrs[training_id % 3]), modes[(training_id // 3) % 3], cfg)
        observation = np.asarray(sim["observation"])
        for w in range(cfg.windows_per_trajectory):
            segment = observation[w * cfg.window_size : (w + 1) * cfg.window_size]
            _, _, spline = spectral_surface(segment, cfg)
            derivative = spline.derivative()
            for source in range(2):
                for _ in range(cfg.training_candidates_per_source):
                    f = rng.uniform(bounds[source][0] + 0.5, bounds[source][1] - 0.5)
                    features.append(_score_features(f, spline, bounds[source]))
                    targets.append(cfg.spectral_weight * float(derivative(f)))

    x = np.vstack(features)
    y = np.asarray(targets)
    order = np.random.default_rng(889).permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = order[:split], order[split:]
    model = SmoothScoreSurrogate(cfg.surrogate_hidden, cfg.surrogate_ridge).fit(x[train_idx], y[train_idx])
    predicted = np.array([model.predict_one(row) for row in x[test_idx]])
    rmse = float(np.sqrt(np.mean((predicted - y[test_idx]) ** 2)))
    correlation = float(np.corrcoef(predicted, y[test_idx])[0, 1])
    return model, {"validation_score_rmse": rmse, "validation_score_correlation": correlation, "training_rows": int(split)}


def _prior_log_score(f: float, previous: float | None, channels: tuple[float, ...]) -> tuple[float, float]:
    if previous is None:
        means = np.asarray(channels)
        variances = np.full(len(means), 3.0**2)
        weights = np.full(len(means), 1.0 / len(means))
    else:
        means = np.r_[previous, np.asarray(channels)]
        variances = np.r_[1.25**2, np.full(len(channels), 2.1**2)]
        weights = np.r_[0.72, np.full(len(channels), 0.28 / len(channels))]
    components = np.log(weights) - 0.5 * np.log(2.0 * np.pi * variances) - 0.5 * (f - means) ** 2 / variances
    log_prior = float(logsumexp(components))
    responsibilities = np.exp(components - log_prior)
    score = float(np.sum(responsibilities * (-(f - means) / variances)))
    return log_prior, score


def _target_and_score(
    state: np.ndarray,
    spl: CubicSpline,
    previous: np.ndarray | None,
    cfg: ExperimentConfig,
    architecture: str,
    surrogate: SmoothScoreSurrogate,
) -> tuple[float, np.ndarray]:
    bounds = ((12.0, 46.0), (46.0, 80.0))
    channels = (cfg.source1_channels_hz, cfg.source2_channels_hz)
    if any(state[i] <= bounds[i][0] or state[i] >= bounds[i][1] for i in range(2)):
        return -math.inf, np.zeros(2)

    derivative = spl.derivative()
    log_target = 0.0
    score = np.zeros(2)
    for source in range(2):
        prior_log, prior_score = _prior_log_score(
            float(state[source]),
            None if previous is None else float(previous[source]),
            channels[source],
        )
        log_target += cfg.spectral_weight * float(spl(state[source])) + prior_log
        if architecture == "Analytical score":
            spectral_score = cfg.spectral_weight * float(derivative(state[source]))
        else:
            spectral_score = surrogate.predict_one(_score_features(float(state[source]), spl, bounds[source]))
        score[source] = spectral_score + prior_score
    return log_target, score


def _ess_1d(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=float)
    n = len(x)
    centered = x - x.mean()
    variance = float(np.dot(centered, centered) / n)
    if n < 4 or variance < 1e-14:
        return 1.0
    rho = []
    for lag in range(1, n):
        rho.append(float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance)))
    positive_sum = 0.0
    for k in range(0, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        positive_sum += pair
    return float(np.clip(n / (1.0 + 2.0 * positive_sum), 1.0, n))


def run_mala_window(
    segment: np.ndarray,
    previous: np.ndarray | None,
    architecture: str,
    surrogate: SmoothScoreSurrogate,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    frequency, log_power, spline = spectral_surface(segment, cfg)
    state = np.array(
        [
            _peak_start(frequency, log_power, (12.0, 46.0)),
            _peak_start(frequency, log_power, (46.0, 80.0)),
        ]
    )
    if previous is not None:
        state = 0.65 * state + 0.35 * previous

    chain = []
    accepted = 0
    for iteration in range(cfg.mala_iterations):
        log_current, score_current = _target_and_score(state, spline, previous, cfg, architecture, surrogate)
        proposal = state + cfg.mala_tau * score_current + math.sqrt(2.0 * cfg.mala_tau) * rng.normal(size=2)
        log_proposal, score_proposal = _target_and_score(proposal, spline, previous, cfg, architecture, surrogate)
        if math.isfinite(log_proposal):
            forward = -float(np.sum((proposal - state - cfg.mala_tau * score_current) ** 2)) / (4.0 * cfg.mala_tau)
            reverse = -float(np.sum((state - proposal - cfg.mala_tau * score_proposal) ** 2)) / (4.0 * cfg.mala_tau)
            log_alpha = log_proposal - log_current + reverse - forward
            if math.log(rng.random()) < min(0.0, log_alpha):
                state = proposal
                accepted += 1
        if iteration >= cfg.mala_burn:
            chain.append(state.copy())

    retained = np.vstack(chain)
    estimate = retained.mean(axis=0)
    ess = min(_ess_1d(retained[:, 0]), _ess_1d(retained[:, 1]))
    return {
        "estimate": estimate,
        "chain": retained,
        "ess": ess,
        "acceptance_rate": accepted / cfg.mala_iterations,
    }


def evaluate_method(
    observation: np.ndarray,
    true_frequencies: np.ndarray,
    architecture: str,
    surrogate: SmoothScoreSurrogate,
    cfg: ExperimentConfig,
    algorithm_seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = np.random.default_rng(algorithm_seed)
    previous = None
    estimates = []
    ess_values = []
    acceptance = []
    trace_rows: list[dict[str, float]] = []
    start = time.perf_counter_ns()
    for window_id in range(cfg.windows_per_trajectory):
        segment = observation[window_id * cfg.window_size : (window_id + 1) * cfg.window_size]
        result = run_mala_window(segment, previous, architecture, surrogate, cfg, rng)
        estimate = np.asarray(result["estimate"])
        estimates.append(estimate)
        ess_values.append(float(result["ess"]))
        acceptance.append(float(result["acceptance_rate"]))
        previous = estimate
        trace_rows.append(
            {
                "window": window_id + 1,
                "true_f1_hz": float(true_frequencies[window_id, 0]),
                "inferred_f1_hz": float(estimate[0]),
                "true_f2_hz": float(true_frequencies[window_id, 1]),
                "inferred_f2_hz": float(estimate[1]),
                "ess": float(result["ess"]),
                "acceptance_rate": float(result["acceptance_rate"]),
            }
        )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    estimates_array = np.vstack(estimates)
    mse = float(np.mean(np.sum((estimates_array - true_frequencies) ** 2, axis=1)))
    return (
        {
            "mse_frequency": mse,
            "latency_ms_per_window": elapsed_ms / cfg.windows_per_trajectory,
            "ess": float(np.mean(ess_values)),
            "acceptance_rate": float(np.mean(acceptance)),
        },
        trace_rows,
    )


def _paired_effects(results: pd.DataFrame) -> pd.DataFrame:
    index = ["seed", "snr_db", "channel"]
    wide = results.pivot(index=index, columns="architecture", values=["mse_frequency", "latency_ms_per_window", "ess"])
    records = []
    group_specs: list[tuple[str, object, pd.DataFrame]] = [("Overall", "All", wide)]
    for snr in sorted(results.snr_db.unique(), reverse=True):
        group_specs.append(("SNR", snr, wide.xs(snr, level="snr_db", drop_level=False)))
    for channel in ("LOS", "MP", "NLOS"):
        group_specs.append(("Channel", channel, wide.xs(channel, level="channel", drop_level=False)))
    for group, level, frame in group_specs:
        for metric in ("mse_frequency", "latency_ms_per_window", "ess"):
            diff = frame[(metric, "Amortized surrogate")] - frame[(metric, "Analytical score")]
            n = len(diff)
            mean = float(diff.mean())
            se = float(diff.std(ddof=1) / math.sqrt(n))
            critical = float(student_t.ppf(0.975, n - 1))
            records.append(
                {
                    "group": group,
                    "level": level,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_difference_surrogate_minus_analytical": mean,
                    "ci95_low": mean - critical * se,
                    "ci95_high": mean + critical * se,
                }
            )
    return pd.DataFrame(records)


def run_experiment(output_dir: Path, repetitions: int, cfg: ExperimentConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    surrogate, validation = train_surrogate(cfg)
    surrogate.save(output_dir / "surrogate_weights.npz")

    records: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    architectures = ("Analytical score", "Amortized surrogate")
    completed = 0
    total_pairs = repetitions * len(cfg.snr_levels_db) * len(cfg.channel_modes)

    for seed in range(repetitions):
        for snr_db in cfg.snr_levels_db:
            for channel in cfg.channel_modes:
                simulation = simulate_trajectory(seed, snr_db, channel, cfg)
                observation = np.asarray(simulation["observation"])
                truth = np.asarray(simulation["true_frequencies"])
                order_rng = np.random.default_rng(_seed("order", seed, snr_db, channel))
                method_order = list(architectures)
                order_rng.shuffle(method_order)
                for architecture in method_order:
                    metrics, traces = evaluate_method(
                        observation,
                        truth,
                        architecture,
                        surrogate,
                        cfg,
                        _seed("mala", seed, snr_db, channel),
                    )
                    records.append(
                        {
                            "seed": seed,
                            "architecture": architecture,
                            "snr_db": snr_db,
                            "channel": channel,
                            "achieved_snr_db": float(simulation["achieved_snr_db"]),
                            **metrics,
                        }
                    )
                    if seed == 0 and snr_db == 5.0 and channel == "MP":
                        for row in traces:
                            example_rows.append({"architecture": architecture, **row})
                completed += 1
                if completed % 100 == 0 or completed == total_pairs:
                    print(f"completed {completed}/{total_pairs} paired trajectories", flush=True)

    results = pd.DataFrame(records).sort_values(["seed", "snr_db", "channel", "architecture"])
    results.to_csv(output_dir / "factorial_results.csv", index=False)
    pd.DataFrame(example_rows).to_csv(output_dir / "example_trace.csv", index=False)

    summary = (
        results.groupby(["architecture", "snr_db", "channel"], observed=True)
        .agg(
            n=("seed", "size"),
            mse_mean=("mse_frequency", "mean"),
            mse_sd=("mse_frequency", "std"),
            latency_mean_ms=("latency_ms_per_window", "mean"),
            latency_sd_ms=("latency_ms_per_window", "std"),
            ess_mean=("ess", "mean"),
            ess_sd=("ess", "std"),
            acceptance_mean=("acceptance_rate", "mean"),
            achieved_snr_mean=("achieved_snr_db", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "cell_summary.csv", index=False)
    _paired_effects(results).to_csv(output_dir / "paired_effects.csv", index=False)

    metadata = {
        "experiment_name": "Signals Project Reference Benchmark",
        "design": "2 x 3 x 3 full factorial with paired architecture comparisons",
        "unique_physical_trajectories": total_pairs,
        "experimental_rows": len(results),
        "repetitions_per_cell": repetitions,
        "config": asdict(cfg),
        "surrogate_validation": validation,
        "interpretation": "Simulation evidence from a spectral pseudo-posterior reference benchmark; not operational validation.",
    }
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    run_experiment(args.output, args.repetitions, ExperimentConfig())


if __name__ == "__main__":
    main()
