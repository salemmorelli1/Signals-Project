"""Executed torus-valued joint state-space benchmark.

The module implements a structure-informed blind filter for two overlapping
frequency-hopping emitters observed as complex baseband I/Q samples.  Each
particle jointly carries amplitude, frequency, discrete hopping state, phase
on the flat torus, latent complex multipath coefficients, and the source lag
buffer required by tapped-delay propagation.

Both score architectures use the same exact time-domain target.  The
analytical architecture differentiates the quadrature likelihood explicitly;
the amortized architecture uses an independently trained SiLU random-feature
network only to form MALA proposals.  Metropolis correction always evaluates
the exact joint target and the actual forward/reverse proposal densities.

The experiment is validation in an operationally representative simulation.
It is not field or operational SIGINT validation; no real receiver recordings
or independent operational test campaign are included.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import logsumexp


TWO_PI = 2.0 * math.pi


def stable_seed(*parts: object) -> int:
    value = 2166136261
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
    return value


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return np.mod(x, TWO_PI)


def wrap_difference(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % TWO_PI - math.pi


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


@dataclass(frozen=True)
class JointConfig:
    sample_rate_hz: float = 128.0
    n_samples: int = 72
    hop_interval: int = 12
    emitters: int = 2
    particles: int = 96
    resample_fraction: float = 0.58
    forced_move_interval: int = 6
    mala_steps: int = 1
    quadrature_nodes: int = 24
    wrapped_lifts: int = 2
    snr_levels_db: tuple[float, ...] = (15.0, 5.0, -5.0)
    channel_modes: tuple[str, ...] = ("LOS", "MP", "NLOS")
    emitter1_channels_hz: tuple[float, ...] = (18.0, 30.0, 42.0, 54.0)
    emitter2_channels_hz: tuple[float, ...] = (42.0, 50.0, 54.0, 62.0)
    hop_probability: float = 0.34
    logamp_mean: float = 0.0
    logamp_reversion: float = 0.035
    logamp_sd: float = 0.035
    frequency_rho: float = 0.82
    frequency_sd: float = 0.42
    phase_sd: float = 0.075
    channel_rho: float = 0.965
    channel_sd: float = 0.050
    cauchy_variance_fraction: float = 0.10
    cauchy_standard_bound: float = 5.0
    surrogate_hidden: int = 72
    surrogate_ridge: float = 0.08
    surrogate_train_rows_per_mode: int = 900
    drift_clip: float = 120.0

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def channel_alphabets(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.emitter1_channels_hz), np.asarray(self.emitter2_channels_hz))


@dataclass(frozen=True)
class ChannelSpec:
    mode: str
    delays: tuple[int, ...]
    means: np.ndarray  # [emitter, tap], complex

    @property
    def taps(self) -> int:
        return len(self.delays)

    @property
    def max_delay(self) -> int:
        return max(self.delays)


def channel_spec(mode: str) -> ChannelSpec:
    if mode == "LOS":
        means = np.array([[1.00 + 0j], [0.92 + 0j]], dtype=np.complex128)
        return ChannelSpec(mode, (0,), means)
    if mode == "MP":
        means = np.array(
            [
                [1.00 + 0j, 0.30 * np.exp(-1j * math.pi / 4)],
                [0.92 + 0j, 0.25 * np.exp(1j * math.pi / 5)],
            ],
            dtype=np.complex128,
        )
        return ChannelSpec(mode, (0, 2), means)
    if mode == "NLOS":
        means = np.array(
            [
                [0.64 * np.exp(-1j * math.pi / 4), 0.34 * np.exp(1j * math.pi / 3)],
                [0.59 * np.exp(1j * math.pi / 5), 0.36 * np.exp(-1j * math.pi / 3)],
            ],
            dtype=np.complex128,
        )
        return ChannelSpec(mode, (1, 4), means)
    raise ValueError(f"Unknown channel mode {mode!r}")


class ConvolutionNoise:
    """Independent Gaussian plus truncated-Cauchy noise for one real component."""

    def __init__(self, sigma_g: float, gamma: float, bound: float, nodes: int):
        self.sigma_g = float(sigma_g)
        self.gamma = float(gamma)
        self.bound = float(bound)
        x, w = np.polynomial.legendre.leggauss(nodes)
        self.tau = self.bound * x
        qweights = self.bound * w
        log_pc = -np.log(
            2.0
            * self.gamma
            * math.atan(self.bound / self.gamma)
            * (1.0 + (self.tau / self.gamma) ** 2)
        )
        self.log_base = np.log(qweights) + log_pc - math.log(self.sigma_g * math.sqrt(2.0 * math.pi))

    @classmethod
    def from_signal_power(cls, signal_power: float, snr_db: float, cfg: JointConfig) -> "ConvolutionNoise":
        total_complex_variance = signal_power / (10.0 ** (snr_db / 10.0))
        component_variance = max(total_complex_variance / 2.0, 1e-8)
        gaussian_variance = (1.0 - cfg.cauchy_variance_fraction) * component_variance
        cauchy_variance = cfg.cauchy_variance_fraction * component_variance
        b = cfg.cauchy_standard_bound
        standard_variance = (b - math.atan(b)) / math.atan(b)
        gamma = math.sqrt(cauchy_variance / standard_variance)
        return cls(math.sqrt(gaussian_variance), gamma, b * gamma, cfg.quadrature_nodes)

    def sample_real(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        gaussian = rng.normal(scale=self.sigma_g, size=size)
        angle = math.atan(self.bound / self.gamma)
        cauchy = self.gamma * np.tan(rng.uniform(-angle, angle, size=size))
        return gaussian + cauchy

    def sample_complex(self, rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
        return self.sample_real(rng, size) + 1j * self.sample_real(rng, size)

    def log_prob_score_real(self, residual: np.ndarray, need_score: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        r = np.asarray(residual, dtype=float)
        expanded = r[..., None] - self.tau
        log_terms = self.log_base - 0.5 * (expanded / self.sigma_g) ** 2
        logp = logsumexp(log_terms, axis=-1)
        if not need_score:
            return logp, None
        posterior_weights = np.exp(log_terms - logp[..., None])
        score = np.sum(posterior_weights * (-expanded / self.sigma_g**2), axis=-1)
        return logp, score

    def log_prob_complex(self, residual: np.ndarray, need_score: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        lr, sr = self.log_prob_score_real(np.real(residual), need_score)
        li, si = self.log_prob_score_real(np.imag(residual), need_score)
        if not need_score:
            return lr + li, None
        return lr + li, sr + 1j * si

    def component_variance(self) -> float:
        a = self.bound / self.gamma
        cauchy_var = self.gamma**2 * (a - math.atan(a)) / math.atan(a)
        return self.sigma_g**2 + cauchy_var


def wrapped_normal_logpdf_score(
    value_minus_location: np.ndarray,
    mean_shift: np.ndarray | float,
    sd: float,
    lifts: int,
    need_score: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    delta = wrap_difference(np.asarray(value_minus_location) - np.asarray(mean_shift))
    winding = np.arange(-lifts, lifts + 1, dtype=float) * TWO_PI
    lifted = delta[..., None] + winding
    log_terms = -0.5 * (lifted / sd) ** 2 - math.log(sd * math.sqrt(2.0 * math.pi))
    logp = logsumexp(log_terms, axis=-1)
    if not need_score:
        return logp, None
    weights = np.exp(log_terms - logp[..., None])
    score_delta = np.sum(weights * (-lifted / sd**2), axis=-1)
    return logp, score_delta


def channel_transition_mean(previous: np.ndarray, spec: ChannelSpec, cfg: JointConfig) -> np.ndarray:
    return cfg.channel_rho * previous + (1.0 - cfg.channel_rho) * spec.means[None, :, :]


def frequency_transition_mean(
    previous_frequency: np.ndarray,
    previous_z: np.ndarray,
    z: np.ndarray,
    cfg: JointConfig,
) -> np.ndarray:
    means = np.empty_like(previous_frequency)
    for k, alphabet in enumerate(cfg.channel_alphabets):
        old_center = alphabet[previous_z[:, k]]
        new_center = alphabet[z[:, k]]
        carry = cfg.frequency_rho * (previous_frequency[:, k] - old_center)
        means[:, k] = new_center + np.where(z[:, k] == previous_z[:, k], carry, 0.0)
    return means


def source_prediction(state: dict[str, np.ndarray], spec: ChannelSpec) -> tuple[np.ndarray, np.ndarray]:
    current_source = np.exp(state["logamp"] + 1j * state["phase"])
    histories = state["history"].copy()
    histories[:, :, 0] = current_source
    prediction = np.zeros(len(current_source), dtype=np.complex128)
    for r, delay in enumerate(spec.delays):
        prediction += np.sum(state["channel"][:, :, r] * histories[:, :, delay], axis=1)
    return prediction, histories


def simulate_joint_trajectory(seed: int, snr_db: float, mode: str, cfg: JointConfig) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(stable_seed("joint-physics", seed, snr_db, mode))
    spec = channel_spec(mode)
    n, k, r = cfg.n_samples, cfg.emitters, spec.taps
    z = np.zeros((n, k), dtype=int)
    frequency = np.zeros((n, k), dtype=float)
    logamp = np.zeros((n, k), dtype=float)
    phase = np.zeros((n, k), dtype=float)
    channel = np.zeros((n, k, r), dtype=np.complex128)

    for emitter, alphabet in enumerate(cfg.channel_alphabets):
        z[0, emitter] = rng.integers(len(alphabet))
        frequency[0, emitter] = alphabet[z[0, emitter]] + rng.normal(scale=cfg.frequency_sd)
    logamp[0] = cfg.logamp_mean + rng.normal(scale=0.18, size=k)
    phase[0] = rng.uniform(0.0, TWO_PI, size=k)
    channel[0] = spec.means + complex_normal(rng, (k, r), 0.12)

    for t in range(1, n):
        z[t] = z[t - 1]
        if t % cfg.hop_interval == 0:
            for emitter, alphabet in enumerate(cfg.channel_alphabets):
                if rng.random() < cfg.hop_probability:
                    options = np.delete(np.arange(len(alphabet)), z[t - 1, emitter])
                    z[t, emitter] = rng.choice(options)
        fmean = frequency_transition_mean(frequency[t - 1 : t], z[t - 1 : t], z[t : t + 1], cfg)[0]
        frequency[t] = fmean + rng.normal(scale=cfg.frequency_sd, size=k)
        amean = logamp[t - 1] + cfg.logamp_reversion * (cfg.logamp_mean - logamp[t - 1])
        logamp[t] = amean + rng.normal(scale=cfg.logamp_sd, size=k)
        phase_mean = phase[t - 1] + TWO_PI * frequency[t] * cfg.dt
        phase[t] = wrap_phase(phase_mean + rng.normal(scale=cfg.phase_sd, size=k))
        hmean = cfg.channel_rho * channel[t - 1] + (1.0 - cfg.channel_rho) * spec.means
        channel[t] = hmean + complex_normal(rng, (k, r), cfg.channel_sd)

    source = np.exp(logamp + 1j * phase)
    clean = np.zeros(n, dtype=np.complex128)
    for tap, delay in enumerate(spec.delays):
        delayed = np.zeros_like(source)
        if delay == 0:
            delayed[:] = source
        else:
            delayed[delay:] = source[:-delay]
        clean += np.sum(channel[:, :, tap] * delayed, axis=1)

    signal_power = float(np.mean(np.abs(clean) ** 2))
    noise_law = ConvolutionNoise.from_signal_power(signal_power, snr_db, cfg)
    noise = noise_law.sample_complex(rng, n)
    observation = clean + noise
    achieved_snr = 10.0 * math.log10(signal_power / float(np.mean(np.abs(noise) ** 2)))
    return {
        "observation": observation,
        "clean": clean,
        "frequency": frequency,
        "z": z,
        "logamp": logamp,
        "phase": phase,
        "channel": channel,
        "noise_law": noise_law,
        "achieved_snr_db": achieved_snr,
        "signal_power": signal_power,
    }


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def weighted_circular_mean(phases: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return wrap_phase(np.angle(np.sum(weights[:, None] * np.exp(1j * phases), axis=0)))


def normal_logpdf(residual: np.ndarray, sd: float | np.ndarray) -> np.ndarray:
    scale = np.asarray(sd, dtype=float)
    return -0.5 * (residual / scale) ** 2 - np.log(scale * math.sqrt(2.0 * math.pi))


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in state.items()}


def take_state(state: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[index].copy() for key, value in state.items()}


def pack_state(state: dict[str, np.ndarray], spec: ChannelSpec) -> np.ndarray:
    return np.concatenate(
        [
            state["logamp"],
            state["frequency"],
            state["phase"],
            np.real(state["channel"]).reshape(len(state["logamp"]), -1),
            np.imag(state["channel"]).reshape(len(state["logamp"]), -1),
        ],
        axis=1,
    )


def unpack_state(packed: np.ndarray, template: dict[str, np.ndarray], spec: ChannelSpec) -> dict[str, np.ndarray]:
    p = len(packed)
    kr = 2 * spec.taps
    state = copy_state(template)
    state["logamp"] = packed[:, 0:2]
    state["frequency"] = packed[:, 2:4]
    state["phase"] = wrap_phase(packed[:, 4:6])
    real_h = packed[:, 6 : 6 + kr].reshape(p, 2, spec.taps)
    imag_h = packed[:, 6 + kr : 6 + 2 * kr].reshape(p, 2, spec.taps)
    state["channel"] = real_h + 1j * imag_h
    state["history"][:, :, 0] = np.exp(state["logamp"] + 1j * state["phase"])
    return state


def pack_score(score: dict[str, np.ndarray], spec: ChannelSpec) -> np.ndarray:
    return np.concatenate(
        [
            score["logamp"],
            score["frequency"],
            score["phase"],
            np.real(score["channel"]).reshape(len(score["logamp"]), -1),
            np.imag(score["channel"]).reshape(len(score["logamp"]), -1),
        ],
        axis=1,
    )


def proposal_steps(spec: ChannelSpec) -> np.ndarray:
    return np.array(
        [0.014, 0.014, 0.10, 0.10, 0.040, 0.040]
        + [0.018] * (2 * spec.taps)
        + [0.018] * (2 * spec.taps),
        dtype=float,
    )


def transition_target(
    candidate: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    observation: complex,
    noise: ConvolutionNoise,
    spec: ChannelSpec,
    cfg: JointConfig,
    need_score: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None, np.ndarray]:
    p = len(candidate["logamp"])
    prediction, histories = source_prediction(candidate, spec)
    residual = observation - prediction
    log_lik, residual_score = noise.log_prob_complex(residual, need_score)

    amp_mean = previous["logamp"] + cfg.logamp_reversion * (cfg.logamp_mean - previous["logamp"])
    amp_residual = candidate["logamp"] - amp_mean
    fmean = frequency_transition_mean(previous["frequency"], previous["z"], candidate["z"], cfg)
    f_residual = candidate["frequency"] - fmean
    phase_location = previous["phase"] + TWO_PI * candidate["frequency"] * cfg.dt
    phase_delta = candidate["phase"] - phase_location
    log_phase, phase_score = wrapped_normal_logpdf_score(
        phase_delta, 0.0, cfg.phase_sd, cfg.wrapped_lifts, need_score
    )
    hmean = channel_transition_mean(previous["channel"], spec, cfg)
    h_residual = candidate["channel"] - hmean

    log_target = (
        log_lik
        + np.sum(normal_logpdf(amp_residual, cfg.logamp_sd), axis=1)
        + np.sum(normal_logpdf(f_residual, cfg.frequency_sd), axis=1)
        + np.sum(log_phase, axis=1)
        + np.sum(normal_logpdf(np.real(h_residual), cfg.channel_sd), axis=(1, 2))
        + np.sum(normal_logpdf(np.imag(h_residual), cfg.channel_sd), axis=(1, 2))
    )
    if not need_score:
        return log_target, None, prediction

    score = {
        "logamp": -amp_residual / cfg.logamp_sd**2,
        "frequency": -f_residual / cfg.frequency_sd**2,
        "phase": phase_score.copy(),
        "channel": -np.real(h_residual) / cfg.channel_sd**2
        + 1j * (-np.imag(h_residual) / cfg.channel_sd**2),
    }
    score["frequency"] += phase_score * (-TWO_PI * cfg.dt)

    # The residual score is d log p(e) / d e.  Since e = y - prediction,
    # the score with respect to prediction has the opposite sign.
    pred_score = -residual_score
    current_source = np.exp(candidate["logamp"] + 1j * candidate["phase"])
    for tap, delay in enumerate(spec.delays):
        delayed_source = histories[:, :, delay]
        derivative_real_h = delayed_source
        derivative_imag_h = 1j * delayed_source
        score["channel"].real[:, :, tap] += (
            np.real(pred_score)[:, None] * np.real(derivative_real_h)
            + np.imag(pred_score)[:, None] * np.imag(derivative_real_h)
        )
        score["channel"].imag[:, :, tap] += (
            np.real(pred_score)[:, None] * np.real(derivative_imag_h)
            + np.imag(pred_score)[:, None] * np.imag(derivative_imag_h)
        )
        if delay == 0:
            derivative_amp = candidate["channel"][:, :, tap] * current_source
            derivative_phase = 1j * derivative_amp
            score["logamp"] += (
                np.real(pred_score)[:, None] * np.real(derivative_amp)
                + np.imag(pred_score)[:, None] * np.imag(derivative_amp)
            )
            score["phase"] += (
                np.real(pred_score)[:, None] * np.real(derivative_phase)
                + np.imag(pred_score)[:, None] * np.imag(derivative_phase)
            )
    return log_target, score, prediction


class JointScoreSurrogate:
    """SiLU random-feature network trained against exact joint target scores."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int, ridge: float, seed: int):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(scale=0.62, size=(input_dim, hidden))
        self.b = rng.normal(scale=0.28, size=hidden)
        self.beta = np.zeros((hidden + 1, output_dim))
        self.x_mean = np.zeros(input_dim)
        self.x_scale = np.ones(input_dim)
        self.y_mean = np.zeros(output_dim)
        self.y_scale = np.ones(output_dim)
        self.ridge = ridge

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        z = np.clip(x, -35.0, 35.0)
        return z / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "JointScoreSurrogate":
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0) + 1e-8
        self.y_mean = y.mean(axis=0)
        self.y_scale = y.std(axis=0) + 1e-8
        xs = (x - self.x_mean) / self.x_scale
        ys = (y - self.y_mean) / self.y_scale
        hidden = self.silu(xs @ self.w + self.b)
        design = np.column_stack([np.ones(len(hidden)), hidden])
        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.beta = np.linalg.solve(design.T @ design + penalty, design.T @ ys)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (x - self.x_mean) / self.x_scale
        hidden = self.silu(xs @ self.w + self.b)
        design = np.column_stack([np.ones(len(hidden)), hidden])
        return self.y_mean + self.y_scale * (design @ self.beta)

    def save(self, path: Path, validation: dict[str, float]) -> None:
        np.savez(
            path,
            w=self.w,
            b=self.b,
            beta=self.beta,
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale,
            validation=json.dumps(validation),
        )


def surrogate_features(
    candidate: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    observation: complex,
    noise: ConvolutionNoise,
    spec: ChannelSpec,
    cfg: JointConfig,
) -> np.ndarray:
    p = len(candidate["logamp"])
    amp_mean = previous["logamp"] + cfg.logamp_reversion * (cfg.logamp_mean - previous["logamp"])
    fmean = frequency_transition_mean(previous["frequency"], previous["z"], candidate["z"], cfg)
    phase_location = previous["phase"] + TWO_PI * candidate["frequency"] * cfg.dt
    phase_residual = wrap_difference(candidate["phase"] - phase_location)
    hmean = channel_transition_mean(previous["channel"], spec, cfg)
    hres = candidate["channel"] - hmean
    prediction, _ = source_prediction(candidate, spec)
    residual = (observation - prediction) / math.sqrt(2.0 * noise.component_variance())
    padded_h = np.zeros((p, 2, 2), dtype=np.complex128)
    padded_h[:, :, : spec.taps] = hres
    mode_code = np.zeros((p, 3))
    mode_code[:, {"LOS": 0, "MP": 1, "NLOS": 2}[spec.mode]] = 1.0
    return np.concatenate(
        [
            (candidate["logamp"] - amp_mean) / cfg.logamp_sd,
            (candidate["frequency"] - fmean) / cfg.frequency_sd,
            np.sin(phase_residual),
            np.cos(phase_residual),
            np.real(padded_h).reshape(p, -1) / cfg.channel_sd,
            np.imag(padded_h).reshape(p, -1) / cfg.channel_sd,
            np.column_stack([np.real(residual), np.imag(residual)]),
            candidate["frequency"] / cfg.sample_rate_hz,
            candidate["logamp"],
            mode_code,
            np.full((p, 1), math.log(noise.component_variance())),
        ],
        axis=1,
    )


def surrogate_output_pad(score: dict[str, np.ndarray], spec: ChannelSpec) -> np.ndarray:
    p = len(score["logamp"])
    padded = np.zeros((p, 2, 2), dtype=np.complex128)
    padded[:, :, : spec.taps] = score["channel"]
    return np.concatenate(
        [
            score["logamp"], score["frequency"], score["phase"],
            np.real(padded).reshape(p, -1), np.imag(padded).reshape(p, -1),
        ], axis=1,
    )


def surrogate_output_unpad(output: np.ndarray, spec: ChannelSpec) -> np.ndarray:
    p = len(output)
    real_h = output[:, 6:10].reshape(p, 2, 2)[:, :, : spec.taps]
    imag_h = output[:, 10:14].reshape(p, 2, 2)[:, :, : spec.taps]
    return np.concatenate(
        [output[:, :6], real_h.reshape(p, -1), imag_h.reshape(p, -1)], axis=1
    )


def random_training_block(
    rng: np.random.Generator,
    mode: str,
    snr_db: float,
    rows: int,
    cfg: JointConfig,
) -> tuple[np.ndarray, np.ndarray]:
    spec = channel_spec(mode)
    p = rows
    previous: dict[str, np.ndarray] = {
        "logamp": rng.normal(cfg.logamp_mean, 0.24, size=(p, 2)),
        "frequency": np.zeros((p, 2)),
        "z": np.zeros((p, 2), dtype=int),
        "phase": rng.uniform(0.0, TWO_PI, size=(p, 2)),
        "channel": spec.means[None, :, :] + complex_normal(rng, (p, 2, spec.taps), 0.18),
        "history": np.zeros((p, 2, spec.max_delay + 1), dtype=np.complex128),
    }
    for k, alphabet in enumerate(cfg.channel_alphabets):
        previous["z"][:, k] = rng.integers(len(alphabet), size=p)
        previous["frequency"][:, k] = alphabet[previous["z"][:, k]] + rng.normal(scale=0.8, size=p)
    for lag in range(spec.max_delay + 1):
        previous["history"][:, :, lag] = np.exp(
            rng.normal(cfg.logamp_mean, 0.25, size=(p, 2))
            + 1j * rng.uniform(0.0, TWO_PI, size=(p, 2))
        )
    candidate = copy_state(previous)
    candidate["z"] = previous["z"].copy()
    hop = rng.random((p, 2)) < 0.22
    for k, alphabet in enumerate(cfg.channel_alphabets):
        proposed = rng.integers(len(alphabet), size=p)
        candidate["z"][:, k] = np.where(hop[:, k], proposed, previous["z"][:, k])
    fmean = frequency_transition_mean(previous["frequency"], previous["z"], candidate["z"], cfg)
    candidate["frequency"] = fmean + rng.normal(scale=cfg.frequency_sd * 1.8, size=(p, 2))
    amean = previous["logamp"] + cfg.logamp_reversion * (cfg.logamp_mean - previous["logamp"])
    candidate["logamp"] = amean + rng.normal(scale=cfg.logamp_sd * 1.8, size=(p, 2))
    phase_mean = previous["phase"] + TWO_PI * candidate["frequency"] * cfg.dt
    candidate["phase"] = wrap_phase(phase_mean + rng.normal(scale=cfg.phase_sd * 1.8, size=(p, 2)))
    hmean = channel_transition_mean(previous["channel"], spec, cfg)
    candidate["channel"] = hmean + complex_normal(rng, (p, 2, spec.taps), cfg.channel_sd * 1.8)
    candidate["history"] = previous["history"].copy()
    candidate["history"][:, :, 1:] = previous["history"][:, :, :-1]
    candidate["history"][:, :, 0] = np.exp(candidate["logamp"] + 1j * candidate["phase"])
    signal_power = float(np.mean(np.abs(source_prediction(candidate, spec)[0]) ** 2))
    noise = ConvolutionNoise.from_signal_power(signal_power, snr_db, cfg)
    observation = source_prediction(candidate, spec)[0] + noise.sample_complex(rng, p)
    _, score, _ = transition_target(candidate, previous, observation, noise, spec, cfg, True)
    x = surrogate_features(candidate, previous, observation, noise, spec, cfg)
    y = surrogate_output_pad(score, spec)
    return x, y


def train_joint_surrogate(cfg: JointConfig) -> tuple[JointScoreSurrogate, dict[str, float]]:
    rng = np.random.default_rng(31977)
    x_parts, y_parts = [], []
    for mode in cfg.channel_modes:
        rows_each = cfg.surrogate_train_rows_per_mode
        snr_values = np.resize(np.asarray(cfg.snr_levels_db), rows_each)
        for snr in cfg.snr_levels_db:
            rows = int(np.sum(snr_values == snr))
            x, y = random_training_block(rng, mode, float(snr), rows, cfg)
            x_parts.append(x); y_parts.append(y)
    x = np.vstack(x_parts); y = np.vstack(y_parts)
    order = rng.permutation(len(x)); split = int(0.82 * len(x))
    train, valid = order[:split], order[split:]
    model = JointScoreSurrogate(x.shape[1], y.shape[1], cfg.surrogate_hidden, cfg.surrogate_ridge, 41821)
    model.fit(x[train], y[train])
    pred = model.predict(x[valid])
    rmse = float(np.sqrt(np.mean((pred - y[valid]) ** 2)))
    corr = float(np.corrcoef(pred.ravel(), y[valid].ravel())[0, 1])
    return model, {"validation_score_rmse": rmse, "validation_score_correlation": corr, "training_rows": int(len(train)), "validation_rows": int(len(valid))}


class JointParticleFilter:
    def __init__(self, cfg: JointConfig, mode: str, architecture: str, surrogate: JointScoreSurrogate | None, seed: int):
        self.cfg = cfg
        self.spec = channel_spec(mode)
        self.architecture = architecture
        self.surrogate = surrogate
        self.rng = np.random.default_rng(seed)

    def initialize(self) -> dict[str, np.ndarray]:
        p, k = self.cfg.particles, self.cfg.emitters
        state: dict[str, np.ndarray] = {
            "logamp": self.rng.normal(self.cfg.logamp_mean, 0.28, size=(p, k)),
            "frequency": np.zeros((p, k)),
            "z": np.zeros((p, k), dtype=int),
            "phase": self.rng.uniform(0.0, TWO_PI, size=(p, k)),
            "channel": self.spec.means[None, :, :] + complex_normal(self.rng, (p, k, self.spec.taps), 0.24),
            "history": np.zeros((p, k, self.spec.max_delay + 1), dtype=np.complex128),
        }
        for emitter, alphabet in enumerate(self.cfg.channel_alphabets):
            state["z"][:, emitter] = self.rng.integers(len(alphabet), size=p)
            state["frequency"][:, emitter] = alphabet[state["z"][:, emitter]] + self.rng.normal(scale=1.0, size=p)
        state["history"][:, :, 0] = np.exp(state["logamp"] + 1j * state["phase"])
        return state

    def propagate(self, previous: dict[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
        cfg, p = self.cfg, cfg_particles(self.cfg)
        state = copy_state(previous)
        state["z"] = previous["z"].copy()
        if t > 0 and t % cfg.hop_interval == 0:
            jump = self.rng.random((p, 2)) < cfg.hop_probability
            for k, alphabet in enumerate(cfg.channel_alphabets):
                draw = self.rng.integers(len(alphabet) - 1, size=p)
                current = previous["z"][:, k]
                alternatives = draw + (draw >= current)
                state["z"][:, k] = np.where(jump[:, k], alternatives, current)
        fmean = frequency_transition_mean(previous["frequency"], previous["z"], state["z"], cfg)
        state["frequency"] = fmean + self.rng.normal(scale=cfg.frequency_sd, size=(p, 2))
        amean = previous["logamp"] + cfg.logamp_reversion * (cfg.logamp_mean - previous["logamp"])
        state["logamp"] = amean + self.rng.normal(scale=cfg.logamp_sd, size=(p, 2))
        phase_mean = previous["phase"] + TWO_PI * state["frequency"] * cfg.dt
        state["phase"] = wrap_phase(phase_mean + self.rng.normal(scale=cfg.phase_sd, size=(p, 2)))
        hmean = channel_transition_mean(previous["channel"], self.spec, cfg)
        state["channel"] = hmean + complex_normal(self.rng, (p, 2, self.spec.taps), cfg.channel_sd)
        state["history"] = previous["history"].copy()
        if self.spec.max_delay > 0:
            state["history"][:, :, 1:] = previous["history"][:, :, :-1]
        state["history"][:, :, 0] = np.exp(state["logamp"] + 1j * state["phase"])
        return state

    def proposal_score(
        self,
        state: dict[str, np.ndarray],
        previous: dict[str, np.ndarray],
        observation: complex,
        noise: ConvolutionNoise,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.architecture == "Analytical joint score":
            target, score, _ = transition_target(state, previous, observation, noise, self.spec, self.cfg, True)
            packed = pack_score(score, self.spec)
        elif self.architecture == "Amortized joint surrogate":
            target, _, _ = transition_target(state, previous, observation, noise, self.spec, self.cfg, False)
            features = surrogate_features(state, previous, observation, noise, self.spec, self.cfg)
            packed = surrogate_output_unpad(self.surrogate.predict(features), self.spec)
        else:
            raise ValueError(self.architecture)
        return target, np.clip(packed, -self.cfg.drift_clip, self.cfg.drift_clip)

    def mala_move(
        self,
        state: dict[str, np.ndarray],
        previous: dict[str, np.ndarray],
        observation: complex,
        noise: ConvolutionNoise,
    ) -> tuple[dict[str, np.ndarray], float]:
        spec, cfg = self.spec, self.cfg
        current = copy_state(state)
        accepted_total = 0
        steps = proposal_steps(spec)
        phase_columns = np.array([4, 5])
        euclidean_columns = np.array([i for i in range(len(steps)) if i not in phase_columns])
        for _ in range(cfg.mala_steps):
            x = pack_state(current, spec)
            target, score = self.proposal_score(current, previous, observation, noise)
            shift = 0.5 * steps**2 * score
            raw = x + shift + self.rng.normal(size=x.shape) * steps
            raw[:, phase_columns] = wrap_phase(raw[:, phase_columns])
            proposal = unpack_state(raw, current, spec)
            prop_target, prop_score = self.proposal_score(proposal, previous, observation, noise)
            prop_shift = 0.5 * steps**2 * prop_score

            delta_forward = raw[:, euclidean_columns] - x[:, euclidean_columns] - shift[:, euclidean_columns]
            delta_reverse = x[:, euclidean_columns] - raw[:, euclidean_columns] - prop_shift[:, euclidean_columns]
            logq_forward = np.sum(normal_logpdf(delta_forward, steps[euclidean_columns]), axis=1)
            logq_reverse = np.sum(normal_logpdf(delta_reverse, steps[euclidean_columns]), axis=1)
            for column in phase_columns:
                lf, _ = wrapped_normal_logpdf_score(
                    raw[:, column] - x[:, column], shift[:, column], steps[column], cfg.wrapped_lifts, False
                )
                lr, _ = wrapped_normal_logpdf_score(
                    x[:, column] - raw[:, column], prop_shift[:, column], steps[column], cfg.wrapped_lifts, False
                )
                logq_forward += lf; logq_reverse += lr
            log_alpha = prop_target - target + logq_reverse - logq_forward
            accept = np.log(self.rng.random(len(x))) < np.minimum(0.0, log_alpha)
            accepted_total += int(np.sum(accept))
            for key in ("logamp", "frequency", "phase", "channel", "history"):
                current[key][accept] = proposal[key][accept]
        return current, accepted_total / (len(state["logamp"]) * cfg.mala_steps)

    def run(self, observation: np.ndarray, noise: ConvolutionNoise) -> dict[str, np.ndarray | float]:
        cfg, p = self.cfg, cfg_particles(self.cfg)
        state = self.initialize()
        log_weights = np.full(p, -math.log(p))
        frequency_est, phase_est, channel_est = [], [], []
        particle_ess, unique_ancestors, acceptance = [], [], []
        resampling_count = 0
        start = time.perf_counter()
        for t, y_t in enumerate(observation):
            previous = copy_state(state)
            state = self.propagate(previous, t)
            prediction, _ = source_prediction(state, self.spec)
            log_lik, _ = noise.log_prob_complex(y_t - prediction, False)
            log_weights = log_weights + log_lik
            log_weights -= logsumexp(log_weights)
            weights = np.exp(log_weights)
            ess = 1.0 / np.sum(weights**2)
            force_move = (t + 1) % cfg.forced_move_interval == 0
            if ess < cfg.resample_fraction * p or force_move:
                index = systematic_resample(weights, self.rng)
                state = take_state(state, index)
                previous = take_state(previous, index)
                unique_ancestors.append(len(np.unique(index)))
                log_weights[:] = -math.log(p)
                resampling_count += 1
                state, acc = self.mala_move(state, previous, y_t, noise)
                acceptance.append(acc)
                weights = np.full(p, 1.0 / p)
                ess = float(p)
            frequency_est.append(np.sum(weights[:, None] * state["frequency"], axis=0))
            phase_est.append(weighted_circular_mean(state["phase"], weights))
            channel_est.append(np.sum(weights[:, None, None] * state["channel"], axis=0))
            particle_ess.append(ess)
        elapsed = time.perf_counter() - start
        return {
            "frequency": np.asarray(frequency_est),
            "phase": np.asarray(phase_est),
            "channel": np.asarray(channel_est),
            "particle_ess": np.asarray(particle_ess),
            "unique_ancestors": np.asarray(unique_ancestors) if unique_ancestors else np.array([p]),
            "mala_acceptance": float(np.mean(acceptance)) if acceptance else math.nan,
            "resampling_count": resampling_count,
            "latency_ms_per_sample": 1000.0 * elapsed / len(observation),
        }


def cfg_particles(cfg: JointConfig) -> int:
    return int(cfg.particles)


def circular_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wrap_difference(estimate - truth) ** 2)))


def channel_nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    numerator = np.mean(np.abs(estimate - truth) ** 2)
    denominator = np.mean(np.abs(truth) ** 2)
    return float(numerator / denominator)


def evaluate_run(
    truth: dict[str, np.ndarray | float],
    output: dict[str, np.ndarray | float],
    burn: int,
) -> dict[str, float]:
    sl = slice(burn, None)
    freq_error = np.asarray(output["frequency"])[sl] - np.asarray(truth["frequency"])[sl]
    return {
        "mse_frequency": float(np.mean(np.sum(freq_error**2, axis=1))),
        "phase_circular_rmse": circular_rmse(np.asarray(output["phase"])[sl], np.asarray(truth["phase"])[sl]),
        "channel_nmse": channel_nmse(np.asarray(output["channel"])[sl], np.asarray(truth["channel"])[sl]),
        "particle_ess_mean": float(np.mean(np.asarray(output["particle_ess"])[sl])),
        "unique_ancestor_mean": float(np.mean(np.asarray(output["unique_ancestors"]))),
        "mala_acceptance": float(output["mala_acceptance"]),
        "resampling_count": float(output["resampling_count"]),
        "latency_ms_per_sample": float(output["latency_ms_per_sample"]),
    }


def paired_contrasts(frame: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    key = ["seed", "snr_db", "channel"]
    wide = frame.pivot(index=key, columns="architecture", values=list(metrics))
    rows: list[dict[str, object]] = []
    groups = [("Overall", "All", np.ones(len(wide), dtype=bool))]
    for snr in sorted(frame.snr_db.unique(), reverse=True):
        groups.append(("SNR", str(snr), wide.index.get_level_values("snr_db") == snr))
    for mode in frame.channel.unique():
        groups.append(("Channel", mode, wide.index.get_level_values("channel") == mode))
    for group, level, mask in groups:
        sub = wide[mask]
        for metric in metrics:
            delta = sub[(metric, "Amortized joint surrogate")] - sub[(metric, "Analytical joint score")]
            se = delta.std(ddof=1) / math.sqrt(len(delta))
            rows.append({
                "group": group,
                "level": level,
                "metric": metric,
                "n_pairs": len(delta),
                "mean_difference_surrogate_minus_analytical": delta.mean(),
                "ci95_low": delta.mean() - 1.96 * se,
                "ci95_high": delta.mean() + 1.96 * se,
            })
    return pd.DataFrame(rows)


def summarize_cells(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mse_frequency", "phase_circular_rmse", "channel_nmse",
        "particle_ess_mean", "unique_ancestor_mean", "mala_acceptance",
        "resampling_count", "latency_ms_per_sample", "achieved_snr_db",
    ]
    grouped = frame.groupby(["architecture", "snr_db", "channel"])
    pieces = [grouped.size().rename("n")]
    for metric in metrics:
        pieces.append(grouped[metric].mean().rename(metric + "_mean"))
        pieces.append(grouped[metric].std(ddof=1).rename(metric + "_sd"))
    return pd.concat(pieces, axis=1).reset_index()


def run_factorial(output: Path, repetitions: int, cfg: JointConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    surrogate, validation = train_joint_surrogate(cfg)
    surrogate.save(output / "joint_surrogate_weights.npz", validation)
    architectures = ("Analytical joint score", "Amortized joint surrogate")
    rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    total = repetitions * len(cfg.snr_levels_db) * len(cfg.channel_modes)
    complete = 0
    for seed in range(repetitions):
        for snr in cfg.snr_levels_db:
            for mode in cfg.channel_modes:
                truth = simulate_joint_trajectory(seed, snr, mode, cfg)
                for architecture in architectures:
                    engine = JointParticleFilter(
                        cfg, mode, architecture, surrogate if "surrogate" in architecture.lower() else None,
                        stable_seed("joint-inference", seed, snr, mode, architecture),
                    )
                    result = engine.run(np.asarray(truth["observation"]), truth["noise_law"])
                    metrics = evaluate_run(truth, result, burn=cfg.hop_interval)
                    rows.append({
                        "seed": seed, "architecture": architecture, "snr_db": snr,
                        "channel": mode, "achieved_snr_db": truth["achieved_snr_db"], **metrics,
                    })
                    if seed == 0 and snr == 5.0 and mode == "NLOS":
                        for t in range(cfg.n_samples):
                            example_rows.append({
                                "architecture": architecture, "sample": t,
                                "true_f1_hz": truth["frequency"][t, 0],
                                "inferred_f1_hz": result["frequency"][t, 0],
                                "true_f2_hz": truth["frequency"][t, 1],
                                "inferred_f2_hz": result["frequency"][t, 1],
                                "true_phase1": truth["phase"][t, 0],
                                "inferred_phase1": result["phase"][t, 0],
                                "particle_ess": result["particle_ess"][t],
                            })
                complete += 1
                if complete % max(1, total // 10) == 0:
                    print(f"completed {complete}/{total} paired joint trajectories", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "joint_factorial_results.csv", index=False)
    summarize_cells(frame).to_csv(output / "joint_cell_summary.csv", index=False)
    contrast_metrics = ["mse_frequency", "phase_circular_rmse", "channel_nmse", "particle_ess_mean", "latency_ms_per_sample"]
    paired_contrasts(frame, contrast_metrics).to_csv(output / "joint_paired_effects.csv", index=False)
    pd.DataFrame(example_rows).to_csv(output / "joint_example_trace.csv", index=False)
    metadata = {
        "experiment_name": "Signals Project Executed Joint Torus SSM",
        "validation_scope": "Operationally representative simulation validation; not field or operational SIGINT validation.",
        "design": "2 x 3 x 3 paired full factorial",
        "unique_physical_trajectories": repetitions * 9,
        "experimental_rows": len(frame),
        "repetitions_per_cell": repetitions,
        "config": asdict(cfg),
        "surrogate_validation": validation,
        "executed_components": [
            "joint sequential particles over amplitude, frequency, hopping state, torus phase, channel, and delay buffer",
            "Gaussian plus truncated-Cauchy convolution likelihood by Gauss-Legendre log-sum-exp quadrature",
            "flat-torus tangent MALA with wrapped-Gaussian forward and reverse proposal densities",
            "latent tapped-delay complex channel propagation",
            "exact-target correction for analytical and amortized score proposals",
        ],
    }
    (output / "joint_experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--particles", type=int, default=96)
    parser.add_argument("--samples", type=int, default=72)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = JointConfig(particles=args.particles, n_samples=args.samples)
    run_factorial(args.output, args.repetitions, cfg)


if __name__ == "__main__":
    main()
