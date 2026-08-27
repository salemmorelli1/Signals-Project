"""Numerical and structural regression tests for the executed joint SSM."""

from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import replace

import numpy as np

from src.signals_project.joint_ssm import (
    ConvolutionNoise,
    JointConfig,
    JointParticleFilter,
    channel_spec,
    pack_score,
    pack_state,
    simulate_joint_trajectory,
    transition_target,
    unpack_state,
    wrap_difference,
    wrapped_normal_logpdf_score,
)


class ExactNoiseLawTests(unittest.TestCase):
    def setUp(self) -> None:
        self.noise = ConvolutionNoise(sigma_g=0.31, gamma=0.17, bound=0.85, nodes=48)

    def test_convolution_density_integrates_to_one(self) -> None:
        grid = np.linspace(-3.5, 3.5, 16001)
        logp, _ = self.noise.log_prob_score_real(grid, False)
        integral = np.trapezoid(np.exp(logp), grid)
        self.assertAlmostEqual(float(integral), 1.0, places=5)

    def test_convolution_score_matches_finite_difference(self) -> None:
        residual = np.array([-0.71, -0.19, 0.03, 0.46, 0.92])
        _, score = self.noise.log_prob_score_real(residual, True)
        step = 1e-6
        plus, _ = self.noise.log_prob_score_real(residual + step, False)
        minus, _ = self.noise.log_prob_score_real(residual - step, False)
        numerical = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(score, numerical, rtol=1e-6, atol=1e-7)


class TorusGeometryTests(unittest.TestCase):
    def test_wrapped_gaussian_integrates_against_haar_volume(self) -> None:
        grid = np.linspace(0.0, 2.0 * math.pi, 20001)
        logp, _ = wrapped_normal_logpdf_score(grid - 1.7, 0.23, 0.31, 4, False)
        integral = np.trapezoid(np.exp(logp), grid)
        self.assertAlmostEqual(float(integral), 1.0, places=5)

    def test_wrapped_score_matches_finite_difference(self) -> None:
        delta = np.array([-2.5, -0.4, 0.2, 2.7])
        _, score = wrapped_normal_logpdf_score(delta, -0.13, 0.42, 4, True)
        step = 1e-6
        plus, _ = wrapped_normal_logpdf_score(delta + step, -0.13, 0.42, 4, False)
        minus, _ = wrapped_normal_logpdf_score(delta - step, -0.13, 0.42, 4, False)
        numerical = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(score, numerical, rtol=1e-6, atol=1e-7)

    def test_wrap_difference_is_periodic(self) -> None:
        x = np.array([-5.2, -0.1, 0.0, 1.4, 7.8])
        np.testing.assert_allclose(wrap_difference(x), wrap_difference(x + 8.0 * math.pi))


class JointTargetTests(unittest.TestCase):
    def test_channel_alphabets_are_nyquist_valid_and_cochannel(self) -> None:
        cfg = JointConfig()
        first, second = cfg.channel_alphabets
        self.assertLess(float(np.max(first)), cfg.sample_rate_hz / 2.0)
        self.assertLess(float(np.max(second)), cfg.sample_rate_hz / 2.0)
        self.assertEqual(set(first).intersection(second), {42.0, 54.0})

    def test_joint_score_matches_all_coordinate_finite_differences(self) -> None:
        cfg = replace(JointConfig(), particles=1, n_samples=12)
        truth = simulate_joint_trajectory(31, 5.0, "MP", cfg)
        engine = JointParticleFilter(cfg, "MP", "Analytical joint score", None, 71)
        previous = engine.initialize()
        candidate = engine.propagate(previous, 1)
        spec = channel_spec("MP")
        observation = np.asarray(truth["observation"])[1]
        noise = truth["noise_law"]
        _, score, _ = transition_target(candidate, previous, observation, noise, spec, cfg, True)
        analytical = pack_score(score, spec)[0]
        packed = pack_state(candidate, spec)[0]
        numerical = np.empty_like(packed)
        step = 1e-6
        for column in range(len(packed)):
            plus = packed.copy()
            minus = packed.copy()
            plus[column] += step
            minus[column] -= step
            plus_state = unpack_state(plus[None, :], candidate, spec)
            minus_state = unpack_state(minus[None, :], candidate, spec)
            f_plus, _, _ = transition_target(plus_state, previous, observation, noise, spec, cfg, False)
            f_minus, _, _ = transition_target(minus_state, previous, observation, noise, spec, cfg, False)
            numerical[column] = (f_plus[0] - f_minus[0]) / (2.0 * step)
        np.testing.assert_allclose(analytical, numerical, rtol=2e-7, atol=2e-7)

    def test_filter_api_has_no_truth_or_true_channel_argument(self) -> None:
        parameters = set(inspect.signature(JointParticleFilter.run).parameters)
        self.assertEqual(parameters, {"self", "observation", "noise"})
        self.assertFalse(any("truth" in name or "true_channel" in name for name in parameters))

    def test_blind_filter_smoke_outputs_are_finite_and_torus_valued(self) -> None:
        cfg = replace(JointConfig(), particles=20, n_samples=18, hop_interval=6)
        truth = simulate_joint_trajectory(44, -5.0, "NLOS", cfg)
        engine = JointParticleFilter(cfg, "NLOS", "Analytical joint score", None, 101)
        output = engine.run(np.asarray(truth["observation"]), truth["noise_law"])
        for key in ("frequency", "phase", "channel", "particle_ess"):
            self.assertTrue(np.all(np.isfinite(np.asarray(output[key]))), key)
        phase = np.asarray(output["phase"])
        self.assertTrue(np.all((phase >= 0.0) & (phase < 2.0 * math.pi)))


if __name__ == "__main__":
    unittest.main()
