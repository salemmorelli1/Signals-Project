# Filter and evidence audit — October 11, 2026

Base reviewed: `3e2b040c848b32ca3a76e4659ad497780c79d2a3`.
Corrected filter protocol: `joint-ssm-v2`.

The existing CI was passing, but it did not test the filter's first observation,
the interpretation of its ESS diagnostic, or invalid statistical covariance.
These were reproducible scientific defects rather than a failed deployment.
The corrected implementation has been rerun on the complete original factorial
design. No superiority or operational validation is established by this audit.

## Findings and changes

| Finding | Reproduced behavior | Correction and regression coverage |
| --- | --- | --- |
| First observation used a nonexistent transition | Propagation at `t=0` moved the initial state and filled a source lag at `t=-1`. In NLOS, that changed the first prediction from zero to a nonzero delayed signal, although the generator's first observation contains only noise. | Weight the initial prior at `t=0`; propagate and use the ancestor-conditioned MALA target starting at `t=1`. Test initial delay history, NLOS prediction, and the first transition/move times. |
| Initial inference prior differed from the generator | Log-amplitude, frequency, and channel standard deviations were respectively 0.28, 1.0, and 0.24 in inference versus 0.18, 0.42, and 0.12 in generation. | Share the configured initial dispersions; validate the new parameters and test initialization against the generator's configuration. |
| ESS concealed weight collapse | A concentrated likelihood produced weighting ESS near 1 but reported the full particle count after resampling. | Retain weighting ESS before resampling and a separate post-resampling trace. Test collapse and both diagnostic meanings. |
| Singular covariance produced invalid ordinary F inference | Adding machine epsilon and using a pseudoinverse produced an enormous F statistic for deterministic contrasts. | Reject zero-variance or numerically rank-deficient contrasts. Solve a standardized nonsingular system and test singular cases and invariance to common unit scaling. |
| Verified WiSig bytes could differ from parsed bytes | The adapter hashed the input pathname, then reopened it for pickle loading; replacing that file between operations bypassed the byte-integrity check. | Hash a private disk-backed snapshot, then parse those exact bytes. Test pathname replacement and wrong-checksum rejection. The trusted-official-pickle acknowledgment remains required. |
| WiSig manifest lineage could be altered | Materialization silently converted some noninteger indices and did not recheck recording labels or split boundaries. | Revalidate indices, source identifiers, distinct transmitters, unique mixture IDs, non-equalized captures, receiver/day splits, and finite SIR. Require usable train/validation/test partitions when building a manifest. |
| Failed WiSig numerical preparation could mix runs | A new manifest was written before checking whether the new recording arrays could be materialized. | Complete numerical checks before replacing preparation files; reject manifest-only reuse beside an old NPZ. Test that an invalid recording leaves a prior preparation unchanged. Save hashes for newly prepared artifacts. This is not a transaction guarantee for arbitrary filesystem write failures. |
| Publication could retain stale conclusions | The HTML/report contained results and conclusions typed independently of the current artifacts; validation checked many statistical ranges without recomputing the complete analysis. | Derive conclusions from the current results, recompute all 35 effect rows and primary summaries during validation, and check source/raw artifact hashes. Test forged inference-summary rejection. |

Production Python changes are in `src/signals_project/joint_ssm.py`,
`src/signals_project/wisig_realdata.py`, `scripts/analyze_joint_results.py`,
`scripts/build_site.py`, `scripts/build_report.py`, and
`scripts/validate_repository.py`.

## Complete corrected replay

The replay retains two methods, three SNRs (+15, +5, -5 dB), three channel modes
(LOS, MP, NLOS), 100 seed blocks, 64 particles, and 60 I/Q samples. It produces
900 physical trajectories and 1,800 method-level records. Matching the old and
new records by seed, architecture, SNR, and channel gives a maximum achieved-SNR
difference of **0.0 dB**. The physical trajectories and factorial design were
kept fixed while correcting inference.

| Metric | Retained v1 | Corrected v2 |
| --- | ---: | ---: |
| Analytical mean frequency MSE (Hz²) | 246.756898 | 218.427920 |
| Surrogate mean frequency MSE (Hz²) | 238.411066 | 219.207013 |
| Analytical mean ESS | 59.050278, after resampling | 28.993613, before resampling |
| Surrogate mean ESS | 59.044773, after resampling | 29.089902, before resampling |

The two ESS columns measure different stages; the lower corrected number is
not evidence of a new deterioration in particle health.

The v2 surrogate-minus-analytical raw frequency-MSE difference is **+0.779093
Hz²**, with a descriptive 95% seed-block t interval **[-14.227943, 15.786129]**.
This interval includes zero. It is not a multiplicity-adjusted superiority test.

The prespecified primary response remains log frequency MSE. Seven omnibus
within-seed effects are Holm-adjusted together; no threshold or favorable
endpoint was substituted during this correction.

| Primary effect | F | df | Holm-adjusted p |
| --- | ---: | --- | ---: |
| Architecture | 0.304165 | 1, 99 | 1.0 |
| SNR | 28.586994 | 2, 98 | 1.161578e-9 |
| Channel | 20.707728 | 2, 98 | 1.892317e-7 |
| Architecture × SNR | 0.004958 | 2, 98 | 1.0 |
| Architecture × Channel | 0.716003 | 2, 98 | 1.0 |
| SNR × Channel | 4.871426 | 4, 96 | 0.006392 |
| Architecture × SNR × Channel | 1.036444 | 4, 96 | 1.0 |

Hotelling's ordinary F reference requires independent seed profiles,
multivariate-normal contrast vectors, and full-rank covariance. The new
implementation rejects invalid covariance but does not prove the normality
assumption. Timing is descriptive: analytical execution precedes surrogate
execution, and recorded durations depend on host and warm-up effects.

## Evidence retention and claim limits

All eight original simulation artifacts are retained byte-for-byte under
`data/baseline_v1/`. `archive_manifest.json` records their SHA-256 hashes and
original source commit. The repository validator checks this archive as well as
the corrected model source, result table, example trace, and surrogate weights.
Old results are retained for audit and comparison; they are not silently relabeled
as v2 evidence.

The simulator/filter assumes the channel mode, frequency alphabets, number of
emitters, and calibrated noise parameters are known. The filter receives
observations and those configured assumptions, not realized latent truth. Truth
is retained for scoring after inference. This is controlled simulation
validation, not unrestricted blind recovery from arbitrary captured RF.

The previously prepared 3,200 WiSig digital overlaps remain preparation
evidence. This change tests the adapter using small synthetic acquisition
fixtures; it does not rerun inference on the private ManyRx recording archive.
Hardware-origin source recordings mixed digitally are not simultaneous RF
overlaps, hardware-in-the-loop validation, or operational SIGINT evidence.
The dataset's license and provenance requirements remain in force.

## Verification and reproduction

Local verification uses Python 3.12 and the exact 27-package CI lock:
50 tests plus 25 pytest subtests, Ruff, mypy on seven source files, compilation,
dependency checks, complete artifact recomputation, and repository contracts.
The report is 27 pages; all pages were rendered and inspected. Derived artifact
rebuilds must leave the committed tree unchanged.

The validation workflow now runs on both Ubuntu 24.04 and Windows. Python source
uses LF checkout rules so source hashes agree across platforms. The smoke replay
writes to the runner's temporary directory instead of an assumed Unix `/tmp`.
GitHub job conclusions must be checked on the published PR's final commit.

To verify the committed evidence, with the locked environment activated and
`PYTHONPATH=.`:

```bash
python -m pytest -q
python scripts/analyze_joint_results.py
python scripts/build_site.py
python scripts/build_report.py
python scripts/validate_repository.py
git diff --exit-code
```

To run a separate complete experiment without overwriting the reviewed evidence:

```bash
python -m src.signals_project.joint_ssm \
  --output ../Signals-Project-replay-v2 \
  --repetitions 100 --particles 64 --samples 60
```

Raw timing will differ on another host. A new experiment should be reviewed as
new evidence before replacing the frozen result table and derived publications.
