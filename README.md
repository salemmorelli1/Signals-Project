# Signals Project

This repository evaluates a blind, joint torus-valued state-space model for
two-source complex I/Q de-mixing. It now has two deliberately separate evidence
lanes:

1. **WiSig external real-data benchmark (primary extension).** Genuine,
   non-equalized WiSig receiver recordings are paired into controlled digital
   overlaps with exact component truth, receiver/day leakage controls, and
   deterministic manifests.
2. **Executed joint-SSM simulation baseline.** The existing 2 × 3 × 3 paired
   factorial benchmark remains frozen and fully reproducible.

The WiSig path uses the public
[`WiSig-dataset/wisig-examples`](https://github.com/WiSig-dataset/wisig-examples)
schema and the official **ManyRx** compact subset distributed by UCLA. The
examples repository contains loaders and notebooks; the 1.2 GB data file must
be downloaded separately from the
[official WiSig dataset page](https://cores.ee.ucla.edu/downloads/datasets/wisig/).

## Evidence status

| Lane | Data source | Status | Defensible statement |
|---|---|---:|---|
| Joint torus SSM baseline | Simulated I/Q | Complete | Executed and numerically verified in simulation |
| WiSig ingestion and overlap construction | Public hardware-captured I/Q | Executed; 3,200 overlaps prepared | Provenance-checked external-evaluation input |
| WiSig model results | Public hardware-captured I/Q | Not yet executed | No performance claim until artifacts are produced |
| Simultaneous RF / hardware-in-the-loop | New controlled capture | Not included | No operational SIGINT claim |

WiSig transmitters were recorded separately. This repository therefore says
**digitally overlapped hardware-captured I/Q**, never “simultaneous over-the-air
co-channel capture.” The WiSig benchmark can externally test receiver/day
transfer and de-mixing reconstruction, but it cannot validate frequency-hop
tracking, true channel taps, or a hardware-in-the-loop deployment.

## Prepare the real-data benchmark

The tested runtime is CPython 3.12. The exact CI environment is recorded in
`requirements-ci-lock.txt`; `requirements.txt` contains the supported direct
runtime dependencies.

1. Download the **ManyRx** compact subset from the official WiSig page.
2. Extract its `.pkl` file to `external_data/wisig/ManyRx.pkl`.
3. Run the adapter:

```bash
python --version  # must report Python 3.12.x
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt

PYTHONPATH=. python -m src.signals_project.wisig_realdata \
  --dataset external_data/wisig/ManyRx.pkl \
  --output external_data/derived/wisig_manyrx \
  --mixtures-per-domain 25 \
  --seed 2026 \
  --expected-sha256 f634d90585167437d196c89b7c5a344903180bf4f5a55d02d175b8074e009d9a \
  --trust-official-pickle
```

The acknowledgement flag exists because Python pickle files can execute code
when loaded. The adapter verifies the audited ManyRx SHA-256 above before
unpickling; both the explicit acknowledgement and matching digest are required.
Use the flag only for the official UCLA file. The raw dataset and derived I/Q
are ignored by Git; only compact aggregate results and provenance metadata
should later be committed.

The command creates:

- `wisig_overlap_manifest.csv`: exact Tx/Rx/day/signal references and SIR;
- `wisig_digital_overlaps.npz`: source A, scaled source B, and their exact sum;
- `wisig_preparation_metadata.json`: SHA-256, split counts, license, and claim scope.

Default construction uses non-equalized I/Q (`equalized=0`) and 25 mixtures per
receiver/day domain, so the working subset is small compared with the original
download. Increase the cap only for the locked confirmatory run.

## Leakage-safe split

- **Train:** training receivers on training days.
- **Validation:** the same training receivers on a held-out day.
- **Test:** receivers never used in training or validation, on all days.

Digital pairs are formed only within a common receiver/day domain and always
use two distinct transmitters. The test component recordings remain sealed
until preprocessing, tuning, and model selection are frozen.

## Executed simulation baseline

The frozen baseline carries amplitude, frequency, hop state, phase on
\(\mathbb T^2\), latent complex tapped-delay channels, and delayed source
history. It compares an analytical score with an amortized SiLU score under an
Metropolis correction against the same configured numerical target and a
Gaussian–truncated-Cauchy convolution likelihood evaluated by fixed quadrature.

Across 900 paired simulated trajectories per architecture, mean frequency MSE
was 246.76 Hz² for the analytical score and 238.41 Hz² for the surrogate. After
averaging the repeated conditions within each of 100 independent seed profiles,
the raw paired difference was −8.35 Hz², 95% <i>t</i> CI [−24.94, 8.25]. These
are simulation results and are not presented as WiSig results.

The committed timing endpoint is descriptive software timing. Every physical
condition ran the analytical method before the surrogate, so the latency
contrast is not a randomized or causal architecture benchmark.

## Reproduce and verify

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/validate_repository.py

# Optional: rerun the frozen simulation factorial
PYTHONPATH=. python -m src.signals_project.joint_ssm \
  --output data --repetitions 100 --particles 64 --samples 60
PYTHONPATH=. python scripts/analyze_joint_results.py

PYTHONPATH=. python scripts/build_site.py
PYTHONPATH=. python scripts/build_report.py
```

## Project links

- Interactive laboratory: <https://salemmorelli1.github.io/Signals-Project/>
- 27-page simulation-baseline report:
  <https://salemmorelli1.github.io/Signals-Project/report/Signals_Project_APA_Report.pdf>
- WiSig real-data protocol: `docs/WISIG_REAL_DATA_PROTOCOL.md`
- Validation ladder: `docs/OPERATIONAL_VALIDATION_PROTOCOL.md`

## Repository map

```text
.
|-- index.html
|-- data/                         # committed aggregate evidence/status
|-- external_data/                # local only; ignored by Git
|-- report/Signals_Project_APA_Report.pdf
|-- src/signals_project/
|   |-- joint_ssm.py
|   `-- wisig_realdata.py
|-- tests/
|-- scripts/
|-- docs/
`-- .github/workflows/
```

## Citation and licenses

For WiSig, cite Hanna, Karunaratne, and Cabric (2022), *IEEE Access*, 10,
22808–22818, <https://doi.org/10.1109/ACCESS.2022.3154790>.

Signals Project code is MIT licensed. The `wisig-examples` code is BSD-3-Clause.
The WiSig dataset is distributed under CC BY-NC-SA 4.0; its license governs the
downloaded and derived capture data.
