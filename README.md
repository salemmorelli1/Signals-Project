# Signals Project

This repository executes a blind, joint state-space filter for two overlapping
frequency-hopping emitters observed as complex baseband I/Q samples. The
filtering distribution includes amplitude, frequency, discrete hop state,
phase on the flat torus, latent complex tapped-delay channels, and delayed
source history.

Two exact-target MALA proposal operators are compared in a paired `2 x 3 x 3`
full factorial design:

1. an analytical score of the joint transition and convolution likelihood;
2. an amortized SiLU random-feature surrogate trained on independent exact
   joint scores.

The completed experiment contains 100 trajectory blocks in each physical
SNR-by-channel cell, 900 shared physical trajectories, and 1,800 method-level
runs.

## Validation claim

The repository establishes **operationally representative simulation
validation**. It does **not** establish field or operational SIGINT validation.
That stronger claim requires lawful controlled receiver recordings,
independently blinded ground truth, calibrated hardware timing, frozen code and
weights, and external replication. The exact validation ladder is documented
in `docs/OPERATIONAL_VALIDATION_PROTOCOL.md` and in the report.

## Executed model

- joint sequential particles over two log amplitudes, frequencies, hop states,
  torus phases, complex channel taps, and source lag buffers;
- switching frequency dynamics over overlapping channel alphabets;
- Gaussian plus truncated-Cauchy **convolution** density evaluated by stable
  Gauss-Legendre log-sum-exp quadrature;
- LOS, minor-multipath, and delayed NLOS complex channels inferred blindly;
- tangent-space phase proposals mapped by the flat-torus exponential map;
- lift-summed wrapped-Gaussian forward and reverse proposal densities;
- exact-target Metropolis correction for both score architectures; and
- permanent numerical, geometric, blindness, and end-to-end tests.

## Main result

Across 900 paired trajectories per architecture, mean frequency MSE was
246.76 Hz squared for the analytical score and 238.41 Hz squared for the
surrogate. The raw paired surrogate-minus-analytical difference was -8.35 Hz
squared (95% CI [-24.83, 8.14]). In the blocked multivariate analysis of log
MSE, SNR, channel complexity, and their interaction survived Holm correction;
the architecture main effect did not. The surrogate added 0.073 ms per I/Q
sample and had lower MALA acceptance.

## View the project

- Interactive laboratory: <https://salemmorelli1.github.io/Signals-Project/>
- 27-page APA-style report:
  <https://salemmorelli1.github.io/Signals-Project/report/Signals_Project_APA_Report.pdf>

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python -m src.signals_project.joint_ssm \
  --output data --repetitions 100 --particles 64 --samples 60

python scripts/analyze_joint_results.py
python scripts/build_site.py
python scripts/build_report.py
```

The factorial execution may take several minutes depending on the processor.
The committed artifacts permit the site and report to be rebuilt without
rerunning the experiment.

## Evidence files

- `data/joint_factorial_results.csv` - all 1,800 method-level outcomes;
- `data/joint_cell_summary.csv` - cell means and standard deviations;
- `data/joint_paired_effects.csv` - paired architecture contrasts;
- `data/joint_factorial_effects.csv` - blocked repeated-measures tests;
- `data/joint_example_trace.csv` - predeclared interactive NLOS trace;
- `data/joint_experiment_metadata.json` - immutable execution settings; and
- `data/joint_surrogate_weights.npz` - fitted surrogate parameters.

## Repository map

```text
.
|-- index.html
|-- data/
|-- report/Signals_Project_APA_Report.pdf
|-- src/signals_project/joint_ssm.py
|-- tests/test_joint_ssm.py
|-- scripts/
|-- docs/
|-- legacy/spectral_reference/
|-- .github/workflows/
|-- CITATION.cff
|-- LICENSE
`-- README.md
```

The earlier spectral pseudo-posterior benchmark is retained under
`legacy/spectral_reference/` for provenance; it is not the current experiment.

## License

Code is released under the MIT License. The report and generated figures are
provided for scholarly and educational use with attribution.
