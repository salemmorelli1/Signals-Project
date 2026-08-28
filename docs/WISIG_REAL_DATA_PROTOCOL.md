# WiSig Real-Data External-Evaluation Protocol

## Purpose

This protocol adds public, hardware-captured complex I/Q to the Signals Project
without conflating source provenance with capture geometry. It is an external
real-data transfer benchmark for the torus/channel/de-mixing components of the
model. It is not a field, hardware-in-the-loop, or operational SIGINT campaign.

## Source and license

- Dataset: WiSig, ManyRx compact subset (reported download: 1.2 GB).
- Official page: <https://cores.ee.ucla.edu/downloads/datasets/wisig/>
- Example code: <https://github.com/WiSig-dataset/wisig-examples>
- Dataset license: CC BY-NC-SA 4.0.
- Example-code license: BSD-3-Clause.
- Citation: Hanna, S., Karunaratne, S., & Cabric, D. (2022). WiSig: A
  large-scale WiFi signal dataset for receiver and channel agnostic RF
  fingerprinting. *IEEE Access, 10*, 22808–22818.
  <https://doi.org/10.1109/ACCESS.2022.3154790>

The repository does not redistribute the compact pickle or derived I/Q.

## Input schema

The official compact object contains `tx_list`, `rx_list`,
`capture_date_list`, `equalized_list`, and nested `data`. A selected cell is

```text
data[tx][rx][day][equalization] -> (signals, 256, 2)
```

The last axis is in-phase and quadrature. Only the non-equalized stratum
(`equalization = 0`) is admissible for the primary analysis because equalization
would partially remove physical receiver/channel structure.

## Security and provenance

The compact distribution is a Python pickle. The adapter refuses to load it
unless the operator supplies `--trust-official-pickle`. Before execution:

1. obtain the file from the official UCLA page;
2. record the download date and published checksum if supplied;
3. compute SHA-256 with the adapter;
4. retain the untouched file read-only; and
5. never load an untrusted replacement with the acknowledgement flag.

## Controlled overlap construction

WiSig does not provide simultaneous co-channel pairs. For each eligible
receiver/day domain, the adapter samples two distinct transmitter recordings,
\(x_a\) and \(x_b\), and constructs

\[
y = x_a + \alpha x_b,\qquad
\alpha = \frac{\operatorname{RMS}(x_a)}
{\operatorname{RMS}(x_b)10^{\mathrm{SIR}/20}}.
\]

The predeclared SIR levels are −5, 0, and +5 dB. The stored component truth is
exactly the pair used in the sum; `mixture == source_a + source_b` is enforced
bit-for-bit after float32 construction. No synthetic channel, phase, or hopping
truth is attached to these recordings.

## Leakage control

The seed-fixed split is blocked at the acquisition-domain level:

- test receivers are absent from both training and validation;
- validation uses a capture day absent from training for non-test receivers;
- training uses the remaining receiver/day domains; and
- pair construction occurs inside a single receiver/day domain.

All normalization and model selection must be fit on training data. Validation
may choose hyperparameters. Test data may be evaluated exactly once after code,
weights, endpoints, and exclusions are locked.

## Endpoints

The WiSig primary endpoints are permutation-invariant component reconstruction
error and scale-invariant signal-to-distortion ratio (SI-SDR). Secondary
endpoints are mixture reconstruction error, transmitter association accuracy,
per-preamble wall-clock latency, failure rate, particle ESS, and MALA
acceptance. Confidence intervals are clustered by receiver/day domain.

Frequency-trajectory MSE, phase-path RMSE, channel-tap NMSE, and hop-detection
accuracy are not identifiable from the compact truth and are therefore not
reported as real-data endpoints.

## Model compatibility gate

WiSig is a packet-preamble dataset, not a frequency-hopping waveform dataset.
Before a confirmatory run, the observation model must be frozen for 20 MHz
complex preambles and its physical units documented. The switching-hop prior
remains validated only by the simulation lane unless a separate dataset with
time-aligned hop truth is added.

## Permitted claim after successful execution

> The joint torus-valued de-mixing pipeline was externally evaluated on
> leakage-controlled, digitally overlapped hardware-captured WiSig I/Q across
> held-out receivers and capture days.

This language still does not imply simultaneous RF overlap, HIL validation, or
operational SIGINT performance.
