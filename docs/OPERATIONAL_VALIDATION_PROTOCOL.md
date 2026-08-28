# Protocol Required for an Operational Validation Claim

The current repository does not claim operational SIGINT validation. WiSig
adds genuine hardware-captured source I/Q, but its controlled digital overlaps
are not simultaneous RF captures and do not supply hop or channel-path truth.
The following gates must still be completed before operational language is
used publicly.

## Public-data bridge: WiSig

The ManyRx compact subset is the predeclared external bridge between simulation
and a new controlled campaign. Receiver-held-out and day-held-out partitions,
non-equalized I/Q, deterministic digital overlaps, and component-level truth
are specified in `WISIG_REAL_DATA_PROTOCOL.md`. This gate can support an
external real-data de-mixing statement after execution, but not an operational
SIGINT statement.

## Gate 1: Frozen analysis

- preregister scenarios, endpoints, exclusions, multiplicity control, and
  success criteria;
- tag and archive source, dependencies, random seeds, and surrogate weights;
- prohibit tuning on the operational test set; and
- identify the hardware and timing method in advance.

## Gate 2: Lawful controlled I/Q collection

- use authorized emitters, receivers, frequencies, power levels, and sites;
- preserve raw complex I/Q with calibration metadata and cryptographic hashes;
- sample multiple receivers, oscillators, gains, temperatures, locations, and
  channel geometries; and
- retain receiver clock, quantization, saturation, and packet-loss records.

## Gate 3: Independent truth and blinding

- maintain emitter schedules, hop truth, channel references, and timing truth
  outside the inference team;
- randomize scenario identifiers before analysis;
- release truth only after locked predictions are timestamped; and
- document every exclusion and failed run.

## Gate 4: Operational endpoints

At minimum, report frequency-trajectory error, emitter association accuracy,
detection and false-alarm rates, calibration, latency distribution, missed
deadlines, memory, throughput, and failure modes. Confidence intervals must
reflect scenario and hardware clustering rather than treating samples from one
recording as independent.

## Gate 5: Stress and ablation

Evaluate unseen hop alphabets, near-far imbalance, clock offset, adjacent
interference, non-modeled modulation, impulsive contamination, saturation,
missing samples, and channel-model misspecification. Compare against
predeclared baselines and conduct component ablations.

## Gate 6: External replication

An independent team should run the frozen pipeline on a separately collected
truth set. Operational language is justified only if the preregistered success
criteria are met on both the primary and replication campaigns.

## Permitted language before completion

> The broader joint torus-valued SSM has been executed and validated in an
> operationally representative simulation. A WiSig external real-data
> evaluation is implemented but remains pending dataset execution. Field,
> simultaneous-RF, HIL, and operational SIGINT validation remain pending.
