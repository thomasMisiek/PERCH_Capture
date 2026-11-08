# DVS LGMD: sensor and benchmark audit

## What the current implementation does

The simulation uses `FrameToDVS` in `src/dvs.py`: consecutive clean 320 × 320 RGB renders are converted to logarithmic-contrast ON/OFF event-count arrays with pixelwise thresholds and sparse background activity, sampled every 50 ms. These are **simulated 20 Hz event bins**, not raw asynchronous data from a Prophesee sensor.

`DVSExpansionLGMD` in `src/dvs_lgmd_expansion.py` downsamples to 160 × 160; it combines retinotopic multi-radius ON+OFF activity with an OFF-polarity gate, delayed coincidence, fixation weighting and feed-forward suppression. Basket alarms are a stricter threshold on the resulting neural response, with a refractory interval.

This is a **custom LGMD-inspired visual expansion circuit**. It is not a biophysically faithful implementation of LGMD1/LGMD2, does not implement the full ON/OFF competition described by Lei et al. (2023), and does not explicitly estimate angular expansion velocity or time-to-contact. The center-weighted response may also miss objects far from the optical axis.

The detector receives **no target location, object size, 3-D distance, segmentation mask, trajectory truth or collision timestamp**. Those quantities are used separately by the simulation benchmark to score alarms.

## Trigger timing benchmark

Twenty independent 3-D flight simulations contain **24 encounters entering the 2 cm capture range**. A timely basket alarm must be produced from the DVS response **0–200 ms before first entry into that range**.

The current conservative operating point yields **4 timely triggers**, **zero off-window alarms**, and **3/20 runs with at least one timely trigger**. The zero false-alarm count applies to these prerecorded simulations only.

## Parameter exploration (8 November 2026)

Replayed and verified detector response signals across all 20 simulations, reproducing the 4/24 result. Evaluated **990 candidate operating points** covering neural response thresholds, OFF-channel gates, relative OFF strength, global feed-forward inhibition, rising-response requirements and activity-dependent thresholds. Parameter selection used ten alternating seeds; the remaining ten were used to assess transfer within the same previously explored simulation environment.

**No candidate improved on four timely contacts with zero or one off-window alarms.** The best apparent results with three off-window alarms reached five timely contacts. Merely lowering the threshold therefore does not deliver reliable collision specificity.

A one-dimensional threshold diagnostic:

| Neural firing threshold | Timely contact triggers / 24 | Off-window alarms |
|---|---:|---:|
| 1.225 | 16 | 63 |
| 1.400 | 5 | 27 |
| 1.450 | 5 | 16 |
| 1.500 | 4 | 9 |
| 1.550 | 4 | 3 |
| **1.600** | **4** | **0** |

This parameter study is exploratory; the 20 simulator seeds have been used in previous development and are **not** a genuinely independent test set. No hardware, new lighting conditions, independent insect trajectories or asynchronous event stream was evaluated.

## Scientific interpretation

LGMD circuits respond to looming-like visual motion, rather than directly measuring a future physical contact timestamp. Translation, background motion, contrast polarity and camera movement can influence the spike response. OFF-event confirmation alone is not equivalent to a validated ON/OFF-competitive LGMD circuit.

Relevant sources:

- Lei et al., *A Robust Visual System for Looming Cue Detection Against Translating Motion*, IEEE TNNLS (2023), DOI: 10.1109/TNNLS.2022.3149832.
- *A Computationally Efficient Neuronal Model for Collision Detection with Contrast Polarity-Specific Feed-Forward Inhibition* (2024), https://pmc.ncbi.nlm.nih.gov/articles/PMC11592146/.
- *An Angular Acceleration Based Looming Detector for Moving UAVs* (2024), https://pmc.ncbi.nlm.nih.gov/articles/PMC11154257/.

The next research issue is **model and sensor validity**, not another basket threshold change: test the direction selectivity and robustness of genuine ON/OFF processing under translational and camera motion, and independently evaluate event-based estimates of approach imminence.
