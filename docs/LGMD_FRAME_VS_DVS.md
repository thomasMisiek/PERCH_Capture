# Frame-based and DVS ON/OFF LGMD — matched experiment

Both computational LGMD-inspired models have separate ON and OFF retinotopic expansion pathways. Frame input comes from photoreceptor brightness differences; DVS input comes from positive and negative simulated event-count maps. Either polarity can activate the basket command. Both models include delayed radial spatial pooling, inhibition, ON/OFF competition and a 1 second refractory period. They do not receive target bearings, object masks, ranges or collision timestamps.

## Test protocol

The same furnished-room 6-DOF drone/insect simulations are used for both detectors at 20 Hz with an unchanged predictive pursuit controller. A simulated basket trigger is successful only when it falls within 0–200 ms before first entry into a 2 cm separation range. Three-dimensional truth is evaluator-only.

## New evaluation: 10 flights, 13 close approaches

| Model | Timely triggers | Off-window triggers | Runs with one or more timed triggers |
|---|---:|---:|---:|
| Frame LGMD | **9/13** | 2 | 6/10 |
| DVS ON/OFF LGMD | **1/13** | 0 | 1/10 |

The 20 older development runs (24 contacts) yielded 13/24 timely frame triggers and 1 false, versus 4/24 timely DVS triggers and no false. The ON/OFF competition threshold was examined after seeing those runs. The **new 10-flight results** were obtained only after model parameters were frozen.

Interpretation: frame differences preserve weak or residual contrast changes not necessarily represented by the coarse 20 Hz event-count approximation. The DVS pathway was more conservative and missed more valid close approaches in this simulation. This result does not establish superiority on actual event-camera hardware. The neural model is simplified rather than an exact neurobiological LGMD1.
