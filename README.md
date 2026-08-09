# Surgical Smoke Removal in Laparoscopic Video (AOD-Net)

## Overview
Electrocautery and laser ablation during laparoscopic surgery produce smoke that
obscures the surgeon's view. This project adapts **AOD-Net** (All-in-One Dehazing
Network, Li et al., ICCV 2017) — originally built for outdoor atmospheric haze —
to remove surgical smoke from real laparoscopic footage.

## Goal
Train a lightweight CNN to take a smoky laparoscopic frame as input and output a
desmoked version, evaluated against real paired ground-truth footage.

## Dataset
**[DesmokeData](https://github.com/wxia43/DesmokeData)** — a real (not synthetic)
paired dataset from the MICCAI 2024 paper *"A New Benchmark In Vivo Paired Dataset
for Laparoscopic Image De-smoking"* (Xia et al.). Built from 21 video sequences
across 63 laparoscopic prostatectomy procedures, using motion tracking to align
smoky and smoke-free frames of the same anatomical scene.

- **961 paired images** (`{id}.png` smoky / `{id}_gt.png` ground truth)
- Real surgical smoke — no synthetic-to-real domain gap to worry about
- Train/val split: 90/10 (865 / 96), random, fixed seed (42) for reproducibility

## Model
**AOD-Net** — an extremely lightweight CNN based on a reformulated atmospheric
scattering model. Chosen as a first pass for its small size and fast training.

Implementation adapted from
[walsvid/AOD-Net-PyTorch](https://github.com/walsvid/AOD-Net-PyTorch)
(archived repo). Since it was originally built for the RESIDE outdoor-haze
dataset, the dataset loader and training script were rewritten to support
DesmokeData's single-folder, suffix-based pairing convention
(`{id}.png` / `{id}_gt.png`) and a proper train/val split.

## Training
- 25 epochs, batch size 8, learning rate 1e-4, MSE loss
- Trained on a Kaggle T4 GPU
  ![Alt text](/loss_evaluation_8epochs.png)
- Loss dropped steadily from ~0.30 to ~0.03 over training

## Results

### Overall (epoch 25, best checkpoint)
| Metric | This model | LSD3K baseline AOD-Net* |
|---|---|---|
| PSNR | 20.55 dB | 16.81 dB |
| SSIM | 0.7381 | 0.8249 |

\* LSD3K baseline is on a *different* dataset (synthetic smoke over Cholec80
cholecystectomy frames) — included as a rough reference point, not a strict
apples-to-apples comparison, since DesmokeData uses real smoke from a
different procedure type.

### Per-epoch trend
PSNR and SSIM climbed steadily across all 25 epochs with no sign of
overfitting (validation metrics tracked training loss improvements
throughout, with only minor, self-correcting dips). The curve was still
gently rising at epoch 25, suggesting a modest amount of headroom remains
with further training.
![Alt text](/validation_by_epoch.png)

### Density-stratified evaluation
Visual inspection showed the model performing very well on lightly-smoked
frames but struggling on heavily-smoked ones. This was quantified by
bucketing the validation set into terciles by a smoke-density proxy
(mean pixel-wise difference between each smoky image and its own ground
truth):

| Smoke density | N | Mean PSNR | Mean SSIM |
|---|---|---|---|
| Low | 32 | 22.99 dB | 0.8088 |
| Medium | 31 | 22.31 dB | 0.7751 |
| High | 33 | 16.52 dB | 0.6346 |

On low/medium smoke, the model performs strongly — matching or exceeding
the LSD3K baseline on both metrics. Performance drops substantially
(~6.5 dB PSNR, ~0.17 SSIM) on dense smoke, which is the clear bottleneck
for this model.

## Visual Results
Validation-set examples (never seen during training), comparing smoky
input / model output / ground truth:
![Alt text](/validation_comparision_grid.png)

- **Light smoke case**: model output closely tracks ground truth, preserving
  fine tissue detail and instrument edges.
- **Heavy smoke case**: model reduces but does not fully remove dense smoke,
  and introduces a visible pink/magenta color cast rather than fully
  reconstructing the underlying scene.

## Limitations
- Small dataset (961 pairs from 63 procedures) limits generalization,
  especially for the sparser dense-smoke examples
- Single lightweight architecture evaluated (AOD-Net); heavier models
  (e.g. GridDehazeNet, DehazeFormer) have shown notably better results
  on related benchmarks and would be a natural next comparison
- Real smoke from one procedure type (laparoscopic prostatectomy) — unclear
  how well this generalizes to other procedures or camera/lighting setups
- No dedicated handling for smoke density; a density-aware loss or training
  strategy (e.g. oversampling dense-smoke examples) could directly target
  the identified weak point

## Possible Next Steps
- Extend training beyond 25 epochs to see where metrics fully plateau
- Compare against a second, heavier dehazing architecture
- Address the dense-smoke weakness directly (loss reweighting, targeted
  data augmentation, or a density-conditioned model)
- Extend to a related surgical visibility task (e.g. blood/fluid region
  segmentation) as a follow-on project
