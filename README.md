# Cloud–Edge Collaborative Multi–Task CKM Construction

This repository provides:

- **A foundation score-based model** trained on **unpaired** CKM data (e.g., gain + AoA-sin) to learn a reusable CKM prior.
- combine the foundation score-based model with **Plug-and-play diffusion posterior sampling (DPS)** for multi-task CKM construction under sparse / noisy / irregular observations using **VP** and **VE** SDE.
- **End-to-end baselines** for supervised comparison, along with **MSE evaluation + visualization** scripts.

All training experiments are conducted using 8 Ascend 910B NPUs.

---

## 1. Environment

### Requirements

- `ml-collections==1.1.0`
- `torch==2.1.0`
- `accelerate==0.31.0`
- `numpy==1.25.2`
- `datasets==2.20.0`
- `Pillow==9.0.0`

## 2. Configuration

Before running, edit the configuration file:

- `default_CKM_gain_AoA_128_configs.py`
  - `training`: settings for training the foundation score model (and baseline training when applicable)
  - `dp_sampling`: settings for diffusion posterior sampling inference / visualization

---

## 3. Train the foundation score model (prior learning)

1) Update `default_CKM_gain_AoA_128_configs.py` → **`training`** section.  
2) Run training:

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  main.py \
  --config "./configs/vp/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" \
  --eval_folder "./eval_folder/" \
  --mode "train" \
  --workdir "./"
```

---

## 4. Inference visualization (DPS using the foundation model)

1) Update `default_CKM_gain_AoA_128_configs.py` → **`dp_sampling`** section.  
2) Run diffusion posterior sampling:

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  main.py \
  --config "./configs/vp/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" \
  --eval_folder "./eval_folder/" \
  --mode "dp_sampling" \
  --workdir "./"
```

---

## 5. Test MSE performance (proposed method) + visualize

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  test_mse.py
```

---

## 6. Train the end-to-end baseline

1) Update `default_CKM_gain_AoA_128_configs.py` → **`training`** section.  
2) Run baseline training:

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  main.py \
  --config "./configs/CKM_gain_AoA_128_ncsnpp_deep_continuous.py" \
  --eval_folder "./eval_folder/" \
  --mode "train" \
  --workdir "./"
```

---

## 7. Test MSE performance (baseline) + visualize

```bash
accelerate launch \
  --num_processes=8 \
  --num_machines=1 \
  --machine_rank=0 \
  --mixed_precision=no \
  test_mse.py
```

---

## 8. Important note (distributed evaluation)

NOTE: The number of test samples must be a **common multiple** of the test batch size and the number of processes (--num_processes); otherwise, distributed evaluation may result in an incomplete final batch, mismatched sample counts across ranks, or runtime errors.

---

## 9. Citation / Reference

If you use this codebase in academic work, please cite the related paper(s) accordingly.
