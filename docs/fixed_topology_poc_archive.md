# Fixed Topology PoC Archive (Dec 2024 - Dec 2025)

This document preserves the detailed experimental notes, ablation studies, and analysis from the fixed topology L-bracket proof of concept. The main CLAUDE.md has been consolidated; this file serves as a historical reference.

---

## PoC Scope: L-Brackets (Fixed Topology)

**Fixed topology simplifies the problem:**
- Single part family: L-brackets with 2 mounting holes
- Constant topology: Always 10 faces (8 bracket + 2 cylindrical holes)
- 8 variable parameters: leg lengths, width, thickness, hole diameters, hole distances
- Face-adjacency graph: 10 nodes × 8 features + 22 edges × 2 features = 124 total features

**L-Bracket Parameters:**

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| `leg1_length` | 50 | 200 | Length along +X (mm) |
| `leg2_length` | 50 | 200 | Length along +Z (mm) |
| `width` | 20 | 60 | Extent along Y (mm) |
| `thickness` | 3 | 12 | Material thickness (mm) |
| `hole1_diameter` | 4 | 12 | Hole in leg 1 (mm) |
| `hole2_diameter` | 4 | 12 | Hole in leg 2 (mm) |
| `hole1_distance` | derived | derived | From end of leg 1 to hole center |
| `hole2_distance` | derived | derived | From end of leg 2 to hole center |

---

## VAE Reconstruction Limit

**Critical Discovery**: The VAE introduces an irreducible information bottleneck.

| Model | Features | Parameter MAE |
|-------|----------|---------------|
| GNN on original graphs | 124D (exact) | ~5mm |
| GNN on VAE-reconstructed graphs | 124D (lossy) | ~12.9mm |
| FeatureRegressor on VAE-reconstructed | 124D (lossy) | ~12.4mm |

**~12mm MAE is the theoretical limit** for any parameter predictor operating on VAE-decoded features. This is not a regressor limitation—it's information lost during encode/decode.

---

## Ablation Studies

### VAE β Parameter

The β parameter controls the trade-off between reconstruction fidelity and latent regularity.

| β Value | area variance | edge_length variance | Result |
|---------|---------------|----------------------|--------|
| 0.1 | 1.8% | 11.1% | Severe variance collapse |
| **0.01** | **89.2%** | **79.0%** | **Optimal** |
| 0.001 | 85.5% | 76.9% | No improvement |

**BKM**: Use β=0.01 for L-bracket training.

### Latent Dimension

8D latent is theoretically sufficient (matches 8 parameters), but 16D provides smoother interpolation:

| Latent Dim | Node MSE | Interpolation Quality |
|------------|----------|-----------------------|
| 8D | 0.000883 | Acceptable |
| **16D** | 0.000891 | **Optimal** |
| 64D | 0.000882 | Equivalent |

**BKM**: Use 16D for best balance of compression and interpolation.

### FeatureRegressor Output Constraints

Attempted fixes for out-of-range predictions (normalized values outside [0,1]):

| Approach | Result |
|----------|--------|
| Sigmoid output | Vanishing gradients, training didn't converge |
| Clamping in denormalize | Training worked, but gradient interpretation changed |
| **No constraint** | Can produce impossible params, but training stable |

**BKM**: No output constraint. Occasional impossible values (e.g., negative lengths) accepted as PoC limitation.

### Parameter Predictor Architecture

Compared GNN vs MLP for predicting parameters from VAE-decoded features:

| Architecture | Input | Parameter MAE |
|--------------|-------|---------------|
| GNN (graph structure) | VAE-reconstructed graph | ~12.9mm |
| **MLP (FeatureRegressor)** | Flattened 124D features | **~12.4mm** |

**BKM**: Simple MLP performs equivalently to GNN on fixed-topology graphs. Use FeatureRegressor for simplicity.

---

## Inference Robustness (Dec 2024 — BKM Training Run)

End-to-end training with BKM parameters shows **improved but inconsistent** results.

### Test Results: "make leg1 20mm longer" (5 samples)

| Original | Edited | Δ leg1 | % of Target | Result |
|----------|--------|--------|-------------|--------|
| 154.64 | 155.98 | +1.34 | 7% | ⚠️ Weak |
| 68.65 | 87.40 | +18.76 | 94% | ✓ Good |
| 113.45 | 124.89 | +11.45 | 57% | ⚠️ Partial |
| 135.49 | 142.66 | +7.18 | 36% | ⚠️ Weak |
| 152.59 | 153.45 | +0.86 | 4% | ⚠️ Weak |

**leg1 summary**: 5/5 correct direction (improved from 3/8), magnitude 4-94% of target.

### Test Results: "make leg2 20mm longer" (6 samples)

| Original | Edited | Δ leg2 | Δ leg1 | Result |
|----------|--------|--------|--------|--------|
| 65.57 | 65.55 | -0.02 | +9.06 | ❌ Wrong param |
| 92.92 | 95.53 | +2.61 | +3.87 | ⚠️ Weak |
| 92.96 | 96.94 | +3.98 | +3.74 | ⚠️ Weak |
| 165.65 | 164.05 | -1.60 | +1.67 | ❌ Wrong direction |
| 166.42 | 164.41 | -2.02 | -0.98 | ❌ Wrong direction |
| 105.90 | 114.34 | +8.45 | +4.16 | ⚠️ Partial (42%) |

**leg2 summary**: 3/6 correct direction, max 42% of target. Model confuses leg1/leg2.

### Key Observations

1. **Asymmetric performance**: leg1 edits work reliably (5/5 correct direction), leg2 does not (3/6)
2. **Starting position sensitivity**: Brackets near parameter bounds (leg~150-170mm) show minimal changes—latent space may saturate
3. **Delta magnitude variance**: LLM produces inconsistent delta magnitudes (0.06-0.87) for identical instructions
4. **Parameter coupling**: Edits to one leg often affect the other due to entangled latent space

### Architecture Coupling

The LLM and FeatureRegressor are **indirectly coupled** through VAE features:

```
LLM Training:                          Inference:
bracket_src → VAE → z_src              z_src → LLM → delta_z
bracket_tgt → VAE → z_tgt              z_edited = z_src + delta_z
delta = z_tgt - z_src                  VAE.decode(z_edited) → features
Train: (z_src, instruction) → delta    FeatureRegressor(features) → params
                                       LBracket(**params) → STEP
```

### Root Causes

1. **Entangled latent space** — leg1/leg2 not well disentangled; dominant "size" direction scales whole bracket
2. **Non-uniform latent coverage** — LLM deltas work on average but don't generalize to all input regions
3. **Boundary saturation** — Brackets with parameters near min/max show reduced edit effectiveness
4. **VAE bottleneck** — ~12mm reconstruction error compounds with edit errors

### Lessons Learned

- **End-to-end training matters**: Components must be trained together or kept in sync
- **Robustness testing**: Always test across multiple random inputs, not just one seed
- **Asymmetric behavior**: Different parameters may have different edit reliability

---

## Auxiliary Parameter Loss VAE (Dec 2024)

### Problem: Latent Space Collapse

Analysis of the original VAE revealed severe latent space collapse:
- **Effective dimensions**: Only ~3 of 16 dimensions were active
- **Missing parameters**: Thickness and hole diameters had near-zero correlation with latent space
- **Entanglement**: leg1/leg2 were highly anti-correlated (-0.80), making them hard to edit independently

### Solution: Auxiliary Parameter Prediction

Added a supervised parameter prediction head during VAE training:
```
z → MLP → 8 predicted parameters
aux_loss = MSE(predicted_params, true_params)
total_loss = reconstruction_loss + β*KL_loss + λ*aux_loss
```

### Results: aux-VAE vs Original VAE

| Metric | Original VAE | aux-VAE | Improvement |
|--------|--------------|---------|-------------|
| Effective dimensions | ~3 | ~8 | ✓ Full utilization |
| Thickness correlation | 0.033 | 0.787 | +2300% |
| Hole1 correlation | 0.048 | 0.673 | +1300% |
| Hole2 correlation | 0.053 | 0.797 | +1400% |
| leg1/leg2 entanglement | -0.80 | -0.46 | Better separation |
| Node MSE | 0.0116 | 0.00073 | 93.7% lower |

**Checkpoint**: `outputs/vae_aux/best_model.pt`

---

## Latent Editor Retraining (Dec 2024)

### Training Runs with aux-VAE

Retrained the latent editor with the improved aux-VAE latent space:

| Run | Epochs | LR | Final Val MSE | Final Val MAE |
|-----|--------|-----|---------------|---------------|
| Baseline | 10 | 2e-4 | 0.00585 | 0.0446 |
| Extended | 20 | 2e-4 | 0.00558 | 0.0434 |
| Lower LR | 20 | 1e-4 | 0.00552 | 0.0433 |

Note: Higher MSE/MAE than original model (0.0037/0.029) because aux-VAE latent space has 8 effective dimensions vs 3, making delta prediction harder.

### End-to-End Evaluation (2000 trials)

| Parameter | Correct Direction | Increase Correct | Decrease Correct |
|-----------|-------------------|------------------|------------------|
| leg1 | 52.0% | 27.5% | 76.5% |
| leg2 | 52.0% | 8.5% | 95.5% |
| width | 60.0% | 74.0% | 46.0% |
| thickness | 52.0% | 100% | 4.0% |
| hole1 | 49.0% | 8.7% | 89.3% |
| hole2 | 51.3% | 28.0% | 74.7% |

### Critical Finding: Directional Bias

The model learned a **shortcut** instead of understanding instructions:
- For leg1/leg2/holes: Always predicts "decrease" direction
- For thickness/width: Always predicts "increase" direction
- Achieves ~52% accuracy by ignoring instruction direction entirely

### Diagnosis: Data is Clean, Model Takes Shortcut

Analyzed training data for bias:
```
Cosine(increase_delta, decrease_delta) per parameter:
  leg1: -0.997 ✓ (nearly opposite, as expected)
  leg2: -0.996 ✓
  thickness: -0.997 ✓
  holes: -0.999 to -1.000 ✓
```

**Conclusion**: Training data has clean, symmetric increase/decrease deltas. The MSE loss allows the model to minimize error by predicting a fixed direction per parameter rather than parsing the instruction.

---

## Shortcut Problem Solutions (Dec 2024)

Two approaches tested in parallel:

### Approach 1: Contrastive Learning

Force the model to distinguish increase vs decrease by training on paired samples:

```python
# For same source bracket:
delta_inc = model(z_src, "make leg1 longer")
delta_dec = model(z_src, "make leg1 shorter")

# Contrastive loss: should be opposite directions
contrastive_loss = cosine_similarity(delta_inc, delta_dec) + 1  # minimize when cos = -1

# Total loss
loss = mse_loss + λ * contrastive_loss
```

**Initial Results (weight=0.5) — DIVERGED:**

| Epoch | Loss | Delta MSE | CosSim |
|-------|------|-----------|--------|
| 8 | 0.271 | 0.0129 | -0.484 |
| 12 | 0.297 | 0.0115 | -0.429 |

- CosSim moved wrong direction (toward 0, less opposite)
- ~95% of loss from contrastive component — **weight too high**

**Fix**: Lower contrastive weight to 0.1.

### Approach 2: Auxiliary Direction Classifier

Add a classification head that explicitly predicts increase vs decrease direction:

```python
# Architecture addition:
hidden_state → DirectionClassifier (MLP: 4096→256→64→1) → direction_logit

# Loss:
direction_loss = BCE(direction_logit, direction_target)  # 1.0=increase, 0.0=decrease
total_loss = delta_weight * MSE + direction_weight * direction_loss
```

**Early Results (3 epochs) — STABLE:**

| Epoch | Dir Acc | Delta MSE | Val Loss |
|-------|---------|-----------|----------|
| 1 | 41.8% | 0.0150 | 0.365 |
| 2 | 67.9% | 0.0102 | 0.211 |
| 3 | 67.3% | 0.0090 | 0.209 |

---

## Parallel Testing Results (Dec 2025)

Both approaches trained for 20 epochs.

**Head-to-Head Comparison:**

| Metric | Direction Classifier | Contrastive (w=0.1) |
|--------|---------------------|---------------------|
| **Overall Accuracy** | **80.2%** ✓ | 64.5% |
| Final Delta MSE | 0.0058 | 0.0059 |
| Final CosSim | N/A | -0.537 ✓ |
| Training Speed | Faster (1 fwd pass) | Slower (2 fwd passes) |

**End-to-End by Parameter:**

| Parameter | Direction | Contrastive | Winner |
|-----------|-----------|-------------|--------|
| leg1_length | 69.8% | 58.0% | Direction |
| leg2_length | 67.0% | 50.2% | Direction |
| width | 88.7% | 65.3% | Direction |
| thickness | 97.3% | 93.3% | Direction |
| hole1_diameter | 81.3% | 67.3% | Direction |
| hole2_diameter | 85.3% | 59.3% | Direction |

**Conclusion:** Direction classifier is the recommended approach.

---

## Final End-to-End Metrics (Dec 2025)

**Overall: 52% → 80.2% (+28 pp improvement)** ✓

| Parameter | Overall | Increase | Decrease |
|-----------|---------|----------|----------|
| leg1_length | 69.8% | 40.5% | 99.0% |
| leg2_length | 67.0% | 37.5% | 96.5% |
| width | 88.7% | 79.3% | 98.0% |
| thickness | **97.3%** | 94.7% | 100.0% |
| hole1_diameter | 81.3% | 62.7% | 100.0% |
| hole2_diameter | 85.3% | 70.7% | 100.0% |

---

## Magnitude Analysis (Dec 2025)

Analysis of 2000 trials revealed the model learned a **fixed edit magnitude prior** rather than scaling to instruction size.

**Overall statistics:**

| Metric | Value |
|--------|-------|
| Mean instruction magnitude | 13.4mm |
| Mean target param change | 7.0mm (52% of instruction) |
| Mean sum of all \|changes\| | 18.7mm |
| Mean spillover to other params | 11.8mm |

**Scaling behavior by instruction magnitude:**

| Instruction | Target % | Sum % | Behavior |
|-------------|----------|-------|----------|
| 1-3mm (holes/thickness) | 37-44% | 600-800% | Small edits amplified |
| 10-20mm (width/legs) | 33-53% | 73-124% | Moderate match |
| 30-50mm (legs) | 55-68% | 90-114% | Sum ≈ instruction ✓ |

**Key insight:** The model produces ~15-20mm total change regardless of instruction:
- Small requests (1-3mm) get **amplified** to ~8-18mm total
- Large requests (50mm) get **compressed** to ~45mm total

**By direction:**

| Direction | Accuracy | Avg Target Change | Avg Sum Change |
|-----------|----------|-------------------|----------------|
| Decrease | 98.8% | 8.7mm | 23.2mm |
| Increase | 61.7% | 5.2mm | 14.3mm |

**Analysis script:** `scripts/analyze_magnitude_spread.py`

---

## Boundary Analysis (Dec 2025)

Investigated whether failures occur because parameters near boundaries can't increase/decrease further.

**Hypothesis:** Failures happen when (high value + increase) or (low value + decrease) — geometric limits.

**Results:**

| Category | Failures | % of all failures |
|----------|----------|-------------------|
| Boundary-related | 89 | 22.5% |
| **Non-boundary** | **306** | **77.5%** |

**Failure rate by parameter region:**

| Region | Increase | Decrease |
|--------|----------|----------|
| Low (0-25%) | 33.3% | 1.9% |
| Mid-low (25-50%) | 37.1% | 1.7% |
| Mid-high (50-75%) | 40.9% | 1.1% |
| High (75-100%) | 41.7% | 0.0% |

**Key finding:** Increase fails at 35-42% **across all regions**, not just at boundaries. 97.4% of non-boundary failures are increase operations.

**Conclusion:** The asymmetry is NOT geometric constraints. The model fails at increase *everywhere*.

---

## VAE Latent Space Symmetry Analysis (Dec 2025)

Investigated whether the VAE encodes increase/decrease directions asymmetrically.

**Method:** For each parameter and magnitude, generated 200 triplets (base, decreased, increased) and measured latent deltas.

**Results:**

| Metric | Finding | Implication |
|--------|---------|-------------|
| **Inc/Dec Cosine** | -0.86 to -0.99 | Near-perfect opposites |
| **Variance Ratio** | Decrease ~10-20% higher | Opposite of expected |
| **Alignment** | 0.87-0.99 for both | Equally consistent |
| **Magnitude** | Decrease ~10-20% larger deltas | Dec direction more pronounced |

**By parameter (50mm edits for legs, max magnitude for others):**

| Parameter | Inc/Dec Cosine | Var Ratio (dec/inc) | Dec Magnitude | Inc Magnitude |
|-----------|----------------|---------------------|---------------|---------------|
| leg1_length | -0.895 | 1.47 | 1.12 | 0.91 |
| leg2_length | -0.865 | 1.15 | 1.05 | 0.89 |
| width | -0.942 | 1.21 | 1.02 | 0.84 |
| thickness | -0.946 | 1.00 | 0.77 | 0.70 |
| hole1_diameter | -0.983 | 0.87 | 0.53 | 0.48 |
| hole2_diameter | -0.979 | 1.44 | 0.48 | 0.36 |

**Key findings:**

1. **VAE is NOT asymmetric** — Inc/dec cosines near -1.0 mean increase and decrease are encoded as nearly perfect mirror directions in latent space.

2. **Decrease has MORE variance** — This is the opposite of what we'd expect if variance caused poor increase performance.

3. **Decrease produces larger latent deltas** — Decrease operations create ~10-20% larger movements in latent space.

**Conclusion:** The VAE latent space structure is NOT the root cause of the increase/decrease asymmetry. The problem is in latent editor training dynamics.

**Analysis script:** `scripts/analyze_vae_asymmetry.py`

---

## Legacy Model Checkpoints

- `outputs/vae_aux/best_model.pt` — VAE with auxiliary parameter loss (16D, β=0.01)
- `outputs/latent_editor_direction/best_model.pt` — LLM Latent Editor with direction classifier (80.2% accuracy)
- `outputs/feature_regressor_aux/best_model.pt` — FeatureRegressor (~12mm MAE)
- `outputs/latent_editor_contrastive_w0.1/best_model.pt` — Contrastive learning approach (64.5% accuracy)
- `outputs/vae_16d_lowbeta/best_model.pt` — Original VAE (collapsed latent space)

---

## Legacy Training Commands

```bash
# VAE with auxiliary parameter loss
python scripts/train_vae.py \
    --epochs 100 --latent-dim 16 --target-beta 0.01 --free-bits 2.0 \
    --aux-param-weight 0.1 \
    --output-dir outputs/vae_aux

# FeatureRegressor
python scripts/train_feature_regressor.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --cache-dir data/feature_regressor_cache \
    --train-size 10000 --epochs 100

# Latent Editor data generation - PAIRED
python scripts/generate_edit_data.py \
    --paired \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 --output data/edit_data_paired

# Latent Editor training WITH direction classifier
python scripts/train_latent_editor.py \
    --data-dir data/edit_data \
    --direction-weight 0.5 \
    --epochs 20 \
    --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_direction
```
