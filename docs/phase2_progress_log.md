# Phase 2 Progress Log: Variable Topology with MLP Decoder

This document contains detailed experimental results and progress from Phase 2.
For the architectural analysis and root cause, see `phase2_mlp_decoder_report.md`.

---

## Variable Topology VAE (13D Features)

### Test Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Node Mask Accuracy | 100% | Perfect topology prediction |
| Edge Mask Accuracy | 100% | Perfect topology prediction |
| Face Type Accuracy | 100% | Perfect PLANAR/HOLE/FILLET classification |
| Reconstruction Loss | 0.0162 | Low |
| Centroid Loss | 0.0041 | Low (critical for geometric solver) |
| Active Latent Dims | 32/32 (100%) | No dimension collapse |
| KL from Prior | 52.4 nats | Good information flow |

**Checkpoint:** `outputs/vae_variable_13d/best_model.pt`

### Latent Space Analysis

**Dimension Health (vs Fixed Topology):**

| Metric | Fixed (aux-VAE) | Variable | Notes |
|--------|-----------------|----------|-------|
| Active dims | 8/16 (50%) | 32/32 (100%) | Major improvement |
| Collapsed dims | 8 | 0 | No collapse |
| Node MSE | 0.00073 | 0.0085 | Expected for harder task |

**Parameter Correlations — Weak:**

| Parameter | Best Latent Dim | Correlation |
|-----------|-----------------|-------------|
| leg1_length | dim 2 | r=0.088 |
| leg2_length | dim 24 | r=0.112 |
| width | dim 22 | r=0.055 |
| thickness | dim 25 | r=-0.073 |

**Why weak?** VAE trained with `aux_weight=0.0` — latent encodes geometry/topology, not explicit parameters.

**Topology Clustering — Good:**
- 10 distinct topology groups detected
- Average inter-group distance: 3.53
- Similar topologies (same face count, hole count, fillet) cluster together

---

## Parameter Prediction Approaches

Three architectures for predicting L-bracket parameters from latent z:

**1. Feature Regressor (via decoded features):**
```
z → VAE.decode() → features (280D) → MLP → 4 params
```
- Test MAE: 21.84mm (poor)
- Compounds decoder reconstruction error

**2. Simple Latent Regressor (4 core params only):**
```
z (32D) → MLP → 4 params (leg1, leg2, width, thickness)
```
- Bypasses decoder entirely
- Cannot generate STEP (missing hole/fillet params)

**3. Full Latent Regressor (all params) — Recommended:**
```
z (32D) → MLP backbone → Multi-head output:
  ├── core_head → 4 params (leg1, leg2, width, thickness)
  ├── fillet_head → radius + exists probability
  ├── hole1_heads → 2 slots × (diameter, distance, exists)
  └── hole2_heads → 2 slots × (diameter, distance, exists)
```

**Full Latent Regressor Results:**

| Metric | Test Value | Interpretation |
|--------|------------|----------------|
| Existence Accuracy | 100% | Perfect fillet/hole detection |
| Core Params RMSE | 20.8% (~18mm) | Moderate |
| Fillet Radius RMSE | 17.2% (1.4mm) | Good |
| Hole Params RMSE | 30-33% | Room to improve |

**Checkpoint:** `outputs/full_latent_regressor_v4/best_model.pt`

---

## Latent Editor — 64% Ceiling Problem

**Results across approaches:**

| Approach | Direction Acc | Notes |
|----------|---------------|-------|
| 9D VAE (direction_weight=0.5) | 64.3% | Original |
| 9D VAE (direction_weight=1.0) | 64.0% | Higher weight didn't help |
| ParameterVAE v2 | 64.0% | Different decoder didn't help |
| **13D VAE + Geometric Solver** | **63.5%** | Same ceiling persists |

**13D Latent Editor Training:**

| Epoch | Direction Accuracy |
|-------|-------------------|
| 1 | 36.1% |
| 2 | 62.9% (rapid rise) |
| 3-20 | 63.0-63.8% (plateau) |
| **Final** | **63.5%** |

---

## Simplified Directional Instructions

Remove numerical reasoning burden:
```
"increase leg1 by 17.3mm" → "make leg1 longer" (fixed 15% change)
```

**Training Results (20 Epochs):**

| Epoch | Train MSE | Val MSE | Val MAE | Val Norm Error |
|-------|-----------|---------|---------|----------------|
| 1 | 6.4e-3 | 4.7e-3 | 0.055 | 38.2% |
| 2 | 2.8e-4 | 4.7e-6 | 0.0016 | 0.65% |
| 5 | 2.8e-5 | 2.0e-6 | 0.0011 | 0.34% |
| 10 | 4.3e-6 | 9.1e-7 | 0.0007 | 0.25% |
| **20** | **6.1e-7** | **5.4e-7** | **0.0004** | **0.20%** |

**Checkpoint:** `outputs/latent_editor_simple/best_model.pt`

**Warning:** 0.2% error may be misleading due to averaging vulnerability with balanced increase/decrease instructions.

---

## Bbox Loss Bug

**Discovery:** Loss function only trained 9/13 features. Features 9-12 (bbox) were never included.

**Fix in `graph_cad/models/losses.py`:**
```python
bbox_diff = node_diff[..., 9:13]
bbox_loss = (bbox_diff * node_mask_exp).sum() / (num_real_nodes * 4)
```

---

## VAE Version History

### VAE v1: Original (Posterior Collapse)
- β=0.01, free_bits=2.0, decoder=256×256×128
- KL collapsed to 0.008 nats
- Mean latent std: 0.13

### VAE v2: Bbox Fixed
- Added bbox to loss
- Still posterior collapse

### VAE v3: aux_weight=0.1
- Parameter correlations still weak (r < 0.3)
- Direction accuracy: 62%

### VAE v4: Collapse Fix ✓
| Parameter | Old | New |
|-----------|-----|-----|
| target-beta | 0.01 | 0.1 |
| free-bits | 2.0 | 0.5 |
| decoder-hidden-dims | 256,256,128 | 128,128,64 |

**Results:**
- Active dims: 32/32 (100%)
- Mean std: 0.59
- KL from prior: 13.26 nats

**Checkpoint:** `outputs/vae_variable_v4/best_model.pt`

---

## ParameterVAE Experiments

Decode directly to parameters instead of graph features.

**v1-v4 Results:**

| Version | aux_weight | leg1 correlation | Direction Acc |
|---------|------------|------------------|---------------|
| v1 | 0 | r < 0.11 | - |
| v2 | 0 | r < 0.11 | 64% |
| v3 | 0.1 | r = 0.089 | 64% |
| v4 | 1.0 | r = 0.104 | 64% |

**Conclusion:** aux_weight doesn't help — fundamental encoder limitation.

**Checkpoints:**
- `outputs/parameter_vae_v2/best_model.pt`
- `outputs/parameter_vae_v3/best_model.pt`
- `outputs/parameter_vae_v4/best_model.pt`

---

## Geometric Solver

Deterministic parameter extraction from decoded features.

**Accuracy on ground truth:**
| Parameter | Error |
|-----------|-------|
| Core params | 0% (exact) |
| Fillet radius | 0% (exact) |
| Hole diameters | 0% (exact) |
| Hole distances | ~2mm |

**Accuracy on decoded features:** ~26% error (decoder noise)

**Location:** `graph_cad/utils/geometric_solver.py`

---

## Model Checkpoints Summary

**VAE v4 (Current):**
- `outputs/vae_variable_v4/best_model.pt`
- `outputs/full_latent_regressor_v4/best_model.pt`

**13D VAE:**
- `outputs/vae_variable_13d/best_model.pt`
- `outputs/latent_editor_simple/best_model.pt`

**Legacy:**
- `outputs/parameter_vae_v4/best_model.pt`
- `outputs/vae_variable/best_model.pt`
- `outputs/vae_aux/best_model.pt` (fixed topology)
- `outputs/latent_editor_direction/best_model.pt` (80.2% accuracy, fixed)

---

## Phase 2 Completed Steps

1. ~~Train ParameterVAE v2~~ — 11.78 nats ✓
2. ~~Train latent editor on v2~~ — 64% accuracy ✓
3. ~~Add auxiliary linear head (v3)~~ — correlations unchanged ✓
4. ~~Try aux_weight=1.0 (v4)~~ — confirms limitation ✓
5. ~~Implement Geometric Solver~~ — exact extraction ✓
6. ~~Retrain VAE with 13D features~~ — 100% accuracy ✓
7. ~~Generate edit data using 13D VAE~~ ✓
8. ~~Train latent editor on 13D data~~ — 63.5% ceiling ✓
9. ~~Discover z-space encodes params~~ — LLM interface problem ✓
10. ~~Train with simplified instructions~~ — 0.2% error ✓
11. ~~Discover bbox loss bug~~ ✓
12. ~~Retrain VAE v2 with bbox fix~~ — posterior collapse ✓
13. ~~Retrain VAE v3 with aux_weight~~ — weak correlations ✓
14. ~~Diagnose posterior collapse~~ — β too low ✓
15. ~~Implement collapse fix (v4)~~ ✓
16. ~~Train VAE v4~~ — 32/32 active dims ✓
17. ~~Test geometric solver on v4~~ — 26% error ✓
18. ~~Train FullLatentRegressor on v4~~ — 26% error ✓
19. ~~Compare regressor vs solver~~ — both ~26% ✓
20. ~~Identify root cause~~ — node ordering problem ✓

---

## Fixed Topology PoC (Phase 1) Summary

| Metric | Target | Achieved |
|--------|--------|----------|
| VAE Node MSE | < 0.01 | 0.00073 ✓ |
| Direction Accuracy | > 70% | **80.2%** ✓ |
| Parameter MAE | — | ~12mm |

**Key Learnings:**
1. Direction classifier required (MSE alone → shortcuts)
2. aux_weight=0.1 enables parameter encoding
3. ~12mm MAE is VAE bottleneck limit

**Checkpoints:**
- `outputs/vae_aux/best_model.pt`
- `outputs/latent_editor_direction/best_model.pt`
- `outputs/feature_regressor_aux/best_model.pt`

---

## Training Commands Reference

```bash
# VAE v4 (recommended)
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/vae_variable_v4

# Edit data generation
python scripts/generate_simple_edit_data.py \
    --vae-checkpoint outputs/vae_variable_13d/best_model.pt \
    --num-samples 50000 --delta-fraction 0.15 \
    --output data/simple_edit_data

# Latent editor
python scripts/train_latent_editor.py \
    --data-dir data/simple_edit_data \
    --latent-dim 32 --direction-weight 0 \
    --epochs 20 --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_simple
```
