# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph_CAD combines Graph Autoencoders with Generative AI for Computer-Aided Design (CAD). The goal is to enable natural language editing of CAD models by operating in a learned latent space.

## Project Structure

```
graph_cad/          # Main package
├── models/         # Graph VAE, FeatureRegressor, LatentEditor
├── data/           # Data loaders, graph extraction, edit dataset
├── utils/          # Visualization, helpers
└── training/       # Training loops, loss functions

scripts/            # CLI scripts for training and inference
tests/              # Unit and integration tests
outputs/            # Saved model checkpoints (gitignored)
data/               # Training data (gitignored)
```

## Development Commands

```bash
# Setup
pip install -e ".[dev]"

# Testing
pytest                              # Run all tests
pytest -v -k "test_specific"        # Run specific test

# Code quality
black graph_cad tests && ruff check graph_cad tests && mypy graph_cad
```

## Architecture Overview

### System Pipeline (Current: Geometric Solver)

```
Input STEP → Graph Extraction (13D features) → VAE Encode → Latent (32D)
                                                              ↓
                                         LLM Latent Editor (instruction)
                                                              ↓
                                                       Edited Latent
                                                              ↓
                                    VAE Decode → Reconstructed Graph Features
                                                              ↓
                                    Geometric Solver → All Parameters (deterministic)
                                                              ↓
                                         VariableLBracket → Output STEP
```

**Key Insight**: Instead of learning parameter regression, we use deterministic geometric solving. The VAE reconstructs graph features (face areas, centroids, normals), and the geometric solver extracts exact parameters from these features using geometric relationships.

### Node Features (13D)

```
[area, dir_x, dir_y, dir_z, cx, cy, cz, curv1, curv2, bbox_diagonal, bbox_cx, bbox_cy, bbox_cz]
```

- `area`: Face area normalized by bbox_diagonal²
- `dir_xyz`: Face normal/axis direction (unit vector)
- `cx, cy, cz`: Centroid normalized as (centroid - bbox_center) / bbox_diagonal
- `curv1, curv2`: Principal curvatures normalized by bbox_diagonal
- `bbox_diagonal`: Bounding box diagonal / 100mm (same for all nodes)
- `bbox_cx, bbox_cy, bbox_cz`: Bounding box center / 100mm (same for all nodes)

The bbox values enable scale-aware reconstruction and deterministic de-normalization.

### Two-Phase Approach

1. **Phase 1 (Complete)**: Fixed Topology PoC
   - L-brackets with exactly 2 holes, 10 faces
   - 16D latent, 8 parameters
   - Achieved 80.2% direction accuracy

2. **Phase 2 (In Progress)**: Variable Topology
   - L-brackets with 0-4 holes, optional fillets, 6-15 faces
   - 32D latent, 13D node features
   - VAE trained ✓, Geometric Solver implemented ✓

---

## Current Status: Variable Topology (Jan 2026)

### Variable Topology VAE (13D Features) — Training Complete ✓

**Test Metrics:**

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

### Parameter Prediction Approaches

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
- Predicts ALL variable topology parameters
- Existence heads handle variable hole counts (0-4 holes)
- Masked loss: param loss only where features exist

**Full Latent Regressor — Training Complete ✓**

| Metric | Test Value | Interpretation |
|--------|------------|----------------|
| Existence Accuracy | 100% | Perfect fillet/hole detection |
| Core Params RMSE | 20.8% (~18mm) | Moderate |
| Fillet Radius RMSE | 17.2% (1.4mm) | Good |
| Hole Params RMSE | 30-33% | Room to improve |

**Limitation:** Fillet radius predictions collapse to mean (~3.2mm). VAE latent doesn't encode fillet size well.

**Conceptual Note:** For parametric templates like L-brackets, we go `params → VariableLBracket() → STEP`. For arbitrary geometry, B-Rep reconstruction from features would be needed — a harder unsolved problem.

### Latent Editor — 64% Ceiling Problem

**Results across approaches:**

| Approach | Direction Acc | Notes |
|----------|---------------|-------|
| 9D VAE (direction_weight=0.5) | 64.3% | Original |
| 9D VAE (direction_weight=1.0) | 64.0% | Higher weight didn't help |
| ParameterVAE v2 | 64.0% | Different decoder didn't help |
| **13D VAE + Geometric Solver** | **63.5%** | Same ceiling persists |

**13D Latent Editor Training (Jan 2026):**

| Epoch | Direction Accuracy |
|-------|-------------------|
| 1 | 36.1% |
| 2 | 62.9% (rapid rise) |
| 3-20 | 63.0-63.8% (plateau) |
| **Final** | **63.5%** |

**Why 13D features didn't help:** The geometric solver extracts exact parameters from *decoded features*, but doesn't change how the latent space encodes information. The LLM still can't determine which direction to push z because "make leg1 longer" has no consistent direction in z-space.

**Direction Classifier Architecture:**
```
LLM hidden (4096D) → MLP (4096→256→64→1) → direction_logit
```
The auxiliary direction head is trained with BCE loss to predict increase (1) vs decrease (0). At 63.5%, it's barely better than random (50%), confirming the hidden state can't encode direction reliably.

**Comparison to Fixed Topology:**
- Fixed VAE used `aux_weight=0.1` → parameters encoded in latent → 80% direction acc
- Variable VAE (any approach) → geometry encoded, not params → 64% ceiling

### Key Discovery: Z-Space DOES Encode Parameters (Jan 2026)

Correlation analysis on 13D VAE revealed **strong linear correlations**:

| Parameter | Best Z Dimension | Correlation |
|-----------|------------------|-------------|
| leg1_length | dim 5 | r = **-0.982** |
| leg2_length | dim 22 | r = **+0.926** |
| width | dim 10 | r = **-0.743** |
| thickness | dim 12 | r = -0.298 |
| bbox_diagonal | dim 21 | r = **-0.980** |

The 13D features (with bbox info) allow the VAE to encode absolute scale. **The problem isn't z-space — it's the LLM interface.**

**Root Cause Refined:**

The LLM receives z as projected tokens but can't extract parameter values from them:
```
LLM sees: [abstract z tokens] + "increase leg1 by 20mm"
LLM doesn't know: leg1 is currently 80mm
LLM can't compute: correct z_delta
```

The numerical reasoning ("increase by 20mm") is too complex to learn through z-token projection.

### Solution: Simplified Directional Instructions (Jan 2026)

Remove numerical reasoning burden. Instead of:
```
"increase leg1 by 17.3mm" → z_delta (varies by magnitude)
```

Use simple directional instructions:
```
"make leg1 longer" → z_delta (fixed 15% proportional change)
```

**Advantages:**
1. LLM leverages pre-trained language understanding ("longer" = increase)
2. Consistent z_delta for same instruction (since z strongly encodes leg1)
3. Direction is explicit in text — no classifier needed
4. More like classification than regression

**Training:**
```bash
# Generate simplified edit data
python scripts/generate_simple_edit_data.py \
    --vae-checkpoint outputs/vae_variable_13d/best_model.pt \
    --num-samples 50000 --delta-fraction 0.15 \
    --output data/simple_edit_data

# Train without direction classifier (direction explicit in instruction)
python scripts/train_latent_editor.py \
    --data-dir data/simple_edit_data \
    --latent-dim 32 --direction-weight 0 \
    --epochs 20 --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_simple
```

**Final Results (20 Epochs) — Training Complete ✓:**

| Epoch | Train MSE | Val MSE | Val MAE | Val Norm Error |
|-------|-----------|---------|---------|----------------|
| 1 | 6.4e-3 | 4.7e-3 | 0.055 | 38.2% |
| 2 | 2.8e-4 | 4.7e-6 | 0.0016 | 0.65% |
| 5 | 2.8e-5 | 2.0e-6 | 0.0011 | 0.34% |
| 10 | 4.3e-6 | 9.1e-7 | 0.0007 | 0.25% |
| 15 | 9.7e-7 | 5.8e-7 | 0.0005 | 0.20% |
| **20** | **6.1e-7** | **5.4e-7** | **0.0004** | **0.20%** |

**Key Observations:**

1. **0.2% relative error** — model predicts delta_z almost exactly
2. **Train ≈ Val by epoch 20** — gap closed, no overfitting
3. **Val < Train early due to dropout** (projector_dropout=0.1, lora_dropout=0.05)
4. **No direction classifier needed** — direction explicit in instruction text

**Comparison to Numerical Instruction Approach:**

| Metric | Numerical + Direction Classifier | Simplified Instructions |
|--------|----------------------------------|------------------------|
| Val MSE | ~1e-6 | 5.4e-7 |
| Direction Accuracy | 64% (auxiliary head) | ~100% (implicit) |
| Relative Error | 10-20% | **0.2%** |
| Meaningful metric? | No (averaging possible) | **Yes** |

**Checkpoint:** `outputs/latent_editor_simple/best_model.pt`

**Why This Works — The Averaging Problem:**

The old numerical approach allowed "averaging" shortcuts:
- "increase by 17.3mm" and "decrease by 12.1mm" both in training
- Dataset balanced: ~50% increase, ~50% decrease → mean delta ≈ 0
- Model could minimize MSE by predicting small/zero deltas

```
True deltas:  [+0.15, -0.12, +0.08, -0.20, ...]  (mean ≈ 0)
Predicted:    [+0.01, -0.01, +0.01, -0.01, ...]  (hedging toward zero)
Result:       Low MSE achieved, but direction essentially random
```

**Critical insight:** Low MSE was misleading in the old approach. The direction classifier (64%) exposed that the model couldn't determine direction — it was achieving low MSE by averaging, not by understanding instructions.

The simplified approach prevents averaging:
- "make leg1 longer" → ALL samples have same direction (increase)
- "make leg1 shorter" → ALL samples have opposite direction (decrease)
- No way to hedge toward zero when all samples agree on direction

```
True deltas:  [+0.15, +0.18, +0.12, ...]  (all positive for "longer")
Predicted:    [+0.1499, +0.1801, +0.1198, ...]  (must commit to direction)
Result:       Low MSE requires correct direction AND magnitude
```

**Why the new low MSE is meaningful:** With ~0.02% relative error, the model must be predicting both direction and magnitude correctly. The metric now measures what we actually care about.

**Open Questions (To Validate):**

1. **Is model personalizing or averaging within instruction type?**
   - If averaging: predicts mean delta for "make leg1 longer" regardless of z_src
   - If personalizing: delta varies with z_src (15% of different leg1 values)
   - Very low MSE (1e-6) suggests personalization, but need to verify

2. **Direction accuracy from deltas?**
   - No explicit direction classifier, but can compute from predicted vs actual delta signs
   - Expected to be near 100% given low MSE

**Planned Analysis (Post-Training):**

1. Scatter plot: predicted vs actual delta_z per dimension
2. Check if predictions vary appropriately with z_src for same instruction
3. Per-instruction breakdown of accuracy
4. Compute implicit direction accuracy from delta signs
5. End-to-end test: z_src + instruction → z_edited → decode → geometric solver → STEP

### Critical Bug Found: Bbox Features Not Trained (Jan 2026)

**Discovery:** End-to-end inference failed — geometric solver returned minimum values (leg1=50mm, width=20mm, etc.) regardless of input.

**Root Cause:** The loss function `variable_reconstruction_loss` only trained on 9D features:
- `[0:1]` = area
- `[1:4]` = direction
- `[4:7]` = centroid
- `[7:9]` = curvature

Features 9-12 (bbox_diagonal, bbox_cx, bbox_cy, bbox_cz) were **never included in the loss**. The decoder learned to output garbage for these dimensions.

**Evidence:**
```
Ground truth bbox_d: 2.08 (=208mm diagonal)
Decoded bbox_d: 0.03-0.25 (=3-25mm diagonal)
```

The geometric solver uses bbox_diagonal to de-normalize centroids. With 10-70x wrong scale, all computed dimensions are tiny and clip to minimums.

**Fix:** Updated `graph_cad/models/losses.py`:
```python
# Added to VariableVAELossConfig:
bbox_weight: float = 1.0

# Added to variable_reconstruction_loss:
bbox_diff = node_diff[..., 9:13]  # bbox_diagonal + bbox_center_xyz
bbox_loss = (bbox_diff * node_mask_exp).sum() / (num_real_nodes * 4)
node_loss = ... + config.bbox_weight * bbox_loss
```

**Status:** Fix committed. VAE v2 trained with bbox loss.

### VAE v2 Results: Bbox Fixed, But Posterior Collapse (Jan 2026)

**Bbox reconstruction improved significantly:**
```
VAE v1: bbox_d = 0.03-0.25 (3-25mm) — 97% error
VAE v2: bbox_d = 1.81 (181mm) vs GT 2.08 (208mm) — 13% error
```

**But geometric solver still has large errors:**

| Metric | Mean Error |
|--------|------------|
| leg1 | 34mm (20-30%) |
| leg2 | 33mm (varies) |
| width | 8mm (15-25%) |
| thickness | 2.8mm (20-50%) |

**Root Cause: Posterior Collapse**

The VAE was trained with `aux_weight=0.0`, so the encoder produces nearly identical latents for different brackets:
```
Bracket A (leg1=100) → z_a, norm=2.78
Bracket B (leg1=115) → z_b, norm=2.78
delta_z = z_b - z_a, norm=0.024 (tiny!)
```

Evidence from training:
- KL loss collapsed to 0.008 (very low)
- Mean latent std: 0.13 (should be ~1.0)
- Delta_z for 15% param changes: ~0.003 (nearly zero)

The decoder reconstructs geometry from minimal latent information, but the latent doesn't encode parameter variations.

**Attempted Solution: Retrain with aux_weight=0.1**

```bash
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --aux-weight 0.1 \
    --output-dir outputs/vae_variable_13d_v3
```

### VAE v3 Results: aux_weight Didn't Help (Jan 2026)

**Training metrics:**
- aux_param_loss: 0.127 → 0.074 (improved but still high)
- Direction accuracy: 62% (same as v2)

**Parameter correlations (still weak):**

| Param | VAE v2 | VAE v3 |
|-------|--------|--------|
| leg1 | r=-0.24 | r=-0.25 |
| leg2 | r=-0.30 | r=+0.31 |
| width | r=+0.25 | r=-0.28 |
| thickness | r=+0.26 | r=-0.24 |

**Why aux_weight=0.1 failed for variable topology:**

The fixed topology VAE achieved 80% direction accuracy with aux_weight=0.1, but it had:
- Fixed 10 faces, 2 holes
- Simpler encoding task
- 8 parameters to encode

Variable topology has:
- 6-15 faces, 0-4 holes, optional fillet
- More complex encoding (topology + geometry)
- Parameters competing with topology info in latent space

The graph encoder architecture inherently encodes graph structure nonlinearly. The aux loss forces a linear mapping from z to params, but the encoder still maps geometry→z in a nonlinear, topology-dependent way.

**Checkpoints:**
- `outputs/vae_variable_13d_v2/best_model.pt` — bbox fixed, collapsed latent
- `outputs/vae_variable_13d_v3/best_model.pt` — bbox fixed, aux_weight=0.1, still weak correlations

### Posterior Collapse Root Cause Analysis (Jan 2026)

The aux_weight approach failed because the real problem was **posterior collapse**, not lack of parameter supervision. Diagnosis revealed multiple interacting factors:

**1. β Too Low (Critical)**
- Old default: `β = 0.01`
- With reconstruction loss ~0.016 and raw KL ~50 nats:
  - KL contribution to loss: 0.01 × 50 = 0.5
  - **KL was only ~3% of total loss** — effectively ignored
- Model learned to minimize reconstruction and ignore KL entirely

**2. Free Bits Too High (Critical)**
- Old default: `free_bits = 2.0`
- This means: "Don't penalize KL below 2.0 nats per dimension"
- With 32 dims: encoder could collapse to near-zero variance and pay zero penalty up to 64 total nats
- Ideal per-dim KL for N(0,1) is ~0.5 nats — far below threshold

**3. Decoder Overpowered**
- Encoder: ~20-30K params → 32D latent bottleneck
- Decoder: ~150K params (256→256→128 hidden dims)
- **Decoder was 5-8x larger than encoder** — could reconstruct from nearly constant z

**4. No KL Annealing**
- β was fixed from epoch 1
- Standard practice: start β=0 to learn good reconstructions, gradually increase

**The Math:**
```
Reconstruction loss: 0.016
KL loss (raw):       50 nats
KL loss (scaled):    0.01 × 50 = 0.5

Total loss ≈ 0.016 + 0.5 = 0.516
KL weight: 0.5 / 0.516 ≈ 3%  ← Model learns to ignore KL
```

### VAE v4: Posterior Collapse Fix (Jan 2026)

Implemented coordinated fixes in `train_variable_vae.py`:

**Parameter Changes:**

| Parameter | Old Default | New Default | Rationale |
|-----------|-------------|-------------|-----------|
| `target-beta` | 0.01 | **0.1** | KL now ~50% of loss |
| `free-bits` | 2.0 | **0.5** | Penalize collapse earlier |
| `decoder-hidden-dims` | (256,256,128) | **(128,128,64)** | Reduce decoder capacity |

**New Features:**

1. **Beta Annealing**: Linear warmup from β=0 to target over first 50% of epochs
   ```python
   # Epoch 1: β=0.002, Epoch 25: β=0.05, Epoch 50+: β=0.1
   beta = get_beta_schedule(epoch, total_epochs, warmup_epochs, start_beta=0.0, target_beta)
   ```

2. **Real-time Collapse Monitoring**: Training now shows latent health indicators
   ```
   Epoch  10/100 | β=0.020 | Val: 0.0234 | Active: 28/32 | Std: 0.85 | KL_prior: 45.2
   Epoch  50/100 | β=0.100 | Val: 0.0189 | Active: 32/32 | Std: 0.92 | KL_prior: 52.1 ⚠️ COLLAPSE
   ```
   - Shows `⚠️ COLLAPSE` if <50% active dims
   - Shows `⚠️ LOW_STD` if mean std < 0.3

3. **New CLI Arguments**:
   - `--beta-warmup-epochs`: Control warmup period (default: 50% of epochs)
   - `--decoder-hidden-dims`: Comma-separated dims (default: "128,128,64")

**Model Size Reduction:**
- Old decoder: ~150K params
- New decoder: ~50K params

**Expected Improvements:**
- KL loss: 2-5 nats (from ~0.008)
- Active dims: 28-32/32 (from 8/32)
- Latent mean std: 0.8-1.0 (from 0.13)

**Checkpoint:** `outputs/vae_variable_v4/best_model.pt` (pending training)

### Fundamental Limitation

The graph-based VAE architecture fundamentally struggles with parameter-aware latent spaces for variable topology:

1. **Encoder**: GNN aggregates node features → nonlinear encoding
2. **Decoder**: MLP reconstructs geometry → doesn't need linear param structure
3. **aux_loss**: Forces linear param prediction from z, but can't change encoder behavior

**Possible directions:**
1. ~~**Much higher aux_weight** (0.5-1.0)~~ — Tried, didn't help (collapse was the real issue)
2. **Fix posterior collapse first** — Now implemented in v4 ✓
3. **ParameterVAE** — decode to params directly (tried, 64% ceiling)
4. **Two-stage**: z-edits for direction, separate regressor for magnitude
5. **Accept limitation**: System encodes geometry well, but latent edits don't map to parameter changes reliably

### Legacy: ParameterVAE

Instead of decoding to graph features, decode directly to parameters. This forces the latent space to encode parameter information.

**Architecture:**
```
Graph → GNN Encoder → z (32D) → Parameter Decoder → All L-bracket params
                                        ↓
                    [leg1, leg2, width, thickness, fillet, holes]
```

**Key Insight:** The graph encoder still sees geometry (centroids, areas, curvatures), but the decoder MUST output parameters. This forces z to encode parameters explicitly.

**Parameter Decoder Outputs (18 values):**

| Component | Values | Description |
|-----------|--------|-------------|
| core_params | 4 | leg1, leg2, width, thickness |
| fillet_radius | 1 | + fillet_exists (1) |
| hole1_params | 4 | 2 slots × (diameter, distance) + exists (2) |
| hole2_params | 4 | 2 slots × (diameter, distance) + exists (2) |

Actual parameters vary by topology: 4 (plain) to 13 (2 holes/leg + fillet).

### ParameterVAE Training Results

**v1 (Posterior Collapse):** Initial training with weak bottleneck (beta=0.01, free_bits=2.0, decoder=3×256) resulted in posterior collapse:
- KL loss: 0.0145 (near zero information in latent)
- Correlations: r < 0.11 (weak)
- The powerful decoder bypassed the latent bottleneck

**v2 (Stricter Bottleneck):** Fixed with beta=0.1, free_bits=0.5, decoder=2×128:
- Raw KL: 11.78 nats (information flowing through z)
- Correlations: r < 0.11 (still weak)
- Latent editor trained on v2: **64% direction accuracy** (same ceiling as VariableGraphVAE)

**v3 (Auxiliary Linear Head):** Added `aux_core_head = Linear(32, 4)` with aux_weight=0.1

**v4 (Strong Linear Constraint):** aux_weight=1.0 (10x higher)

| Metric | v3 (aux=0.1) | v4 (aux=1.0) |
|--------|--------------|--------------|
| Raw KL | 11.94 | 12.26 |
| Aux loss | 0.083 | 0.083 |
| Core loss | 0.083 | 0.083 |

**Correlations (unchanged despite 10x aux_weight):**

| Parameter | v3 | v4 |
|-----------|-----|-----|
| leg1 | 0.089 | 0.104 |
| leg2 | 0.039 | 0.030 |
| width | 0.072 | 0.072 |
| thickness | 0.080 | 0.083 |

### Fundamental Limitation Identified

Even with aux_weight=1.0, correlations remain weak (r ~ 0.03-0.10). This confirms a **fundamental architectural limitation**:

The GNN encoder extracts geometry features (centroids, areas, curvatures) that encode construction parameters through **inherently nonlinear** combinations. "leg1_length" manifests in face positions, areas, and edges in complex ways that cannot be captured by linear directions in z-space — regardless of loss weighting.

**The graph-based latent space encodes *geometry*, not *parameters*.**

**Solution: Geometric Solver (Jan 2026)**

Instead of learning to predict parameters from a nonlinearly-encoded latent, we implemented a **deterministic geometric solver** that extracts parameters directly from decoded graph features:

```python
from graph_cad.utils.geometric_solver import solve_params_from_features

# After VAE decoding
params = solve_params_from_features(
    node_features=decoded_features,  # (num_nodes, 13)
    face_types=predicted_face_types,  # (num_nodes,)
    edge_index=decoded_edges,
    edge_features=decoded_edge_features,
)
# Returns SolvedParams with leg1, leg2, width, thickness, fillet, holes
```

**Geometric Solver Accuracy (on ground truth features):**

| Parameter | Error |
|-----------|-------|
| Core params (leg1, leg2, width, thickness) | **0%** (exact) |
| Fillet radius | **0%** (exact) |
| Hole diameters | **0%** (exact) |
| Hole distances | **~2mm** (centroid precision) |

This approach sidesteps the 64% ceiling entirely by not requiring the latent to encode parameters linearly. The VAE just needs to reconstruct geometry accurately; the solver handles parameter extraction deterministically.

**Legacy checkpoints:** `outputs/parameter_vae_v4/best_model.pt`

### Training Commands (Variable Topology with Geometric Solver)

```bash
# 1. Train VariableGraphVAE with 13D features (v4 with collapse fix)
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/vae_variable_v4
# New defaults: target-beta=0.1, free-bits=0.5, decoder-hidden-dims=128,128,64
# Beta annealing: 0 → 0.1 over first 50 epochs

# For old behavior (not recommended - causes posterior collapse):
python scripts/train_variable_vae.py \
    --target-beta 0.01 --free-bits 2.0 --decoder-hidden-dims 256,256,128 \
    --beta-warmup-epochs 0 ...

# 2. Generate edit data using 13D VAE
python scripts/generate_variable_edit_data.py \
    --vae-checkpoint outputs/vae_variable_v4/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_v4

# 3. Train latent editor
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_v4 \
    --latent-dim 32 --direction-weight 0.5 \
    --epochs 20 --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_v4
```

**Legacy (ParameterVAE approach - limited by 64% ceiling):**
```bash
python scripts/train_parameter_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/parameter_vae_v2
```

**Legacy (VariableGraphVAE 9D approach):**
```bash
# Original VAE (9D features, no geometric solver support)
# Superseded by 13D version above
```

---

## Fixed Topology PoC Summary (Dec 2025) — Complete

### Final Results

| Metric | Target | Achieved |
|--------|--------|----------|
| VAE Node MSE | < 0.01 | 0.00073 ✓ |
| Direction Accuracy | > 70% | **80.2%** ✓ |
| Parameter MAE | — | ~12mm (VAE limit) |

**Verdict: SUCCESS** — Proved LLMs can edit CAD geometry in latent space.

### Key Learnings (Transferable)

1. **Direction Classifier Required**
   - MSE loss alone → model learns shortcuts (always predict decrease)
   - Adding BCE direction head → 80.2% accuracy
   - Use `--direction-weight 0.5` for latent editor training

2. **Auxiliary Parameter Loss Helps VAE**
   - Without: Only 3/16 dimensions active, parameters not encoded
   - With `--aux-param-weight 0.1`: 8/16 active, strong parameter correlations

3. **VAE Bottleneck is Real**
   - ~12mm MAE is theoretical limit for any predictor on decoded features
   - Information lost in encode/decode cannot be recovered

4. **Increase/Decrease Asymmetry**
   - Model better at decrease (96-100%) than increase (37-95%)
   - Not caused by VAE (latent space is symmetric)
   - Caused by training dynamics — direction classifier fixes it

5. **β and Free-Bits (Fixed Topology)**
   - β=0.01 worked for fixed topology (simpler task)
   - **Note**: Variable topology requires different settings (see VAE v4 section)
   - Variable topology with β=0.01 causes posterior collapse due to more complex encoding task

### Ablation Results Summary

| Experiment | Finding |
|------------|---------|
| β parameter | 0.01 optimal; 0.1 causes collapse |
| Latent dim | 16D sufficient; 8D too small, 64D no benefit |
| GNN vs MLP regressor | MLP equivalent for fixed topology |
| Contrastive vs Direction | Direction classifier wins (80% vs 65%) |

---

## Component Reference

### VariableLBracket

```python
VariableLBracket(
    leg1_length, leg2_length, width, thickness,  # Core params
    fillet_radius=0.0,          # 0 = no fillet
    hole1_diameters=(6, 8),     # 0-2 holes per leg
    hole1_distances=(20, 60),
    hole2_diameters=(6,),
    hole2_distances=(30,),
)
```

Topology: 6-15 faces depending on holes/fillet.

### Face Types

| Code | Type | Detection |
|------|------|-----------|
| 0 | PLANAR | Flat faces |
| 1 | HOLE | Cylinder with arc ≥ 180° |
| 2 | FILLET | Cylinder with arc < 180°, or torus |

### VariableGraphVAE (Recommended with Geometric Solver)

- **Encoder**: GNN with face type embeddings, attention pooling
- **Latent**: 32D
- **Decoder**: MLP → node features (13D) + edge features + masks + face type logits
- **Node features**: area, dir_xyz, centroid_xyz, curvatures, bbox_diagonal, bbox_center_xyz
- **Usage**: Decode to features, then use Geometric Solver for parameters

### Geometric Solver

- **Input**: Decoded node features (13D), face types, edge features
- **Output**: All L-bracket parameters (deterministic)
- **Algorithm**:
  1. De-normalize using bbox_diagonal and bbox_center
  2. Identify face roles by normal directions (Y-facing = front/back, etc.)
  3. Core params: From face centroids and areas
  4. Holes: From HOLE face areas (π×d×thickness) and centroids
  5. Fillet: From FILLET face area ((π/2)×r×width)
- **Accuracy**: Exact for all parameters (within ~2mm for hole distances)
- **Location**: `graph_cad/utils/geometric_solver.py`

### ParameterVAE (Legacy - Limited by 64% ceiling)

- **Encoder**: Same GNN as VariableGraphVAE
- **Latent**: 32D
- **Decoder**: Multi-head MLP → 18 parameter outputs (4 core + fillet + holes)
- **Limitation**: Latent editor accuracy capped at 64% due to nonlinear encoding

### Latent Regressor (Legacy)

- **Input**: z (32D)
- **Architecture**: MLP (32 → 256 → 128 → 64 → 4)
- **Output**: Normalized [leg1, leg2, width, thickness]
- **Note**: Superseded by Geometric Solver for deterministic reconstruction

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `train_variable_vae.py` | **Train VariableGraphVAE with 13D features (recommended)** |
| `generate_simple_edit_data.py` | **Generate simplified directional edit data (recommended)** |
| `generate_variable_edit_data.py` | Generate numerical edit data (legacy) |
| `train_latent_editor.py` | Train LLM latent editor |
| `evaluate_variable_vae.py` | Analyze latent space (collapse, correlations, clustering) |
| `infer_latent_editor.py` | End-to-end inference with geometric solver |
| `train_parameter_vae.py` | Train ParameterVAE (legacy - 64% ceiling) |
| `train_full_latent_regressor.py` | Train z → all params (legacy) |

### Inference

```bash
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 longer" \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --editor-checkpoint outputs/latent_editor_variable/best_model.pt \
    --latent-regressor-checkpoint outputs/latent_regressor/best_model.pt
```

**Regressor Options (priority order):**

| Flag | Regressor Type | Params | STEP Output |
|------|----------------|--------|-------------|
| `--full-latent-regressor-checkpoint` | FullLatentRegressor | All (4 core + fillet + 4 holes) | **Yes** |
| `--latent-regressor-checkpoint` | LatentRegressor | 4 core only | No |
| `--regressor-checkpoint` | FeatureRegressor | 4 or 8 (auto-detect) | Fixed only |

Example with full regressor (generates STEP):
```bash
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 longer" \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --editor-checkpoint outputs/latent_editor_variable/best_model.pt \
    --full-latent-regressor-checkpoint outputs/full_latent_regressor/best_model.pt
```

---

## Model Checkpoints

**Current (Simplified Instructions + 13D VAE):**
- `outputs/vae_variable_13d/best_model.pt` — VariableGraphVAE with 13D features ✓
- `outputs/latent_editor_simple/best_model.pt` — **0.2% relative error** ✓

**Legacy (13D VAE + Numerical Instructions):**
- `outputs/latent_editor_variable_13d/best_model.pt` — 63.5% direction accuracy (averaging problem)

**Legacy (ParameterVAE approach - 64% ceiling):**
- `outputs/parameter_vae_v4/best_model.pt` — ParameterVAE aux_weight=1.0
- `outputs/parameter_vae_v3/best_model.pt` — ParameterVAE aux_weight=0.1
- `outputs/parameter_vae_v2/best_model.pt` — ParameterVAE stricter bottleneck
- `outputs/latent_editor_parameter_vae_v2/best_model.pt` — 64% direction acc

**Legacy (VariableGraphVAE 9D approach):**
- `outputs/vae_variable/best_model.pt` — Variable topology VAE (9D features)
- `outputs/full_latent_regressor/best_model.pt` — Multi-head regressor, 100% exist acc
- `outputs/latent_editor_variable/best_model.pt` — 64% direction acc

**Legacy (Fixed Topology):**
- `outputs/vae_aux/best_model.pt` — Fixed topology VAE with aux loss (16D)
- `outputs/latent_editor_direction/best_model.pt` — 80.2% accuracy
- `outputs/feature_regressor_aux/best_model.pt` — ~12mm MAE

---

## Technical Stack

- **ML Framework**: PyTorch
- **GNN Library**: PyTorch Geometric
- **CAD Kernel**: CadQuery
- **LLM**: Mistral 7B (via transformers + peft)
- **Quantization**: bitsandbytes (4-bit QLoRA)

---

## Next Steps

1. ~~**Train ParameterVAE v2**~~ — Done, 11.78 nats ✓
2. ~~**Train latent editor on v2**~~ — Done, 64% accuracy ✓
3. ~~**Add auxiliary linear head (v3)**~~ — Done, correlations unchanged ✓
4. ~~**Try aux_weight=1.0 (v4)**~~ — Done, confirms fundamental limitation ✓
5. ~~**Implement Geometric Solver**~~ — Done, exact parameter extraction ✓
6. ~~**Retrain VariableGraphVAE with 13D features**~~ — Done, 100% accuracy, 0.0041 centroid loss ✓
7. ~~**Generate edit data using 13D VAE**~~ — Done ✓
8. ~~**Train latent editor on 13D edit data**~~ — Done, 63.5% accuracy (same ceiling) ✓
9. ~~**Discover z-space encodes params (r=0.98)**~~ — Problem is LLM interface, not z-space ✓
10. ~~**Train latent editor with simplified instructions**~~ — Done, 0.2% relative error ✓
11. ~~**Discover bbox loss bug**~~ — Loss function only trained 9/13 features, bbox never learned ✓
12. ~~**Retrain VAE v2 with bbox loss fix**~~ — Done, bbox 13% error, but posterior collapse ✓
13. ~~**Retrain VAE v3 with aux_weight=0.1**~~ — Done, 62% direction acc, correlations still weak ✓
14. ~~**Diagnose posterior collapse root cause**~~ — Done, β too low, free_bits too high, decoder overpowered ✓
15. ~~**Implement collapse fix (v4)**~~ — Done, beta annealing + reduced decoder + new defaults ✓
16. **Train VAE v4 with collapse fix** — Pending

**Current focus:**
- Train VAE v4 and verify collapse is fixed (expect: active_dims > 28/32, std > 0.8)
- If collapse fixed, retrain latent editor and test end-to-end pipeline
- Validate geometric solver works with non-collapsed latents

**Possible directions forward:**
- Two-stage approach: latent edits for direction, separate regressor for params
- Alternative architecture that doesn't rely on graph→latent→params pipeline
- Focus on fixed topology where 80% accuracy was achieved
