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

### System Pipeline

```
Input STEP → Graph Extraction → VAE Encode → Latent (32D)
                                                ↓
                            LLM Latent Editor (instruction)
                                                ↓
                                         Edited Latent
                                                ↓
                         LatentRegressor → 4 Core Parameters
                                                ↓
                          VariableLBracket → Output STEP
```

### Two-Phase Approach

1. **Phase 1 (Complete)**: Fixed Topology PoC
   - L-brackets with exactly 2 holes, 10 faces
   - 16D latent, 8 parameters
   - Achieved 80.2% direction accuracy

2. **Phase 2 (In Progress)**: Variable Topology
   - L-brackets with 0-4 holes, optional fillets, 6-15 faces
   - 32D latent, 4 core parameters + fillet + holes
   - VAE trained ✓, Full Latent Regressor trained ✓, Latent Editor 64% (tuning)

---

## Current Status: Variable Topology (Jan 2026)

### Variable Topology VAE — Training Complete ✓

**Test Metrics:**

| Metric | Value | Assessment |
|--------|-------|------------|
| Node Mask Accuracy | 100% | Perfect topology prediction |
| Edge Mask Accuracy | 100% | Perfect topology prediction |
| Face Type Accuracy | 100% | Perfect PLANAR/HOLE/FILLET classification |
| Reconstruction Loss | 0.0173 | Low |
| Active Latent Dims | 32/32 (100%) | No dimension collapse |

**Checkpoint:** `outputs/vae_variable/best_model.pt`

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

**Results across direction_weight values:**

| direction_weight | Direction Acc | Plateau Epoch |
|------------------|---------------|---------------|
| 0.5 | 64.3% | ~5 |
| 1.0 | 64% | ~3 |
| 2.0 | 63.8% | ~3 |

**Root Cause:** The VariableGraphVAE decodes to **graph features** (centroids, areas, normals), not parameters. The latent space encodes geometry, but parameters are only weakly correlated (r<0.12). "Make leg1 longer" has no consistent direction in z-space.

**Comparison to Fixed Topology:**
- Fixed VAE used `aux_weight=0.1` → parameters encoded in latent → 80% direction acc
- Variable VAE used `aux_weight=0.0` → geometry encoded, not params → 64% ceiling

### Solution: ParameterVAE

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

**Options going forward:**
1. **Accept 64% ceiling** for variable topology (still useful)
2. **Hybrid approach**: Add explicit parameter inputs alongside geometry
3. **Direct prediction**: Skip latent editing, predict params and regenerate
4. **Fixed topology**: Use proven 80% approach where applicable

**Checkpoint:** `outputs/parameter_vae_v4/best_model.pt`

### Training Commands (Variable Topology)

```bash
# ParameterVAE v2 (stricter bottleneck - recommended)
python scripts/train_parameter_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/parameter_vae_v2

# Generate edit data (auto-detects VAE type from checkpoint)
python scripts/generate_variable_edit_data.py \
    --vae-checkpoint outputs/parameter_vae_v2/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_parameter_vae_v2

# Train latent editor (same script for both VAE types)
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_parameter_vae_v2 \
    --latent-dim 32 --direction-weight 0.5 \
    --epochs 20 --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_parameter_vae_v2
```

**Legacy (VariableGraphVAE approach):**
```bash
# Original VAE (weak parameter correlations)
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/vae_variable

# Full Latent Regressor (for z → params after VAE)
python scripts/train_full_latent_regressor.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --train-size 10000 --epochs 100 \
    --output-dir outputs/full_latent_regressor
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

5. **β and Free-Bits**
   - β=0.01 optimal (β=0.1 causes variance collapse)
   - Free-bits=2.0 prevents posterior collapse

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

### VariableGraphVAE (Legacy)

- **Encoder**: GNN with face type embeddings, attention pooling
- **Latent**: 32D
- **Decoder**: MLP → node features + edge features + masks + face type logits
- **Limitation**: Weak parameter correlations (r<0.12)

### ParameterVAE (Recommended)

- **Encoder**: Same GNN as VariableGraphVAE
- **Latent**: 32D
- **Decoder**: Multi-head MLP → 18 parameter outputs (4 core + fillet + holes)
- **Advantage**: Strong parameter correlations, meaningful edit directions

### Latent Regressor

- **Input**: z (32D)
- **Architecture**: MLP (32 → 256 → 128 → 64 → 4)
- **Output**: Normalized [leg1, leg2, width, thickness]

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `train_parameter_vae.py` | **Train ParameterVAE (recommended)** |
| `train_variable_vae.py` | Train VariableGraphVAE (legacy) |
| `evaluate_variable_vae.py` | Analyze latent space (collapse, correlations, clustering) |
| `generate_variable_edit_data.py` | Generate paired edit data (auto-detects ParameterVAE or VariableGraphVAE) |
| `train_latent_editor.py` | Train LLM latent editor with direction classifier |
| `train_full_latent_regressor.py` | Train z → all params (for VariableGraphVAE) |
| `infer_latent_editor.py` | End-to-end inference with regressor selection |

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

**Current (ParameterVAE approach):**
- `outputs/parameter_vae_v4/best_model.pt` — ParameterVAE aux_weight=1.0 (confirms limitation) ✓
- `outputs/parameter_vae_v3/best_model.pt` — ParameterVAE aux_weight=0.1 ✓
- `outputs/parameter_vae_v2/best_model.pt` — ParameterVAE stricter bottleneck ✓
- `outputs/latent_editor_parameter_vae_v2/best_model.pt` — 64% direction acc

**Legacy (VariableGraphVAE approach):**
- `outputs/vae_variable/best_model.pt` — Variable topology VAE (32D)
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
5. **Decision point**: Accept 64% or try alternative approaches (hybrid, direct prediction)
6. **Future**: More complex geometry / multiple part families
