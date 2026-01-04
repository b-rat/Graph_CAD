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
   - 32D latent, 4 core parameters
   - VAE trained, latent editor training in progress

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
- Smoke test: 100% fillet acc, 95%+ hole acc after 2 epochs

**Conceptual Note:** For parametric templates like L-brackets, we go `params → VariableLBracket() → STEP`. For arbitrary geometry, B-Rep reconstruction from features would be needed — a harder unsolved problem.

### Training Commands (Variable Topology)

```bash
# VAE (complete)
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 --latent-dim 32 \
    --output-dir outputs/vae_variable

# Edit data generation + Latent Editor
# Note: --paired is for contrastive learning; direction classifier doesn't need it
python scripts/generate_variable_edit_data.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_variable && \
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_variable \
    --latent-dim 32 --direction-weight 0.5 \
    --epochs 20 --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_variable

# Full Latent Regressor (all params, recommended)
python scripts/train_full_latent_regressor.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --train-size 10000 --epochs 100 \
    --cache-dir data/full_latent_regressor_cache \
    --output-dir outputs/full_latent_regressor

# Simple Latent Regressor (4 core params only)
python scripts/train_latent_regressor.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --train-size 10000 --epochs 100 \
    --output-dir outputs/latent_regressor
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

### VariableGraphVAE

- **Encoder**: GNN with face type embeddings, attention pooling
- **Latent**: 32D
- **Decoder**: MLP → node features + edge features + masks + face type logits

### Latent Regressor

- **Input**: z (32D)
- **Architecture**: MLP (32 → 256 → 128 → 64 → 4)
- **Output**: Normalized [leg1, leg2, width, thickness]

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `train_variable_vae.py` | Train variable topology VAE |
| `evaluate_variable_vae.py` | Analyze latent space (collapse, correlations, clustering) |
| `generate_variable_edit_data.py` | Generate paired edit data for latent editor |
| `train_latent_editor.py` | Train LLM latent editor with direction classifier |
| `train_full_latent_regressor.py` | Train z → all params (multi-head with existence) |
| `train_latent_regressor.py` | Train z → 4 core params only |
| `train_variable_feature_regressor.py` | Train decoded features → params MLP |
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

**Current (Variable Topology):**
- `outputs/vae_variable/best_model.pt` — Variable topology VAE (32D)
- `outputs/latent_editor_variable/` — In progress
- `outputs/full_latent_regressor/` — Multi-head regressor (all params)
- `outputs/latent_regressor/` — Simple regressor (4 core params)

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

1. **Complete latent editor training** with variable topology VAE
2. **Train latent regressor** (z → params directly)
3. **End-to-end evaluation** on variable topology brackets
4. **If successful**: Expand to more complex geometry / multiple part families
