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

### Two-Phase Approach

1. **Phase 1 (Complete)**: Graph Autoencoder
   - STEP file → Graph → Latent vector → Graph → Parameters → STEP file
   - Enables compression and reconstruction of CAD geometry

2. **Phase 2 (Complete)**: LLM Latent Editor
   - Natural language instruction + latent → edited latent
   - "make leg1 20mm longer" → modified geometry

### System Pipeline

```
Input STEP → Graph Extraction → VAE Encode → Latent (16D)
                                                ↓
                            LLM Latent Editor (instruction)
                                                ↓
                                         Edited Latent
                                                ↓
                              VAE Decode → Graph Features
                                                ↓
                      FeatureRegressor → 8 Parameters
                                                ↓
                            LBracket → Output STEP
```

## PoC Scope: L-Brackets

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

## Key Findings

### VAE Reconstruction Limit

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

#### Test Results: "make leg1 20mm longer" (5 samples)

| Original | Edited | Δ leg1 | % of Target | Result |
|----------|--------|--------|-------------|--------|
| 154.64 | 155.98 | +1.34 | 7% | ⚠️ Weak |
| 68.65 | 87.40 | +18.76 | 94% | ✓ Good |
| 113.45 | 124.89 | +11.45 | 57% | ⚠️ Partial |
| 135.49 | 142.66 | +7.18 | 36% | ⚠️ Weak |
| 152.59 | 153.45 | +0.86 | 4% | ⚠️ Weak |

**leg1 summary**: 5/5 correct direction (improved from 3/8), magnitude 4-94% of target.

#### Test Results: "make leg2 20mm longer" (6 samples)

| Original | Edited | Δ leg2 | Δ leg1 | Result |
|----------|--------|--------|--------|--------|
| 65.57 | 65.55 | -0.02 | +9.06 | ❌ Wrong param |
| 92.92 | 95.53 | +2.61 | +3.87 | ⚠️ Weak |
| 92.96 | 96.94 | +3.98 | +3.74 | ⚠️ Weak |
| 165.65 | 164.05 | -1.60 | +1.67 | ❌ Wrong direction |
| 166.42 | 164.41 | -2.02 | -0.98 | ❌ Wrong direction |
| 105.90 | 114.34 | +8.45 | +4.16 | ⚠️ Partial (42%) |

**leg2 summary**: 3/6 correct direction, max 42% of target. Model confuses leg1/leg2.

#### Key Observations

1. **Asymmetric performance**: leg1 edits work reliably (5/5 correct direction), leg2 does not (3/6)
2. **Starting position sensitivity**: Brackets near parameter bounds (leg~150-170mm) show minimal changes—latent space may saturate
3. **Delta magnitude variance**: LLM produces inconsistent delta magnitudes (0.06-0.87) for identical instructions
4. **Parameter coupling**: Edits to one leg often affect the other due to entangled latent space

#### Architecture Coupling

The LLM and FeatureRegressor are **indirectly coupled** through VAE features:

```
LLM Training:                          Inference:
bracket_src → VAE → z_src              z_src → LLM → delta_z
bracket_tgt → VAE → z_tgt              z_edited = z_src + delta_z
delta = z_tgt - z_src                  VAE.decode(z_edited) → features
Train: (z_src, instruction) → delta    FeatureRegressor(features) → params
                                       LBracket(**params) → STEP
```

#### Root Causes

1. **Entangled latent space** — leg1/leg2 not well disentangled; dominant "size" direction scales whole bracket
2. **Non-uniform latent coverage** — LLM deltas work on average but don't generalize to all input regions
3. **Boundary saturation** — Brackets with parameters near min/max show reduced edit effectiveness
4. **VAE bottleneck** — ~12mm reconstruction error compounds with edit errors

#### Lessons Learned

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

## Contrastive Learning (In Progress)

### Approach: Paired Batch Contrastive Loss

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

### Implementation

1. **Data generation**: `--paired` flag generates matched increase/decrease samples
2. **Dataset**: `PairedLatentEditDataset` holds paired samples
3. **Training**: `train_epoch_paired` does two forward passes per sample
4. **Loss**: Combines MSE for accuracy + contrastive for direction

### Training Commands

```bash
# Generate paired data
python scripts/generate_edit_data.py \
    --paired \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_paired

# Train with contrastive loss
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_paired \
    --contrastive-weight 0.5 \
    --epochs 20 \
    --output-dir outputs/latent_editor_contrastive
```

### Expected Outcome

- `mean_cos_sim` should decrease toward -1.0 (opposite directions)
- Increase/decrease accuracy should become symmetric (~75%+ both)
- May trade some MSE accuracy for directional control

---

## Model Checkpoints

**Current best checkpoints:**
- `outputs/vae_aux/best_model.pt` — VAE with auxiliary parameter loss (16D, β=0.01)
- `outputs/latent_editor_aux_ep20_lr1e4/best_model.pt` — LLM Latent Editor (aux-VAE, pre-contrastive)
- `outputs/feature_regressor/best_model.pt` — FeatureRegressor

**Legacy checkpoints:**
- `outputs/vae_16d_lowbeta/best_model.pt` — Original VAE (collapsed latent space)
- `outputs/latent_editor_vae16d_lowbeta/best_model.pt` — Original LLM Latent Editor

## Training Commands

```bash
# VAE with auxiliary parameter loss (recommended)
python scripts/train_vae.py \
    --epochs 100 --latent-dim 16 --target-beta 0.01 --free-bits 2.0 \
    --aux-param-weight 0.1 \
    --output-dir outputs/vae_aux

# FeatureRegressor (cloud GPU recommended)
python scripts/train_feature_regressor.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --cache-dir data/feature_regressor_cache \
    --train-size 10000 --epochs 100

# Latent Editor data generation - PAIRED for contrastive learning
python scripts/generate_edit_data.py \
    --paired \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 --output data/edit_data_paired

# Latent Editor training WITH contrastive loss (cloud GPU, A40 recommended)
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_paired \
    --contrastive-weight 0.5 \
    --epochs 20 \
    --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_contrastive
```

## Inference

```bash
# Full pipeline
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 20mm longer" \
    --regressor-checkpoint outputs/feature_regressor/best_model.pt \
    --output outputs/inference/edited.step

# VAE-only mode (testing without LLM)
python scripts/infer_latent_editor.py \
    --random-bracket --vae-only --verbose
```

## Component Summary

### Graph VAE
- **Encoder**: 3× GAT layers (4 heads, 64 hidden) + global mean pooling
- **Latent**: 16D with free-bits constraint (prevents posterior collapse)
- **Decoder**: MLP producing 124 graph features
- **Performance**: Node MSE ~0.001, Edge MSE ~0.001

### FeatureRegressor
- **Input**: Flattened VAE-decoded features (124D)
- **Architecture**: MLP (124 → 256 → 128 → 64 → 8)
- **Output**: 8 L-bracket parameters
- **Performance**: ~12mm overall MAE (at VAE's theoretical limit)

### LLM Latent Editor
- **Base model**: Mistral 7B with QLoRA (4-bit)
- **Input**: Latent token (16D → 4096D) + text instruction
- **Output**: Delta prediction (residual editing)
- **Performance**: Delta MSE 0.0037, Delta MAE 0.029 (~2.9% error per latent dimension)

## MVP Expansion Path

When extending beyond L-brackets:

1. **Variable Topology**: GNN encoder with attention pooling → fixed latent
2. **Multiple Part Families**: Universal autoencoder trained on mixed dataset
3. **Real CAD Data**: Preprocessing pipeline for ABC Dataset, Fusion 360 Gallery
4. **Complex Features**: Richer node/edge features (fillets, chamfers, curves)

**Key insight**: The LLM can reason over entangled latent spaces. Perfect disentanglement is not required—the editor learns the semantics during training.

## PoC Success Criteria (Updated — Dec 2024)

### Component Metrics (aux-VAE)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| VAE Node MSE | < 0.01 | 0.00073 | ✓ |
| VAE Effective Dims | > 50% | 100% (8/8 params encoded) | ✓ |
| LLM Delta MSE | < 0.01 | 0.0055 | ✓ |
| LLM Delta MAE | — | 0.043 | — |

### End-to-End Metrics (Pre-Contrastive)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Overall direction accuracy | > 70% | 52% | ❌ Blocked by shortcut |
| Increase accuracy | > 70% | 8-100% (param dependent) | ❌ Asymmetric |
| Decrease accuracy | > 70% | 4-95% (param dependent) | ❌ Asymmetric |

### Targets for Contrastive Learning

| Metric | Target | Notes |
|--------|--------|-------|
| mean_cos_sim | < -0.5 | Opposite deltas for opposite instructions |
| Inc/Dec symmetry | < 20% gap | Both directions ~equally accurate |
| Direction accuracy | > 70% | For all parameters |

**Current blocker**: Model takes shortcut (always predict fixed direction per parameter). Contrastive learning should fix this by explicitly penalizing same-direction predictions for opposite instructions.

## Technical Stack

- **ML Framework**: PyTorch
- **GNN Library**: PyTorch Geometric
- **CAD Kernel**: CadQuery
- **LLM**: Mistral 7B (via transformers + peft)
- **Quantization**: bitsandbytes (4-bit QLoRA)
