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
docs/               # Reports and progress logs
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

## GPU Execution (RunPod)

For tasks requiring GPU (model training, inference with LLM, exploration studies), use the configured RunPod instance:

```bash
# Connect to RunPod
ssh runpod

# Sync code changes before running
git pull  # on RunPod, after pushing from local

# For long-running tasks, use tmux to survive disconnections
ssh runpod
tmux new -s train  # or tmux attach -t train to reconnect
cd /workspace/Graph_CAD && python scripts/train_transformer_vae.py --epochs 100
# Ctrl+B, D to detach; tmux attach -t train to reconnect later

# Quick one-off commands can run directly
ssh runpod "cd /workspace/Graph_CAD && python scripts/explore_instruction_domain.py --num-brackets 10"
```

The local machine (MPS) is too slow for LLM inference. Always use RunPod for:
- Latent editor training/inference (requires Mistral 7B)
- Exploration studies
- VAE training with large datasets

## Technical Stack

- **ML Framework**: PyTorch
- **GNN Library**: PyTorch Geometric
- **CAD Kernel**: CadQuery
- **LLM**: Mistral 7B (via transformers + peft)
- **Quantization**: bitsandbytes (4-bit QLoRA)

---

## Project Phases

| Phase | Status | Result |
|-------|--------|--------|
| 1. Fixed Topology PoC | Complete | 80.2% direction accuracy |
| 2. Variable Topology (MLP Decoder) | Complete | 64% ceiling (architectural limitation) |
| 3. DETR Transformer Decoder | Complete | 100% face/edge accuracy, leg1/leg2 r>0.98 |
| 3b. Attention Pooling | Failed | Did not improve width/thickness encoding |
| **3c. Direct Latent Supervision** | **Complete** | width r=0.998, thickness r=0.355 |

**Documentation:**
- `docs/phase2_mlp_decoder_report.md` — Root cause analysis
- `docs/phase2_progress_log.md` — Detailed experimental results

---

## Phase 2 Summary: Why MLP Decoder Failed

The MLP decoder's fixed-slot output is incompatible with variable topology:

1. **OCC ordering is inconsistent** — `TopExp_Explorer` order depends on B-Rep internals
2. **Boolean operations scramble order** — Adding holes/fillets reconstructs B-Rep
3. **Same slot = different faces** — Slot 5 might be hole1, fillet, planar, or padding

**Result:** Noisy training signal → encoder can't learn face-specific encoding → 64% ceiling

| Component | Graph-Aware | Permutation Property |
|-----------|-------------|---------------------|
| **Encoder (GAT)** | Yes | Invariant (global pooling) |
| **Decoder (MLP)** | No | NOT invariant (fixed slots) |

---

## Phase 3: DETR Transformer Decoder (In Progress)

### Objective

Replace MLP decoder with permutation-invariant transformer to break the 64% ceiling.

### Current Status

**Reconstruction works perfectly:**
- Face type accuracy: 100%
- Edge accuracy: 100%
- All 4 core parameters now encoded in latent space

**SOLVED: Width Now Encoded via Direct Latent Supervision**

| Parameter | Baseline | Direct Latent | Status |
|-----------|----------|---------------|--------|
| leg1 | r = 0.992 | r = 0.999 | ✓ Excellent |
| leg2 | r = 0.989 | r = 0.999 | ✓ Excellent |
| width | r = 0.155 | r = 0.998 | ✓ **FIXED** |
| thickness | r = 0.102 | r = 0.355 | ↑ Improved |

**Root Cause: Symmetric Cancellation in Mean Pooling**

The encoder uses **global mean pooling** after GAT layers. This loses width/thickness information due to the L-bracket's geometry:

**Why leg lengths survive pooling:**
- L-shape is asymmetric in X and Z directions
- When leg1 extends, faces shift in one direction (not balanced)
- The mean centroid shifts detectably

**Why width/thickness cancel out:**
- Width changes are symmetric in Y direction
- Y=0 face shifts -y, Y=width face shifts +y
- After bbox_center normalization: equal and opposite centroids
- Mean pooling: `(-Δ + Δ) / 2 = 0` — signal lost!

```
Width change example:
  Y=0 face centroid:     (0 - width/2) / diagonal = -0.14
  Y=width face centroid: (width - width/2) / diagonal = +0.14
  Mean: cancels to ~0 regardless of width!
```

The encoder has the information (individual face positions), but mean pooling destroys it before reaching the latent space.

### Attempted Solutions

**1. Auxiliary Parameter Head (param_head)**
```
z (latent) → param_head (MLP) → predicted [leg1, leg2, width, thickness]
                                        ↓
                                   Loss vs ground truth
```

**2. Loss Function Experiments:**

| Approach | aux_weight | width r | thickness r | Issue |
|----------|------------|---------|-------------|-------|
| MSE (raw) | 1.0 | 0.252 | 0.107 | Leg MSE dominates (~300x larger) |
| MSE (normalized) | 1.0 | 0.074 | 0.107 | Loss too small vs reconstruction |
| MSE (normalized) | 100.0 | 0.057 | 0.107 | Still dominated by reconstruction |
| Correlation | 1.0 | - | - | Collapse (negative loss dominates) |
| Correlation | 0.1 | 0.155 | 0.102 | Best so far, but still insufficient |

**3. Why Aux Head Approach Fails:**
- Reconstruction gradients overwhelm param_head gradients
- Encoder learns what decoder needs (leg lengths), ignores aux head signal
- param_head can only predict what's already in latent space

### Failed Experiment: Multi-Head Attention Pooling

**Hypothesis:** Replace mean pooling with learned attention pooling to break symmetric cancellation.

**Result:** Did NOT improve width/thickness encoding. Attention pooling achieved similar or worse correlations than mean pooling across multiple configurations (3 GAT/64D, 6 GAT/128D, 2 GAT/64D).

| Configuration | width r | thickness r |
|---------------|---------|-------------|
| Baseline (3 GAT, Mean) | 0.155 | 0.102 |
| Attention (3 GAT, 4 heads) | 0.068 | 0.104 |
| Scaled (6 GAT, 128D, 8 heads) | 0.056 | 0.093 |

**Why it failed:** Attention pooling changes *how* faces are weighted but doesn't fundamentally solve the information bottleneck. The symmetric cancellation happens in the normalized face features, not the pooling weights.

### Successful Solution: Direct Latent Supervision with KL Exclusion

**Approach:** Force first 4 latent dimensions to directly encode normalized parameters, and exclude these dimensions from KL divergence.

```python
# In losses.py
def normalize_params_for_latent(params: torch.Tensor) -> torch.Tensor:
    """Scale [0, 1] normalized params to [-2, 2] for latent space."""
    return params * 4.0 - 2.0

# Direct supervision on mu[:, :4]
direct_loss = F.mse_loss(mu[:, :4], normalize_params_for_latent(target_params))

# KL exclusion: don't penalize first 4 dims toward N(0,1)
kl_per_dim = kl_per_dim[:, 4:]  # Skip supervised dims
```

**Training command:**
```bash
python scripts/train_transformer_vae.py --epochs 100 \
    --aux-weight 1.0 --aux-loss-type direct \
    --output-dir outputs/vae_direct_kl_exclude_v2
```

**Why KL exclusion is critical:**
- Without exclusion: KL pushes mu[:, :4] toward N(0,1), conflicting with direct supervision
- With exclusion: First 4 dims are free to encode parameters at any value
- Remaining 28 dims still follow N(0,1) for sampling

**Final Results:**

| Parameter | Best Dim | Correlation |
|-----------|----------|-------------|
| leg1 | dim 0 | r = 0.999 |
| leg2 | dim 1 | r = 0.999 |
| width | dim 2 | r = 0.998 |
| thickness | dim 3 | r = 0.355 |

**Thickness limitation:** Still moderate correlation. Possible causes:
- Thickness has smallest range (3-12mm vs 50-200mm for legs)
- May need stronger supervision or separate treatment

### Alternative Approaches (Not Needed)

**Option B: Two-Phase Training** — Not attempted (direct supervision worked)

**Option C: Larger param_head** — Not needed (direct supervision bypasses param_head)

### Implemented Architecture

```
z (32D) ─→ [Linear + LayerNorm] ─→ Memory (1 × 256)
                                        ↓
Learned Queries (20 × 256) ─→ [Transformer Decoder × 4 layers]
  + Positional Embeddings              ↓
                              Node Embeddings (20 × 256)
                                        ↓
                    ┌───────────┬───────────┬───────────┐
                    ↓           ↓           ↓           ↓
             Node Features  Face Types  Existence   Edge Logits
              (20 × 13)     (20 × 3)     (20,)      (20 × 20)
                    ↓           ↓           ↓           ↓
              ──────────── Hungarian Matching ────────────
                                        ↓
                           Permutation-Invariant Loss
```

### Key Components (All Implemented)

1. **Learned Node Queries**: 20 interchangeable vectors (like DETR object queries)
2. **Cross-Attention**: Queries attend to projected latent z
3. **Self-Attention**: 4-layer transformer decoder with 8 heads
4. **Hungarian Matching**: scipy.optimize.linear_sum_assignment for optimal GT assignment
5. **Edge Prediction**: Pairwise MLP on concatenated node embeddings (binary, symmetric)
6. **Output Heads**: Separate MLPs for node features, face types, existence, edges

### Implementation Files

| File | Description |
|------|-------------|
| `graph_cad/models/transformer_decoder.py` | TransformerGraphDecoder, TransformerGraphVAE |
| `graph_cad/models/graph_vae.py` | GAT encoder + **MultiHeadAttentionPooling** |
| `graph_cad/models/losses.py` | Hungarian matching loss functions (added ~400 lines) |
| `scripts/train_transformer_vae.py` | Training script (supports `--pooling-type attention`) |
| `tests/test_transformer_decoder.py` | 24 unit tests (all passing) |

### Training Commands

```bash
# Basic training with mean pooling (default)
python scripts/train_transformer_vae.py --epochs 100 --train-size 5000

# RECOMMENDED: Direct latent supervision (encodes all 4 params)
python scripts/train_transformer_vae.py --epochs 100 \
    --aux-weight 1.0 --aux-loss-type direct \
    --output-dir outputs/vae_direct_latent

# Training with auxiliary parameter head (leg1/leg2 only)
python scripts/train_transformer_vae.py --epochs 100 --aux-weight 0.1 --aux-loss-type correlation

# Test parameter correlations after training
python scripts/test_tvae.py outputs/vae_direct_kl_exclude_v2/best_model.pt
```

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 32 |
| Encoder GAT layers | 3 |
| Encoder hidden dim | 64 |
| **Pooling type** | mean (options: mean, **attention**) |
| **Attention heads** | 4 (for attention pooling) |
| Decoder hidden dim | 256 |
| Decoder layers | 4 |
| Decoder attention heads | 8 |
| Max nodes | 20 |
| Learning rate | 1e-4 |
| Beta (KL weight) | 0.1 (with 30% warmup) |
| Aux weight | 1.0 (for direct), 0.1 (for correlation) |
| Aux loss type | **direct** (recommended), correlation, mse, mse_normalized |

### Hungarian Matching Cost Weights

| Component | Weight |
|-----------|--------|
| Face type (classification) | 2.0× |
| Node features (L2 distance) | 1.0× |
| Existence probability | 1.0× |
| Edge loss | 1.0× |

### Success Criteria

| Metric | Phase 2 (MLP) | Phase 3 Final | Phase 3 Target |
|--------|---------------|---------------|----------------|
| Direction accuracy | 64% | Pending | **>80%** |
| leg1/leg2 correlation | r < 0.3 | r = 0.999 ✓ | r > 0.7 |
| width correlation | r < 0.3 | r = 0.998 ✓ | **r > 0.7** |
| thickness correlation | r < 0.3 | r = 0.355 ↑ | **r > 0.7** |
| Face type accuracy | ~85% | 100% ✓ | >95% |
| Edge prediction accuracy | N/A | 100% ✓ | >90% |

**Status:** Width encoding solved via direct latent supervision. Thickness partially improved but below target.

---

## Current Architecture (Encoder — Keep As-Is)

### GAT Encoder

```python
class VariableGraphVAEEncoder(nn.Module):
    # Input: node_features (N, 13), face_types (N,), edge_index, edge_attr
    # Output: mu (batch, 32), logvar (batch, 32)

    # Uses:
    # - Face type embeddings (PLANAR=0, HOLE=1, FILLET=2)
    # - 3 GAT layers with message passing (64D hidden)
    # - Pooling: mean (default) or attention (4 heads)
```

**Pooling options:**
- `pooling_type="mean"`: Global mean pooling (loses symmetric info)
- `pooling_type="attention"`: Multi-head attention pooling (preserves symmetric info)

**Location:** `graph_cad/models/graph_vae.py`

### Node Features (13D) — Decoder Output Spec

```
[area, dir_x, dir_y, dir_z, cx, cy, cz, curv1, curv2, bbox_diagonal, bbox_cx, bbox_cy, bbox_cz]
```

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | area | / bbox_diagonal² |
| 1-3 | normal (dir_xyz) | unit vector |
| 4-6 | centroid (cx,cy,cz) | (c - bbox_center) / bbox_diagonal |
| 7-8 | curvatures | × bbox_diagonal |
| 9 | bbox_diagonal | / 100mm |
| 10-12 | bbox_center | / 100mm |

### Face Types

| Code | Type | Detection |
|------|------|-----------|
| 0 | PLANAR | Flat faces |
| 1 | HOLE | Cylinder with arc ≥ 180° |
| 2 | FILLET | Cylinder with arc < 180°, or torus |

### VariableLBracket

```python
VariableLBracket(
    leg1_length, leg2_length, width, thickness,  # Core params (50-200, 50-200, 20-60, 3-12 mm)
    fillet_radius=0.0,          # 0 = no fillet (0-8mm)
    hole1_diameters=(6, 8),     # 0-2 holes per leg (4-12mm)
    hole1_distances=(20, 60),   # Distance from corner
    hole2_diameters=(6,),
    hole2_distances=(30,),
)
```

Topology: 6-15 faces depending on holes/fillet.

---

## Key Files

| File | Purpose |
|------|---------|
| `graph_cad/models/transformer_decoder.py` | **Phase 3** Transformer decoder + VAE wrapper + param_head |
| `graph_cad/models/graph_vae.py` | GAT encoder (used by both Phase 2 & 3) |
| `graph_cad/models/latent_editor.py` | Mistral 7B + LoRA for latent space editing |
| `graph_cad/models/losses.py` | All loss functions including Hungarian matching + aux loss |
| `graph_cad/data/dataset.py` | VariableLBracketDataset |
| `graph_cad/data/edit_dataset.py` | Edit instruction dataset for latent editor |
| `graph_cad/data/graph_extraction.py` | Graph extraction from STEP |
| `scripts/train_transformer_vae.py` | **Phase 3** training script (supports --aux-weight) |
| `scripts/train_latent_editor.py` | Train latent editor on edit data |
| `scripts/train_latent_regressor.py` | Train z → params regressor |
| `scripts/generate_edit_data_transformer.py` | Generate leg-length edit training data |
| `scripts/infer_latent_editor.py` | **Inference** Full pipeline: STEP → edit → STEP |
| `scripts/test_tvae.py` | Test parameter correlations in latent space |
| `scripts/train_variable_vae.py` | Phase 2 training script (MLP decoder) |

---

## Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `outputs/vae_direct_kl_exclude_v2/best_model.pt` | **RECOMMENDED** Direct latent supervision (width r=0.998) |
| `outputs/vae_transformer_aux2_w100/best_model.pt` | Transformer VAE with aux head (leg1/leg2 r>0.98, width r=0.155) |
| `outputs/latent_editor_tvae/best_model.pt` | Latent editor for leg length operations |
| `outputs/latent_regressor_tvae/best_model.pt` | z → params regressor (2-3mm MAE for legs) |
| `outputs/vae_variable_v4/best_model.pt` | Phase 2 VAE v4 with collapse fix (32/32 active dims) |
| `outputs/vae_aux/best_model.pt` | Phase 1 fixed topology VAE (80.2% direction accuracy) |
| `outputs/vae_transformer/best_model.pt` | Phase 3 Transformer VAE (no aux head) |

**Best model for latent editing:** `outputs/vae_direct_kl_exclude_v2/best_model.pt` — first 4 dims encode [leg1, leg2, width, thickness].

---

## Latent Editor Pipeline

### Current Working Configuration

The latent editor was trained on **leg length operations** using the old VAE checkpoint. With the new direct latent supervision model, width editing should also be possible (requires retraining latent editor).

| Component | Checkpoint | Description |
|-----------|------------|-------------|
| VAE | `outputs/vae_transformer_aux2_w100/best_model.pt` | Transformer VAE with aux head (w=100) |
| Latent Editor | `outputs/latent_editor_tvae/best_model.pt` | Mistral 7B + LoRA, trained on leg edits |
| Latent Regressor | `outputs/latent_regressor_tvae/best_model.pt` | z → 4 core params (bypasses decoder) |

### Training Data

Edit data location: `data/edit_data_legs/`

```json
{
  "vae_checkpoint": "outputs/vae_transformer_aux2_w100/best_model.pt",
  "latent_dim": 32,
  "edit_types": ["single_leg", "both_legs", "noop"],
  "parameters": ["leg1_length", "leg2_length"]
}
```

### CRITICAL: Instruction Format

**The latent editor learned to rely on explicit `+/-` signs in instructions.**

| Instruction | Result |
|-------------|--------|
| `make leg1 +20mm longer` | ✓ Correct direction |
| `make leg1 20mm longer` | ✗ WRONG direction (sign flipped) |
| `change leg1 length by +20mm` | ✓ Correct direction |
| `leg1 +20mm` | ✓ Correct direction |

**Always use `+` for positive changes and `-` for negative changes in instructions.**

This is a training data artifact — the model overfit to `+/-` tokens rather than learning "longer/shorter" semantics.

### Inference Commands

```bash
# Full inference with latent editor
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 +20mm longer" \
    --seed 123

# VAE-only mode (no LLM, for testing encoder/decoder)
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "test" \
    --vae-only

# All defaults are set correctly for the transformer pipeline
```

### Latent Space Correlations

**New model (`vae_direct_kl_exclude_v2`):** First 4 dimensions directly encode parameters:

| Dimension | Parameter | Correlation |
|-----------|-----------|-------------|
| dim 0 | leg1 | r = 0.999 |
| dim 1 | leg2 | r = 0.999 |
| dim 2 | width | r = 0.998 |
| dim 3 | thickness | r = 0.355 |

For latent editing, directly modify `z[0:4]` to change parameters.

**Old model (`vae_transformer_aux2_w100`):** Parameters distributed across dimensions:

| Dimension | Correlation with leg1_delta |
|-----------|----------------------------|
| dim 23 | r = -0.967 |
| dim 27 | r = -0.965 |
| dim 11 | r = +0.962 |

For the old model, a +20mm leg1 change expected:
- `delta_z[11]`: ~+0.22 (positive)
- `delta_z[23]`: ~-0.22 (negative)

### Known Limitations

1. **Latent editor needs retraining**: Current editor trained on old VAE; needs retraining with `vae_direct_kl_exclude_v2` for width editing
2. **Thickness encoding moderate**: r = 0.355 in new model (improved from 0.102 but below target)
3. **Instruction format sensitivity**: Must use explicit `+/-` signs
4. **Entanglement**: Editing leg1 may cause spurious changes to leg2/width
5. **Latent regressor accuracy**: ~2-3mm MAE for leg lengths, ~8mm for width

### Latent Regressor vs Geometric Solver

| Method | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| Latent Regressor | ~2-3mm MAE (legs) | Fast | Predicts from z directly |
| Geometric Solver | Variable | Fast | Extracts from decoded features, can be lossy |

The latent regressor is preferred when available (set as default in inference script).

### Exploration Study Results (50 brackets, 1300 trials)

Results from `outputs/exploration/full_study_260117.json`:

**Direction Accuracy:**

| Parameter | Correct Direction | Notes |
|-----------|-------------------|-------|
| leg1_length | **100%** | Perfect |
| leg2_length | **100%** | Perfect |
| both_legs | **98.7%** | Near-perfect |
| noop | **17.5%** | Fails to hold still |

**Magnitude Scaling (target achievement):**

| Magnitude | leg1 | leg2 | Notes |
|-----------|------|------|-------|
| 10mm | 93% | 105% | Good |
| 20mm | 87% | 94% | Good |
| 30mm | 91% | 96% | Good |
| 50mm | 75% | 78% | Undershoots on large requests |

**Cross-talk (Entanglement):**

When editing one parameter, others change unintentionally:
- Editing leg1 → leg2 changes by ~3.6mm
- Editing leg1 → width changes by ~5.7mm
- Thickness is stable (~0.2mm)

**No-op Problem:**

The model doesn't understand "no change" instructions:
- "keep it the same" → 11.2mm change, 0% correct
- "don't modify anything" → 7.0mm change, 38% correct (best)
- Model produces non-zero deltas (0.16-0.26 norm) for all no-op instructions

**Both Legs Asymmetry:**

When requesting both legs change together:
- leg1 average change: 0.2mm
- leg2 average change: 2.4mm
- Model favors leg2 over leg1

**Summary:**
- Excellent at direction (100% for single legs)
- Good at magnitude (75-95% of target)
- Struggles with no-op (produces changes when it shouldn't)
- Has entanglement (editing one param affects others)
