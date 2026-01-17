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
| **3. DETR Transformer Decoder** | **In Progress** | 100% face/edge accuracy, aux head added |

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
- Leg parameters encoded: r > 0.98

**Unsolved Problem: Width/Thickness Not Encoded**

| Parameter | Correlation | Status |
|-----------|-------------|--------|
| leg1 | r = 0.992 | ✓ Excellent |
| leg2 | r = 0.989 | ✓ Excellent |
| width | r = 0.155 | ✗ Poor |
| thickness | r = 0.102 | ✗ Poor |

**Root Cause: Geometric Dominance**

Reconstruction loss is dominated by leg lengths (they affect many faces' areas, centroids, edges). Width and thickness have smaller geometric footprints, so the encoder ignores them — it achieves low reconstruction loss without encoding them.

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

### Next Steps to Try

**Option A: Direct Latent Supervision**
Force first 4 latent dims to equal normalized parameters:
```python
direct_loss = F.mse_loss(mu[:, :4], normalize_params(target_params))
```
Guarantees parameters are encoded — no competition with reconstruction.

**Option B: Two-Phase Training**
1. Phase 1: Train encoder + param_head only (no decoder)
2. Phase 2: Freeze encoder, train decoder only

**Option C: Larger param_head**
Current: 32 → 64 → 4. Try: 32 → 128 → 64 → 4 with residual.

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
| `graph_cad/models/losses.py` | Hungarian matching loss functions (added ~400 lines) |
| `scripts/train_transformer_vae.py` | Training script for Phase 3 |
| `tests/test_transformer_decoder.py` | 16 unit tests (all passing) |

### Training Commands

```bash
# Basic training (without aux head)
python scripts/train_transformer_vae.py --epochs 100 --train-size 5000

# Training with auxiliary parameter head
python scripts/train_transformer_vae.py --epochs 100 --aux-weight 0.1 --aux-loss-type correlation

# Alternative loss types
python scripts/train_transformer_vae.py --aux-weight 1.0 --aux-loss-type mse
python scripts/train_transformer_vae.py --aux-weight 100.0 --aux-loss-type mse_normalized

# Test parameter correlations after training
python scripts/test_tvae.py outputs/vae_transformer/best_model.pt
```

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 32 |
| Decoder hidden dim | 256 |
| Decoder layers | 4 |
| Attention heads | 8 |
| Max nodes | 20 |
| Learning rate | 1e-4 |
| Beta (KL weight) | 0.1 (with 30% warmup) |
| Aux weight | 0.1 (when enabled) |
| Aux loss type | correlation (options: mse, mse_normalized) |

### Hungarian Matching Cost Weights

| Component | Weight |
|-----------|--------|
| Face type (classification) | 2.0× |
| Node features (L2 distance) | 1.0× |
| Existence probability | 1.0× |
| Edge loss | 1.0× |

### Success Criteria

| Metric | Phase 2 (MLP) | Phase 3 Current | Phase 3 Target |
|--------|---------------|-----------------|----------------|
| Direction accuracy | 64% | Pending | **>80%** |
| leg1/leg2 correlation | r < 0.3 | r > 0.99 ✓ | r > 0.7 |
| width correlation | r < 0.3 | r = 0.155 ✗ | **r > 0.7** |
| thickness correlation | r < 0.3 | r = 0.102 ✗ | **r > 0.7** |
| Face type accuracy | ~85% | 100% ✓ | >95% |
| Edge prediction accuracy | N/A | 100% ✓ | >90% |

**Blocker:** Aux head approach insufficient. Need direct latent supervision or two-phase training.

---

## Current Architecture (Encoder — Keep As-Is)

### GAT Encoder

```python
class VariableGraphVAEEncoder(nn.Module):
    # Input: node_features (N, 13), face_types (N,), edge_index, edge_attr
    # Output: mu (batch, 32), logvar (batch, 32)

    # Uses:
    # - Face type embeddings (PLANAR=0, HOLE=1, FILLET=2)
    # - GAT layers with message passing
    # - Masked global mean pooling
```

**Location:** `graph_cad/models/graph_vae.py` lines 406-535

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
| `graph_cad/models/losses.py` | All loss functions including Hungarian matching + aux loss |
| `graph_cad/data/dataset.py` | VariableLBracketDataset |
| `graph_cad/data/graph_extraction.py` | Graph extraction from STEP |
| `scripts/train_transformer_vae.py` | **Phase 3** training script (supports --aux-weight) |
| `scripts/test_tvae.py` | Test parameter correlations in latent space |
| `scripts/train_variable_vae.py` | Phase 2 training script (MLP decoder) |

---

## Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `outputs/vae_variable_v4/best_model.pt` | Phase 2 VAE v4 with collapse fix (32/32 active dims) |
| `outputs/vae_aux/best_model.pt` | Phase 1 fixed topology VAE (80.2% direction accuracy) |
| `outputs/vae_transformer/best_model.pt` | Phase 3 Transformer VAE (leg1/leg2 encoded, width/thickness not) |

**Note:** Current aux head approach insufficient for width/thickness. Need architectural change.
