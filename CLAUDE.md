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
| **3. DETR Transformer Decoder** | **Current** | Target: >80% direction accuracy |

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

## Phase 3: DETR Transformer Decoder

### Objective

Replace MLP decoder with permutation-invariant transformer to break the 64% ceiling.

### Architecture

```
z (32D) → Learned Node Queries (max_nodes × hidden)
                    ↓
          Cross-Attention to z
                    ↓
          Self-Attention Layers
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
Node Features   Edge Logits    Existence Mask
 (N × 13D)       (N × N)         (N × 1)
                    ↓
          Hungarian Matching to GT
                    ↓
           Permutation-Invariant Loss
```

### Key Components

1. **Learned Node Queries**: Interchangeable vectors (like DETR object queries)
2. **Cross-Attention**: Queries attend to latent z
3. **Self-Attention**: Queries attend to each other for geometric consistency
4. **Hungarian Matching**: Optimal assignment of predictions to ground truth
5. **Edge Prediction**: Pairwise attention scores or dedicated edge head

### Implementation Steps

1. Implement `TransformerGraphDecoder` in `graph_cad/models/`
2. Implement Hungarian matching loss in `graph_cad/models/losses.py`
3. Train VAE with new decoder (keep existing GAT encoder)
4. Validate parameter correlations improve (target: r > 0.7)
5. Train latent editor (target: >80% direction accuracy)
6. End-to-end validation: instruction → edited STEP file

### Success Criteria

| Metric | Phase 2 (MLP) | Phase 3 Target |
|--------|---------------|----------------|
| Direction accuracy | 64% | **>80%** |
| Parameter correlations | r < 0.3 | **r > 0.7** |
| Parameter RMSE | ~26% | **<15%** |
| Topology detection | 100% | 100% |

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

## Key Files for Phase 3

| File | Purpose |
|------|---------|
| `graph_cad/models/graph_vae.py` | Current VAE (encoder to keep, decoder to replace) |
| `graph_cad/models/losses.py` | Loss functions (add Hungarian matching) |
| `graph_cad/data/dataset.py` | VariableLBracketDataset |
| `graph_cad/data/graph_extraction.py` | Graph extraction from STEP |
| `scripts/train_variable_vae.py` | Training script (modify for new decoder) |

---

## Current Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `outputs/vae_variable_v4/best_model.pt` | VAE v4 with collapse fix (32/32 active dims) |
| `outputs/vae_aux/best_model.pt` | Fixed topology VAE (80.2% direction accuracy) |
