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
outputs/            # Phase 4 checkpoints (gitignored)
outputs_phase3/     # Phase 3 archived checkpoints (gitignored)
data/               # Training data (gitignored)
docs/               # Reports and progress logs
```

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | **Phase 4** (current) — new training runs go here |
| `outputs_phase3/` | **Phase 3** (archived) — transformer VAE, latent editor, exploration results |
| `outputs_phase2/` | **Phase 2** (archived) — MLP decoder experiments |

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

# For long-running tasks, use nohup with unbuffered output (preferred over tmux)
ssh runpod "cd /workspace/Graph_CAD && PYTHONUNBUFFERED=1 nohup python scripts/train_hetero_vae.py --epochs 100 --output-dir outputs/hetero_vae > outputs/hetero_vae_train.log 2>&1 &"

# Monitor training progress
ssh runpod "tail -f /workspace/Graph_CAD/outputs/hetero_vae_train.log"

# Check if training is running
ssh runpod "ps aux | grep python | grep train"

# Quick one-off commands
ssh runpod "cd /workspace/Graph_CAD && python scripts/explore_instruction_domain.py --num-brackets 10"
```

**Important:** Always use `PYTHONUNBUFFERED=1` with nohup to ensure logs are written in real-time.

**Important:** Never transfer code files directly to RunPod (e.g., via `scp` or `rsync`). Always commit and push changes locally, then `git pull` on RunPod. Direct transfers cause branch divergence.

**Note:** `HF_HOME=/workspace/.cache/huggingface` is configured via symlink. Large caches go to `/workspace` (network volume).

## Technical Stack

- **ML Framework**: PyTorch + PyTorch Geometric
- **CAD Kernel**: CadQuery
- **LLM**: Mistral 7B (4-bit QLoRA via transformers + peft + bitsandbytes)

---

## Project Phases

| Phase | Status | Result | Documentation |
|-------|--------|--------|---------------|
| 1. Fixed Topology PoC | Complete | 80.2% direction accuracy | `docs/fixed_topology_poc_archive.md` |
| 2. Variable Topology (MLP) | Complete | 64% ceiling | `docs/phase2_mlp_decoder_report.md` |
| 3. DETR Transformer | Complete | 100% direction accuracy | `docs/phase3_transformer_decoder_report.md` |
| 4. Multi-Geometry B-Rep | **Complete** | 100% type acc, ~2mm MAE | `docs/phase-4.md` |

---

## Phase 3 Summary

**Problem:** MLP decoder's fixed-slot output incompatible with variable topology (64% ceiling).

**Solution:** DETR-style transformer decoder with Hungarian matching + direct latent supervision.

**Results:**

| Metric | Phase 2 | Phase 3 |
|--------|---------|---------|
| Direction accuracy | 64% | **100%** |
| Width correlation | r < 0.3 | r = 0.998 |
| Thickness correlation | r < 0.3 | r = 0.355 |
| Face/edge accuracy | ~85% | 100% |

**Key Insight:** Mean pooling causes symmetric cancellation for width/thickness. Direct latent supervision bypasses this by forcing `mu[0:4]` to encode parameters directly.

**Remaining Limitation:** Thickness magnitude poor (2.4% target achievement) despite correct direction.

See `docs/phase3_transformer_decoder_report.md` for full details.

---

## Current Architecture

### Pipeline

```
STEP → Graph Extraction → GAT Encoder → z (32D) → Transformer Decoder → Graph
                                           ↓
                               Mistral 7B (Latent Editor)
                                           ↓
                                     Edited z → STEP
```

### Node Features (13D)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | area | / bbox_diagonal² |
| 1-3 | normal (xyz) | unit vector |
| 4-6 | centroid (xyz) | (c - bbox_center) / bbox_diagonal |
| 7-8 | curvatures | × bbox_diagonal |
| 9 | bbox_diagonal | / 100mm |
| 10-12 | bbox_center | / 100mm |

### Face Types

| Code | Type | Detection |
|------|------|-----------|
| 0 | PLANAR | Flat faces |
| 1 | HOLE | Cylinder with arc ≥ 180° |
| 2 | FILLET | Cylinder with arc < 180°, or torus |

### SimpleBracket (Phase 4)

```python
SimpleBracket(
    leg1_length,    # 50-200 mm
    leg2_length,    # 50-200 mm
    width,          # 20-60 mm
    thickness,      # 3-12 mm
)
```

Simple L-bracket with consistent 6-face topology. Replaced VariableLBracket (which had holes/fillets causing 6-15 variable faces) to improve training stability.

### VariableLBracket (Phase 3, deprecated)

```python
VariableLBracket(
    leg1_length, leg2_length, width, thickness,  # 50-200, 50-200, 20-60, 3-12 mm
    fillet_radius=0.0,          # 0-8mm
    hole1_diameters=(6, 8),     # 0-2 holes per leg
    hole1_distances=(20, 60),
    hole2_diameters=(6,),
    hole2_distances=(30,),
)
```

Topology: 6-15 faces depending on holes/fillet. Deprecated in favor of SimpleBracket.

---

## Key Files

| File | Purpose |
|------|---------|
| `graph_cad/models/transformer_decoder.py` | Transformer decoder + VAE wrapper |
| `graph_cad/models/graph_vae.py` | GAT encoder |
| `graph_cad/models/latent_editor.py` | Mistral 7B + LoRA |
| `graph_cad/models/losses.py` | Hungarian matching + aux losses |
| `graph_cad/data/dataset.py` | VariableLBracketDataset |
| `graph_cad/data/graph_extraction.py` | Graph extraction from STEP |
| `scripts/train_transformer_vae.py` | VAE training |
| `scripts/train_latent_editor.py` | Latent editor training |
| `scripts/infer_latent_editor.py` | Full inference pipeline |

### Phase 4 Files

| File | Purpose |
|------|---------|
| `graph_cad/data/brep_types.py` | Edge/face/geometry type constants |
| `graph_cad/data/brep_extraction.py` | Full B-Rep (V/E/F) graph extraction |
| `graph_cad/data/geometry_generators.py` | SimpleBracket, Tube, Channel, Block, Cylinder, BlockHole |
| `graph_cad/data/multi_geometry_dataset.py` | Unified dataset for all 6 geometry types |
| `graph_cad/data/param_normalization.py` | Per-type min-max normalization |
| `graph_cad/models/hetero_vae.py` | HeteroGNN encoder + VAE wrapper |
| `graph_cad/models/hetero_decoder.py` | Geometry-aware decoder + param heads |
| `graph_cad/models/extended_latent_editor.py` | LLM with classification + regression |
| `scripts/train_hetero_vae.py` | HeteroVAE training |
| `scripts/train_llm_pretrain.py` | LLM pre-training (latent -> class + params) |
| `scripts/train_llm_instruct.py` | LLM instruction following training |
| `scripts/infer_phase4.py` | Phase 4 inference (all geometry types) |
| `scripts/study_param_fidelity.py` | Parameter reconstruction accuracy study |

---

## Phase 3 Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `outputs_phase3/vae_direct_kl_exclude_v2/best_model.pt` | **RECOMMENDED** VAE (width r=0.998) |
| `outputs_phase3/latent_editor_all_params/best_model.pt` | **RECOMMENDED** Latent editor |

**Latent Space:** First 4 dims directly encode parameters (no regressor needed):
- `mu[0]` = leg1, `mu[1]` = leg2, `mu[2]` = width, `mu[3]` = thickness

---

---

## Phase 4: Multi-Geometry B-Rep

**Goal:** Extend Graph_CAD to support 6 geometry types using full B-Rep graph representation.

**Status:** Complete (v2 with SimpleBracket).

### Geometry Types

| ID | Type | Parameters | Count |
|----|------|------------|-------|
| 0 | Bracket (SimpleBracket) | leg1, leg2, width, thickness | 4 |
| 1 | Tube | length, outer_dia, inner_dia | 3 |
| 2 | Channel | width, height, length, thickness | 4 |
| 3 | Block | length, width, height | 3 |
| 4 | Cylinder | length, diameter | 2 |
| 5 | BlockHole | length, width, height, hole_dia, hole_x, hole_y | 6 |

### Training Progress (v1 - VariableLBracket)

Initial training with VariableLBracket (complex bracket with holes/fillets):

| Stage | Checkpoint | GeoAcc | ParamMAE |
|-------|------------|--------|----------|
| 1. HeteroVAE | `outputs/hetero_vae/best_model.pt` | 100% | 0.0246 |
| 2. LLM Pre-train | `outputs/llm_pretrain/best_model.pt` | 100% | 0.0431 |
| 3. LLM Instruct | `outputs/llm_instruct/best_model.pt` | 100% | 0.0133 |

**Issue:** Parameter fidelity study revealed Bracket performed poorly:
- Bracket: 60% type accuracy, 14mm MAE
- Other types: 100% type accuracy, ~2mm MAE

**Solution:** Replaced VariableLBracket with SimpleBracket (no holes, no fillets) for cleaner topology.

### Training Progress (v2 - SimpleBracket)

Retraining with SimpleBracket complete:

| Stage | Checkpoint | GeoAcc | ParamMAE |
|-------|------------|--------|----------|
| 1. HeteroVAE | `outputs/hetero_vae_v2/best_model.pt` | 100% | 0.0285 |
| 2. LLM Pre-train | `outputs/llm_pretrain_v2/best_model.pt` | 100% | 0.0423 |
| 3. LLM Instruct | `outputs/llm_instruct_v2/best_model.pt` | 100% | 0.0135 |

**Parameter Fidelity (v2, 100 samples/type):**

| Geometry | Type Acc | MAE (mm) | Std |
|----------|----------|----------|-----|
| Bracket | 100% | 1.70 | ±0.76 |
| Tube | 100% | 2.74 | ±1.37 |
| Channel | 100% | 2.03 | ±0.83 |
| Block | 100% | 2.37 | ±1.11 |
| Cylinder | 100% | 2.78 | ±1.51 |
| BlockHole | 100% | 2.38 | ±0.88 |
| **Overall** | **100%** | **2.33** | ±1.18 |

### Architecture

```
STEP → B-Rep Extraction → HeteroGNN (V↔E↔F) → z (32D) → Transformer Decoder
                                                 ↓
                           Extended LLM (class + params)
                                                 ↓
                                    Geometry Type + Parameters
```

### B-Rep HeteroGraph Features

| Node Type | Features | Dimension |
|-----------|----------|-----------|
| Vertex | Normalized xyz coordinates | 3 |
| Edge | length, tangent_xyz, curvatures | 6 |
| Face | area, normal_xyz, centroid_xyz, curvatures, bbox | 13 |

### Training Pipeline

```bash
# Stage 1: Train HeteroVAE on all 6 geometries
python scripts/train_hetero_vae.py \
    --samples-per-type 5000 \
    --epochs 100 \
    --output-dir outputs/hetero_vae_v2

# Stage 2: Pre-train LLM (classification + regression)
python scripts/train_llm_pretrain.py \
    --vae-checkpoint outputs/hetero_vae_v2/best_model.pt \
    --epochs 50 \
    --output-dir outputs/llm_pretrain_v2

# Stage 3: Train instruction following (requires GPU)
python scripts/train_llm_instruct.py \
    --vae-checkpoint outputs/hetero_vae_v2/best_model.pt \
    --llm-checkpoint outputs/llm_pretrain_v2/best_model.pt \
    --epochs 30 \
    --output-dir outputs/llm_instruct_v2
```

### Phase 4 Inference

```bash
# VAE-only mode (test parameter reconstruction)
python scripts/infer_phase4.py --random --geometry-type bracket --vae-only

# Full LLM instruction following
python scripts/infer_phase4.py --random --geometry-type tube \
    --instruction "make it +20mm longer"

# Test all geometry types (VAE only)
python scripts/infer_phase4.py --test-all
```

---

## Latent Editor Usage

### Inference

```bash
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 +20mm longer" \
    --seed 123
```

### Critical: Instruction Format

**Must use explicit `+/-` signs** (training data artifact):

| Instruction | Result |
|-------------|--------|
| `make leg1 +20mm longer` | Correct |
| `make leg1 20mm longer` | WRONG (sign flipped) |

### Training

```bash
# VAE with direct latent supervision
python scripts/train_transformer_vae.py --epochs 100 \
    --aux-weight 1.0 --aux-loss-type direct \
    --output-dir outputs/vae_new

# Generate edit data
python scripts/generate_edit_data_transformer.py \
    --vae-checkpoint outputs/vae_new/best_model.pt \
    --output-dir data/edit_data_new

# Train latent editor
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_new \
    --output-dir outputs/latent_editor_new
```
