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
outputs/            # Current phase checkpoints (gitignored)
outputs_phase3/     # Phase 3 archived checkpoints (gitignored)
data/               # Training data (gitignored)
docs/               # Reports and progress logs
CAD/                # Face-annotated prototype STEP files for synthetic dataset generation
```

## Output Directories

| Directory | Contents |
|-----------|----------|
| `outputs/` | **Phase 5** (current) — new training runs go here |
| `outputs_phase4/` | **Phase 4** (archived) — multi-geometry B-Rep, HeteroVAE |
| `outputs_phase3/` | **Phase 3** (archived) — transformer VAE, latent editor |
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
| 4. Multi-Geometry B-Rep | Complete | 99.6% dir acc, 2.3mm MAE | `docs/phase4_multi_geometry_report.md` |
| 5. Geometric Comprehension | Planning | — | `docs/Faeno Phase 5 ammended.md` |

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

### Node Features (V/E/F Heterogeneous Graph)

**Vertex Features (3D)**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-2 | position (xyz) | (coord - bbox_center) / bbox_diagonal |

**Edge Features (6D)**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | length | / bbox_diagonal |
| 1-3 | tangent at midpoint (xyz) | unit vector |
| 4 | curvature at start | × bbox_diagonal, clipped to [-10, 10] |
| 5 | curvature at end | × bbox_diagonal, clipped to [-10, 10] |

**Edge Types**

| Code | Type | Detection |
|------|------|-----------|
| 0 | LINE | Straight edge |
| 1 | ARC | Circular arc (partial circle) |
| 2 | CIRCLE | Full circle (closed curve) |
| 3 | OTHER | B-spline, ellipse, etc. |

**Face Features (13D)**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | area | / bbox_diagonal² |
| 1-3 | normal (xyz) | unit vector |
| 4-6 | centroid (xyz) | (c - bbox_center) / bbox_diagonal |
| 7-8 | curvatures | × bbox_diagonal, clipped to [-10, 10] |
| 9 | bbox_diagonal | / 100mm |
| 10-12 | bbox_center | / 100mm |

**Face Types**

| Code | Type | Detection |
|------|------|-----------|
| 0 | PLANAR | Flat faces |
| 1 | HOLE | Cylinder with arc ≥ 180° |
| 2 | FILLET | Cylinder with arc < 180°, or torus |

**Topology Connections**

| Relation | Direction | Description |
|----------|-----------|-------------|
| vertex_to_edge | V → E | Which vertices bound each edge |
| edge_to_face | E → F | Which edges bound each face |
| (reverse edges added for bidirectional message passing) | | |

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

### Models
| File | Purpose |
|------|---------|
| `graph_cad/models/hetero_vae.py` | HeteroGNN encoder + VAE wrapper |
| `graph_cad/models/extended_latent_editor.py` | LLM with classification + regression |
| `graph_cad/models/latent_editor.py` | Mistral 7B + LoRA base |

### Data
| File | Purpose |
|------|---------|
| `graph_cad/data/brep_extraction.py` | Full B-Rep (V/E/F) graph extraction |
| `graph_cad/data/geometry_generators.py` | 6 geometry generators |
| `graph_cad/data/multi_geometry_dataset.py` | Unified dataset |
| `graph_cad/data/param_normalization.py` | Per-type min-max normalization |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/train_hetero_vae.py` | HeteroVAE training |
| `scripts/train_llm_pretrain.py` | LLM pre-training |
| `scripts/train_llm_instruct.py` | LLM instruction tuning |
| `scripts/infer_phase4.py` | Full inference pipeline |
| `scripts/study_inference_space.py` | Pipeline analysis study |

---

## Phase 4 Summary

**Problem:** Phase 3 limited to single geometry type (L-bracket) with 4 parameters.

**Solution:** HeteroGNN encoder with full B-Rep (Vertex-Edge-Face) representation + Extended LLM with geometry classification and per-type parameter heads.

**Results:**

| Metric | Phase 3 | Phase 4 |
|--------|---------|---------|
| Geometry types | 1 | **6** |
| Type classification | N/A | **100%** |
| Parameter MAE | ~2mm | **2.3mm** |
| Direction accuracy | 100% | **99.6%** |
| Parameter isolation | Poor | **1.2mm drift** |

**Geometry Types:** SimpleBracket, Tube, Channel, Block, Cylinder, BlockHole (2-6 params each)

**Key Insight:** Magnitude calibration varies by parameter range — small-range params (thickness: 12%) underachieve while large-range params (leg1: 132%) overshoot. Root cause: normalized training compresses/amplifies by range.

**Remaining Limitation:** Magnitude accuracy 12-132% depending on parameter range.

See `docs/phase4_multi_geometry_report.md` for full details.

---

## Current Checkpoints (Phase 4)

| Checkpoint | Description |
|------------|-------------|
| `outputs/hetero_vae_v2/best_model.pt` | HeteroVAE encoder (6 geometry types) |
| `outputs/llm_instruct_v2/best_model.pt` | LLM instruction-tuned |
| `outputs/llm_instruct_v2/best_model_lora/` | LoRA adapter weights |

**Inference:**
```bash
python scripts/infer_phase4.py --random --geometry-type bracket \
    --instruction "make leg1 +20mm longer"
```

**Critical:** Must use explicit `+/-` signs in instructions.

---

## CAD Folder: Annotated STEP Files

The `CAD/` folder contains face-annotated STEP files that serve as prototypes for synthetic dataset generation. These files have semantic labels on `ADVANCED_FACE` entities (e.g., `'x0'`, `'slot.bottom'`, `'z_length'`) that enable natural language geometry editing.

### Purpose

- **Direct STEP editing**: Map instructions like "increase z_length by 20mm" to specific coordinate changes
- **Synthetic dataset generation**: Programmatically vary labeled parameters to create (geometry, instruction, delta) training pairs
- **Ground truth for training**: Labels define which faces/parameters the model should associate with instructions

### STEP File Structure

STEP files use `ADVANCED_FACE` entities with labels:
```
#349 = ADVANCED_FACE ( 'z_length', ( #289 ), #135, .F. ) ;
#38 = ADVANCED_FACE ( 'bottom', ( #201 ), #140, .F. ) ;
#195 = ADVANCED_FACE ( 'slot.planar_1', ( #286 ), #329, .F. ) ;
```

Face geometry is defined by: `ADVANCED_FACE` → `PLANE` → `AXIS2_PLACEMENT_3D` → `CARTESIAN_POINT` + `DIRECTION`

### Modifying STEP Files

**Simple coordinate changes** (dimensions, positions):
1. Read the STEP file and identify the dimension to change
2. Find all `CARTESIAN_POINT` entries with coordinates matching the current value
3. Use `replace_all` to update coordinates (e.g., change z=3.0" to z=4.0")

Example - change z_length from 3" to 4" (file uses inches):
```python
# Replace all z-coordinates at 3.0 with 4.0
Edit(file_path, old_string="3.000000000000000000", new_string="4.000000000000000000", replace_all=True)
```

**Topology changes** (adding/removing features like blind slots):
1. Use CadQuery to generate new geometry programmatically
2. Post-process the STEP file to add semantic face labels
3. See `CAD/generate_blind_slot.py` for a complete example

### Labeling Faces Programmatically

When generating STEP files with CadQuery, faces have empty labels. Post-process to add labels:

```python
def get_face_label(plane_point, plane_normal):
    """Assign label based on plane position and normal direction."""
    px, py, pz = plane_point
    nx, ny, nz = plane_normal

    # Block boundary faces
    if abs(px) < tol and abs(nx) > 0.9: return "x0"
    if abs(px - BLOCK_X) < tol and abs(nx) > 0.9: return "x_width"
    if abs(pz - BLOCK_Z) < tol and abs(nz) > 0.9: return "z_length"

    # Slot faces (interior)
    if abs(px - SLOT_LEFT_X) < tol and abs(nx) > 0.9: return "slot.wall_left"
    if abs(py - SLOT_BOTTOM_Y) < tol and abs(ny) > 0.9: return "slot.bottom"
    # etc.
```

Parse STEP to find: `PLANE` → `AXIS2_PLACEMENT_3D` → get point and normal → assign label.

### Current Prototype Files

| File | Description |
|------|-------------|
| `block_slot_annot.STEP` | Original block with through slot (SolidWorks export) |
| `block_slot_annot_100mm.STEP` | Modified dimensions via coordinate editing |
| `block_slot_annot_blind.STEP` | Blind slot variant (CadQuery + post-process labels) |
| `generate_blind_slot.py` | Script to generate labeled blind slot geometry |

### Face Label Conventions

| Label Pattern | Meaning |
|---------------|---------|
| `x0`, `x_width` | Faces at x=0 and x=max |
| `z0`, `z_length` | Faces at z=0 and z=max |
| `bottom`, `top.planar_N` | Faces at y=0 and y=max |
| `slot.wall_left/right` | Vertical slot walls |
| `slot.bottom` | Horizontal slot floor |
| `slot.end` | Blind end cap (if applicable) |
| `hole.N` | Hole cylindrical surfaces |
| `fillet.N` | Fillet surfaces |
