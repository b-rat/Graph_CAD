# Phase 4 Report: Multi-Geometry B-Rep

## Executive Summary

Phase 4 extended Graph_CAD from single-geometry (L-bracket) to 6 geometry types using full B-Rep heterogeneous graph representation. The system achieves 100% geometry type classification, ~2mm parameter reconstruction accuracy, and 99.6% edit direction accuracy.

| Metric | Phase 3 | Phase 4 | Target |
|--------|---------|---------|--------|
| Geometry types | 1 | **6** | Multiple |
| Type classification | N/A | **100%** | >95% |
| Parameter MAE | ~2mm (4 params) | **2.3mm** (2-6 params) | <5mm |
| Direction accuracy | 100% | **99.6%** | >90% |
| Parameter isolation | Poor | **1.2mm drift** | Minimal |

**Key Achievements:**
- HeteroGNN encoder handles variable topology across geometry types
- Extended LLM predicts geometry type + per-type parameters
- 99.6% direction accuracy with good parameter isolation
- Full B-Rep (Vertex-Edge-Face) graph representation

**Remaining Limitations:**
- Magnitude calibration varies by parameter range (12-132%)
- Small-range parameters (thickness, hole_dia) underachieve
- Cylinder has highest parameter leakage (3.59mm)

---

## 1. Problem Statement

### 1.1 Phase 3 Limitations

Phase 3 achieved 100% direction accuracy but was limited to a single geometry type (L-bracket) with 4 parameters. Real CAD applications require:

1. **Multiple geometry types** — Different primitives (tubes, channels, blocks, etc.)
2. **Variable parameter counts** — 2-6 parameters depending on type
3. **Type classification** — Identify geometry before editing
4. **Scalable architecture** — Handle diverse topologies

### 1.2 Phase 4 Objectives

1. Support 6 geometry types with 2-6 parameters each
2. Achieve >95% geometry type classification accuracy
3. Maintain direction accuracy >90% across all types
4. Enable parameter-specific editing with isolation

---

## 2. Geometry Types

| ID | Type | Parameters | Range | Count |
|----|------|------------|-------|-------|
| 0 | SimpleBracket | leg1, leg2, width, thickness | 50-200, 50-200, 20-60, 3-12 mm | 4 |
| 1 | Tube | length, outer_dia, inner_dia | 50-200, 20-80, 10-70 mm | 3 |
| 2 | Channel | width, height, length, thickness | 30-100, 20-80, 50-200, 2-10 mm | 4 |
| 3 | Block | length, width, height | 20-150, 20-150, 20-150 mm | 3 |
| 4 | Cylinder | length, diameter | 30-200, 20-100 mm | 2 |
| 5 | BlockHole | length, width, height, hole_dia, hole_x, hole_y | 50-150, 50-150, 20-80, 5-25, *, * mm | 6 |

### 2.1 SimpleBracket (replaced VariableLBracket)

Initial training used VariableLBracket with holes and fillets, causing 6-15 variable faces. This resulted in poor performance:
- 60% type accuracy
- 14mm MAE

SimpleBracket (no holes, no fillets) provides consistent 6-face topology:
- 100% type accuracy
- 1.7mm MAE

---

## 3. Architecture

### 3.1 Pipeline Overview

```
STEP File
    ↓
B-Rep Extraction (CadQuery)
    ↓
HeteroGraph (Vertex ↔ Edge ↔ Face)
    ↓
HeteroGNN Encoder (GAT per node type)
    ↓
z (32D latent)
    ↓
┌─────────────────────────────────────┐
│     Extended LLM (Mistral 7B)       │
│  ┌─────────────┬─────────────────┐  │
│  │ Class Head  │ Param Heads (6) │  │
│  │ (6 types)   │ (per geometry)  │  │
│  └─────────────┴─────────────────┘  │
└─────────────────────────────────────┘
    ↓
Geometry Type + Edited Parameters
```

### 3.2 B-Rep HeteroGraph Features

| Node Type | Features | Dimension |
|-----------|----------|-----------|
| Vertex | Normalized xyz coordinates | 3 |
| Edge | length, tangent_xyz, curvatures | 6 |
| Face | area, normal_xyz, centroid_xyz, curvatures, bbox | 13 |

| Edge Type | Description |
|-----------|-------------|
| vertex_to_edge | Vertex belongs to edge |
| edge_to_face | Edge bounds face |
| face_to_face | Faces share an edge |

### 3.3 HeteroVAE Encoder

- Separate GAT layers for each node type
- Message passing across heterogeneous edges
- Global pooling per node type, concatenated
- MLP to latent space (32D)

### 3.4 Extended Latent Editor

Two-stage training:
1. **Pre-training**: Latent → geometry type + parameters (no instruction)
2. **Instruction tuning**: Latent + instruction → edited parameters

Architecture:
- Mistral 7B with 4-bit QLoRA
- Latent projector (32D → 4096D)
- Classification head (6 geometry types)
- Per-type parameter heads (separate MLP per geometry)

---

## 4. Training Pipeline

### 4.1 Stage 1: HeteroVAE

```bash
python scripts/train_hetero_vae.py \
    --samples-per-type 5000 \
    --epochs 100 \
    --output-dir outputs/hetero_vae_v2
```

| Metric | Value |
|--------|-------|
| Training samples | 30,000 (5k × 6 types) |
| Epochs | 100 |
| Final loss | 0.0285 |
| Geometry accuracy | 100% |

### 4.2 Stage 2: LLM Pre-training

```bash
python scripts/train_llm_pretrain.py \
    --vae-checkpoint outputs/hetero_vae_v2/best_model.pt \
    --epochs 50 \
    --output-dir outputs/llm_pretrain_v2
```

| Metric | Value |
|--------|-------|
| Task | Latent → type + params |
| Epochs | 50 |
| Final ParamMAE | 0.0423 |
| Geometry accuracy | 100% |

### 4.3 Stage 3: Instruction Tuning

```bash
python scripts/train_llm_instruct.py \
    --vae-checkpoint outputs/hetero_vae_v2/best_model.pt \
    --llm-checkpoint outputs/llm_pretrain_v2/best_model.pt \
    --epochs 30 \
    --output-dir outputs/llm_instruct_v2
```

| Metric | Value |
|--------|-------|
| Task | Latent + instruction → delta |
| Epochs | 30 |
| Final ParamMAE | 0.0135 |
| Geometry accuracy | 100% |

---

## 5. Results

### 5.1 Parameter Fidelity (VAE Reconstruction)

100 samples per geometry type, VAE-only (no LLM):

| Geometry | Type Acc | MAE (mm) | Std |
|----------|----------|----------|-----|
| Bracket | 100% | 1.70 | ±0.76 |
| Tube | 100% | 2.74 | ±1.37 |
| Channel | 100% | 2.03 | ±0.83 |
| Block | 100% | 2.37 | ±1.11 |
| Cylinder | 100% | 2.78 | ±1.51 |
| BlockHole | 100% | 2.38 | ±0.88 |
| **Overall** | **100%** | **2.33** | ±1.18 |

### 5.2 Inference Space Study

500 samples across 27 instructions, full pipeline (VAE + LLM):

| Geometry | Dir Acc | VAE MAE | Target Δ | Non-Target Δ |
|----------|---------|---------|----------|--------------|
| Bracket | 100% | 1.68 mm | 15.70 mm | 0.27 mm |
| Tube | 100% | 2.47 mm | 12.03 mm | 0.93 mm |
| Channel | 100% | 2.00 mm | 10.74 mm | 0.31 mm |
| Block | 97.5% | 2.39 mm | 18.21 mm | 0.35 mm |
| Cylinder | 100% | 2.92 mm | 11.63 mm | 3.59 mm |
| BlockHole | 100% | 2.30 mm | 12.27 mm | 1.95 mm |
| **Overall** | **99.6%** | **2.27 mm** | **13.52 mm** | **1.20 mm** |

**Key Findings:**
- Direction accuracy: 99.6% (498/500 correct)
- VAE introduces ~2.3mm baseline error before LLM
- Non-target parameters drift only ~1.2mm (good isolation)
- Cylinder has highest leakage (3.59mm) — only 2 parameters

### 5.3 Magnitude Calibration

Comparison of requested vs achieved parameter deltas:

| Requested | N | Mean Achieved | Ratio |
|-----------|---|---------------|-------|
| 2-3mm | 2 | 0.3mm | 12% |
| 5mm | 2 | 2.3mm | 45% |
| 8-10mm | 4 | 6.9mm | 76% |
| 15mm | 6 | 14.7mm | 98% |
| 20mm | 5 | 22.0mm | 110% |
| 25mm | 4 | 17.8mm | 71% |
| 30mm | 2 | 17.9mm | 60% |

**Per-Instruction Breakdown:**

| Geometry | Instruction | Requested | Achieved | Ratio |
|----------|-------------|-----------|----------|-------|
| Bracket | leg1 +20mm | +20mm | +26.3mm | 132% |
| Bracket | leg1 -15mm | -15mm | -19.8mm | 132% |
| Bracket | thickness +3mm | +3mm | +0.4mm | 12% |
| Channel | thickness +2mm | +2mm | +0.2mm | 12% |
| BlockHole | hole_dia +5mm | +5mm | +0.9mm | 18% |

**Root Cause:** LLM trained on normalized deltas (0-1 range). Parameters with small absolute ranges (thickness: 9mm) compress small mm changes; parameters with large ranges (leg1: 150mm) amplify them.

---

## 6. Error Attribution

### 6.1 Error Sources

| Source | Magnitude | Description |
|--------|-----------|-------------|
| VAE reconstruction | ~2.3mm | Baseline error before LLM |
| Non-target drift | ~1.2mm | Unintended changes to other params |
| Magnitude miscalibration | 12-132% | Varies by parameter range |

### 6.2 Parameter-Specific Issues

| Parameter | Issue | Cause |
|-----------|-------|-------|
| thickness | 12% achievement | Small range (3-12mm = 9mm) |
| hole_dia | 18% achievement | Small range |
| leg1/leg2 | 132% overshoot | Large range (50-200mm = 150mm) |

### 6.3 Geometry-Specific Issues

| Geometry | Issue | Cause |
|----------|-------|-------|
| Cylinder | 3.59mm non-target drift | Only 2 params, edits leak |
| BlockHole | 1.95mm drift | 6 params, more complex |

---

## 7. Checkpoints

### 7.1 Recommended (v2 - SimpleBracket)

| Checkpoint | Description |
|------------|-------------|
| `outputs/hetero_vae_v2/best_model.pt` | HeteroVAE encoder |
| `outputs/llm_pretrain_v2/best_model.pt` | LLM pre-trained |
| `outputs/llm_instruct_v2/best_model.pt` | LLM instruction-tuned |
| `outputs/llm_instruct_v2/best_model_lora/` | LoRA adapter weights |

### 7.2 Deprecated (v1 - VariableLBracket)

| Checkpoint | Issue |
|------------|-------|
| `outputs/hetero_vae/best_model.pt` | 60% bracket accuracy |
| `outputs/llm_pretrain/best_model.pt` | Poor bracket performance |
| `outputs/llm_instruct/best_model.pt` | 14mm bracket MAE |

---

## 8. Key Files

| File | Purpose |
|------|---------|
| `graph_cad/data/brep_types.py` | Edge/face/geometry type constants |
| `graph_cad/data/brep_extraction.py` | Full B-Rep (V/E/F) graph extraction |
| `graph_cad/data/geometry_generators.py` | 6 geometry generators |
| `graph_cad/data/multi_geometry_dataset.py` | Unified dataset |
| `graph_cad/data/param_normalization.py` | Per-type min-max normalization |
| `graph_cad/models/hetero_vae.py` | HeteroGNN encoder + VAE |
| `graph_cad/models/hetero_decoder.py` | Geometry-aware decoder |
| `graph_cad/models/extended_latent_editor.py` | LLM with classification + regression |
| `scripts/train_hetero_vae.py` | HeteroVAE training |
| `scripts/train_llm_pretrain.py` | LLM pre-training |
| `scripts/train_llm_instruct.py` | LLM instruction tuning |
| `scripts/infer_phase4.py` | Full inference pipeline |
| `scripts/study_param_fidelity.py` | VAE reconstruction study |
| `scripts/study_inference_space.py` | Full pipeline analysis |

---

## 9. Inference Commands

```bash
# VAE-only mode (test parameter reconstruction)
python scripts/infer_phase4.py --random --geometry-type bracket --vae-only

# Full LLM instruction following
python scripts/infer_phase4.py --random --geometry-type tube \
    --instruction "make it +20mm longer"

# Test all geometry types (VAE only)
python scripts/infer_phase4.py --test-all
```

**Critical: Instruction Format**

Must use explicit `+/-` signs (training data artifact):

| Instruction | Result |
|-------------|--------|
| `make leg1 +20mm longer` | Correct |
| `make leg1 20mm longer` | May fail |

---

## 10. Future Improvements

### 10.1 Magnitude Calibration

1. **Denormalized training targets** — Train on absolute mm deltas
2. **Parameter-aware loss weighting** — Upweight small-range params
3. **Post-hoc calibration layer** — Learn per-parameter scaling
4. **Separate magnitude head** — Predict direction and magnitude independently

### 10.2 Architecture

1. **More geometry types** — Extend to complex assemblies
2. **Real CAD import** — Support arbitrary STEP files
3. **Iterative refinement** — Multi-step editing for precision

---

## 11. Conclusions

Phase 4 successfully extended Graph_CAD to multi-geometry support with:

- **100% geometry type classification** across 6 types
- **~2.3mm parameter reconstruction** accuracy
- **99.6% edit direction accuracy** with good parameter isolation
- **Full B-Rep representation** (Vertex-Edge-Face heterogeneous graph)

The main remaining challenge is magnitude calibration, which varies from 12% to 132% depending on parameter range. This is a known issue caused by normalized training and can be addressed with the strategies outlined above.

**Phase 4 Status: Complete**
