# Phase 2 Report: MLP Decoder Architecture Analysis

## Executive Summary

This report documents the findings from Phase 1 (Fixed Topology) and Phase 2 (Variable Topology) of the Graph_CAD project, which aims to enable natural language editing of CAD models through learned latent spaces.

**Key Finding:** The MLP decoder architecture fundamentally limits parameter extraction for variable topology because it enforces fixed slot ordering on outputs, while graph extraction produces inconsistent node orderings across different geometries and topologies.

| Phase | Topology | Direction Accuracy | Parameter Error | Status |
|-------|----------|-------------------|-----------------|--------|
| Phase 1 | Fixed (10 faces, 2 holes) | **80.2%** | ~12mm MAE | Success |
| Phase 2 | Variable (6-15 faces, 0-4 holes) | **64%** ceiling | ~26% RMSE | Architectural limitation |

---

## 1. Architecture Overview

### 1.1 System Pipeline

```
STEP File → Graph Extraction → GNN Encoder → z (latent) → MLP Decoder → Node Features
                                                ↓
                                    LLM Latent Editor (instruction)
                                                ↓
                                         Edited z → Decode → Parameters → STEP
```

### 1.2 Current VAE Architecture

**Encoder (Graph Attention Network):**
- Input: Node features (13D per face), edge index, edge features
- Architecture: GAT with message passing + attention pooling
- Output: μ, σ for 32D latent via reparameterization
- **Property: Theoretically permutation-invariant** (uses graph structure, not node order)

**Decoder (MLP):**
- Input: z (32D latent vector)
- Architecture: MLP backbone → fixed-size output
- Output: `(batch, max_nodes, 13)` node features + masks + face types
- **Property: NOT permutation-invariant** (outputs to fixed slots)

### 1.3 Node Features (13D)

```
[area, dir_x, dir_y, dir_z, cx, cy, cz, curv1, curv2, bbox_diagonal, bbox_cx, bbox_cy, bbox_cz]
```

---

## 2. Phase 1: Fixed Topology Results

### 2.1 Configuration

- **Topology:** Always 10 faces, 2 holes (fixed L-bracket template)
- **Latent dimension:** 16D
- **Parameters:** 8 (leg1, leg2, width, thickness, 2 hole diameters, 2 hole distances)

### 2.2 Results

| Metric | Value |
|--------|-------|
| VAE Node MSE | 0.00073 |
| Direction Accuracy | **80.2%** |
| Parameter MAE | ~12mm |
| Active Latent Dims | 8/16 (with aux_weight=0.1) |

### 2.3 Why Fixed Topology Worked

The MLP decoder's fixed-slot output was **inadvertently compatible** with fixed topology:

1. **Consistent OCC Ordering:** Same parametric template → same construction sequence → same face ordering from OCC's `TopExp_Explorer`

2. **Implicit Slot Semantics:** The decoder learned meaningful slot assignments:
   - Slot 0 → front face (always)
   - Slot 3 → first hole cylinder (always)
   - Slot 8 → back face (always)

3. **Pseudo-Graph Structure:** The ordered feature list acted as a consistent representation:
   ```
   [front, back, left, right, top, bottom, hole1, hole2, inner1, inner2]
   ```
   This consistency allowed "increase leg1" to have a stable meaning in latent space.

4. **Parameter Correlations:** With `aux_weight=0.1`, strong correlations emerged:
   - leg1_length ↔ z_dim_5: r = 0.85+
   - The latent encoded parameters in interpretable dimensions

---

## 3. Phase 2: Variable Topology Results

### 3.1 Configuration

- **Topology:** 6-15 faces, 0-4 holes, optional fillet
- **Latent dimension:** 32D
- **Parameters:** 4-13 depending on topology

### 3.2 Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Topology Detection | **100%** | Hole/fillet presence |
| Face Type Accuracy | **100%** | PLANAR/HOLE/FILLET |
| Direction Accuracy | **64%** | Stuck at ceiling |
| Parameter RMSE | **~26%** | Both solver and regressor |
| Active Latent Dims | 32/32 | After collapse fix |

### 3.3 Approaches Attempted

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Higher aux_weight (0.1 → 1.0) | 64% ceiling | Doesn't change encoder behavior |
| ParameterVAE (decode to params) | 64% ceiling | Same GNN encoder limitation |
| Geometric Solver | 26% error | Decoder reconstruction noise |
| FullLatentRegressor | 26% error | z doesn't encode params |
| Simplified Instructions | 0.2% z_delta error | May be averaging artifact |
| Increase-Only Training | Pending | Addresses averaging |

### 3.4 The 64% Direction Accuracy Ceiling

Across all approaches, direction accuracy plateaued at ~64%:

```
Epoch 1:  36.1%
Epoch 2:  62.9% (rapid rise)
Epoch 3+: 63.0-64.0% (plateau)
```

This is barely better than random (50%), indicating the model cannot reliably determine edit direction from the latent representation.

---

## 4. Root Cause Analysis: Node Ordering Problem

### 4.1 The Fundamental Issue

The MLP decoder outputs to **fixed slots**, but graph extraction produces **inconsistent node orderings**:

```python
# Decoder assumption
output = decoder(z)  # Shape: (batch, max_nodes=15, features=13)
# Slot 0 should always mean "front face"
# Slot 5 should always mean "hole 1"

# Reality: OCC extraction order varies
Bracket A: [front, back, hole1, left, right, ...]
Bracket B: [left, hole2, front, back, hole1, ...]  # Different order!
```

### 4.2 Sources of Ordering Inconsistency

**1. OCC Internal Representation:**
- `TopExp_Explorer` iterates faces based on internal B-Rep structure
- Order depends on construction history, not geometric semantics
- Same topology can produce different orderings

**2. Boolean Operations Scramble Order:**
Adding features (holes, fillets) reconstructs the B-Rep:
```
Before fillet: [front, back, left, right, top, bottom]
After fillet:  [left, fillet, front, bottom, back, right, top]  # Scrambled!
```

**3. Variable Topology Compounds the Problem:**
```
6-face bracket:  slots 0-5 = real faces, 6-14 = padding
10-face bracket: slots 0-9 = real faces, 10-14 = padding
15-face bracket: slots 0-14 = all real faces

Slot 8 might be: hole1 OR hole2 OR fillet OR planar face OR padding
```

### 4.3 The Vicious Training Cycle

```
Inconsistent slot targets (same slot = different face types across samples)
        ↓
Decoder learns to predict averages (minimize MSE across all samples)
        ↓
Gradients to z are noisy (conflicting supervision)
        ↓
Encoder can't learn face-specific encoding
        ↓
z becomes topology-only (global properties survive, details lost)
        ↓
Decoder gets uninformative z
        ↓
Decoder relies more on slot averages
        (cycle continues)
```

### 4.4 Evidence from Code Analysis

**Graph Extraction (`graph_cad/data/graph_extraction.py`):**
```python
# Faces extracted in OCC order - no canonical sorting
explorer = TopExp_Explorer(shape, TopAbs_FACE)
while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    faces.append(face)  # Order determined by OCC internals
```

**Decoder (`graph_cad/models/graph_vae.py`):**
```python
# MLP outputs to fixed slots - assumes consistent ordering
node_flat = self.node_head(h)  # (batch, max_nodes * features)
node_features = node_flat.view(batch_size, max_nodes, features)
```

**No shuffling or canonical ordering is applied anywhere in the pipeline.**

### 4.5 Why Topology Detection Still Works

Topology is **order-invariant**:
- "How many holes?" → Count HOLE-type faces (order doesn't matter)
- "Is there a fillet?" → Check if any FILLET type exists

The encoder uses global pooling (order-invariant), so topology info survives the noisy training signal.

### 4.6 Why Parameter Extraction Fails

Parameters require **face identity**:
- "What is leg1_length?" → Need to identify which faces belong to leg1
- With scrambled ordering, the decoder can't consistently reconstruct specific faces
- The latent can't encode "leg1" cleanly because leg1's faces land in different slots

---

## 5. Why Fixed Topology Succeeded (Retrospective)

Fixed topology worked **not by design, but by accident**:

1. **Same Template:** All brackets built with identical `LBracket()` constructor
2. **Deterministic Construction:** Same boolean operation sequence
3. **Consistent OCC Order:** Same construction → same internal B-Rep → same face order
4. **Stable Slot Semantics:** Slot N always corresponded to the same conceptual face

This created an **implicit canonical ordering** that was never explicitly enforced.

When we moved to variable topology:
- Different numbers of holes → different construction sequences
- Optional fillet → different boolean operations
- The implicit ordering broke down

---

## 6. Potential Solutions Within MLP Decoder Framework

### 6.1 Canonical Face Ordering (Recommended for MLP)

Sort faces by deterministic geometric criteria before encoding:

```python
def canonicalize_faces(faces, features):
    """Sort faces by geometric properties for consistent ordering."""
    # Primary: Face type (PLANAR=0, HOLE=1, FILLET=2)
    # Secondary: Normal direction (which axis)
    # Tertiary: Centroid position

    sort_keys = []
    for i, (face, feat) in enumerate(zip(faces, features)):
        face_type = feat[0]  # or separate face_types array
        normal = feat[1:4]
        centroid = feat[4:7]

        # Determine primary axis of normal
        axis = np.argmax(np.abs(normal))
        axis_sign = np.sign(normal[axis])

        sort_keys.append((
            face_type,           # Group by type
            axis,                # Then by axis
            axis_sign,           # Then by direction
            centroid[0],         # Then by position
            centroid[1],
            centroid[2],
        ))

    order = sorted(range(len(faces)), key=lambda i: sort_keys[i])
    return [faces[i] for i in order], features[order]
```

**Advantages:**
- Minimal code change
- Same MLP decoder architecture
- Consistent slot semantics across all topologies

**Disadvantages:**
- Requires careful design of sort criteria
- May not generalize beyond L-brackets
- Still fundamentally relies on fixed slots

### 6.2 Face Type Grouping

Organize slots by face type:
```
Slots 0-5:  PLANAR faces (sorted by centroid)
Slots 6-9:  HOLE faces (sorted by position)
Slots 10-11: FILLET faces
Slots 12-14: Padding
```

This creates predictable slot semantics even with variable topology.

### 6.3 Augmentation with Shuffling

Train with random node permutations to force order-invariance:

```python
def augment_with_shuffle(node_features, edge_index):
    perm = torch.randperm(node_features.size(0))
    node_features = node_features[perm]
    # Update edge_index to reflect permutation
    inv_perm = torch.argsort(perm)
    edge_index = inv_perm[edge_index]
    return node_features, edge_index
```

**Problem:** The MLP decoder still outputs to fixed slots, so this doesn't solve the fundamental issue — it just adds noise to training.

---

## 7. Fundamental Solution: Graph-Aware Decoder

The MLP decoder is architecturally incompatible with variable topology. A graph-aware decoder solves this by making output permutation-invariant.

### 7.1 DETR-Style Transformer Decoder (Recommended)

```
z → Learned Node Queries → Cross-Attention to z → Self-Attention → Node Features
                                                         ↓
                                              Hungarian Matching to GT
```

**Key Properties:**
- Node queries are interchangeable (no fixed slot semantics)
- Self-attention lets nodes find relationships
- Hungarian matching assigns predictions to ground truth optimally
- Order-invariant by design

### 7.2 Architecture Comparison

| Aspect | MLP Decoder | DETR Decoder |
|--------|-------------|--------------|
| Output order | Fixed slots | Order-invariant |
| Node relationships | None (independent) | Self-attention |
| Training signal | Per-slot MSE | Hungarian-matched MSE |
| Variable topology | Problematic | Native support |
| Edge prediction | Separate head | Natural (attention scores) |

### 7.3 Expected Benefits

1. **Consistent training signal:** Each GT face matched to best prediction
2. **No slot semantics:** Predictions are a set, not an ordered list
3. **Geometric consistency:** Self-attention enforces relationships
4. **Clean latent space:** Encoder receives consistent gradients

---

## 8. Summary of Findings

### 8.1 What Worked

| Component | Fixed Topology | Variable Topology |
|-----------|---------------|-------------------|
| VAE reconstruction | 0.00073 MSE | 0.0085 MSE |
| Topology detection | N/A | 100% |
| Face type classification | N/A | 100% |
| Latent health (active dims) | 8/16 | 32/32 (v4) |

### 8.2 What Didn't Work

| Issue | Fixed Topology | Variable Topology |
|-------|---------------|-------------------|
| Direction accuracy | 80.2% | 64% ceiling |
| Parameter correlations | Strong (r > 0.8) | Weak (r < 0.3) |
| Parameter extraction | ~12mm MAE | ~26% RMSE |
| Interpretable latent | Yes | No |

### 8.3 Root Causes Identified

1. **MLP decoder enforces fixed slot ordering** — incompatible with variable topology
2. **OCC face ordering is inconsistent** — depends on construction history
3. **Boolean operations scramble face order** — adding holes/fillets changes ordering
4. **No canonical ordering enforced** — implicit ordering in fixed topology, absent in variable
5. **Inconsistent targets poison training** — decoder learns averages, encoder receives noise

### 8.4 Key Insight

The fixed topology success was **accidental**, relying on implicit ordering from consistent template construction. This ordering broke down with variable topology, exposing the fundamental incompatibility between MLP decoders and graph-structured data.

---

## 9. Recommendations

### 9.1 Short-Term (MLP Decoder)

If continuing with MLP decoder:
1. **Implement canonical face ordering** before graph extraction
2. **Group slots by face type** for predictable semantics
3. **Validate ordering consistency** with unit tests

### 9.2 Long-Term (Recommended)

Replace MLP decoder with **DETR-style transformer decoder**:
1. Permutation-invariant output (node queries)
2. Hungarian matching for training
3. Self-attention for geometric consistency
4. Native support for variable topology

### 9.3 Phase 3 Objectives

1. Implement DETR transformer decoder
2. Train VAE with Hungarian matching loss
3. Validate parameter correlations improve
4. Test latent editor on new architecture
5. Achieve >80% direction accuracy on variable topology

---

## 10. Appendix: Code References

### Key Files

| File | Purpose |
|------|---------|
| `graph_cad/data/graph_extraction.py` | Face extraction (lines 96-220) |
| `graph_cad/data/dataset.py` | Dataset with padding (lines 212-416) |
| `graph_cad/models/graph_vae.py` | Encoder (406-535), Decoder (537-631) |
| `graph_cad/models/losses.py` | Reconstruction loss (180-250) |

### Relevant Tests

| Test | Location |
|------|----------|
| `test_deterministic_extraction` | `tests/unit/test_graph_extraction.py:188` |
| `test_deterministic_extraction_variable` | `tests/unit/test_graph_extraction.py:421` |

---

## 11. Conclusion

The MLP decoder architecture, while simple and effective for fixed topology, is fundamentally incompatible with variable topology CAD editing. The root cause is the mismatch between fixed-slot outputs and inconsistent graph node orderings.

Phase 3 will implement a DETR-style transformer decoder that is permutation-invariant by design, enabling robust parameter encoding for variable topology L-brackets and beyond.

---

*Report generated: January 2026*
*Project: Graph_CAD*
*Phase: 2 (Variable Topology with MLP Decoder)*
