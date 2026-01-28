# Phase 3 Report: DETR Transformer Decoder

## Executive Summary

Phase 3 replaced the MLP decoder with a permutation-invariant DETR-style transformer decoder, breaking through the 64% ceiling from Phase 2. Combined with direct latent supervision, this achieved near-perfect reconstruction and enabled editing of all 4 core parameters.

| Metric | Phase 2 (MLP) | Phase 3 Final | Target |
|--------|---------------|---------------|--------|
| Direction accuracy | 64% | **100%** | >80% |
| leg1/leg2 correlation | r < 0.3 | r = 0.999 | r > 0.7 |
| width correlation | r < 0.3 | r = 0.998 | r > 0.7 |
| thickness correlation | r < 0.3 | r = 0.355 | r > 0.7 |
| Face type accuracy | ~85% | 100% | >95% |
| Edge prediction accuracy | N/A | 100% | >90% |

**Key Achievements:**
- Width editing: Previously impossible, now 78% magnitude accuracy
- Direction accuracy: 100% for all parameters
- Reconstruction: Perfect face type and edge prediction

**Remaining Limitation:** Thickness magnitude poor (2.4%) despite correct direction

---

## 1. Problem Statement

### 1.1 Why MLP Decoder Failed (Phase 2)

The MLP decoder's fixed-slot output is incompatible with variable topology:

1. **OCC ordering is inconsistent** — `TopExp_Explorer` order depends on B-Rep internals
2. **Boolean operations scramble order** — Adding holes/fillets reconstructs B-Rep
3. **Same slot = different faces** — Slot 5 might be hole1, fillet, planar, or padding

| Component | Graph-Aware | Permutation Property |
|-----------|-------------|---------------------|
| **Encoder (GAT)** | Yes | Invariant (global pooling) |
| **Decoder (MLP)** | No | NOT invariant (fixed slots) |

### 1.2 Phase 3 Objective

Replace MLP decoder with permutation-invariant transformer to break the 64% ceiling while preserving the working GAT encoder.

---

## 2. Transformer Decoder Architecture

### 2.1 DETR-Style Design

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

### 2.2 Key Components

1. **Learned Node Queries**: 20 interchangeable vectors (like DETR object queries)
2. **Cross-Attention**: Queries attend to projected latent z
3. **Self-Attention**: 4-layer transformer decoder with 8 heads
4. **Hungarian Matching**: `scipy.optimize.linear_sum_assignment` for optimal GT assignment
5. **Edge Prediction**: Pairwise MLP on concatenated node embeddings (binary, symmetric)
6. **Output Heads**: Separate MLPs for node features, face types, existence, edges

### 2.3 Hungarian Matching Cost Weights

| Component | Weight |
|-----------|--------|
| Face type (classification) | 2.0× |
| Node features (L2 distance) | 1.0× |
| Existence probability | 1.0× |
| Edge loss | 1.0× |

---

## 3. Width/Thickness Encoding Problem

### 3.1 Root Cause: Symmetric Cancellation in Mean Pooling

The encoder uses **global mean pooling** after GAT layers. This loses width/thickness information due to the L-bracket's geometry.

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

### 3.2 Attempted Solutions

**Approach 1: Auxiliary Parameter Head (param_head)**

```
z (latent) → param_head (MLP) → predicted [leg1, leg2, width, thickness]
                                        ↓
                                   Loss vs ground truth
```

**Loss Function Experiments:**

| Approach | aux_weight | width r | thickness r | Issue |
|----------|------------|---------|-------------|-------|
| MSE (raw) | 1.0 | 0.252 | 0.107 | Leg MSE dominates (~300x larger) |
| MSE (normalized) | 1.0 | 0.074 | 0.107 | Loss too small vs reconstruction |
| MSE (normalized) | 100.0 | 0.057 | 0.107 | Still dominated by reconstruction |
| Correlation | 1.0 | - | - | Collapse (negative loss dominates) |
| Correlation | 0.1 | 0.155 | 0.102 | Best so far, but still insufficient |

**Why Aux Head Approach Fails:**
- Reconstruction gradients overwhelm param_head gradients
- Encoder learns what decoder needs (leg lengths), ignores aux head signal
- param_head can only predict what's already in latent space

**Approach 2: Multi-Head Attention Pooling (Phase 3b)**

**Hypothesis:** Replace mean pooling with learned attention pooling to break symmetric cancellation.

**Result:** Did NOT improve width/thickness encoding.

| Configuration | width r | thickness r |
|---------------|---------|-------------|
| Baseline (3 GAT, Mean) | 0.155 | 0.102 |
| Attention (3 GAT, 4 heads) | 0.068 | 0.104 |
| Scaled (6 GAT, 128D, 8 heads) | 0.056 | 0.093 |

**Why it failed:** Attention pooling changes *how* faces are weighted but doesn't fundamentally solve the information bottleneck. The symmetric cancellation happens in the normalized face features, not the pooling weights.

---

## 4. Solution: Direct Latent Supervision

### 4.1 Approach

Force first 4 latent dimensions to directly encode normalized parameters, and exclude these dimensions from KL divergence.

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

### 4.2 Why KL Exclusion is Critical

- Without exclusion: KL pushes mu[:, :4] toward N(0,1), conflicting with direct supervision
- With exclusion: First 4 dims are free to encode parameters at any value
- Remaining 28 dims still follow N(0,1) for sampling

### 4.3 Results

| Parameter | Baseline | Direct Latent | Status |
|-----------|----------|---------------|--------|
| leg1 | r = 0.992 | r = 0.999 | Excellent |
| leg2 | r = 0.989 | r = 0.999 | Excellent |
| width | r = 0.155 | r = 0.998 | **FIXED** |
| thickness | r = 0.102 | r = 0.355 | Improved |

**Latent Space Mapping:**
- `mu[0]` = leg1_length (normalized to [-2, 2])
- `mu[1]` = leg2_length
- `mu[2]` = width
- `mu[3]` = thickness

### 4.4 Thickness Limitation

Still moderate correlation (r=0.355). Possible causes:
- Thickness has smallest range (3-12mm vs 50-200mm for legs)
- May need stronger supervision or separate treatment

---

## 5. Latent Editor Results

### 5.1 Configuration

| Component | Checkpoint | Description |
|-----------|------------|-------------|
| VAE | `outputs_phase3/vae_direct_kl_exclude_v2/best_model.pt` | Direct latent supervision |
| Latent Editor | `outputs_phase3/latent_editor_all_params/best_model.pt` | Mistral 7B + LoRA |

Training data: 15,000 samples across 6 edit types (leg1, leg2, both_legs, width, thickness, noop)

### 5.2 Exploration Study Results

Results from `outputs_phase3/exploration/exploration_all_params.json` (20 brackets, 800 trials):

**Direction Accuracy:**

| Parameter | Accuracy | Notes |
|-----------|----------|-------|
| leg1_length | **100%** | Perfect |
| leg2_length | **100%** | Perfect |
| both_legs | **100%** | Perfect |
| width | **100%** | Previously impossible |
| thickness | **100%** | Correct direction |
| noop | **100%** | Fixed (was 17.5%) |

**Magnitude Accuracy:**

| Parameter | Achievement | Notes |
|-----------|-------------|-------|
| leg1 | 73.2% | Good |
| leg2 | 89.4% | Excellent |
| both_legs | 81.3% | Good |
| width | **78.4%** | Major breakthrough |
| thickness | 2.4% | Poor (VAE encoding limitation) |

### 5.3 Critical: Instruction Format

**The latent editor learned to rely on explicit `+/-` signs in instructions.**

| Instruction | Result |
|-------------|--------|
| `make leg1 +20mm longer` | Correct direction |
| `make leg1 20mm longer` | WRONG direction (sign flipped) |

**Always use `+` for positive changes and `-` for negative changes.**

This is a training data artifact — the model overfit to `+/-` tokens rather than learning "longer/shorter" semantics.

### 5.4 Known Limitations

1. **Thickness magnitude poor**: VAE encodes thickness (r=0.355), editor achieves only 2.4% target
2. **Instruction format sensitivity**: Must use explicit `+/-` signs
3. **Entanglement**: Editing one parameter may cause small spurious changes to others
4. **Large edits undershoot**: Requests >30mm achieve only 75-80% of target

---

## 6. Implementation Details

### 6.1 Files

| File | Description |
|------|-------------|
| `graph_cad/models/transformer_decoder.py` | TransformerGraphDecoder, TransformerGraphVAE |
| `graph_cad/models/graph_vae.py` | GAT encoder + MultiHeadAttentionPooling |
| `graph_cad/models/losses.py` | Hungarian matching loss functions |
| `scripts/train_transformer_vae.py` | Training script |
| `tests/test_transformer_decoder.py` | 24 unit tests |

### 6.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 32 |
| Encoder GAT layers | 3 |
| Encoder hidden dim | 64 |
| Pooling type | mean |
| Decoder hidden dim | 256 |
| Decoder layers | 4 |
| Decoder attention heads | 8 |
| Max nodes | 20 |
| Learning rate | 1e-4 |
| Beta (KL weight) | 0.1 (with 30% warmup) |
| Aux weight | 1.0 (for direct latent) |

### 6.3 Training Commands

```bash
# RECOMMENDED: Direct latent supervision
python scripts/train_transformer_vae.py --epochs 100 \
    --aux-weight 1.0 --aux-loss-type direct \
    --output-dir outputs_phase3/vae_direct_kl_exclude_v2

# Test parameter correlations
python scripts/test_tvae.py outputs_phase3/vae_direct_kl_exclude_v2/best_model.pt
```

### 6.4 Inference Commands

```bash
# Full inference with latent editor
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 +20mm longer" \
    --seed 123

# Width editing
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make the bracket +10mm wider"

# VAE-only mode (no LLM)
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "test" \
    --vae-only
```

---

## 7. Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `outputs_phase3/vae_direct_kl_exclude_v2/best_model.pt` | **RECOMMENDED** VAE with direct latent supervision |
| `outputs_phase3/latent_editor_all_params/best_model.pt` | **RECOMMENDED** Latent editor for all 4 params |
| `outputs_phase3/vae_transformer_aux2_w100/best_model.pt` | Old VAE with aux head (leg1/leg2 only) |
| `outputs_phase3/latent_editor_tvae/best_model.pt` | Old latent editor (leg length only) |
| `outputs_phase3/vae_transformer/best_model.pt` | Phase 3 VAE (no aux head) |

**Best pipeline:** `vae_direct_kl_exclude_v2` + `latent_editor_all_params` with `--direct-latent` flag.

---

## 8. Conclusions

Phase 3 successfully solved the permutation-invariance problem and enabled width editing through direct latent supervision. The remaining challenge is thickness magnitude, which requires either:
- Expanded thickness range in training data
- Stronger supervision signal
- Alternative encoding approach for thin dimensions

**Phase 3 Status: Complete**
