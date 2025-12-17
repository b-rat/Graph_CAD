# Graph VAE Architecture

This document provides a detailed reference for the Graph Variational Autoencoder implementation in `graph_cad/models/graph_vae.py`.

## Overview

The Graph VAE encodes L-bracket face-adjacency graphs into a compact latent space and reconstructs graph features. It's designed for fixed-topology graphs (always 10 faces, 22 edges).

```
Input STEP → Graph Extraction → VAE Encode → Latent (16D) → VAE Decode → Graph Features
```

## Components

### 1. Configuration (`GraphVAEConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `node_features` | 8 | Per-node features: face_type, area, dir_xyz (3), centroid_xyz (3) |
| `edge_features` | 2 | Per-edge features: edge_length, dihedral_angle |
| `num_nodes` | 10 | Fixed L-bracket topology (8 bracket faces + 2 hole faces) |
| `num_edges` | 22 | Fixed adjacency relationships |
| `hidden_dim` | 64 | Hidden dimension for encoder layers |
| `num_gat_layers` | 3 | Number of Graph Attention layers |
| `num_heads` | 4 | Attention heads per GAT layer |
| `latent_dim` | 16 | Latent space dimensionality (BKM: 16D) |
| `decoder_hidden_dims` | (256, 256, 128) | MLP decoder layer sizes |
| `use_param_head` | False | Enable auxiliary parameter prediction |

**Total input features**: 10 nodes × 8 features + 22 edges × 2 features = **124 features**

---

### 2. Encoder (`GraphVAEEncoder`)

**Purpose**: Transform a graph into latent distribution parameters (μ, log σ²).

#### Architecture Diagram

```
Node features (10×8) ─→ Linear(8→64) ─→ ReLU ─→ Dropout(0.1) ─┐
                                                               │
Edge features (22×2) ─→ Linear(2→64) ─→ ReLU ─────────────────┤
                                                               ↓
                                                 ┌─────────────────────────┐
                                                 │   3× GAT Layers         │
                                                 │   (64D, 4 heads each)   │
                                                 │   with edge attributes  │
                                                 └─────────────────────────┘
                                                               ↓
                                                 Global Mean Pooling
                                                 (10 nodes → 1 vector)
                                                               ↓
                                                 ┌─────────┴─────────┐
                                                 ↓                   ↓
                                           mu_head(64→16)    logvar_head(64→16)
                                                 ↓                   ↓
                                                 μ                log(σ²)
```

#### Key Implementation Details

**Node Embedding** (lines 65-69):
```python
self.node_encoder = nn.Sequential(
    nn.Linear(config.node_features, config.hidden_dim),  # 8 → 64
    nn.ReLU(),
    nn.Dropout(config.encoder_dropout),
)
```

**Edge Embedding** (lines 72-75):
```python
self.edge_encoder = nn.Sequential(
    nn.Linear(config.edge_features, config.hidden_dim),  # 2 → 64
    nn.ReLU(),
)
```

**GAT Layers** (lines 78-89):
```python
GATConv(
    in_channels=config.hidden_dim,           # 64
    out_channels=config.hidden_dim // config.num_heads,  # 64/4 = 16 per head
    heads=config.num_heads,                  # 4 heads
    dropout=config.encoder_dropout,          # 0.1
    edge_dim=config.hidden_dim,              # 64 (edge features in attention)
    concat=True,                             # Output: 4×16 = 64
)
```

**Why GAT?** Graph Attention Networks learn to weight neighbor contributions dynamically. For CAD graphs, this helps the model learn which adjacent faces are most informative for understanding the overall shape.

**Global Mean Pooling** (lines 124-129):
```python
if batch is None:
    h = h.mean(dim=0, keepdim=True)  # Single graph
else:
    h = global_mean_pool(h, batch)   # Batched graphs
```

This aggregates all node embeddings into a single graph-level representation. Mean pooling is permutation-invariant—the order of nodes doesn't affect the output.

---

### 3. Decoder (`GraphVAEDecoder`)

**Purpose**: Reconstruct graph features from a latent vector.

#### Architecture Diagram

```
z (16D) ─→ Linear(16→256) ─→ ReLU ─→ Dropout(0.1)
        ─→ Linear(256→256) ─→ ReLU ─→ Dropout(0.1)
        ─→ Linear(256→128) ─→ ReLU ─→ Dropout(0.1)
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
            node_head(128→80)                           edge_head(128→44)
                    ↓                                           ↓
            reshape(10×8)                               reshape(22×2)
                    ↓                                           ↓
            Node features                               Edge features
```

#### Why MLP Instead of GNN?

Since topology is **fixed** for L-brackets:
- Always 10 nodes (faces)
- Always 22 edges (adjacencies)
- Same connectivity pattern

There's no graph structure to predict—only 124 continuous feature values. An MLP is simpler and equally effective for this task.

#### Output Reshaping (lines 200-206):
```python
node_features = node_flat.view(
    batch_size, self.config.num_nodes, self.config.node_features
)  # (B, 10, 8)

edge_features = edge_flat.view(
    batch_size, self.config.num_edges, self.config.edge_features
)  # (B, 22, 2)
```

---

### 4. Full VAE (`GraphVAE`)

Combines encoder and decoder with the reparameterization trick.

#### Reparameterization Trick (lines 239-260)

```python
def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if self.training:
        std = torch.exp(0.5 * logvar)  # σ = exp(log(σ²)/2)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + eps * std           # z = μ + σ·ε
    else:
        return mu  # Deterministic in eval mode
```

**Why this matters**: Sampling from N(μ, σ²) isn't differentiable. The reparameterization trick moves the randomness to ε, making z a deterministic function of μ, σ, and ε. This allows gradients to flow through the sampling operation.

#### Auxiliary Parameter Head (lines 230-237)

```python
if self.config.use_param_head:
    self.param_head = nn.Sequential(
        nn.Linear(self.config.latent_dim, 64),  # 16 → 64
        nn.ReLU(),
        nn.Linear(64, self.config.num_params),   # 64 → 8
    )
```

**Purpose**: Forces the latent space to encode all 8 L-bracket parameters.

**Problem it solves**: Without this, the VAE exhibited severe latent space collapse:
- Only ~3 of 16 dimensions were active
- Thickness and hole diameters had near-zero correlation with z
- leg1/leg2 were highly entangled (correlation -0.80)

**Results with aux-VAE**:

| Metric | Original VAE | aux-VAE |
|--------|--------------|---------|
| Effective dimensions | ~3 | ~8 |
| Thickness correlation | 0.033 | 0.787 |
| Hole1 correlation | 0.048 | 0.673 |
| leg1/leg2 entanglement | -0.80 | -0.46 |

#### Forward Pass (lines 276-316)

```python
def forward(self, x, edge_index, edge_attr, batch=None):
    # 1. Encode to distribution parameters
    mu, logvar = self.encode(x, edge_index, edge_attr, batch)

    # 2. Sample latent vector
    z = self.reparameterize(mu, logvar)

    # 3. Decode to graph features
    node_recon, edge_recon = self.decode(z)

    # 4. (Optional) Predict parameters directly from z
    if self.param_head is not None:
        result["param_pred"] = self.param_head(z)

    return result
```

---

## Utility Methods

### Sampling from Prior

```python
def sample(self, num_samples: int, device: str = "cpu"):
    z = torch.randn(num_samples, self.config.latent_dim, device=device)
    return self.decode(z)
```

Generates new graphs by sampling z ~ N(0, I) and decoding. Useful for exploring what shapes the latent space has learned.

### Latent Interpolation

```python
def interpolate(self, z1, z2, num_steps=10):
    alphas = torch.linspace(0, 1, num_steps, device=z1.device)
    z_interp = [(1 - alpha) * z1 + alpha * z2 for alpha in alphas]
    return self.decode(torch.stack(z_interp))
```

Linear interpolation between two latent vectors. Useful for:
- Visualizing smooth transitions between bracket shapes
- Verifying the latent space is well-structured (no "holes")

---

## Training Configuration (BKM)

Based on ablation studies documented in CLAUDE.md:

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `latent_dim` | 16 | 8D theoretically sufficient, 16D gives smoother interpolation |
| `target_beta` | 0.01 | Balances reconstruction vs. KL divergence; 0.1 causes variance collapse |
| `free_bits` | 2.0 | Prevents posterior collapse by ensuring minimum KL per dimension |
| `aux_param_weight` | 0.1 | Weight for auxiliary parameter loss |

### Training Command

```bash
python scripts/train_vae.py \
    --epochs 100 \
    --latent-dim 16 \
    --target-beta 0.01 \
    --free-bits 2.0 \
    --aux-param-weight 0.1 \
    --output-dir outputs/vae_aux
```

---

## Loss Function

The VAE is trained with a composite loss:

```
L = L_recon + β · L_KL + λ · L_aux
```

Where:
- **L_recon**: MSE between original and reconstructed node/edge features
- **L_KL**: KL divergence from prior N(0, I), with free-bits constraint
- **L_aux**: MSE between predicted and true L-bracket parameters (if enabled)

### Free-Bits KL (prevents posterior collapse)

```python
# Standard KL: -0.5 * (1 + logvar - mu² - exp(logvar))
# Free-bits: max(KL_per_dim, free_bits) then sum
```

This ensures each latent dimension contributes at least `free_bits` nats to the KL term, preventing the model from ignoring dimensions.

---

## Performance Metrics

### Reconstruction Quality

| Metric | Target | Achieved (aux-VAE) |
|--------|--------|-------------------|
| Node MSE | < 0.01 | 0.00073 |
| Edge MSE | < 0.01 | ~0.001 |

### Latent Space Quality

| Metric | Original VAE | aux-VAE |
|--------|--------------|---------|
| Effective dimensions | ~3/16 | ~8/16 |
| Parameter correlations | 0.03-0.05 (thickness, holes) | 0.67-0.80 |

### Information Bottleneck

**Critical finding**: The VAE introduces an irreducible information bottleneck.

| Model | Parameter MAE |
|-------|---------------|
| GNN on original graphs | ~5mm |
| Any predictor on VAE-reconstructed graphs | ~12mm |

This ~12mm MAE is the **theoretical limit** for downstream parameter prediction. It's information lost during encode/decode, not a limitation of the FeatureRegressor.

---

## Checkpoint Location

**Current best**: `outputs/vae_aux/best_model.pt`

This checkpoint uses:
- 16D latent space
- β = 0.01
- Auxiliary parameter prediction enabled
- Full utilization of latent dimensions

---

## Extension Notes

For extending beyond L-brackets to variable topology:

1. **Encoder**: Keep GNN-based (handles variable node counts)
2. **Pooling**: Use attention-based pooling instead of mean pooling
3. **Decoder**: Replace MLP with graph generative model (e.g., autoregressive node/edge prediction)
4. **Latent dim**: May need to increase for more complex shapes

The current fixed-topology design is a deliberate simplification for the PoC.
