# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph_CAD combines Graph Autoencoders with Generative AI for Computer-Aided Design (CAD). The project aims to leverage graph-based representations of CAD models for machine learning tasks.

## Project Structure

```
graph_cad/          # Main package
├── models/         # Graph autoencoder and generative model definitions
├── data/           # Data loaders, preprocessors, CAD parsers
├── utils/          # Graph operations, visualization, helpers
└── training/       # Training loops, loss functions, optimizers

tests/              # Test suite
├── unit/           # Unit tests for individual components
└── integration/    # End-to-end workflow tests

data/               # Data directory (gitignored)
├── raw/            # Original CAD files
├── processed/      # Preprocessed graph representations
└── models/         # Saved model checkpoints

notebooks/          # Jupyter notebooks for experimentation
scripts/            # CLI scripts for training, evaluation, preprocessing
docs/               # Documentation and architecture diagrams
```

## Development Commands

**Setup environment:**
```bash
# Install dependencies
pip install -e .                    # Install package in development mode
pip install -r requirements-dev.txt # Install development tools

# Or install everything at once
pip install -e ".[dev]"
```

**Testing:**
```bash
pytest                              # Run all tests
pytest tests/unit                   # Run unit tests only
pytest tests/integration            # Run integration tests only
pytest -v -k "test_specific"        # Run specific test
pytest --cov=graph_cad              # Run with coverage report
```

**Code quality:**
```bash
black graph_cad tests               # Format code
ruff check graph_cad tests          # Lint code
ruff check --fix graph_cad tests    # Auto-fix linting issues
mypy graph_cad                      # Type checking
```

**Run from scripts directory:**
```bash
# Example commands (create scripts as needed)
python scripts/train.py --config config.yaml
python scripts/evaluate.py --checkpoint path/to/model.pt
python scripts/preprocess_cad.py --input data/raw --output data/processed
```

## Architecture Overview & Design Decisions

### Core System Design

**Two-Phase Approach:**
1. **Phase 1 (Current Focus)**: Train graph autoencoder on CAD B-Rep data
   - Encoder: STEP file → Graph representation → Compressed latent vector
   - Decoder: Latent vector → Graph → STEP file
   - Goal: Perfect reconstruction, smooth latent space

2. **Phase 2 (Future)**: Integrate pre-trained LLM for geometric reasoning
   - Freeze trained autoencoder
   - LLM operates in latent space (reasoning about geometry)
   - Use case: "add mounting holes", "make it symmetric", etc.

**Key Insight**: The autoencoder acts as a "geometric codec" that compresses complex STEP files into a latent representation that LLMs can reason over, similar to how vision encoders work for image-language models.

### Proof of Concept (PoC) Constraints

**Starting small with focused scope:**

**1. Fixed Topology, Variable Geometry**
- Single part family: L-brackets with 2 mounting holes
- Constant topology: Always 10 faces (8 bracket body + 2 cylindrical holes)
- Only dimensions vary: 8 parameters (leg lengths, width, thickness, hole diameters, hole distances)
- Range constraints per parameter (e.g., leg_length: [50, 200] mm)

**2. Simplified Graph Representation**
- Face-adjacency graph (nodes = faces, edges = face adjacency)
- Node features: [face_type, area, normal_vector, centroid]
- Edge features: [edge_length, dihedral_angle]
- Fixed graph structure (10 nodes for L-bracket)

**3. Synthetic Training Data**
- Generate 5k-10k parametric models programmatically
- Use CAD kernel (CadQuery/pythonocc) for generation
- Ground truth parameters available for validation
- Controlled distribution (uniform/gaussian over parameter ranges)

**4. Fixed-Size Latent Vector**
- Latent dimension: z ∈ ℝ³² to ℝ⁶⁴
- Fixed size simplifies architecture (no sequence modeling yet)
- Should capture ~8 geometric parameters + learned features

**5. Success Metrics**
- Geometric accuracy: Parameter reconstruction error < 1%
- Vertex position error < 0.5mm (mean distance)
- Topology preservation: 100% accuracy (same face/edge/vertex count)
- Latent space quality: Interpolation produces valid CAD models
- Training efficiency: Converges in < 50 epochs, < 4 hours on single GPU

### PoC to MVP Expansion Path

**Key expansions needed for practical autoencoder:**

**1. Variable Topology (Critical)**
- Current: Fixed 6-node graphs
- Target: Variable graphs (6-50 nodes)
- Solution: Graph Neural Networks with global pooling
  ```
  Variable graph → GNN layers → Global attention pooling → Fixed latent
  ```
- Requires PyTorch Geometric or DGL
- Decoder options:
  - Simple: Predict max_nodes, mask unused (acceptable for MVP)
  - Advanced: Autoregressive generation (better long-term)

**2. Multiple Part Families**
- Current: L-brackets only
- Target: 5+ part types (brackets, flanges, blocks, shafts, gears)
- Approach: Single universal autoencoder trained on mixed dataset
- Dataset composition: 20% each type, 50k total models
- Latent space should naturally cluster by part type

**3. Real CAD Data**
- Current: Perfect synthetic data
- Target: Real-world CAD files (ABC Dataset, Fusion 360 Gallery)
- Challenges:
  - Messy topology (tiny faces, degenerate edges)
  - No ground truth parameters
  - Scale/orientation variance
- Solutions:
  - Preprocessing pipeline (clean, normalize, validate)
  - Geometry-based reconstruction loss (not parameter-based)
  - Normalize to unit bounding box, align to principal axes

**4. Complex Features**
- Current: Planar faces, cylindrical holes, right angles
- Target: Fillets, chamfers, pockets, patterns, curved surfaces
- Solution: Richer node/edge features (expand from 4D to 16D)
- Node features include: surface_type, curvature, control points
- Edge features include: edge_type (sharp/fillet/chamfer), curvature

**5. Latent Space Quality**
- Semantic organization: Similar parts cluster together
- Smooth interpolation: Blending parts produces valid intermediates
- Disentanglement: Individual dimensions control specific features
- Training objectives:
  ```python
  loss = reconstruction_loss
       + β * kl_divergence(z, N(0,I))      # VAE prior
       + λ * contrastive_loss              # cluster similar
       + γ * cycle_consistency             # smooth paths
  ```

### MVP Success Criteria

**Quantitative:**
- Reconstruction: Chamfer distance < 2mm on 95% of test parts
- Topology: Face count ±2, edge count ±5
- Generalization: Works on 5+ part families
- Latent quality: >90% interpolation validity, silhouette score > 0.6

**Qualitative:**
- Recognizability: Output clearly resembles input (may be simplified)
- Robustness: Handles messy real-world CAD without crashes

**Demo Capability:**
1. Upload diverse parts → encode → decode → recognizable reconstruction
2. Interpolate two parts → smooth morph between them
3. Sample nearby latent → generate variations of same part type

### LLM Integration Options (Phase 2)

**Three approaches for connecting latent space to LLM:**

**Option A: Latent-as-tokens** (Recommended)
- Encoder produces sequence: [z₁, z₂, ..., zₙ]
- LLM processes latent tokens + text prompt
- Variable-length latents match variable-complexity CAD
- Requires training LLM on latent semantics

**Option B: Latent-as-continuous-embeddings**
- Fixed-size latent vector
- Projection layers: latent ↔ token embeddings
- Works with frozen pre-trained LLM + adapter
- Limited by fixed size

**Option C: VQ-VAE style** (Best for pre-trained LLMs)
- Continuous latent → discrete codebook indices
- LLM operates on discrete tokens (natural)
- Codebook learned during autoencoder training
- Minimal LLM fine-tuning needed

### Implementation Timeline

**PoC Phase (Month 1):**
- Week 1: Data generation pipeline, graph extraction
- Week 2: Autoencoder architecture, initial training
- Week 3: Optimization, tune to <1% reconstruction error
- Week 4: Validation, latent space visualization, metrics

**MVP Phase (Months 2-4):**
- Month 2: Variable topology (GNN architecture), 3 part families
- Month 3: Real data integration, preprocessing pipeline
- Month 4: Feature complexity, latent space optimization

**LLM Integration (Months 5-7):**
- Month 5: Basic LLM connection, simple tasks ("make bigger")
- Month 6: Co-evolve autoencoder + LLM capabilities
- Month 7: Complex reasoning tasks, full system demo

**Key Decision**: Don't wait for perfect autoencoder before LLM integration. Start LLM experiments after PoC works, then iterate both together.

### Technical Stack Decisions

**ML Framework**: PyTorch (reasoning: best GNN support, most flexible)

**GNN Library**: PyTorch Geometric (for MVP variable topology)

**CAD Kernel Options**:
- CadQuery (Pythonic, good for parametric generation)
- pythonocc (Full OpenCASCADE bindings, more powerful)
- FreeCAD (Python API, good for complex operations)

**Data Sources**:
- Synthetic: Custom generators for PoC
- Real: ABC Dataset (1M+ models), Fusion 360 Gallery (20k models)

### L-Bracket Generator Decisions

**Geometry & Coordinate System:**
```
        Z
        ↑
        │   ┌─────────┐
        │   │         │
        │   │    ○    │  ← Leg 2 (Y-Z plane, extends +Z)
        │   │  hole2  │
        │   │         │
        ├───┼─────────┘
        │   │thickness
 origin ●━━━┿━━━━━━━━━━━━━━━┓
        │   │               ┃
        │   │      ○        ┃  ← Leg 1 (X-Y plane, extends +X)
        │   │    hole1      ┃
        └───┴───────────────┻────→ X
            └── thickness ──┘

        Y goes into the page (width direction)
```

- Origin at outer corner where leg 1 and leg 2 meet (all dimensions positive)
- Leg 1 on X-Y plane, extends along +X
- Leg 2 on Y-Z plane, extends along +Z
- Width along Y axis

**Parameters (8 total):**

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

**Derived constraints:**
- Holes on centerline (Y = width/2), so no lateral position parameters
- `hole_distance` min: 1 diameter from end
- `hole_distance` max: 1 diameter from corner (`leg_length - thickness - diameter`)
- `width` min: `2 × max(hole_diameters)` to ensure 1 diameter clearance from edges

**B-Rep Topology (OpenCASCADE/CadQuery):**
- **10 faces total**: 8 bracket body + 2 cylindrical hole faces
- Full cylinder = 1 face with 1 seam edge (unlike Creo which uses 2 half-cylinders)
- Through-holes: each adds 1 cylindrical face, modifies planar faces with inner edge loops
- Edges shared between adjacent faces (manifold geometry)

**Implementation Decisions:**
- **CAD Kernel**: CadQuery (Pythonic API, sufficient for PoC)
- **Units**: Millimeters throughout
- **Sampling**: Uniform distribution over parameter ranges
- **Design Pattern**: Class-based with validation on construction
- **Output**: STEP files + metadata CSV with ground truth parameters

**Usage:**
```bash
# Generate 5000 training samples
python scripts/generate_l_brackets.py --output data/raw --count 5000 --seed 42
```

### Graph Extraction Decisions

**Graph Structure:**
- Face-adjacency graph: nodes = faces, edges = shared topological edges
- L-bracket produces 10 nodes (8 planar + 2 cylindrical) and 22 edges
- Seam edges (cylinder self-loops) excluded—only face-to-face adjacency tracked
- COO sparse format for edge indices (PyTorch Geometric compatible)

**Node Features (8D per face):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | face_type | Integer: 0=planar, 1=cylindrical, 2=other |
| 1 | area | Normalized by bbox_diagonal² |
| 2-4 | dir_x/y/z | Planar: surface normal; Cylindrical: axis direction |
| 5-7 | centroid | Normalized position (centered on bbox) |

**Edge Features (2D per edge):**
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | edge_length | Total shared edge length, normalized |
| 1 | dihedral_angle | Angle between face normals (radians, 0 to π) |

**Key Design Decisions:**
- **Cylinder axis vs normal**: Cylindrical faces store axis direction (meaningful) rather than a sampled surface normal (arbitrary point on cylinder has no single normal)
- **Normalization**: All lengths/positions normalized by bounding box diagonal for scale invariance
- **Sufficiency for reconstruction**: With fixed topology, edge lengths + face features act like "origami constraints"—should uniquely determine shape (up to rigid transformation)

**Reconstruction Path (Graph → CAD):**
- For PoC: Graph features are over-determined (80+ values → 8 parameters), so a parameter regressor can recover L-bracket dimensions
- For MVP: More complex—will need direct B-Rep generation from graph or learned implicit reconstruction

**Usage:**
```python
from graph_cad.data import extract_graph, extract_graph_from_solid

# From STEP file
graph = extract_graph("bracket.step")

# From CadQuery solid (skip file I/O)
graph = extract_graph_from_solid(bracket.to_solid())
```

### Parameter Regressor Results

**Purpose:** Validation experiment to confirm that the graph representation contains sufficient information to uniquely determine L-bracket geometry. This is NOT a core component—it answers the question: "Is our feature design adequate?"

**Key Validation:** If Graph → Parameters is learnable with reasonable accuracy, then the graph features (face types, areas, normals/axes, centroids, edge lengths, dihedral angles) capture the essential geometry. This GNN predicts the 8 L-bracket dimensions directly from the face-adjacency graph.

**Architecture:**
- 3× GAT layers (4 attention heads, 64 hidden dim)
- Global mean pooling → MLP → 8 parameters
- ~32K trainable parameters

**Training Configuration:**
- 5000 training samples, 500 val, 500 test
- 100 epochs, batch size 32
- AdamW optimizer, cosine LR schedule
- First epoch: ~16 min (graph generation), subsequent: <1s (cached)

**Test Results (5000 samples, 100 epochs):**

| Parameter | MAE (mm) | Error % |
|-----------|----------|---------|
| leg1_length | 10.83 | 7.2% |
| leg2_length | 10.77 | 7.2% |
| width | 3.55 | 8.9% |
| thickness | 0.68 | 7.5% |
| hole1_distance | 7.40 | 4.2% |
| hole1_diameter | 0.80 | 10.0% |
| hole2_distance | 7.28 | 4.1% |
| hole2_diameter | 0.82 | 10.2% |
| **Overall** | **5.27** | **~7%** |

**Key Findings:**
1. **Graph representation is sufficient** — model successfully learns parameter mapping
2. **Hole distances easiest** (~4% error) — well-captured by face centroids
3. **Hole diameters hardest** (~10% error) — smallest parameter range (4-12mm)
4. **Validation complete** — ~7% error proves concept; further refinement deferred

**Conclusion:** The regressor validated our feature design. For the autoencoder, the primary loss is graph feature reconstruction (MSE on nodes + edges). The regressor becomes an optional evaluation metric to confirm reconstructed graphs encode valid geometry. No further regressor refinement needed before building the autoencoder.

**Usage:**
```bash
python scripts/train_regressor.py --train-size 5000 --epochs 100
```

### PoC Autoencoder Architecture Decision

**Key Insight:** Fixed topology dramatically simplifies the decoder.

For L-brackets, topology is constant (always 10 nodes, 22 edges, same adjacency). The decoder doesn't need to *generate* structure—only predict feature values for a known structure.

**PoC Architecture (Simplified):**
```
Input Graph (10×8D nodes, 22×2D edges, fixed adjacency)
    ↓
Graph Encoder (GAT layers + global pooling)
    ↓
Latent Vector z ∈ ℝ³² to ℝ⁶⁴
    ↓
Feature Decoder (MLP)
    ↓
Predicted Features (10×8D nodes, 22×2D edges)
    ↓
[Same fixed adjacency as input]
```

**Decoder = MLP** outputting 10×8 + 22×2 = 124 values, reshaped into node/edge features. No transformer, no autoregressive generation needed.

**Loss Function:**
```
Loss = λ₁ × MSE(predicted_node_features, true_node_features)
     + λ₂ × MSE(predicted_edge_features, true_edge_features)
     + λ₃ × MSE(regressor(predicted_graph), true_parameters)  [optional]
```

The parameter regressor can serve as:
1. **Evaluation metric**: Validate predicted graphs encode correct geometry
2. **Training signal**: Add semantic loss ensuring latent captures meaningful parameters

**MVP Architecture (Future - Variable Topology):**
When topology varies (6-50 nodes), decoder becomes generative:
- Predict number of nodes
- Autoregressive node generation (transformer decoder)
- Predict edge existence (attention/pointer mechanism)
- This is deferred complexity—PoC proves the concept first

### Key Architectural Notes

**Why Graph Representation?**
- CAD topology is inherently graph-structured (faces, edges, vertices)
- Captures both geometry (node/edge features) and topology (adjacency)
- Variable-size graphs handle different part complexities
- GNNs naturally process this representation

**Why Fixed Topology for PoC?**
- Eliminates variable-size complexity during initial development
- Allows standard neural networks (faster iteration)
- Still proves core concept (geometric reasoning in latent space)
- Variable topology is architectural expansion, not conceptual change

**Critical Success Factor**:
The quality of the latent space determines everything. It must be:
1. Compact enough for LLM to process
2. Expressive enough to capture geometric details
3. Smooth enough for interpolation and reasoning
4. Structured enough for LLM to learn semantics

The graph-aware attention mechanism is designed to ensure these properties.

### Graph VAE Implementation

**Status**: Implemented and tested (40/40 tests passing)

**Architecture Decisions:**
- **Type**: Variational Autoencoder (VAE) — chosen for smooth latent space and interpolation capability
- **Latent dim**: 64D — 8× the 8 L-bracket parameters, good capacity without overfitting
- **Encoder**: Reuses GAT architecture from parameter regressor (3× GAT layers, 4 heads, 64 hidden)
- **Decoder**: MLP with split heads — separate outputs for nodes (80D) and edges (44D)
- **Beta schedule**: Warmup strategy (β=0 for 10 epochs, then linear ramp to target)

**Loss Function:**
```
Loss = λ_node × MSE(node_features)
     + λ_edge × MSE(edge_features)
     + β × KL(z || N(0,I))
     + λ_semantic × MSE(regressor(recon), regressor(orig))  [optional]
```

**Files:**
- `graph_cad/models/graph_vae.py` — GraphVAE, GraphVAEEncoder, GraphVAEDecoder, GraphVAEConfig
- `graph_cad/models/losses.py` — reconstruction_loss, kl_divergence, vae_loss, semantic_loss
- `graph_cad/training/vae_trainer.py` — BetaScheduler, train_epoch, evaluate, compute_latent_metrics
- `scripts/train_vae.py` — CLI training script
- `scripts/evaluate_vae.py` — Evaluation and latent space analysis

**Usage:**
```bash
# Basic training
python scripts/train_vae.py --epochs 100 --latent-dim 64

# With semantic loss (uses pre-trained regressor)
python scripts/train_vae.py --use-semantic-loss

# Evaluate trained model
python scripts/evaluate_vae.py --checkpoint outputs/vae/best_model.pt --all
```

**Success Metrics (targets):**
| Metric | Target |
|--------|--------|
| Node MSE | < 0.01 (normalized) |
| Edge MSE | < 0.01 (normalized) |
| Active latent dims | > 32 of 64 |
| Interpolation validity | > 95% |

### VAE Training Results

**Training Configuration:**
- 5000 training samples, 500 val, 500 test
- 100 epochs, batch size 32
- Device: Apple M3 GPU (MPS backend)
- Free bits: 2.0 (prevents posterior collapse)
- Target beta: 0.1

**Key Finding: Posterior Collapse Prevention**

Initial training with default settings resulted in **posterior collapse** - the VAE ignored the latent space entirely (0/64 active dims). The model learned to predict an "average" graph regardless of input.

**Solution**: Added `--free-bits 2.0` parameter which guarantees minimum information flow through each latent dimension before KL penalty applies.

**Final Results (with free bits + semantic loss):**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Node MSE | 0.000882 | < 0.01 | ✅ 11× better |
| Edge MSE | 0.001246 | < 0.01 | ✅ 8× better |
| Active dims | 62/64 (97%) | > 32 | ✅ Excellent |
| Mean ‖z‖ | 3.99 | > 0 | ✅ Active latent |

**Per-Feature Reconstruction Quality:**
| Feature | MAE | Assessment |
|---------|-----|------------|
| face_type | 0.000075 | Excellent |
| area | 0.020 | Good |
| direction (x/y/z) | ~0.0002 | Excellent |
| centroid (x/y/z) | 0.006-0.022 | Good |

**Latent Space Analysis - Parameter Correlations:**

| Parameter | Best Latent Dim | Correlation | Assessment |
|-----------|-----------------|-------------|------------|
| leg1_length | dim 25 | r=0.74 | ✅ Strong |
| leg2_length | dim 53 | r=0.77 | ✅ Strong |
| hole1_dist | dim 58 | r=0.37 | ⚠️ Moderate |
| hole2_dist | dim 50 | r=0.45 | ⚠️ Moderate |
| width | dim 53 | r=0.10 | ❌ Weak |
| thickness | dim 40 | r=0.10 | ❌ Weak |
| hole1_diam | dim 54 | r=0.06 | ❌ Weak |
| hole2_diam | dim 46 | r=0.04 | ❌ Weak |

**Key Insights:**

1. **Reconstruction is excellent** - All metrics significantly exceed targets
2. **Major geometry well-encoded** - Leg lengths (major features) strongly correlate with individual latent dims
3. **Small parameters entangled** - Width, thickness, hole diameters don't map to single dimensions
4. **Semantic loss marginal benefit** - Improved width correlation slightly (0.06→0.10)
5. **Compression achieved** - 124 graph features → 64 latent dims → reconstructs 8 parameters

**Disentanglement Conclusion:**

The VAE successfully compresses and reconstructs L-bracket geometry, but the latent space is **entangled** (parameters spread across multiple dimensions). True disentanglement would require:
- β-VAE with higher β
- FactorVAE or β-TCVAE
- Explicit supervision on latent structure

For PoC purposes, the current model is sufficient - geometry is captured accurately even if not disentangled. Phase 2 (LLM integration) can reason over the entangled latent space.

**Recommended Training Command:**
```bash
python scripts/train_vae.py --epochs 100 --device mps --free-bits 2.0 --target-beta 0.1
```

### Latent Dimension Ablation Study

**Question:** What is the minimum latent dimension needed for accurate reconstruction?

L-brackets have 8 parameters, so we tested 64D (8×), 32D (4×), and 16D (2×) to find the optimal compression.

**Results:**

| Latent Dim | Node MSE | Edge MSE | Active Dims | Best Epoch |
|------------|----------|----------|-------------|------------|
| **64D** | 0.000882 | 0.001246 | 62/64 (97%) | 81 |
| **32D** | 0.000879 | 0.001229 | 31/32 (97%) | 96 |
| **16D** | 0.000891 | 0.001292 | 16/16 (100%) | 91 |

**Parameter Correlations at 16D:**

| Parameter | Best Dim | Correlation |
|-----------|----------|-------------|
| leg1_length | dim 7 | r=0.74 |
| leg2_length | dim 12 | r=-0.76 |
| hole1_dist | dim 7 | r=0.37 |
| hole2_dist | dim 12 | r=-0.43 |
| width | dim 2 | r=-0.06 |
| thickness | dim 15 | r=-0.11 |

**Key Findings:**

1. **16D is sufficient** — Only ~1% MSE increase vs 64D, still far exceeds targets
2. **No reconstruction degradation** — All configurations achieve Node MSE < 0.001
3. **Efficient compression** — 16D uses 100% of capacity (no wasted dimensions)
4. **2× parameters = optimal** — 16D latent for 8-parameter geometry is ideal
5. **Correlations preserved** — Same major/minor parameter patterns across all sizes

**Recommendation:** Use 16D latent for L-brackets. This achieves:
- 8:1 compression ratio (124 graph features → 16 latent → 8 parameters)
- Near-perfect reconstruction
- Minimal computational cost
- All latent dimensions utilized

**Command for 16D training:**
```bash
python scripts/train_vae.py --epochs 100 --device mps --free-bits 2.0 --target-beta 0.1 --latent-dim 16
```
