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
