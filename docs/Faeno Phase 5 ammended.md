---
parent: "[[Hub - Faeno.ai]]"
---

# Phase 5 — Geometric Comprehension Layer

## Objective

Build a system that enables Claude Code (or a future CAD harness) to reason about B-Rep geometry the way a trained human engineer does — understanding function, design intent, feature relationships, and manufacturing implications — without requiring manual re-explanation of design context each session.

This is not a generative system. Claude Code handles geometry generation natively via CadQuery. This is a **comprehension layer**: a read interface that translates geometry into engineering language.

---

## Target Domain

The PoC focuses on **power electronics mechanical design** rather than attempting broad coverage. This domain is a strong fit:

- **Rich functional variety** within a coherent domain — thermal management, current carrying, insulation, mounting, enclosure
- **Natural manufacturing diversity** — the same functional role is commonly realized in stamped, extruded, cast, and machined variants
- **Deep design intent** — every feature exists for a specific reason (thermal path, creepage distance, current capacity, vibration resistance) which provides high-quality description signal
- **CadQuery-friendly geometry** — mostly prismatic + holes + fins + pockets, manageable script complexity

Four design intent axes define this domain:

- **Thermal:** heat flow path, interface area, fin efficiency, contact pressure
- **Electrical:** current capacity, creepage/clearance, insulation coordination
- **Structural:** vibration resistance, mounting force, clamping load
- **Serviceability:** access for assembly/disassembly, connector mating direction

### Geometric Complexity Target

The PoC targets parts with **8-20 B-Rep faces**. This range is driven by what contrastive training needs to succeed:

| Face Count | Assessment |
|------------|------------|
| 4-8 | Too simple. Manufacturing variants produce nearly identical topology — the contrastive loss has nothing to push apart, so it has nothing to learn. |
| **8-20** | **Sweet spot.** Manufacturing variants produce meaningfully different topology (a sheet metal bracket with bends has different face counts and adjacency than a cast bracket with draft and fillets). The encoder must learn that face count and adjacency pattern aren't the signal — function is. |
| 20-40 | Stretch goal. Ribbed housings, pocketed plates, multiple hole patterns. Richer signal but slower to generate, train, and label. Worth including a few to test generalization. |
| 40+ | Too much for PoC. Noise floor rises — the encoder must sort through too many features to find functional signal, requiring a much larger dataset. |

The contrastive loss needs positive pairs (same function, different manufacturing) to be *hard*. At 8-20 faces, manufacturing variants are different enough that the encoder can't cheat with simple statistics (face count, total area), but similar enough that there is a learnable functional signal underneath.

---

## Architecture Overview

### 1. Graph-Aware Transformer (GAT) Encoder

Carries forward the Phase 4 HeteroGNN encoder. The B-Rep is represented as a heterogeneous graph with three node types:

- **Vertices (3D):** normalized (x, y, z) coordinates
- **Edges (6D):** normalized length, tangent vector, start/end curvature; typed as LINE, ARC, CIRCLE, or OTHER
- **Faces (13D):** area, normal, centroid, principal curvatures, bounding box; typed as PLANAR, HOLE, or FILLET

Topology is encoded as bidirectional message-passing connections: V↔E (vertices bound edges) and E↔F (edges bound faces). The encoder uses `HeteroConv` layers with graph attention to produce a fixed-dimension embedding vector representing the part.

### 2. Cross-Modal Semantic Alignment (CLIP-style)

The embedding space must be organized semantically — proximity should reflect functional similarity, not geometric similarity. A stamped bus bar and a machined bus bar should be neighbors; a stamped bus bar and a stamped EMI shield should not.

To achieve this:

- Generate structured descriptions of training parts (functional role, load paths, feature purposes, manufacturing constraints, design intent)
- Embed those descriptions using a text encoder (frozen language model embedding layer)
- Train the GAT encoder with a contrastive loss (InfoNCE) that pulls geometry embeddings toward their corresponding text embeddings and pushes mismatched pairs apart

This produces an embedding space organized by engineering meaning, not raw shape.

### 3. Interpreter LLM (Language Decoder)

A fine-tuned language model (Mistral 7B for PoC; 70B+ target) trained to:

- Accept a geometry embedding as input (via a projector MLP that maps the embedding to the LLM's hidden dimension)
- Produce structured descriptions of the part in engineering terms
- Handle conversational queries from an agent or human

Example queries the interpreter should handle:

- "What is this part's functional role?"
- "What manufacturing process does this geometry suggest?"
- "What is the critical dimension and why?"
- "What changes if wall thickness increases by 2mm?"
- "What are the thermal implications of this geometry?"

The interpreter is the **language interface** to the embedding space. It is not a general-purpose LLM — it is fine-tuned specifically to read geometric semantic embeddings and speak about them in engineering terms.

---

## Training Data Strategy

### Core Principle

A general geometry dataset (e.g., ABC dataset) trains the encoder to cluster by shape. A domain-specific engineering dataset trains the encoder to cluster by function. Shape-based clustering is not useful for engineering reasoning. Function-based clustering is.

The dataset is built from (geometry, description) pairs where both elements are generated together — the geometry via CadQuery, the description from the design conversation that produced it. This grounds every label in actual design intent, not post-hoc shape inference.

### Description Schema

Each part is described using a structured YAML schema that is both human-readable and efficient as context input for Claude Code:

```yaml
function: right-angle structural bracket
load_path: bending moment at corner, bolt shear at mounting holes
critical_dimensions:
  - leg1_length: sets mounting span (load-bearing)
  - hole_diameter: M6 bolt clearance (functional interface)
non_critical_dimensions:
  - fillet_radius: stress relief, not positional
  - wall_thickness: driven by manufacturing process
manufacturing: sheet_metal | cast | machined
manufacturing_constraints:
  - minimum bend radius 2x thickness (sheet metal)
  - draft angles on interior faces (casting)
features:
  - name: mounting_hole_pattern
    purpose: bolt attachment to mating structure
    faces: [hole.1, hole.2]
  - name: stiffening_rib
    purpose: resist bending under load
    faces: [rib.1, rib.2]
design_intent: low-cost structural connection between orthogonal panels
```

**Key fields and their role in training:**

- **function** — the primary clustering axis. This is what pulls geometrically different parts together when they serve the same purpose (e.g., a cast bracket and a sheet metal bracket).
- **load_path** — how force flows through the part. Two parts with the same load path are functionally equivalent regardless of form.
- **features with purpose** — ties face labels to function, not geometry. A "mounting hole" on a casting and a "pierced hole" on sheet metal serve the same purpose and should be recognized as equivalent.
- **manufacturing** — explicitly labeled so the model learns this is a *variant axis*, not a *similarity axis*. The same function should cluster together across different manufacturing values.
- **critical vs. non-critical dimensions** — tells the interpreter which dimensions matter for function and which are process-driven. This is the information Claude Code needs to reason about design changes.

### Contrastive Training Pairs

The schema enables precise construction of contrastive pairs:

- **Positive pairs:** same function, different manufacturing process (bracket-cast, bracket-sheet-metal). These are pulled together in embedding space.
- **Hard negatives:** same manufacturing process, different function (sheet-metal bracket vs. sheet-metal EMI shield). These are pushed apart.

This is the core training signal for functional clustering. A casting, a machined part, and a sheet metal part that all serve as a right-angle structural bracket should be neighbors in the embedding space despite having very different B-Rep topology, face counts, and geometric features.

### Seed Part Generation: Guided Parametric Construction

The user walks Claude Code through creating each seed part using CadQuery, explaining design intent along the way. This approach is preferred over providing pre-made STEP files for three reasons:

1. **The CadQuery script is the parametric generator.** The script already takes parameters (lengths, thicknesses, hole diameters, etc.) as arguments. Generating variants is a deterministic parameter sweep — no additional scripting or manual intervention.

2. **The design conversation is the description source.** During construction, the user explains *why* — why the wall is that thick, what the holes are for, what manufacturing process the geometry assumes. Claude Code synthesizes this conversation into the YAML description at the end of the session. The description is grounded in actual design decisions, not inferred from shape.

3. **Variant descriptions are deterministic too.** Because the CadQuery script defines which parameters vary and the description schema maps those parameters to functional roles (critical vs. non-critical), variant descriptions can be templated automatically. A variant with `leg1_length=150mm` instead of `100mm` updates the critical_dimensions entry; the rest carries forward unchanged.

**Workflow for each seed:**

```
1. User + Claude Code design a part interactively via CadQuery
   - User provides engineering context during construction
   - Claude Code builds the parametric CadQuery script

2. Claude Code synthesizes the conversation into YAML description
   - Functional role, load paths, feature purposes, manufacturing constraints
   - Maps CadQuery parameters to critical/non-critical dimensions

3. Claude Code defines the parameter sweep space
   - Which parameters vary, over what range, with what step size
   - Constraints (e.g., hole diameter < wall thickness, fillet radius < edge length)

4. Deterministic variant generation
   - Sweep parameters → STEP files
   - Template descriptions → YAML files
   - Result: N (geometry, description) pairs per seed
```

**Manufacturing variants** follow the same workflow but require a separate CadQuery script per manufacturing process. A sheet metal bracket, a cast bracket, and a machined bracket are three different seeds with three different scripts — but they share the same `function`, `load_path`, and `features.purpose` fields in their descriptions. The geometric differences (bend radii vs. draft angles vs. sharp edges) live in `manufacturing_constraints` and in the geometry itself. This is exactly the signal the contrastive loss needs: same function, different form.

### Dataset Scale

The dataset is structured as a generation matrix:

- **Rows:** functional roles (~15-20 within the power electronics domain)
- **Columns:** manufacturing processes (2-3 per role — not every combination makes engineering sense)
- **Cells:** each cell contains a parametric CadQuery script plus a defined parameter space

This gives ~30-50 seed parts, each producing 5-15 parametric variants, for a total of several hundred (geometry, description) pairs.

### Generation Matrix (partial)

| Function | Approx. Faces | Sheet Metal | Cast | Machined |
|----------|---------------|-------------|------|----------|
| Bus bar | 6-10 | stamped + bent | — | milled from plate |
| Heatsink (simple fin) | 12-20 | — | draft + pin fins | milled fins |
| Heatsink (extruded) | 12-20 | — | — | extruded profile |
| DIN rail bracket | 8-14 | formed + snap-fit | — | milled + bolted |
| Terminal block body | 10-18 | — | molded | machined |
| Power module baseplate | 8-16 | — | cast + thermal pad | milled + flatness ground |
| Connector mounting bracket | 8-12 | bent + pierced | — | milled + tapped |
| EMI shield cover | 10-16 | formed + finger stock | cast lid | milled cover |
| Capacitor clamp | 8-12 | formed spring clip | — | milled saddle |
| Inductor bobbin mount | 8-14 | bent tabs | molded base | milled platform |

Cells marked "—" are combinations that don't make engineering sense. The matrix is intentionally sparse — only real-world plausible combinations are included.

---

## Integration with Claude Code / CAD Harness

Intended workflow:

1. A B-Rep part (STEP file) is passed to the GAT encoder
2. The encoder produces a geometry embedding
3. The interpreter LLM is queried (by Claude Code or a human) about the part
4. The interpreter's language output is passed to Claude Code as engineering context
5. Claude Code reasons from that context — generates instructions, modifications, or parametric variants via CadQuery

The interpreter eliminates the manual translation step: no more re-explaining design intent at the start of each Claude Code session.

---

## What This System Is Not

- **Not a generative system.** Claude Code handles geometry generation via CadQuery. This system reads geometry; it does not produce it.
- **Not a replacement for the CAD harness.** The harness handles task execution. This system provides geometric context — the understanding layer beneath the action layer.
- **Not a VAE.** There is no reconstruction loss, no KL regularization, no sampling from a learned distribution. The variational framing from earlier phases has been dropped entirely. This is a GAT encoder + cross-modal alignment + language decoder.

---

## Reuse Mapping from Phases 1-4

Phase 5 replaces the geometric decoder with a language decoder and adds cross-modal alignment. The encoder architecture carries forward. The output target changes from geometry reconstruction to language generation.

### Carries Forward Directly

| Component | Source File | Notes |
|-----------|-------------|-------|
| B-Rep graph extraction | `graph_cad/data/brep_extraction.py` | STEP → heterogeneous V/E/F graph. No changes needed. |
| Node features | Same | Vertex (3D), Edge (6D), Face (13D) — all still relevant. |
| HeteroGNN encoder layers | `graph_cad/models/hetero_vae.py` | `HeteroConv` with bidirectional V↔E, E↔F message passing and attention pooling. |
| `LatentProjector` | `graph_cad/models/latent_editor.py` | MLP mapping embedding → LLM hidden dim. Same role (geometry embedding → Mistral input). |
| Mistral + LoRA/QLoRA infrastructure | `graph_cad/models/latent_editor.py` | Model loading, 4-bit quantization, LoRA config, adapter setup. Mechanical plumbing, fully reusable. |
| B-Rep type constants | `graph_cad/data/brep_types.py` | Edge types, face types, feature dimensions. |

### Carries Forward with Modification

| Component | Source File | What Changes |
|-----------|-------------|--------------|
| VAE wrapper | `graph_cad/models/hetero_vae.py` | Drop `mu`/`logvar` projection, KL loss, reparameterization. Encoder produces a deterministic embedding. Simplification, not rewrite. |
| `OutputProjector` | `graph_cad/models/latent_editor.py` | Phase 4: LLM hidden state → latent delta. Phase 5: not needed — Mistral's built-in language head handles text generation. Gets simpler. |
| Training loop skeleton | `scripts/train_hetero_vae.py`, `scripts/train_llm_instruct.py` | Loss functions change (contrastive for encoder, language generation for interpreter). Loop structure (data loading, optimization, checkpointing, logging) carries forward. |
| Geometry type system | `graph_cad/data/brep_types.py` | Phase 4: 6 parametric types (bracket, tube, etc.). Phase 5: functional roles × manufacturing processes — more types, different taxonomy. Constants expand. |

### New Components

| Component | Purpose |
|-----------|---------|
| Text encoder | Frozen embedding layer for CLIP-style contrastive alignment. Can reuse Mistral's own embedding layer (already in the codebase). |
| Contrastive loss (InfoNCE) | Pulls geometry embeddings toward matched text embeddings, pushes mismatched pairs apart. ~20 lines of code. |
| Description schema parser | YAML loader for structured (geometry, description) pairs. |
| Description templater | Generates variant descriptions from seed descriptions by updating dimension values and manufacturing constraints. |
| Parameter sweep runner | Drives CadQuery scripts across parameter ranges, pairs STEP outputs with templated YAML descriptions. |
| Seed construction harness | Workflow tooling for guided parametric construction sessions (CadQuery script + conversation → description). |

### What Gets Dropped

| Component | Reason |
|-----------|--------|
| DETR-style transformer decoder | Reconstruction objective no longer exists. |
| Hungarian matching loss | No set prediction — output is language, not face sets. |
| Parameter regression heads | Phase 4 predicted geometric parameters. Phase 5 generates language descriptions. |
| Geometry classification head | Functional role is richer than a 6-class label. The interpreter LLM handles this via language. |
| `param_normalization.py` | Per-type min-max normalization for parameter regression. Not needed without regression targets. |
| `geometry_generators.py` | Phase 4 generators (SimpleBracket, Tube, etc.). Replaced by seed CadQuery scripts per the generation matrix. |

---

## Proof of Concept Milestones

1. **Seed part construction:** Guided parametric construction sessions for ~30-50 seed parts across the generation matrix, producing CadQuery scripts + YAML descriptions
2. **Dataset generation:** Deterministic parameter sweeps to produce several hundred (geometry, description) pairs
3. **Encoder training:** GAT encoder with CLIP-style contrastive alignment on (geometry, description) pairs
4. **Interpreter fine-tuning:** Mistral 7B on (embedding, description) pairs
5. **Evaluation:** Can the interpreter correctly describe held-out parts? Can it answer follow-up engineering questions? Does the embedding space cluster by function across manufacturing processes?
6. **Integration test:** Pass a STEP file through the full pipeline; query the interpreter from a Claude Code session; verify the output provides useful engineering context
