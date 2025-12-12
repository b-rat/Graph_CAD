# Latent Space Analysis Findings

**Date**: December 2024
**Analysis**: Systematic exploration of instruction domain and VAE latent space structure

## Executive Summary

The LLM latent editor achieves only ~53% correct direction (near random chance) because **the VAE latent space has collapsed to ~3 effective dimensions**, losing critical parameter information. Thickness and hole diameters are not encoded at all, and leg1/leg2 are anti-correlated in the latent space, making independent control impossible.

---

## 1. Instruction Domain Exploration

### Methodology

Used `scripts/explore_instruction_domain.py` to systematically test:
- **Parameters**: leg1_length, leg2_length, width, thickness, hole1_diameter, hole2_diameter
- **Directions**: increase, decrease
- **Magnitudes**: Parameter-appropriate (10-50mm for legs, 5-15mm for width, 1-3mm for thickness/holes)
- **Brackets**: 50 random brackets, stratified across small/medium/large regions

### Results Summary (2000 trials)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Overall correct direction | 52.8% | Near random (50%) |
| Mean target achieved | 0.4% | Essentially zero effect |
| Mean inference time | 110ms | Fast once loaded |

### By Parameter Performance

| Parameter | Correct % | Target % | Inc % | Dec % |
|-----------|-----------|----------|-------|-------|
| leg1_length | 49.5 | 0.5 | 36.5 | 62.5 |
| leg2_length | 50.2 | 0.1 | 35.0 | 65.5 |
| width | **67.0** | 1.9 | 88.0 | 46.0 |
| thickness | 50.7 | -0.0 | 47.3 | 54.0 |
| hole1_diameter | 50.7 | -0.0 | 46.7 | 54.7 |
| hole2_diameter | 50.7 | -0.0 | 61.3 | 40.0 |

### Key Observations

1. **Width is the only parameter with above-chance performance** (67% correct)
2. **Thickness and hole diameters show zero effect** (max change ~0.02mm)
3. **Legs have strong decrease bias**: Both increase and decrease instructions yield negative changes
4. **Strong parameter coupling**: Editing leg1 affects leg2 inversely

### Actual Change Magnitudes

| Parameter | Mean Actual Δ | Requested Δ | Max |Δ| |
|-----------|---------------|-------------|---------|
| leg1_length | -2.46mm | ±27.5mm | 25.0mm |
| leg2_length | -5.16mm | ±27.5mm | 25.7mm |
| width | +0.28mm | ±10.0mm | 2.7mm |
| thickness | -0.00mm | ±2.0mm | 0.02mm |
| hole1_diameter | +0.00mm | ±2.0mm | 0.01mm |
| hole2_diameter | +0.00mm | ±2.0mm | 0.00mm |

### Asymmetry Analysis

The model has learned a **default bias**, not instruction semantics:

| Parameter | Increase Instruction | Decrease Instruction | Actual Mean Δ |
|-----------|---------------------|---------------------|---------------|
| leg1_length | Request +27.5mm | Request -27.5mm | **-2.5mm always** |
| leg2_length | Request +27.5mm | Request -27.5mm | **-5.2mm always** |

The ~50% "correct direction" comes from random alignment with decrease instructions, not learned behavior.

---

## 2. VAE Latent Space Structure Analysis

### Effective Dimensionality

| Principal Component | Variance Explained | Cumulative |
|--------------------|-------------------|------------|
| PC_0 | **72.8%** | 72.8% |
| PC_1 | 19.6% | 92.3% |
| PC_2 | 7.5% | 99.9% |
| PC_3 through PC_15 | ~0% | 100% |

**Critical Finding**: The 16D latent space has collapsed to effectively **2-3 dimensions**. This is a severe information bottleneck that cannot represent 8 independent parameters.

### Parameter Encoding in Latent Space

#### Well-Encoded Parameters

| Parameter | Best Latent Dim | Correlation |
|-----------|-----------------|-------------|
| leg1_length | z_7 | r = +0.812 |
| leg2_length | z_0 | r = +0.785 |
| width | z_8 | r = +0.579 |
| hole1_distance | z_6 | r = -0.508 |
| hole2_distance | z_0 | r = +0.435 |

#### NOT Encoded (Lost Information)

| Parameter | Best Correlation | Status |
|-----------|-----------------|--------|
| thickness | r = 0.033 | **Not encoded** |
| hole1_diameter | r = -0.074 | **Not encoded** |
| hole2_diameter | r = -0.038 | **Not encoded** |

These parameters have essentially **zero representation** in the latent space. The VAE discards them during compression.

### Edit Direction Analysis

Computed mean latent direction vectors for parameter edits:

#### Cosine Similarity Between Edit Directions

| Comparison | Cosine Similarity | Interpretation |
|------------|-------------------|----------------|
| leg1 vs leg2 | **-0.802** | Strongly anti-correlated |
| leg1 vs width | -0.403 | Moderately anti-correlated |
| leg2 vs width | -0.216 | Weakly anti-correlated |
| leg1 vs thickness | -0.413 | Moderately anti-correlated |

**Critical Finding**: leg1 and leg2 edit directions are **almost opposite** in latent space (cosine = -0.80). Making leg1 longer pushes the latent vector in nearly the opposite direction of making leg2 longer.

### Latent-Parameter Correlation Matrix (Key Dimensions)

```
Dim     leg1_len  leg2_len    width   thickness
------------------------------------------------
z_0       -0.43    +0.79*    -0.26      -0.02
z_2       +0.66*   -0.69*    -0.08      -0.01
z_4       -0.79*   +0.50*    +0.24      +0.02
z_6       -0.81*   +0.36     +0.42      +0.02
z_7       +0.81*   -0.05     -0.53*     -0.03
z_8       -0.80*   +0.05     +0.58*     +0.03

* = |r| > 0.5 (strong correlation)
```

Note: Multiple latent dimensions encode **mixtures** of leg1 and leg2 with opposite signs.

---

## 3. Root Cause Diagram

```
                  8 L-Bracket Parameters
                          │
                          ▼
              ┌─────────────────────────┐
              │      VAE Encoder        │
              │  (Graph → 16D Latent)   │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   INFORMATION LOSS      │
              │                         │
              │  16D nominal latent     │
              │  collapses to ~3D       │
              │  effective space        │
              └─────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
      ┌───────┐      ┌───────┐      ┌───────────┐
      │ leg1  │      │ leg2  │      │   width   │
      │ (z_7) │◄────►│ (z_0) │      │   (z_8)   │
      └───────┘      └───────┘      └───────────┘
          │    ANTI-     │
          │  CORRELATED  │
          │   r=-0.80    │
          │              │
          └──────┬───────┘
                 │
      ┌──────────┴──────────┐
      │    NOT ENCODED      │
      │  ┌────────────────┐ │
      │  │ thickness      │ │
      │  │ hole1_diameter │ │
      │  │ hole2_diameter │ │
      │  └────────────────┘ │
      └─────────────────────┘
```

---

## 4. Why the LLM Editor Fails

### The LLM Cannot Edit What Isn't Encoded

1. **Thickness**: r = 0.033 correlation → No latent direction exists
2. **Hole diameters**: r < 0.08 → No latent direction exists
3. **Result**: Any delta the LLM produces has ~zero effect on these parameters

### Leg1/Leg2 Interference

1. The LLM learns that "make leg1 longer" requires moving in direction D
2. Direction D is **opposite** to the "make leg2 longer" direction
3. When the LLM tries to edit leg1, it inadvertently affects leg2
4. The entanglement makes independent control impossible

### Default Bias Explanation

The LLM has learned to output a **generic delta** that:
- Makes legs shorter (the dominant learned pattern)
- Slightly increases width
- Has no effect on thickness/holes

This explains:
- ~50% correct direction (random alignment)
- Strong decrease bias for legs
- Width being the "best" parameter (67% correct)

---

## 5. Comparison with Previous Findings

| Metric | Dec 2024 (Pre-Analysis) | Dec 2024 (Post-Analysis) |
|--------|------------------------|--------------------------|
| leg1 correct direction | 5/5 (100%) | 49.5% (large sample) |
| leg2 correct direction | 3/6 (50%) | 50.2% (large sample) |
| Root cause | Unknown | **Latent space collapse** |

The small initial sample (5-6 brackets) showed high variance. The systematic 2000-trial study reveals the true ~50% baseline.

---

## 6. Recommendations

### Short-Term (Diagnostic)

1. **Verify VAE reconstruction loss per parameter**
   - Check if VAE can reconstruct thickness/hole diameters at all
   - Expect high reconstruction error for unencoded parameters

2. **Visualize latent space**
   - 2D/3D PCA plots colored by each parameter
   - Will show leg1/leg2 anti-correlation visually

### Medium-Term (Architecture)

1. **Increase KL weight (β)**
   - Current β=0.01 may allow posterior collapse
   - Try β=0.1 or β=1.0 to force latent utilization

2. **Add supervised signal to VAE**
   - Joint training with parameter prediction head
   - Force latent to encode all 8 parameters

3. **Disentanglement methods**
   - β-VAE with annealing
   - FactorVAE or β-TCVAE for better disentanglement

### Long-Term (Redesign)

1. **Richer graph features**
   - Current features may not distinguish thickness/holes
   - Add explicit geometric features per face

2. **Hierarchical latent space**
   - Separate latents for global shape vs local features
   - May improve parameter independence

3. **Direct parameter conditioning**
   - Bypass VAE for parameter prediction
   - Use graph features directly for editing

---

## 7. Files and Data

### Exploration Results
- `outputs/exploration/results.json` - 10-bracket study (400 trials)
- `outputs/exploration/full_study.json` - 50-bracket study (2000 trials)

### Scripts
- `scripts/explore_instruction_domain.py` - Systematic exploration tool

### Model Checkpoints Analyzed
- `outputs/vae_16d_lowbeta/best_model.pt` - VAE (β=0.01, 16D)
- `outputs/latent_editor/best_model.pt` - LLM Latent Editor
- `outputs/feature_regressor/best_model.pt` - Feature Regressor

---

## Appendix: Raw Correlation Data

### Full Latent-Parameter Correlation Matrix

```
Dim     leg1_len  leg2_len     width  thickness  hole1_dist  hole1_diam  hole2_dist  hole2_diam
------------------------------------------------------------------------------------------------
z_0       -0.426    +0.785    -0.264     -0.015      -0.249      +0.055      +0.435      -0.019
z_1       +0.006    +0.530    -0.555     -0.033      +0.050      +0.007      +0.308      -0.038
z_2       +0.660    -0.690    -0.078     -0.005      +0.415      -0.074      -0.374      +0.002
z_3       -0.228    +0.613    -0.020     +0.000      -0.188      +0.058      +0.302      +0.022
z_4       -0.794    +0.503    +0.244     +0.015      -0.485      +0.067      +0.282      -0.005
z_5       -0.382    -0.295    +0.251     +0.007      -0.184      -0.009      -0.122      -0.027
z_6       -0.807    +0.364    +0.415     +0.023      -0.508      +0.072      +0.199      +0.006
z_7       +0.812    -0.053    -0.534     -0.032      +0.491      -0.046      -0.044      +0.002
z_8       -0.803    +0.045    +0.579     +0.033      -0.496      +0.049      +0.034      +0.006
z_9       -0.680    +0.673    +0.080     +0.005      -0.422      +0.073      +0.365      -0.004
z_10      -0.593    +0.749    -0.057     -0.007      -0.366      +0.067      +0.408      -0.008
z_11      +0.374    -0.765    +0.318     +0.017      +0.211      -0.047      -0.428      +0.022
z_12      +0.790    -0.504    -0.284     -0.018      +0.492      -0.070      -0.277      -0.002
z_13      +0.714    -0.636    -0.150     -0.009      +0.447      -0.071      -0.347      +0.000
z_14      -0.243    -0.642    +0.478     +0.021      -0.119      -0.027      -0.325      -0.007
z_15      +0.598    -0.750    +0.021     -0.001      +0.377      -0.070      -0.407      +0.004
```

### Edit Direction Vectors (Mean Δz for +30mm leg / +10mm width / +3mm thickness)

```
Dim      leg1 Δ     leg2 Δ    width Δ    thick Δ
------------------------------------------------
z_0     -0.1464    +0.2453    -0.1175    -0.0066
z_1     -0.0089    +0.1528    -0.1761    -0.0082
z_2     +0.2153    -0.2089    -0.0067    +0.0005
z_3     -0.0512    +0.0916    -0.0179    -0.0043
z_4     -0.2664    +0.1668    +0.0749    +0.0065
z_5     -0.0581    -0.0290    +0.0561    +0.0062
z_6     -0.2572    +0.1177    +0.1371    +0.0086
z_7     +0.2143    -0.0268    -0.1703    -0.0127
z_8     -0.2278    +0.0202    +0.1937    +0.0132
z_9     -0.2419    +0.2284    +0.0107    -0.0001
z_10    -0.1715    +0.1977    -0.0416    -0.0031
z_11    +0.1079    -0.1997    +0.1126    +0.0063
z_12    +0.2439    -0.1494    -0.0814    -0.0059
z_13    +0.1866    -0.1536    -0.0256    -0.0010
z_14    -0.0429    -0.1347    +0.1468    +0.0119
z_15    +0.2079    -0.2441    +0.0349    +0.0030
```
