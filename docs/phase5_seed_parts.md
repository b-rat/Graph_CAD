# Phase 5 Seed Part Suggestions

Seed parts for the Geometric Comprehension Layer dataset, organized by functional role within the power electronics domain. All parts target **8-20 B-Rep faces**. Manufacturing variants provide contrastive training signal (same function, different form).

---

## Tier 1 — High-Value Seeds

Rich contrastive signal with multiple manufacturing variants.

### Bus Bar (Electrical)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Stamped + bent | 8-10 | Bent tabs, pierced bolt holes |
| Machined | Milled from plate | 10-14 | Tapped holes, chamfered edges |

**Contrastive value:** Same current-carrying function, very different topology. Bend radii vs. sharp edges is a clean signal.

### Right-Angle Mounting Bracket (Structural)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Bent from flat | 8-12 | Pierced holes, optional stiffening bead |
| Cast | Cast + machined | 12-18 | Draft angles, bosses around holes, filleted corner |
| Machined | Milled from block | 10-16 | Sharp internal corner, tapped holes, weight-reduction pocket |

**Contrastive value:** Three manufacturing variants of the same function. Largest contrastive batch in the dataset.

### Simple Fin Heatsink (Thermal)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Cast | Cast + finished | 14-20 | Pin or tapered fins with draft, thermal pad recess |
| Machined | Slot-cut | 12-18 | Straight fins, flat mounting surface, counterbored holes |

**Contrastive value:** Thermal function is dominant. Fin geometry differs significantly between processes but thermal intent (maximize surface area, maintain base contact) is shared.

### EMI Shield Cover (Electrical / Structural)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Formed | 12-16 | Finger stock tabs, ventilation slots |
| Cast | Cast lid | 10-14 | Tongue-and-groove seam, mounting lugs |

**Contrastive value:** Shielding function is identical; form factor diverges sharply.

---

## Tier 2 — Good Contrastive Pairs

Two manufacturing variants each.

### DIN Rail Clip Bracket (Structural / Serviceability)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Formed spring clip | 8-14 | Snap-fit tab retention |
| Machined | Milled body | 10-14 | Bolted DIN rail clamp |

**Contrastive value:** Same mounting interface, completely different retention mechanism.

### Capacitor Clamp (Structural)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Formed spring clip | 8-12 | Saddle profile, bent mounting tabs |
| Machined | Milled saddle | 10-14 | Bolt-down flanges |

**Contrastive value:** Clamping load path is the same; geometry is very different.

### Connector Mounting Bracket (Structural / Serviceability)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Sheet metal | Bent + pierced | 8-12 | Alignment tabs for mating direction |
| Machined | Milled + tapped | 10-14 | Datum surfaces for connector alignment |

**Contrastive value:** Mating direction and access clearance are functional, not geometric.

### Power Module Baseplate (Thermal / Structural)

| Variant | Process | Approx. Faces | Key Features |
|---------|---------|---------------|--------------|
| Cast | Cast + machined | 10-16 | Thermal pad recess, mounting bosses, draft on walls |
| Machined | Ground + bored | 12-18 | Ground flat surface, counterbored holes, O-ring groove |

**Contrastive value:** Flatness and thermal contact are the critical function; manufacturing determines how that's achieved.

---

## Tier 3 — Single-Variant, High Intent Density

One plausible manufacturing process but rich in functional features. Useful as **hard negatives** against each other.

### Terminal Block Body (Electrical / Serviceability)

- **Process:** Machined or molded
- **Approx. faces:** 12-18
- **Key features:** Wire entry channels, clamping screw bosses, creepage ribs
- **Intent density:** Creepage distance, current capacity, and wire gauge all drive geometry.

### Inductor Bobbin Mount (Structural / Electrical)

- **Process:** Sheet metal (bent tabs) or molded base
- **Approx. faces:** 8-14
- **Key features:** Mounting platform, insulation standoffs, vibration-resistant base
- **Intent density:** Insulation coordination + vibration resistance in the same part.

### Extruded Heatsink Profile (Thermal)

- **Process:** Extrusion only
- **Approx. faces:** 14-20
- **Key features:** Fin array with base, mounting slots
- **Intent density:** Contrasts against cast/machined heatsinks — same thermal function, fundamentally different manufacturing constraint (constant cross-section).

---

## Suggested Build Order

Prioritized to maximize early training signal:

| Order | Seed Part | Variants | Est. Sweep Samples | Rationale |
|-------|-----------|----------|---------------------|-----------|
| 1 | Right-angle bracket | 3 | 15-45 | Biggest contrastive batch, familiar geometry |
| 2 | Bus bar | 2 | 10-30 | Simple geometry, strong electrical intent |
| 3 | Simple fin heatsink | 2 | 10-30 | Thermal axis, higher face count |
| 4 | Capacitor clamp | 2 | 10-30 | Structural, very different form factor |
| 5 | EMI shield cover | 2 | 10-30 | Electrical, moderate complexity |

**First batch total:** 11 seeds, ~55-165 variants from parameter sweeps. Enough to validate whether contrastive alignment learns functional clustering before investing in the full matrix.

---

## Design Intent Axes Coverage

| Axis | Primary Seeds | Supporting Seeds |
|------|--------------|-----------------|
| Thermal | Heatsinks (simple fin, extruded), power module baseplate | Bus bar (I²R heating) |
| Electrical | Bus bar, EMI shield, terminal block | Connector bracket (interface) |
| Structural | Bracket, capacitor clamp, DIN rail clip | Bobbin mount, baseplate |
| Serviceability | DIN rail clip, connector bracket, terminal block | Capacitor clamp (replacement access) |
