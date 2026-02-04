#!/usr/bin/env python3
"""
Generate a block with a blind slot using CadQuery, with semantic face labels.

Geometry (in inches, converted to mm for CadQuery):
- Block: 2.0 x 1.0 x 4.0 inches
- Slot width: 0.2 inches (x = 1.20 to 1.40, centered at 1.30)
- Slot depth: 0.625 inches (from top y=1.0 down to y=0.375)
- Blind slot ending 0.25 inches from z0 face (slot runs z=0.25 to z=4.0)
"""

import cadquery as cq
import re
from pathlib import Path

# Conversion factor
INCH_TO_MM = 25.4

# Block dimensions (inches -> mm)
BLOCK_X = 2.0 * INCH_TO_MM  # 50.8 mm
BLOCK_Y = 1.0 * INCH_TO_MM  # 25.4 mm
BLOCK_Z = 4.0 * INCH_TO_MM  # 101.6 mm

# Slot parameters (inches -> mm)
SLOT_WIDTH = 0.2 * INCH_TO_MM  # 5.08 mm
SLOT_CENTER_X = 1.30 * INCH_TO_MM  # 33.02 mm
SLOT_DEPTH = 0.625 * INCH_TO_MM  # 15.875 mm
SLOT_START_Z = 0.25 * INCH_TO_MM  # 6.35 mm - blind end distance from z0
SLOT_END_Z = BLOCK_Z  # open end at z_length face

# Derived slot positions
SLOT_LEFT_X = SLOT_CENTER_X - SLOT_WIDTH / 2  # 30.48 mm (1.2")
SLOT_RIGHT_X = SLOT_CENTER_X + SLOT_WIDTH / 2  # 35.56 mm (1.4")
SLOT_LENGTH = SLOT_END_Z - SLOT_START_Z
SLOT_BOTTOM_Y = BLOCK_Y - SLOT_DEPTH  # 9.525 mm (0.375")

print(f"Block dimensions: {BLOCK_X/INCH_TO_MM:.2f} x {BLOCK_Y/INCH_TO_MM:.2f} x {BLOCK_Z/INCH_TO_MM:.2f} inches")
print(f"  ({BLOCK_X:.1f} x {BLOCK_Y:.1f} x {BLOCK_Z:.1f} mm)")
print(f"Slot: width={SLOT_WIDTH/INCH_TO_MM:.2f}\", depth={SLOT_DEPTH/INCH_TO_MM:.3f}\", length={SLOT_LENGTH/INCH_TO_MM:.2f}\"")
print(f"Slot position: x={SLOT_LEFT_X/INCH_TO_MM:.2f}\" to {SLOT_RIGHT_X/INCH_TO_MM:.2f}\", z={SLOT_START_Z/INCH_TO_MM:.2f}\" to {SLOT_END_Z/INCH_TO_MM:.2f}\"")
print(f"Slot walls at x={SLOT_LEFT_X:.2f}mm and x={SLOT_RIGHT_X:.2f}mm")
print(f"Slot bottom at y={SLOT_BOTTOM_Y:.2f}mm")
print(f"Slot end at z={SLOT_START_Z:.2f}mm")

# Create the block - position so origin is at (0, 0, 0) corner
block = (
    cq.Workplane("XY")
    .box(BLOCK_X, BLOCK_Y, BLOCK_Z, centered=False)
)

# Create the slot as a box to subtract
# Slot is cut from the top, positioned at the correct x and z
slot_box = (
    cq.Workplane("XY")
    .transformed(offset=(SLOT_LEFT_X, SLOT_BOTTOM_Y, SLOT_START_Z))
    .box(SLOT_WIDTH, SLOT_DEPTH, SLOT_LENGTH, centered=False)
)

# Subtract slot from block
result = block.cut(slot_box)

# Export to temporary STEP file
temp_path = Path("block_blind_slot_temp.STEP")
output_path = Path("block_slot_annot_blind.STEP")
cq.exporters.export(result, str(temp_path))
print(f"\nGenerated geometry, now adding semantic labels...")


def get_face_label(plane_point, plane_normal, tol=1.0):
    """
    Assign semantic label based on plane origin point and normal direction.

    Uses position and normal to identify faces:
    - Main block faces at boundaries
    - Slot faces inside the block
    """
    px, py, pz = plane_point
    nx, ny, nz = plane_normal

    # Tolerance for position matching (in mm)
    pos_tol = tol

    # Check for main block faces first (at boundaries)

    # x0: plane at x~0 with normal in x direction
    if abs(px) < pos_tol and abs(nx) > 0.9 and abs(ny) < 0.1 and abs(nz) < 0.1:
        return "x0"

    # x_width: plane at x~BLOCK_X with normal in x direction
    if abs(px - BLOCK_X) < pos_tol and abs(nx) > 0.9:
        return "x_width"

    # z0: plane at z~0 with normal in z direction
    if abs(pz) < pos_tol and abs(nz) > 0.9 and abs(px) < pos_tol:
        return "z0"

    # z_length: plane at z~BLOCK_Z with normal in z direction
    if abs(pz - BLOCK_Z) < pos_tol and abs(nz) > 0.9:
        return "z_length"

    # bottom: plane at y~0 with normal in y direction
    if abs(py) < pos_tol and abs(ny) > 0.9 and abs(px) < pos_tol and abs(pz) < pos_tol:
        return "bottom"

    # top faces: plane at y~BLOCK_Y with normal in y direction
    if abs(py - BLOCK_Y) < pos_tol and abs(ny) > 0.9:
        return "top.planar_1"  # Will distinguish later if needed

    # Slot faces (inside the block, not at boundaries)

    # slot.wall_left: at x~SLOT_LEFT_X with normal in x direction
    if abs(px - SLOT_LEFT_X) < pos_tol and abs(nx) > 0.9:
        return "slot.wall_left"

    # slot.wall_right: at x~SLOT_RIGHT_X with normal in x direction
    if abs(px - SLOT_RIGHT_X) < pos_tol and abs(nx) > 0.9:
        return "slot.wall_right"

    # slot.bottom: at y~SLOT_BOTTOM_Y with normal in y direction
    if abs(py - SLOT_BOTTOM_Y) < pos_tol and abs(ny) > 0.9:
        return "slot.bottom"

    # slot.end: at z~SLOT_START_Z with normal in z direction (the blind end)
    if abs(pz - SLOT_START_Z) < pos_tol and abs(nz) > 0.9:
        return "slot.end"

    return ""  # Unknown face


def add_labels_to_step(input_path, output_path):
    """
    Post-process STEP file to add semantic labels to ADVANCED_FACE entities.
    """
    with open(input_path, 'r') as f:
        content = f.read()

    # Regex patterns
    face_pattern = re.compile(r"(#(\d+)\s*=\s*ADVANCED_FACE\s*\(\s*)''\s*,")
    axis_pattern = re.compile(r"#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\([^,]*,\s*#(\d+)\s*,\s*#(\d+)")
    point_pattern = re.compile(r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\([^,]*,\s*\(\s*([^)]+)\s*\)\s*\)")
    dir_pattern = re.compile(r"#(\d+)\s*=\s*DIRECTION\s*\([^,]*,\s*\(\s*([^)]+)\s*\)\s*\)")
    plane_pattern = re.compile(r"#(\d+)\s*=\s*PLANE\s*\([^,]*,\s*#(\d+)\s*\)")
    adv_face_pattern = re.compile(r"#(\d+)\s*=\s*ADVANCED_FACE\s*\([^,]*,\s*\([^)]*\)\s*,\s*#(\d+)")

    # Build lookup tables
    points = {}
    for match in point_pattern.finditer(content):
        pid = match.group(1)
        coords_str = match.group(2)
        # Handle both 2D and 3D points
        coords = [float(x.strip()) for x in coords_str.split(',')]
        if len(coords) == 3:
            points[pid] = coords

    directions = {}
    for match in dir_pattern.finditer(content):
        did = match.group(1)
        coords_str = match.group(2)
        coords = [float(x.strip()) for x in coords_str.split(',')]
        if len(coords) == 3:
            directions[did] = coords

    axes = {}
    for match in axis_pattern.finditer(content):
        aid = match.group(1)
        point_id = match.group(2)
        dir_id = match.group(3)
        if point_id in points and dir_id in directions:
            axes[aid] = {'point': points[point_id], 'normal': directions[dir_id]}

    planes = {}
    for match in plane_pattern.finditer(content):
        pid = match.group(1)
        axis_id = match.group(2)
        if axis_id in axes:
            planes[pid] = axes[axis_id]

    # Map ADVANCED_FACE to surface (plane)
    face_surfaces = {}
    for match in adv_face_pattern.finditer(content):
        face_id = match.group(1)
        surface_id = match.group(2)
        if surface_id in planes:
            face_surfaces[face_id] = planes[surface_id]

    # Debug: print plane info
    print("\nPlane geometry detected:")
    for face_id, info in sorted(face_surfaces.items(), key=lambda x: int(x[0])):
        pt = info['point']
        nm = info['normal']
        label = get_face_label(pt, nm)
        print(f"  Face #{face_id}: point=({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}), "
              f"normal=({nm[0]:.1f}, {nm[1]:.1f}, {nm[2]:.1f}) -> '{label}'")

    # Replace empty labels with semantic ones
    def replace_face_label(match):
        full_match = match.group(0)
        prefix = match.group(1)
        face_id = match.group(2)

        if face_id in face_surfaces:
            info = face_surfaces[face_id]
            label = get_face_label(info['point'], info['normal'])
            if label:
                return f"{prefix}'{label}',"

        return full_match

    new_content = face_pattern.sub(replace_face_label, content)

    with open(output_path, 'w') as f:
        f.write(new_content)

    # Count labeled faces
    labeled = len(re.findall(r"ADVANCED_FACE\s*\(\s*'[^']+'\s*,", new_content))
    total = len(re.findall(r"ADVANCED_FACE\s*\(", new_content))
    print(f"\nLabeled {labeled}/{total} faces")


# Add semantic labels
add_labels_to_step(temp_path, output_path)

# Clean up temp file
temp_path.unlink()

print(f"\nExported to: {output_path}")

# Show the labels
print("\nFace labels in output file:")
with open(output_path) as f:
    for line in f:
        if 'ADVANCED_FACE' in line:
            print(f"  {line.strip()}")
