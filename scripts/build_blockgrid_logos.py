#!/usr/bin/env python3
"""
Generate 7 distinct, precision BlockGrid logos matching the dimensions, typography,
layout, and studio line-art aesthetic of dioptra_logo.jpg (1376x768).

Layout:
  [Left: Mathematical / Architectural Emblem] | BlockGrid
                                                Stratum
                                                Research
                                                Group

Concepts:
  01 - Planar Virtual Grid & Factorized Tensor Tile (Sub-quadratic grid transport)
  02 - 4D Hypercube Tesseract Projection (Orthogonal multi-dimensional tensor manifold)
  03 - Multi-Head Cross-Mesh Routing Loom (Dynamic token transport across virtual cells)
  04 - Concentric Toroidal Context Flow (Continuous cyclic O(N) context scaling)
  05 - Hyper-Dimensional Simplex Tensor Lattice (Barycentric coordinate frame & 22D hypercube)
  06 - Cellular Matrix-in-Matrix Micro-Engine (Two-sided factorized intra-cell operator)
  07 - Hexagonal Systolic Tensor Array (Massive parallel grid computing & Triton kernel acceleration)
"""

import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions (exact match to dioptra_logo.jpg)
WIDTH = 1376
HEIGHT = 768
SCALE = 2  # 2x supersampling for ultra-crisp anti-aliasing
W_S = WIDTH * SCALE
H_S = HEIGHT * SCALE

# Typography & Layout anchors (in 1x coordinates, matching dioptra / metis / imf standard)
DIVIDER_X = 631
DIVIDER_TOP = 185
DIVIDER_BOT = 582
DIVIDER_WIDTH = 10

TEXT_X = 688
SUB_X = 690

EMBLEM_CX = 390
EMBLEM_CY = 384

FONT_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)


def _get_font(size: int):
    if FONT_PATH and os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


def create_studio_background(w, h, seed=42):
    """Generate subtle studio paper off-white background with micro-grain noise."""
    rng = np.random.default_rng(seed)
    y_grad = np.linspace(231.8, 230.5, h)[:, None]
    x_grad = np.linspace(231.2, 230.8, w)[None, :]
    base_luma = (y_grad + x_grad) / 2.0

    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = base_luma
    bg[:, :, 1] = base_luma
    bg[:, :, 2] = base_luma - 1.2  # subtle warm paper undertone

    noise = rng.normal(0.0, 1.8, size=(h, w, 3))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(bg)


def render_base_template(draw, scale=SCALE):
    """Render the vertical divider bar and standard text."""
    # Divider line
    div_x = DIVIDER_X * scale
    div_w = DIVIDER_WIDTH * scale
    y0 = DIVIDER_TOP * scale
    y1 = DIVIDER_BOT * scale
    draw.rounded_rectangle([div_x - div_w // 2, y0, div_x + div_w // 2, y1], radius=scale, fill=(0, 0, 0, 255))

    # Font sizing calibrated to fit BlockGrid perfectly into the frame
    font_title = _get_font(int(128 * scale))
    font_sub = _get_font(int(70 * scale))

    # Title & Subtitle coordinates
    title_y = 205 * scale
    sub_y1 = 368 * scale
    sub_y2 = 444 * scale
    sub_y3 = 520 * scale

    # Text rendering
    draw.text((int(TEXT_X * scale), int(title_y)), "BlockGrid", font=font_title, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(sub_y1)), "Stratum", font=font_sub, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(sub_y2)), "Research", font=font_sub, fill=(0, 0, 0, 255))
    draw.text((int(SUB_X * scale), int(sub_y3)), "Group", font=font_sub, fill=(0, 0, 0, 255))


# ---------------------------------------------------------------------------
# EMBLEM GENERATORS (Render at 2x scale directly onto transparent overlay)
# ---------------------------------------------------------------------------

def draw_concept_01_planar_grid(overlay, cx, cy, s):
    """Concept 01: Planar Virtual Grid & Factorized Tensor Tile."""
    draw = ImageDraw.Draw(overlay)
    span = 140 * s
    step = span / 2.0  # 4x4 cells -> 5 grid lines

    # Grid lines
    for i in range(-2, 3):
        pos = i * step
        w = int(8 * s) if i == 0 else int(5 * s)
        draw.line([cx - span, cy + pos, cx + span, cy + pos], fill=(0, 0, 0, 255), width=w)
        draw.line([cx + pos, cy - span, cx + pos, cy + span], fill=(0, 0, 0, 255), width=w)

    # Active tensor blocks (highlighted sub-matrices)
    tile_half = (step - 8 * s) / 2.0
    for (gx, gy) in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
        bx = cx + (gx * 0.5) * step
        by = cy + (gy * 0.5) * step
        draw.rounded_rectangle([bx - tile_half, by - tile_half, bx + tile_half, by + tile_half], radius=int(3 * s), outline=(0, 0, 0, 255), width=int(5 * s))

    # Center factorized nucleus (outer ring + solid core)
    r_core = 36 * s
    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(8 * s))
    draw.ellipse([cx - 14 * s, cy - 14 * s, cx + 14 * s, cy + 14 * s], fill=(0, 0, 0, 255))

    # Grid intersection nodes
    for i in [-2, 0, 2]:
        for j in [-2, 0, 2]:
            if i == 0 and j == 0:
                continue
            px, py = cx + i * step, cy + j * step
            draw.ellipse([px - 8 * s, py - 8 * s, px + 8 * s, py + 8 * s], fill=(0, 0, 0, 255))


def draw_concept_02_hypercube_tesseract(overlay, cx, cy, s):
    """Concept 02: 4D Hypercube Tesseract Projection (Orthogonal Tensor Manifold)."""
    draw = ImageDraw.Draw(overlay)
    r_out = 142 * s
    r_in = 68 * s

    # 8 Outer cube vertices (isometric 2D projection)
    outer_v = []
    inner_v = []
    for i in range(8):
        angle = i * (math.pi / 4.0) + (math.pi / 8.0)
        outer_v.append((cx + r_out * math.cos(angle), cy + r_out * math.sin(angle)))
        inner_v.append((cx + r_in * math.cos(angle), cy + r_in * math.sin(angle)))

    # Outer octagon boundary
    for i in range(8):
        next_i = (i + 1) % 8
        draw.line([outer_v[i], outer_v[next_i]], fill=(0, 0, 0, 255), width=int(7 * s))

    # Inner octagon boundary
    for i in range(8):
        next_i = (i + 1) % 8
        draw.line([inner_v[i], inner_v[next_i]], fill=(0, 0, 0, 255), width=int(6 * s))

    # Hypercube diagonal projection rays connecting outer to inner
    for i in range(8):
        draw.line([outer_v[i], inner_v[i]], fill=(0, 0, 0, 255), width=int(5 * s))

    # Cross-dimensional tensor diagonals
    for i in [0, 2, 4, 6]:
        opp = (i + 4) % 8
        draw.line([inner_v[i], inner_v[opp]], fill=(0, 0, 0, 255), width=int(4 * s))

    # Outer nodes
    for px, py in outer_v:
        draw.ellipse([px - 9 * s, py - 9 * s, px + 9 * s, py + 9 * s], fill=(0, 0, 0, 255))

    # Inner nodes (ring nodes)
    for px, py in inner_v:
        draw.ellipse([px - 8 * s, py - 8 * s, px + 8 * s, py + 8 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(4 * s))

    # Central nucleus
    draw.ellipse([cx - 15 * s, cy - 15 * s, cx + 15 * s, cy + 15 * s], fill=(0, 0, 0, 255))


def draw_concept_03_mesh_routing_loom(overlay, cx, cy, s):
    """Concept 03: Multi-Head Cross-Mesh Routing Loom."""
    draw = ImageDraw.Draw(overlay)
    r = 148 * s

    # Boundary circular chassis
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 0, 0, 255), width=int(9 * s))
    draw.ellipse([cx - r + 16 * s, cy - r + 16 * s, cx + r - 16 * s, cy + r - 16 * s], outline=(0, 0, 0, 255), width=int(4 * s))

    # 12 Perimeter pins
    pins = []
    for i in range(12):
        angle = i * (math.pi / 6.0)
        px = cx + (r - 8 * s) * math.cos(angle)
        py = cy + (r - 8 * s) * math.sin(angle)
        pins.append((px, py))

    # Cross-head chord routing (braided chord chords)
    # Head 1 (step 5)
    for i in range(12):
        target = (i + 5) % 12
        draw.line([pins[i], pins[target]], fill=(0, 0, 0, 255), width=int(3 * s))

    # Head 2 (step 4 - prominent triangle harmonics)
    for i in range(0, 12, 4):
        draw.line([pins[i], pins[(i + 4) % 12]], fill=(0, 0, 0, 255), width=int(7 * s))
        draw.line([pins[i + 2], pins[(i + 6) % 12]], fill=(0, 0, 0, 255), width=int(7 * s))

    # Central routing hub
    r_hub = 42 * s
    draw.ellipse([cx - r_hub, cy - r_hub, cx + r_hub, cy + r_hub], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(7 * s))
    draw.ellipse([cx - 16 * s, cy - 16 * s, cx + 16 * s, cy + 16 * s], fill=(0, 0, 0, 255))

    # Perimeter pin marks
    for px, py in pins:
        draw.ellipse([px - 6 * s, py - 6 * s, px + 6 * s, py + 6 * s], fill=(0, 0, 0, 255))


def draw_concept_04_toroidal_context_flow(overlay, cx, cy, s):
    """Concept 04: Concentric Toroidal Context Flow (Continuous Cyclic O(N) Context)."""
    draw = ImageDraw.Draw(overlay)
    r_outer = 146 * s
    r_mid = 104 * s
    r_inner = 62 * s
    r_core = 24 * s

    # Concentric circulation rings
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], outline=(0, 0, 0, 255), width=int(9 * s))
    draw.ellipse([cx - r_mid, cy - r_mid, cx + r_mid, cy + r_mid], outline=(0, 0, 0, 255), width=int(6 * s))
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], outline=(0, 0, 0, 255), width=int(6 * s))
    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], outline=(0, 0, 0, 255), fill=(0, 0, 0, 255), width=int(4 * s))

    # Toroidal spiral streamlines (orthogonal cross-axis loops)
    for rot in [0, 45, 90, 135]:
        rad = math.radians(rot)
        dx = math.cos(rad)
        dy = math.sin(rad)
        draw.line([cx - r_outer * dx, cy - r_outer * dy, cx + r_outer * dx, cy + r_outer * dy], fill=(0, 0, 0, 255), width=int(4 * s))

    # Tangent flow indicators (circular beads at quantized phase positions)
    for i in range(8):
        angle = i * (math.pi / 4.0) + (math.pi / 8.0)
        bx = cx + r_mid * math.cos(angle)
        by = cy + r_mid * math.sin(angle)
        draw.ellipse([bx - 10 * s, by - 10 * s, bx + 10 * s, by + 10 * s], fill=(0, 0, 0, 255))
        draw.ellipse([bx - 4 * s, by - 4 * s, bx + 4 * s, by + 4 * s], fill=(232, 232, 230, 255))


def draw_concept_05_simplex_tensor_lattice(overlay, cx, cy, s):
    """Concept 05: Hyper-Dimensional Simplex Tensor Lattice."""
    draw = ImageDraw.Draw(overlay)
    r_hex = 144 * s

    # 6 Hexagonal outer vertices + 6 inner vertices + 1 center
    nodes_outer = []
    nodes_inner = []
    for i in range(6):
        a = i * (math.pi / 3.0) - (math.pi / 6.0)
        nodes_outer.append((cx + r_hex * math.cos(a), cy + r_hex * math.sin(a)))
        nodes_inner.append((cx + (r_hex * 0.52) * math.cos(a), cy + (r_hex * 0.52) * math.sin(a)))

    # Outer boundary
    for i in range(6):
        draw.line([nodes_outer[i], nodes_outer[(i + 1) % 6]], fill=(0, 0, 0, 255), width=int(8 * s))

    # Inner boundary
    for i in range(6):
        draw.line([nodes_inner[i], nodes_inner[(i + 1) % 6]], fill=(0, 0, 0, 255), width=int(6 * s))

    # Radials from center through inner to outer
    for i in range(6):
        draw.line([(cx, cy), nodes_outer[i]], fill=(0, 0, 0, 255), width=int(5 * s))

    # Cross simplex diagonals (interlaced Star of David in inner lattice)
    for i in range(3):
        draw.line([nodes_inner[i * 2], nodes_inner[((i + 1) * 2) % 6]], fill=(0, 0, 0, 255), width=int(5 * s))
        draw.line([nodes_inner[i * 2 + 1], nodes_inner[((i + 1) * 2 + 1) % 6]], fill=(0, 0, 0, 255), width=int(5 * s))

    # Outer nodes
    for px, py in nodes_outer:
        draw.ellipse([px - 10 * s, py - 10 * s, px + 10 * s, py + 10 * s], fill=(0, 0, 0, 255))

    # Inner nodes
    for px, py in nodes_inner:
        draw.ellipse([px - 7 * s, py - 7 * s, px + 7 * s, py + 7 * s], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(4 * s))

    # Center hub
    draw.ellipse([cx - 18 * s, cy - 18 * s, cx + 18 * s, cy + 18 * s], fill=(0, 0, 0, 255))
    draw.ellipse([cx - 7 * s, cy - 7 * s, cx + 7 * s, cy + 7 * s], fill=(232, 232, 230, 255))


def draw_concept_06_cellular_matrix_in_matrix(overlay, cx, cy, s):
    """Concept 06: Cellular Matrix-in-Matrix Micro-Engine (Two-Sided Intra-Cell Operators)."""
    draw = ImageDraw.Draw(overlay)
    w_box = 138 * s

    # Outer bounding matrix bracket
    draw.rounded_rectangle([cx - w_box, cy - w_box, cx + w_box, cy + w_box], radius=int(14 * s), outline=(0, 0, 0, 255), width=int(9 * s))

    # 4 Quadrant cells (2x2 nested sub-matrices)
    q_span = 54 * s
    q_gap = 16 * s
    quads = [
        (-1, -1), (1, -1),
        (-1, 1), (1, 1)
    ]
    for qx, qy in quads:
        center_x = cx + qx * (q_span / 2.0 + q_gap)
        center_y = cy + qy * (q_span / 2.0 + q_gap)
        # Quadrant frame
        draw.rounded_rectangle([center_x - q_span / 2.0, center_y - q_span / 2.0, center_x + q_span / 2.0, center_y + q_span / 2.0],
                               radius=int(6 * s), outline=(0, 0, 0, 255), width=int(5 * s))
        # Micro 2x2 dots inside each cell
        dot_span = 14 * s
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                px = center_x + dx * dot_span
                py = center_y + dy * dot_span
                draw.ellipse([px - 4 * s, py - 4 * s, px + 4 * s, py + 4 * s], fill=(0, 0, 0, 255))

    # Central crossing channels (horizontal & vertical transport buses)
    draw.line([cx - w_box, cy, cx + w_box, cy], fill=(0, 0, 0, 255), width=int(5 * s))
    draw.line([cx, cy - w_box, cx, cy + w_box], fill=(0, 0, 0, 255), width=int(5 * s))

    # Central operator aperture
    r_core = 26 * s
    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(6 * s))
    draw.ellipse([cx - 10 * s, cy - 10 * s, cx + 10 * s, cy + 10 * s], fill=(0, 0, 0, 255))


def draw_concept_07_systolic_tensor_array(overlay, cx, cy, s):
    """Concept 07: Hexagonal Systolic Tensor Array (Massive Parallel Grid Computing)."""
    draw = ImageDraw.Draw(overlay)
    r_ring = 145 * s
    draw.ellipse([cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring], outline=(0, 0, 0, 255), width=int(8 * s))

    # Isometric 3-axis systolic grid lines (horizontal, +60 deg, -60 deg)
    step = 42 * s
    for k in [-2, -1, 0, 1, 2]:
        # Horizontal lines clipped within circle
        h = k * step
        if abs(h) < r_ring - 10 * s:
            w_len = math.sqrt(r_ring**2 - h**2) - 8 * s
            draw.line([cx - w_len, cy + h, cx + w_len, cy + h], fill=(0, 0, 0, 255), width=int(5 * s))

    # Diagonal grid lines at 60 and 120 degrees
    for angle_deg in [60, 120]:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        for k in [-2, -1, 0, 1, 2]:
            offset = k * step
            # Perpendicular vector is (-sin, cos)
            ox = cx - offset * sin_a
            oy = cy + offset * cos_a
            # Ray along (cos, sin)
            len_ray = math.sqrt(max(0, r_ring**2 - offset**2)) - 8 * s
            if len_ray > 10 * s:
                x1 = ox - len_ray * cos_a
                y1 = oy - len_ray * sin_a
                x2 = ox + len_ray * cos_a
                y2 = oy + len_ray * sin_a
                draw.line([x1, y1, x2, y2], fill=(0, 0, 0, 255), width=int(5 * s))

    # Core systolic array processing unit (central diamond hex)
    r_core = 36 * s
    draw.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], outline=(0, 0, 0, 255), fill=(232, 232, 230, 255), width=int(7 * s))
    draw.ellipse([cx - 14 * s, cy - 14 * s, cx + 14 * s, cy + 14 * s], fill=(0, 0, 0, 255))


CONCEPTS = [
    {
        "id": "blockgrid_logo_01_planar_grid",
        "title": "Planar Virtual Grid & Factorized Tensor Tile",
        "desc": "Sub-quadratic virtual coordinate grid with 2x2 factorized tensor tiles and central crossing nucleus",
        "draw_fn": draw_concept_01_planar_grid,
    },
    {
        "id": "blockgrid_logo_02_hypercube_tesseract",
        "title": "4D Hypercube Tesseract Projection",
        "desc": "Orthogonal multi-dimensional tensor manifold projecting hypercube coordinates to 2D grid space",
        "draw_fn": draw_concept_02_hypercube_tesseract,
    },
    {
        "id": "blockgrid_logo_03_mesh_routing_loom",
        "title": "Multi-Head Cross-Mesh Routing Loom",
        "desc": "Chassis circle with multi-head chord routing chords and central token routing hub",
        "draw_fn": draw_concept_03_mesh_routing_loom,
    },
    {
        "id": "blockgrid_logo_04_toroidal_context_flow",
        "title": "Concentric Toroidal Context Flow",
        "desc": "Continuous cyclic O(N) context scaling with concentric rings, toroidal streamlines, and phase indicators",
        "draw_fn": draw_concept_04_toroidal_context_flow,
    },
    {
        "id": "blockgrid_logo_05_simplex_tensor_lattice",
        "title": "Hyper-Dimensional Simplex Tensor Lattice",
        "desc": "Barycentric coordinate frame and Star of David tensor simplex routing between grid vertices",
        "draw_fn": draw_concept_05_simplex_tensor_lattice,
    },
    {
        "id": "blockgrid_logo_06_cellular_matrix_in_matrix",
        "title": "Cellular Matrix-in-Matrix Micro-Engine",
        "desc": "Nested two-sided factorized intra-cell operator with quadrant sub-matrices and transport buses",
        "draw_fn": draw_concept_06_cellular_matrix_in_matrix,
    },
    {
        "id": "blockgrid_logo_07_systolic_tensor_array",
        "title": "Hexagonal Systolic Tensor Array",
        "desc": "Isometric 3-axis systolic grid array for massive parallel grid computing and Triton kernel speedups",
        "draw_fn": draw_concept_07_systolic_tensor_array,
    },
]


def main():
    repo_dir = Path("/home/akarta/Projects/0.WiP/block-grid-vga")
    assets_dir = repo_dir / "assets" / "logos"
    assets_dir.mkdir(parents=True, exist_ok=True)
    branding_dir = Path("/home/akarta/Projects/z.branding")

    print(f"[*] Building 7 precision BlockGrid logos at {WIDTH}x{HEIGHT} (2x supersampled to {W_S}x{H_S})...")

    # Base studio texture at 1x
    base_bg = create_studio_background(WIDTH, HEIGHT, seed=201)

    for i, c in enumerate(CONCEPTS):
        print(f"[{i+1}/7] Rendering {c['id']}: {c['title']}...")

        # Create 2x transparent overlay
        overlay = Image.new("RGBA", (W_S, H_S), (0, 0, 0, 0))

        # 1. Render Left Emblem
        c["draw_fn"](overlay, EMBLEM_CX * SCALE, EMBLEM_CY * SCALE, SCALE)

        # 2. Render layout template (Divider + Typography) on top
        draw_overlay = ImageDraw.Draw(overlay)
        render_base_template(draw_overlay, scale=SCALE)

        # 3. Downsample overlay with Lanczos anti-aliasing
        overlay_1x = overlay.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

        # 4. Composite onto studio paper background
        composite = base_bg.copy()
        composite.paste(overlay_1x, (0, 0), overlay_1x)

        # 5. Save outputs: high-quality JPEG (matching dioptra_logo.jpg) and lossless PNG
        jpg_path = assets_dir / f"{c['id']}.jpg"
        png_path = assets_dir / f"{c['id']}.png"
        composite.save(jpg_path, quality=98, subsampling=0)
        composite.save(png_path)

        # Copy concept 01 as default blockgrid_logo.jpg/png in assets
        if i == 0:
            default_jpg = repo_dir / "assets" / "blockgrid_logo.jpg"
            default_png = repo_dir / "assets" / "blockgrid_logo.png"
            composite.save(default_jpg, quality=98, subsampling=0)
            composite.save(default_png)

        # Centralized branding copy
        if branding_dir.is_dir():
            import shutil
            shutil.copy(jpg_path, branding_dir / jpg_path.name)
            shutil.copy(png_path, branding_dir / png_path.name)
            if i == 0:
                shutil.copy(jpg_path, branding_dir / "blockgrid_logo.jpg")
                shutil.copy(png_path, branding_dir / "blockgrid_logo.png")

        print(f"      -> Saved {jpg_path} (and {png_path})")

    print("[+] All 7 BlockGrid logos rendered successfully!")


if __name__ == "__main__":
    main()
