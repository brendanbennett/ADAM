"""
ADAM shape model visualiser using PyVista.

Usage:
    python plot_shape.py                        # final shape
    python plot_shape.py --shapefile output/outshape.txt.005
    python plot_shape.py --animate              # loop over intermediate shapes
    python plot_shape.py --animate --delay 0.3
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.exit("pyvista not found — run:  pip install pyvista")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_shape(path):
    """Return (vertices, faces) from an ADAM outshape file.

    Format:
        NVertices NFacets
        x y z          (NVertices lines, km)
        v1 v2 v3       (NFacets  lines, 1-indexed)
    """
    with open(path) as f:
        nv, nf = map(int, f.readline().split())
        verts = np.array([f.readline().split() for _ in range(nv)], dtype=float)
        faces = np.array([f.readline().split() for _ in range(nf)], dtype=int) - 1
    return verts, faces


def read_pole(path):
    """Return (beta_deg, lambda_deg, period_hours) from a param_1 file."""
    try:
        vals = np.loadtxt(path)
        return float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        return None


def shape_to_mesh(verts, faces):
    """Convert numpy arrays to a pyvista PolyData mesh."""
    face_conn = np.hstack([np.full((len(faces), 1), 3, dtype=int), faces])
    return pv.PolyData(verts, face_conn.ravel())


def pole_vector(beta_deg, lambda_deg):
    """Unit vector of the spin pole in ecliptic Cartesian coordinates."""
    b = math.radians(beta_deg)
    l = math.radians(lambda_deg)
    return np.array([math.cos(b) * math.cos(l),
                     math.cos(b) * math.sin(l),
                     math.sin(b)])


def intermediate_shapes(output_dir):
    """Sorted list of outshape.txt.NNN files."""
    pattern = os.path.join(output_dir, "outshape.txt.[0-9][0-9][0-9]")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

WINDOW_SIZE = (900, 750)

def add_pole_arrow(plotter, verts, beta_deg, lambda_deg):
    """Draw the spin-pole axis through the centre of mass."""
    centre = verts.mean(axis=0)
    radius = np.linalg.norm(verts - centre, axis=1).max()
    pv_dir = pole_vector(beta_deg, lambda_deg)
    tip = centre + pv_dir * radius * 1.4
    tail = centre - pv_dir * radius * 1.4
    arrow = pv.Arrow(start=tail, direction=pv_dir, tip_length=0.12,
                     tip_radius=0.04, shaft_radius=0.015,
                     scale=np.linalg.norm(tip - tail))
    plotter.add_mesh(arrow, color="tomato", label=f"Pole  β={beta_deg:.1f}°  λ={lambda_deg:.1f}°")


def scalar_field(mesh):
    """Add surface curvature as a colour scalar (purely cosmetic)."""
    mesh = mesh.compute_normals(cell_normals=False, point_normals=True)
    mesh["Mean_Curvature"] = mesh.curvature("mean")
    return mesh


def plot_single(shapefile, paramfile=None):
    verts, faces = read_shape(shapefile)
    mesh = scalar_field(shape_to_mesh(verts, faces))

    pole = read_pole(paramfile) if paramfile and os.path.exists(paramfile) else None

    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.add_mesh(mesh, scalars="Mean_Curvature", cmap="coolwarm",
                     show_scalar_bar=True, scalar_bar_args={"title": "Mean curvature"},
                     smooth_shading=True)
    plotter.add_mesh(mesh, style="wireframe", color="black",
                     opacity=0.12, line_width=0.4)

    if pole:
        beta, lam, period = pole
        add_pole_arrow(plotter, verts, beta, lam)
        plotter.add_text(
            f"β = {beta:.2f}°   λ = {lam:.2f}°   P = {period:.5f} h",
            position="upper_left", font_size=11, color="white"
        )

    name = os.path.basename(shapefile)
    plotter.add_title(f"ADAM shape — {name}", font_size=13)
    plotter.show_axes()
    plotter.show()


def plot_animate(output_dir, paramfile=None, delay=0.4):
    import time

    files = intermediate_shapes(output_dir)
    if not files:
        sys.exit(f"No outshape.txt.NNN files found in {output_dir}")

    final = os.path.join(output_dir, "outshape.txt")
    if os.path.exists(final):
        files.append(final)

    pole = read_pole(paramfile) if paramfile and os.path.exists(paramfile) else None

    # Build live meshes from the first frame — points are updated in-place each step
    verts, faces = read_shape(files[0])
    mesh = shape_to_mesh(verts, faces)
    mesh["Mean_Curvature"] = mesh.curvature("mean")
    wire_mesh = shape_to_mesh(verts, faces)

    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.add_mesh(mesh, scalars="Mean_Curvature", cmap="coolwarm",
                     show_scalar_bar=True, smooth_shading=True,
                     scalar_bar_args={"title": "Mean curvature"})
    plotter.add_mesh(wire_mesh, style="wireframe", color="black",
                     opacity=0.12, line_width=0.4)

    if pole:
        add_pole_arrow(plotter, verts, pole[0], pole[1])

    plotter.add_title("ADAM optimisation — shape evolution", font_size=13)
    plotter.show_axes()
    label_actor = plotter.add_text("", position="upper_left", font_size=11, color="white")
    plotter.show(auto_close=False, interactive_update=True)

    for path in files:
        iteration = os.path.splitext(path)[1].lstrip(".")
        label = f"Iteration {iteration}" if iteration.isdigit() else "Final"
        text = (f"{label}   β = {pole[0]:.2f}°  λ = {pole[1]:.2f}°  P = {pole[2]:.5f} h"
                if pole else label)

        new_verts, _ = read_shape(path)

        # Update vertex positions in-place — preserves camera and avoids actor churn
        mesh.points = new_verts
        mesh["Mean_Curvature"] = mesh.curvature("mean")
        wire_mesh.points = new_verts

        plotter.remove_actor(label_actor)
        label_actor = plotter.add_text(text, position="upper_left",
                                       font_size=11, color="white")
        plotter.render()
        time.sleep(delay)

    plotter.show(interactive=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise ADAM shape model output")
    parser.add_argument("--shapefile", default="output/outshape.txt",
                        help="Path to shape file (default: output/outshape.txt)")
    parser.add_argument("--paramfile", default="output/param_1",
                        help="Path to param file for pole arrow (default: output/param_1)")
    parser.add_argument("--outputdir", default="output",
                        help="Directory containing outshape.txt.NNN files")
    parser.add_argument("--animate", action="store_true",
                        help="Step through intermediate shapes")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between animation frames (default: 0.4)")
    args = parser.parse_args()

    if args.animate:
        plot_animate(args.outputdir, args.paramfile, args.delay)
    else:
        if not os.path.exists(args.shapefile):
            sys.exit(f"Shape file not found: {args.shapefile}")
        plot_single(args.shapefile, args.paramfile)


if __name__ == "__main__":
    main()
