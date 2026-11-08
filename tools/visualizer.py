#!/usr/bin/env python3
"""
Render Tessera 3D volume dumps as an animation with PyVista.

Each frame file (frames3d/vol_XXXX.bin) is:
    uint32 w, uint32 h, uint32 d, uint32 stride, uint64 step   (24-byte header, little-endian)
    float32[d][h][w]                                            (x fastest, ghost layer included)

Modes:
    contour   isosurfaces at +/- level: red and blue shells (default, clearest)
    volume    translucent volume rendering, near-zero pressure transparent
    slices    three orthogonal cuts through the centre

Usage:
    python3 viz3d.py                                   # frames3d/ -> wave3d.mp4 (or .gif)
    python3 viz3d.py /path/to/frames3d out.mp4 --mode volume --clip 0.2
    python3 viz3d.py frames3d out.gif --mode contour --level 0.05

Install: pip3 install numpy pyvista imageio imageio-ffmpeg
"""
import argparse
import glob
import os
import sys

import numpy as np
import pyvista as pv

HEADER = np.dtype([("w", "<u4"), ("h", "<u4"), ("d", "<u4"), ("stride", "<u4"), ("step", "<u8")])
CMAP = "RdBu_r"   # negative blue, zero white, positive red


def load_volume(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=HEADER, count=1)[0]
        w, h, d = int(hdr["w"]), int(hdr["h"]), int(hdr["d"])
        stride, step = int(hdr["stride"]), int(hdr["step"])
        data = np.fromfile(f, dtype="<f4", count=w * h * d)
    if data.size != w * h * d:
        raise ValueError(f"{path}: expected {w * h * d} floats, got {data.size}")
    grid = pv.ImageData(dimensions=(w, h, d), spacing=(stride, stride, stride))
    grid.point_data["p"] = data   # file order is x fastest, which is VTK's point order
    return step, grid


def draw(plotter, grid, mode, vmax, level):
    """Add this frame's actors and return them so they can be removed next frame."""
    actors = []
    if mode == "contour":
        # Cut away the octant facing the iso camera so the nested shells are visible
        x0, x1, y0, y1, z0, z1 = grid.bounds
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        cutaway = [cx, x1 + 1, cy, y1 + 1, cz, z1 + 1]
        for value, color in ((level, "#e8553f"), (-level, "#3d8fe0")):   # positive red, negative blue
            iso = grid.contour(isosurfaces=[value], scalars="p")
            if iso.n_points:
                iso = iso.clip_box(cutaway)
            if iso.n_points:
                actors.append(plotter.add_mesh(iso, color=color, smooth_shading=True,
                                               specular=0.3, show_scalar_bar=False))
    elif mode == "volume":
        actors.append(plotter.add_volume(grid, scalars="p", cmap=CMAP, clim=[-vmax, vmax],
                                         opacity=[0.9, 0.5, 0.0, 0.5, 0.9], show_scalar_bar=False))
    elif mode == "slices":
        actors.append(plotter.add_mesh(grid.slice_orthogonal(), scalars="p", cmap=CMAP,
                                       clim=[-vmax, vmax], show_scalar_bar=False))
    return actors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", nargs="?", default="frames3d")
    ap.add_argument("output", nargs="?", default=None, help="wave3d.mp4 by default (.gif also works)")
    ap.add_argument("--mode", choices=["contour", "volume", "slices"], default="contour")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--clip", type=float, default=0.25,
                    help="colour range as a fraction of the first frame's peak (volume, slices)")
    ap.add_argument("--level", type=float, default=0.05,
                    help="isosurface level as a fraction of the first frame's peak (contour)")
    ap.add_argument("--size", type=int, default=800, help="output resolution in pixels (square)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames_dir, "vol_*.bin")))
    if not paths:
        sys.exit(f"no vol_*.bin files in {args.frames_dir}/")

    _, first = load_volume(paths[0])
    peak = max(float(np.abs(first["p"]).max()), 1e-12)
    vmax, level = args.clip * peak, args.level * peak

    out = args.output or "wave3d.mp4"
    plotter = pv.Plotter(off_screen=True, window_size=(args.size, args.size))
    plotter.set_background("#111111")
    if out.endswith(".gif"):
        plotter.open_gif(out, fps=args.fps)
    else:
        try:
            plotter.open_movie(out, framerate=args.fps, quality=8)
        except Exception as e:
            sys.exit(f"cannot write {out} ({e}). Install imageio-ffmpeg, or use an output ending in .gif")

    plotter.add_mesh(first.outline(), color="#777777")      # the simulation box, drawn once
    actors = []
    for i, path in enumerate(paths):
        step, grid = load_volume(path)
        for a in actors:
            plotter.remove_actor(a)
        actors = draw(plotter, grid, args.mode, vmax, level)
        actors.append(plotter.add_text(f"step {step}", position="upper_left", font_size=12, color="white"))
        if i == 0:
            plotter.camera_position = "iso"
            plotter.camera.zoom(1.1)
        plotter.write_frame()
        print(f"\rframe {i + 1}/{len(paths)}", end="", flush=True)

    plotter.close()
    print(f"\nwrote {out} ({len(paths)} frames, mode={args.mode})")


if __name__ == "__main__":
    main()