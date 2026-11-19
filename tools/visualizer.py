#!/usr/bin/env python3
"""
Render Tessera 3D volume dumps as an animation with PyVista.

Each frame file (frames/vol_XXXX.bin, written by Simulator::DumpVolume) is:
    uint32 w, uint32 h, uint32 d, uint32 stride, uint64 step   (24-byte header, little-endian)
    float32[d][h][w]                                            (x fastest, ghost layer included)

Modes:
    contour   red/blue isosurfaces at +/- level, with the octant facing the camera cut away (default)
    volume    translucent volume rendering, near-zero pressure transparent

Speed: frames are loaded and contoured in parallel worker processes (--jobs); the main process
only draws. --stride N downsamples the volume N times per axis (N^3 fewer cells) for quick previews.

Usage:
    python3 tools/visualizer.py                                        # frames/ -> wave3d.mp4
    python3 tools/visualizer.py frames out.mp4 --mask tools/bunny_256.nrrd
    python3 tools/visualizer.py frames out.mp4 --mode volume --clip 0.2
    python3 tools/visualizer.py frames preview.gif --stride 2 --size 500

Install: pip3 install numpy pyvista imageio imageio-ffmpeg pynrrd   (pynrrd only for --mask)
"""
import argparse
import glob
import itertools
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyvista as pv

HEADER = np.dtype([("w", "<u4"), ("h", "<u4"), ("d", "<u4"), ("stride", "<u4"), ("step", "<u8")])
CMAP = "RdBu_r"                                   # negative blue, zero white, positive red
SHELL_COLORS = ("#e8553f", "#3d8fe0")             # +level red, -level blue


def read_frame(path, stride=1):
    """-> (step, spacing, float32 array indexed [z, y, x])"""
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=HEADER, count=1)[0]
        w, h, d = int(hdr["w"]), int(hdr["h"]), int(hdr["d"])
        data = np.fromfile(f, dtype="<f4", count=w * h * d)
    if data.size != w * h * d:
        raise ValueError(f"{path}: expected {w * h * d} floats, got {data.size}")
    data = data.reshape(d, h, w)
    if stride > 1:
        data = np.ascontiguousarray(data[::stride, ::stride, ::stride])
    return int(hdr["step"]), int(hdr["stride"]) * stride, data


def to_grid(data, spacing):
    d, h, w = data.shape
    grid = pv.ImageData(dimensions=(w, h, d), spacing=(spacing,) * 3)
    grid.point_data["p"] = data.ravel()           # x fastest, which is VTK's point order
    return grid


def contour_frame(job):
    """Worker: load one frame and build both shells (the expensive part). Returns plain arrays."""
    path, stride, level = job
    step, spacing, data = read_frame(path, stride)
    grid = to_grid(data, spacing)
    x0, x1, y0, y1, z0, z1 = grid.bounds
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])

    shells = []
    for value in (level, -level):
        # flying_edges: VTK's fast isosurface for image data; normals come almost for free
        iso = grid.contour([value], scalars="p", method="flying_edges", compute_normals=True)
        if iso.n_points:
            # Cutaway: drop the octant facing the iso camera so the nested shells are visible.
            # Removing points keeps a PolyData (clip_box made a slower unstructured grid).
            inside = np.all(iso.points > centre, axis=1)
            if inside.any():
                iso, _ = iso.remove_points(inside)
        if iso.n_points:
            shells.append((np.asarray(iso.points), np.asarray(iso.faces),
                           np.asarray(iso.point_data["Normals"])))
        else:
            shells.append(None)
    return step, shells


def load_frame(job):
    """Worker for volume mode: just the (parallel) file read and downsampling."""
    path, stride, _ = job
    return read_frame(path, stride)


def ordered_results(pool, fn, jobs, ahead):
    """Like pool.map, but keeps at most `ahead` frames in flight so memory stays bounded."""
    jobs = iter(jobs)
    pending = deque(pool.submit(fn, j) for j in itertools.islice(jobs, ahead))
    while pending:
        result = pending.popleft().result()
        nxt = next(jobs, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))
        yield result


def load_mask_surface(path):
    """Surface of the solid cells in a Tessera mask (.nrrd), in simulation cell coordinates."""
    import nrrd                                   # only needed for --mask
    data, _ = nrrd.read(path, index_order="C")    # indexed [z, y, x], like the simulation
    grid = to_grid(data.astype(np.float32), 1)    # cell i sits at coordinate i, like the dumps
    surface = grid.contour([0.5], scalars="p", method="flying_edges")
    if surface.n_points:
        surface = surface.smooth_taubin(n_iter=30)  # soften the voxel staircase
    return surface


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", nargs="?", default="frames")
    ap.add_argument("output", nargs="?", default="wave3d.mp4", help=".mp4 (needs imageio-ffmpeg) or .gif")
    ap.add_argument("--mode", choices=["contour", "volume"], default="contour")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--level", type=float, default=0.05,
                    help="isosurface level as a fraction of the first frame's peak (contour)")
    ap.add_argument("--clip", type=float, default=0.25,
                    help="colour range as a fraction of the first frame's peak (volume)")
    ap.add_argument("--size", type=int, default=800, help="output resolution in pixels (square)")
    ap.add_argument("--mask", default=None, help="obstacle mask (.nrrd) to draw as a grey model")
    ap.add_argument("--stride", type=int, default=1, help="extra downsampling per axis (2 = 8x fewer cells)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="worker processes (default: CPU count - 2)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames_dir, "vol_*.bin")))
    if not paths:
        sys.exit(f"no vol_*.bin files in {args.frames_dir}/")

    _, spacing, first = read_frame(paths[0], args.stride)
    peak = max(float(np.abs(first).max()), 1e-12)
    level, vmax = args.level * peak, args.clip * peak
    jobs = [(p, args.stride, level) for p in paths]

    # Start the workers before any rendering window exists (forking an OpenGL context is unsafe)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        plotter = pv.Plotter(off_screen=True, window_size=(args.size, args.size))
        plotter.set_background("#111111")
        if args.output.endswith(".gif"):
            plotter.open_gif(args.output, fps=args.fps)
        else:
            try:
                plotter.open_movie(args.output, framerate=args.fps, quality=8)
            except Exception as e:
                sys.exit(f"cannot write {args.output} ({e}). Install imageio-ffmpeg, or use .gif")

        # Drawn once: the simulation box and the obstacle model
        plotter.add_mesh(to_grid(first, spacing).outline(), color="#777777")
        if args.mask:
            model = load_mask_surface(args.mask)
            if model.n_points:
                plotter.add_mesh(model, color="#bbbbbb", smooth_shading=True, specular=0.2)
            else:
                print(f"warning: {args.mask} has no solid cells", file=sys.stderr)

        worker = contour_frame if args.mode == "contour" else load_frame
        actors = []
        for i, result in enumerate(ordered_results(pool, worker, jobs, ahead=2 * args.jobs)):
            for a in actors:
                plotter.remove_actor(a)
            actors = []

            if args.mode == "contour":
                step, shells = result
                for shell, color in zip(shells, SHELL_COLORS):
                    if shell is None:
                        continue
                    points, faces, normals = shell
                    mesh = pv.PolyData(points, faces)
                    mesh.point_data["Normals"] = normals
                    actors.append(plotter.add_mesh(mesh, color=color, smooth_shading=True,
                                                   specular=0.3, show_scalar_bar=False))
            else:
                step, frame_spacing, data = result
                actors.append(plotter.add_volume(to_grid(data, frame_spacing), scalars="p", cmap=CMAP,
                                                 clim=[-vmax, vmax], opacity=[0.9, 0.5, 0.0, 0.5, 0.9],
                                                 show_scalar_bar=False))

            actors.append(plotter.add_text(f"step {step}", position="upper_left", font_size=12, color="white"))
            if i == 0:
                plotter.camera_position = "iso"
                plotter.camera.zoom(1.1)
            plotter.write_frame()
            print(f"\rframe {i + 1}/{len(paths)}", end="", flush=True)

        plotter.close()
    print(f"\nwrote {args.output} ({len(paths)} frames, mode={args.mode})")


if __name__ == "__main__":
    main()
