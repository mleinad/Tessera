#!/usr/bin/env python3
"""
Voxelize a triangle mesh (STL, OBJ, PLY, ...) into a Tessera obstacle mask (.nrrd).

The output is a 3D uint8 grid of (n+2)^3 cells (ghost layer included): 1 = solid, 0 = air,
x fastest in memory, matching wave_idx and Simulator::LoadMask.

Usage:
    python3 voxelize.py bunny.stl 256 bunny_256.nrrd
    python3 voxelize.py bunny.stl 256 bunny_256.nrrd --fraction 0.3

Install: pip3 install numpy trimesh scipy pynrrd
"""
import argparse
import sys

import numpy as np
import nrrd
import trimesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", help="input mesh file (STL, OBJ, PLY, ...)")
    ap.add_argument("n", type=int, help="simulation grid size (interior cells per axis)")
    ap.add_argument("output", help="output .nrrd file")
    ap.add_argument("--fraction", type=float, default=0.4,
                    help="the model's longest side as a fraction of the box (default 0.4)")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    if not mesh.is_watertight:
        print("warning: mesh is not watertight; the interior fill may be wrong", file=sys.stderr)

    # Cell size in model units, so the longest side spans fraction * n cells
    pitch = mesh.extents.max() / (args.fraction * args.n)
    vox = mesh.voxelized(pitch).fill().matrix          # bool array indexed [x, y, z]
    vox = vox.transpose(2, 1, 0)                       # -> [z, y, x], the simulation's layout

    side = args.n + 2
    if any(s > args.n for s in vox.shape):
        sys.exit(f"model ({vox.shape} cells) doesn't fit in n={args.n}; lower --fraction")

    # Centre the model in the box; the ghost layer (index 0 and n+1) stays air
    grid = np.zeros((side, side, side), dtype=np.uint8)
    start = [(side - s) // 2 for s in vox.shape]
    grid[start[0]:start[0] + vox.shape[0],
         start[1]:start[1] + vox.shape[1],
         start[2]:start[2] + vox.shape[2]] = vox

    # raw: the C++ loader doesn't read compressed data; C order: array is [z, y, x], x fastest
    nrrd.write(args.output, grid, header={"encoding": "raw"}, index_order="C")

    solid = int(grid.sum())
    zs, ys, xs = np.nonzero(grid)
    print(f"wrote {args.output}: {side}^3 cells, {solid} solid ({100.0 * solid / grid.size:.1f}%)")
    print(f"model occupies x {xs.min()}..{xs.max()}, y {ys.min()}..{ys.max()}, z {zs.min()}..{zs.max()}")


if __name__ == "__main__":
    main()