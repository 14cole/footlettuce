"""
demo_2d_to_3d.py
================

Minimal usage example for rcs_2d_to_3d.expand_2d_to_3d.

Drop a real 2D .grim file next to this script (or pass its path on the CLI)
and run:

    python demo_2d_to_3d.py path/to/my_2d_file.grim
"""
import sys
import numpy as np

from grim_dataset import RcsGrid
from rcs_2d_to_3d import expand_2d_to_3d, rcs_3d_at


def main(path: str, length_m: float = 1.5):
    # ---- Load the 2D .grim file ----
    grid_2d = RcsGrid.load(path)
    print(f"Loaded 2D grid: {path}")
    print(f"  azimuths:      {len(grid_2d.azimuths)} values "
          f"[{grid_2d.azimuths.min()} .. {grid_2d.azimuths.max()}]")
    print(f"  elevations:    {grid_2d.elevations} (must be length-1 and ~0)")
    print(f"  frequencies:   {grid_2d.frequencies} "
          f"({(grid_2d.units or {}).get('frequency', '?')})")
    print(f"  polarizations: {list(grid_2d.polarizations)}")

    # ---- Choose output axes ----
    requested_az = np.linspace(0.0, 360.0, 73, endpoint=False)  # 5-deg az
    requested_el = np.linspace(-30.0, 30.0, 61)                 # 1-deg el, +/-30
    requested_f = grid_2d.frequencies                           # keep source freqs
    requested_pol = list(grid_2d.polarizations)

    # ---- Expand ----
    grid_3d = expand_2d_to_3d(
        grid_2d,
        length=length_m,
        azimuths_deg=requested_az,
        elevations_deg=requested_el,
        frequencies=requested_f,
        polarizations=requested_pol,
        extrusion_axis="z",
        input_domain="scattering_width",   # matches load_out convention
        preserve_phase=True,
    )
    print(f"\nExpanded 3D grid shape: "
          f"az={len(grid_3d.azimuths)} x el={len(grid_3d.elevations)} "
          f"x f={len(grid_3d.frequencies)} x pol={len(grid_3d.polarizations)}")

    # ---- Save back to .grim ----
    out_path = grid_3d.save(path.replace(".grim", "") + "_3d")
    print(f"Saved: {out_path}")

    # ---- Single-point query ----
    pol0 = requested_pol[0]
    f0 = float(requested_f[0])
    c = rcs_3d_at(
        grid_2d, length_m,
        azimuth_deg=45.0, elevation_deg=10.0,
        frequency=f0, polarization=pol0,
        return_complex=False,
    )
    print(f"\nSingle-point query  (az=45, el=10, f={f0}, pol={pol0!r}):")
    for k, v in c.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    length = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.5
    main(sys.argv[1], length_m=length)
