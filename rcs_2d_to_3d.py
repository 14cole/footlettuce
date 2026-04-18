"""
rcs_2d_to_3d.py
================

2D-to-3D RCS expansion for .grim files.

Phase 1 (implemented here)
--------------------------
Given a 2D .grim file (``elevations == [0.0]``) containing 2D scattering
width  sigma_2D(azimuth, frequency, polarization)  and an extrusion length
L, compute the equivalent 3D RCS  sigma_3D(azimuth, elevation, frequency,
polarization)  on user-requested axes.

Physical relation (far-field, high-frequency, translationally-invariant body):

    sigma_3D(phi, theta, f) = (2 L^2 / lambda)
                             * sinc^2(k L sin(theta))
                             * sigma_2D(phi, f)

    where
        phi    = in-plane azimuth (same axis as the 2D file; deg)
        theta  = elevation from broadside (deg); theta=0 is in the 2D plane,
                 theta = +/-90 is along the extrusion axis (endfire)
        k      = 2 pi / lambda
        sinc(x)= sin(x)/x,  sinc(0) = 1

The ``2 L^2 / lambda`` factor and the sinc-squared "array factor" together
carry the 2D scattering width (units m) into 3D RCS (units m^2), and
naturally produce a main lobe of angular width ~ lambda / L around
broadside in the elevation plane.

Phase 2 (scaffolded, not implemented)
-------------------------------------
An STL mesh and a list of placement points (x, y, z) define where the 2D
template is applied in 3D. Shadowing is evaluated by ray-casting against
the mesh. See :func:`apply_2d_to_mesh`. STL is recommended over STEP
because it is already a triangle mesh and integrates directly with
``trimesh`` for occlusion queries.

Author: built to sit next to grim_dataset.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from grim_dataset import RcsGrid, C0


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

_FREQ_SCALES_TO_HZ = {
    "hz": 1.0,
    "khz": 1.0e3,
    "mhz": 1.0e6,
    "ghz": 1.0e9,
}


def _frequency_to_hz(frequency_values, unit: str) -> np.ndarray:
    """Convert stored frequency values to Hz based on the grid's unit tag."""
    freq = np.asarray(frequency_values, dtype=float)
    u = (unit or "GHz").strip().lower()
    scale = _FREQ_SCALES_TO_HZ.get(u, 1.0e9)  # default GHz if unknown
    return freq * scale


def _wavelength_m(frequency_values, unit: str) -> np.ndarray:
    f_hz = _frequency_to_hz(frequency_values, unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(f_hz > 0.0, C0 / f_hz, np.nan)
    return lam


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _sinc(x: np.ndarray) -> np.ndarray:
    """sin(x)/x with sinc(0) = 1 (NOT numpy's normalized sinc)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = np.abs(x) > 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def _sinc_sq(x: np.ndarray) -> np.ndarray:
    s = _sinc(x)
    return s * s


def _interp_clip_axis(values: np.ndarray, x_old, x_new, axis: int) -> np.ndarray:
    """Linear interp along ``axis``; out-of-range requests clamp to end values."""
    x_old = np.asarray(x_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x_old.size == 0:
        raise ValueError("source axis is empty")
    if x_old.size == 1:
        # nothing to interpolate over; broadcast
        moved = np.moveaxis(values, axis, 0)
        tile_shape = (x_new.size,) + moved.shape[1:]
        out = np.broadcast_to(moved, tile_shape).copy()
        return np.moveaxis(out, 0, axis)
    x_new_clipped = np.clip(x_new, x_old.min(), x_old.max())
    order = np.argsort(x_old)
    x_sorted = x_old[order]
    moved = np.moveaxis(values, axis, 0)[order]
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((x_new.size, flat.shape[1]), dtype=values.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(x_new_clipped, x_sorted, flat[:, i])
    out = out.reshape((x_new.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def _interp_wrap_azimuth(values: np.ndarray, az_old_deg, az_new_deg, axis: int) -> np.ndarray:
    """Periodic (mod-360) linear interpolation along ``axis``."""
    az_old = np.mod(np.asarray(az_old_deg, dtype=float), 360.0)
    az_new = np.mod(np.asarray(az_new_deg, dtype=float), 360.0)

    order = np.argsort(az_old)
    az_sorted = az_old[order]
    moved = np.moveaxis(values, axis, 0)[order]

    # Extend by one period on each side to handle wrap-around cleanly.
    az_ext = np.concatenate([az_sorted - 360.0, az_sorted, az_sorted + 360.0])
    data_ext = np.concatenate([moved, moved, moved], axis=0)

    flat = data_ext.reshape(data_ext.shape[0], -1)
    out = np.empty((az_new.size, flat.shape[1]), dtype=values.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(az_new, az_ext, flat[:, i])
    out = out.reshape((az_new.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def _interp_wrap_azimuth_phase(phase: np.ndarray, az_old_deg, az_new_deg, axis: int) -> np.ndarray:
    """Interp phase via unit-complex decomposition to avoid 2*pi jump artefacts."""
    # NaNs in phase -> NaN real/imag -> NaN output (desired).
    real = _interp_wrap_azimuth(np.cos(phase), az_old_deg, az_new_deg, axis=axis)
    imag = _interp_wrap_azimuth(np.sin(phase), az_old_deg, az_new_deg, axis=axis)
    return np.arctan2(imag, real)


# ---------------------------------------------------------------------------
# Public API — 2D -> 3D expansion
# ---------------------------------------------------------------------------

@dataclass
class ExpansionMeta:
    """Metadata describing how a 2D grid was expanded to 3D."""
    length: float
    extrusion_axis: str
    input_domain: str                 # "scattering_width" or "rcs"
    broadside_elevation_deg: float    # where broadside sits on the elevation axis


def expand_2d_to_3d(
    grid_2d: RcsGrid,
    length: float,
    *,
    azimuths_deg: Optional[Sequence[float]] = None,
    elevations_deg: Optional[Sequence[float]] = None,
    frequencies: Optional[Sequence[float]] = None,
    polarizations: Optional[Sequence] = None,
    extrusion_axis: str = "z",
    input_domain: str = "scattering_width",
    broadside_elevation_deg: float = 0.0,
    preserve_phase: bool = True,
) -> RcsGrid:
    """Expand a 2D .grim grid to 3D assuming translational symmetry along ``extrusion_axis``.

    Parameters
    ----------
    grid_2d : RcsGrid
        A 2D RCS grid. Must have exactly one elevation sample.
    length : float
        Extrusion length L in meters (along the symmetry axis).
    azimuths_deg : sequence of float, optional
        Requested output azimuths (deg). Defaults to the source grid's azimuths.
    elevations_deg : sequence of float, optional
        Requested output elevations (deg), measured from broadside
        (``broadside_elevation_deg``). Defaults to ``np.linspace(-90, 90, 181)``.
    frequencies : sequence of float, optional
        Requested output frequencies in the *same unit* as the source grid.
        Defaults to the source grid's frequencies.
    polarizations : sequence, optional
        Subset of source polarizations to keep. Defaults to all.
    extrusion_axis : str
        Symbolic label for the extrusion axis (metadata only).
    input_domain : {"scattering_width", "rcs"}
        - "scattering_width": source power is 2D scattering width sigma_2D (m).
          This is the default and matches ``RcsGrid.load_out``.
        - "rcs": source power is already 3D-like RCS (m^2) at broadside; the
          expansion only applies the sinc^2 shape (useful if you've already
          scaled for an assumed length).
    broadside_elevation_deg : float
        Elevation angle that corresponds to "broadside to the extrusion axis"
        (i.e. in the 2D plane). Defaults to 0 deg.
    preserve_phase : bool
        If True, carry 2D phase through to 3D, adding the 0/pi sign of sinc.
        If False, phase is set to NaN.

    Returns
    -------
    RcsGrid
        A new 3D RcsGrid on the requested (az, el, f, pol) axes, ready to
        save back to a .grim file via :meth:`RcsGrid.save`.
    """
    # ---------------- Validation ----------------
    elev_in = np.asarray(grid_2d.elevations, dtype=float)
    if elev_in.size != 1:
        raise ValueError(
            f"expand_2d_to_3d expects a single-elevation (2D) grid; got "
            f"{elev_in.size} elevation samples. Use a 2D .grim file with "
            "elevations=[0.0]."
        )
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be finite and positive, got {length!r}")
    if input_domain not in ("scattering_width", "rcs"):
        raise ValueError("input_domain must be 'scattering_width' or 'rcs'")

    az_in = np.asarray(grid_2d.azimuths, dtype=float)
    f_in = np.asarray(grid_2d.frequencies, dtype=float)
    pol_in = np.asarray(grid_2d.polarizations)

    if az_in.size == 0 or f_in.size == 0 or pol_in.size == 0:
        raise ValueError("source grid has an empty axis")

    # ---------------- Build output axes ----------------
    az_out = np.asarray(list(azimuths_deg) if azimuths_deg is not None else az_in, dtype=float)
    el_out = np.asarray(
        list(elevations_deg) if elevations_deg is not None else np.linspace(-90.0, 90.0, 181),
        dtype=float,
    )
    f_out = np.asarray(list(frequencies) if frequencies is not None else f_in, dtype=float)
    # Force unicode str dtype so the produced .grim can be reloaded with
    # allow_pickle=False (np.savez pickles object-dtype arrays, which breaks
    # RcsGrid.load). Unicode strings are saved natively.
    _pol_src = list(polarizations) if polarizations is not None else list(pol_in)
    pol_out = np.asarray([str(p) for p in _pol_src], dtype="<U")

    if az_out.size == 0 or el_out.size == 0 or f_out.size == 0 or pol_out.size == 0:
        raise ValueError("all output axes must have at least one sample")

    # ---------------- Collapse 2D grid to (az, f, pol) ----------------
    power_in = np.asarray(grid_2d.rcs_power, dtype=np.float64)[:, 0, :, :]    # (az, f, pol)
    phase_in = np.asarray(grid_2d.rcs_phase, dtype=np.float64)[:, 0, :, :]    # (az, f, pol)

    # Polarization alignment (label match, string-based to be dtype-agnostic)
    pol_in_str = np.asarray([str(p) for p in pol_in])
    pol_idx = []
    for p in pol_out:
        matches = np.where(pol_in_str == str(p))[0]
        if matches.size == 0:
            raise ValueError(f"requested polarization {p!r} not present in 2D grid")
        pol_idx.append(int(matches[0]))
    power_in = power_in[:, :, pol_idx]
    phase_in = phase_in[:, :, pol_idx]

    # ---------------- Interpolate source to requested az/f ----------------
    power_az = _interp_wrap_azimuth(power_in, az_in, az_out, axis=0)
    phase_az = _interp_wrap_azimuth_phase(phase_in, az_in, az_out, axis=0)
    power_f = _interp_clip_axis(power_az, f_in, f_out, axis=1)        # (az_out, f_out, pol)
    phase_f = _interp_clip_axis(phase_az, f_in, f_out, axis=1)

    # ---------------- Elevation / frequency expansion ----------------
    freq_unit = (grid_2d.units or {}).get("frequency", "GHz")
    lam = _wavelength_m(f_out, freq_unit)                             # (f_out,)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(lam > 0.0, 2.0 * np.pi / lam, np.nan)            # (f_out,)

    # theta relative to broadside
    theta_rel_deg = el_out - float(broadside_elevation_deg)
    sin_theta = np.sin(np.deg2rad(theta_rel_deg))                     # (el_out,)

    # kL sin(theta) over (el, f)
    kL_sintheta = np.outer(sin_theta, k * length)                     # (el, f)
    sinc_vals = _sinc(kL_sintheta)                                    # (el, f)
    sinc2_vals = sinc_vals * sinc_vals                                # (el, f)

    if input_domain == "scattering_width":
        with np.errstate(divide="ignore", invalid="ignore"):
            two_L2_over_lam = np.where(lam > 0.0, (2.0 * length ** 2) / lam, np.nan)
        multiplier = two_L2_over_lam[np.newaxis, :] * sinc2_vals      # (el, f)
    else:  # "rcs"
        multiplier = sinc2_vals

    # Broadcast: power_f[az, f, pol] * multiplier[el, f]  ->  (az, el, f, pol)
    power_out = power_f[:, np.newaxis, :, :] * multiplier[np.newaxis, :, :, np.newaxis]

    # Phase:  arg(sigma_3D) = arg(sigma_2D) + arg(sinc), where arg(sinc) is 0 or pi.
    if preserve_phase:
        sign_phase = np.where(sinc_vals >= 0.0, 0.0, np.pi)           # (el, f)
        phase_out = (
            phase_f[:, np.newaxis, :, :]
            + sign_phase[np.newaxis, :, :, np.newaxis]
        )
        phase_out = np.asarray(phase_out, dtype=np.float32)
    else:
        phase_out = np.full(power_out.shape, np.nan, dtype=np.float32)

    # ---------------- Package as a new RcsGrid ----------------
    new_units = dict(grid_2d.units or {})
    new_units.setdefault("azimuth", "deg")
    new_units["elevation"] = "deg"  # force 3D elevation to deg

    history_prev = grid_2d.history or ""
    expansion_tag = (
        f"2D->3D expansion | L={length:g} m | axis={extrusion_axis} "
        f"| input_domain={input_domain} | broadside_el={broadside_elevation_deg} deg"
    )
    new_history = f"{history_prev} | {expansion_tag}" if history_prev else expansion_tag

    return RcsGrid(
        azimuths=az_out,
        elevations=el_out,
        frequencies=f_out,
        polarizations=pol_out,
        rcs_power=power_out.astype(np.float32),
        rcs_phase=phase_out,
        rcs_domain="power_phase",
        source_path=grid_2d.source_path,
        history=new_history,
        units=new_units,
    )


# ---------------------------------------------------------------------------
# Convenience: single-point query
# ---------------------------------------------------------------------------

def rcs_3d_at(
    grid_2d: RcsGrid,
    length: float,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    frequency: float,
    polarization,
    return_complex: bool = True,
    **kwargs,
):
    """Return the expanded 3D RCS at a single (az, el, f, pol) point.

    If ``return_complex`` is True, returns ``complex`` (sqrt(power)*exp(j*phase)).
    Otherwise returns a dict with linear power, phase, and dBsm.
    """
    sub = expand_2d_to_3d(
        grid_2d,
        length,
        azimuths_deg=[azimuth_deg],
        elevations_deg=[elevation_deg],
        frequencies=[frequency],
        polarizations=[polarization],
        preserve_phase=True,
        **kwargs,
    )
    power = float(sub.rcs_power[0, 0, 0, 0])
    phase = float(sub.rcs_phase[0, 0, 0, 0])
    if return_complex:
        if not np.isfinite(power):
            return complex("nan")
        if not np.isfinite(phase):
            phase = 0.0
        return complex(np.sqrt(power) * np.exp(1j * phase))
    return {
        "rcs_linear_m2": power,
        "phase_rad": phase,
        "rcs_dbsm": 10.0 * np.log10(max(power, 1e-12)),
    }


# ---------------------------------------------------------------------------
# Phase 2 scaffolding — STL mesh + placement + shadowing
# ---------------------------------------------------------------------------

@dataclass
class MeshContext:
    """Triangle mesh loaded for Phase 2 ray-casting / shadowing."""
    vertices: np.ndarray        # (N, 3)
    faces: np.ndarray           # (M, 3) vertex-index triples
    face_normals: Optional[np.ndarray] = None  # (M, 3), unit-length

    @property
    def centroids(self) -> np.ndarray:
        v = self.vertices[self.faces]            # (M, 3, 3)
        return v.mean(axis=1)


def load_stl(path: str) -> MeshContext:
    """Load an STL into a :class:`MeshContext` using ``trimesh``.

    STL is recommended over STEP because it is already a triangle mesh
    and ``trimesh`` provides out-of-the-box ray intersection tests
    (embree / pyembree backends for speed).
    """
    try:
        import trimesh  # type: ignore
    except ImportError as err:
        raise ImportError(
            "load_stl requires the 'trimesh' package. Install with: pip install trimesh"
        ) from err
    mesh = trimesh.load(path, force="mesh")
    return MeshContext(
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=int),
        face_normals=np.asarray(mesh.face_normals, dtype=float),
    )


def _incident_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector pointing *toward* the radar from the target, using
    azimuth measured CCW from +X in the XY plane and elevation from XY up.
    """
    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elevation_deg)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], dtype=float)


def apply_2d_to_mesh(
    grid_2d: RcsGrid,
    mesh: MeshContext,
    placement_xyz: np.ndarray,
    *,
    length_per_segment: Optional[np.ndarray] = None,
    azimuths_deg=None,
    elevations_deg=None,
    frequencies=None,
    polarizations=None,
    include_shadowing: bool = True,
    shadow_epsilon: float = 1.0e-4,
) -> RcsGrid:
    """Apply a 2D RCS template along placement points in a 3D mesh.

    This is **Phase 2** — currently a typed stub with the intended interface
    and the algorithm outline below. Raises ``NotImplementedError`` until
    wired up.

    Intended algorithm
    ------------------
    1. For each requested incidence direction ``(phi, theta)``:
       a. Build the unit incident vector ``d_hat`` via
          :func:`_incident_direction`.
       b. For each placement point ``p_k`` in ``placement_xyz``:
          - Cast a ray from ``p_k + shadow_epsilon * d_hat`` toward the
            radar (direction ``d_hat``). If the ray intersects any mesh
            face before escaping the bounding box, the point is shadowed
            and contributes zero (or an attenuated value per a PO model).
          - Otherwise, evaluate the 2D->3D expansion contribution of this
            point using the local segment length (``length_per_segment[k]``)
            and the 2D RCS at the rotated azimuth that corresponds to this
            incident direction in the local segment frame.
       c. Sum contributions incoherently (power) or coherently (complex
          field) depending on mode. Incoherent is appropriate when the
          inter-point spacing is >> lambda and phases are uncorrelated;
          coherent is appropriate for a single rigid body.

    2. Pack the resulting (az, el, f, pol) grid into an :class:`RcsGrid`.

    Parameters
    ----------
    grid_2d : RcsGrid
        2D RCS template (elevations == [0.0]).
    mesh : MeshContext
        Loaded STL mesh representing the full object.
    placement_xyz : np.ndarray, shape (K, 3)
        Centerline / feature points along which to apply the 2D template.
    length_per_segment : np.ndarray, shape (K,), optional
        Per-point effective length. Defaults to the spacing between
        consecutive points (``np.diff`` with edge replication).
    include_shadowing : bool
        Ray-cast against ``mesh`` to drop shadowed contributions.
    shadow_epsilon : float
        Offset to avoid self-intersection at the ray origin.
    """
    raise NotImplementedError(
        "apply_2d_to_mesh is Phase 2. The API above is the planned interface; "
        "implement by (1) using trimesh.ray.ray_pyembree for fast occlusion "
        "tests, and (2) summing per-segment expand_2d_to_3d contributions "
        "using length_per_segment[k] as L."
    )


__all__ = [
    "ExpansionMeta",
    "MeshContext",
    "apply_2d_to_mesh",
    "expand_2d_to_3d",
    "load_stl",
    "rcs_3d_at",
]
