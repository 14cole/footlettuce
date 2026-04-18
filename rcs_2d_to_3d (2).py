"""
rcs_2d_to_3d.py
================

2D -> 3D RCS expansion for .grim files.

Convention (per the input-file spec)
------------------------------------
* The 2D .grim file has ``elevations == [0.0]`` (vestigial slot).
* Its ``azimuths`` field stores values that represent **3D elevation** angles,
  measured along a great circle in the vertical east-zenith plane:
    0  deg = east   (+X, horizon)
   90  deg = zenith (+Z, overhead)
  180  deg = west   (-X, horizon)
  270  deg = nadir  (-Z, below)
  values progress CCW in that plane.
* Frequencies and polarizations are conventional.

Output 3D grid
--------------
* ``azimuths``    in [-180, 180] (signed), 0 = east, +90 = north, CCW.
* ``elevations``  in [-180, 180] (signed) with the same great-circle meaning
  as the stored 2D-azimuth values (i.e. same mapping, just renamed).

Phase 1 -- bulk extrusion expansion
-----------------------------------
Under the model "translationally-invariant body extruded by length L along the
Y axis", physical optics yields:

    sigma_3D(az, el, f) = (2 L^2 / lambda)
                         * sinc^2(k L sin(az - az_broadside))
                         * sigma_2D(el mod 360, f)

The sinc^2 is the finite-length array factor in azimuth; its main lobes sit at
az = 0 (east, broadside) and az = +/-180 (west, equivalent broadside), with a
~lambda/L angular width. The elevation dependence is inherited directly from
the 2D data.

Phase 2 -- mesh-aware swept expansion with shadowing
----------------------------------------------------
* The target is described by an STL hull (triangle mesh) and a centerline of
  placement points  {r_k in R^3, k = 0..K-1}  along which the 2D template is
  applied. Per-point tangents and segment lengths are derived from the
  centerline.
* For each requested incidence direction d_hat(az, el):
  - for each placement point r_k, compute
      alpha_k = arcsin(d_hat . t_hat_k)        (aspect from local broadside)
      phi_k   = angle of d_hat's projection on the plane perpendicular to
                t_hat_k, measured in a local (east, up) frame
      shadow_k = any_hit(origin=r_k, dir=d_hat) on the STL mesh
  - amplitude_k = sqrt(2/lambda) * dL_k * sinc(kw*dL_k*sin(alpha_k))
                   * sqrt(sigma_2D(phi_k, f, pol))                        (0 if shadowed)
  - phase_k     = arg(sigma_2D(phi_k, f, pol)) - 2*kw*(d_hat . r_k)
                   + {0 or pi per sign of sinc}
  - E_total = sum_k amplitude_k * exp(j * phase_k)
  - sigma_3D(az, el, f, pol) = |E_total|^2         (coherent, default)
                               or sum_k |..|^2     (incoherent, optional)

The ``sqrt(2/lambda) * dL_k * sinc(kw dL_k sin alpha_k)`` per-element form lets
the coherent sum reproduce the continuous (2 L^2/lambda) sinc^2(kw L sin alpha)
result in the dense-centerline limit, while remaining robust to coarse
sampling (the per-element sinc suppresses aliasing grating lobes).

STL I/O and shadowing are pure numpy -- no external dependencies.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from grim_dataset import RcsGrid, C0


# ============================================================================
# Shared helpers
# ============================================================================

_FREQ_SCALES_TO_HZ = {"hz": 1.0, "khz": 1.0e3, "mhz": 1.0e6, "ghz": 1.0e9}


def _frequency_to_hz(frequency_values, unit: str) -> np.ndarray:
    freq = np.asarray(frequency_values, dtype=float)
    scale = _FREQ_SCALES_TO_HZ.get((unit or "GHz").strip().lower(), 1.0e9)
    return freq * scale


def _wavelength_m(frequency_values, unit: str) -> np.ndarray:
    f_hz = _frequency_to_hz(frequency_values, unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f_hz > 0.0, C0 / f_hz, np.nan)


def _normalize_to_range(angles, range_type: str = "signed") -> np.ndarray:
    """Map angles (deg) into [-180, 180) for 'signed' or [0, 360) for 'positive'."""
    a = np.asarray(angles, dtype=float)
    if range_type == "signed":
        return ((a + 180.0) % 360.0) - 180.0
    if range_type == "positive":
        return a % 360.0
    raise ValueError(f"range_type must be 'signed' or 'positive', got {range_type!r}")


def _sinc(x: np.ndarray) -> np.ndarray:
    """sin(x)/x with sinc(0) = 1 (NOT numpy's normalized pi-sinc)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = np.abs(x) > 0.0
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def _sinc_sq(x: np.ndarray) -> np.ndarray:
    s = _sinc(x)
    return s * s


def _interp_clip_axis(values, x_old, x_new, axis: int) -> np.ndarray:
    x_old = np.asarray(x_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x_old.size == 0:
        raise ValueError("source axis is empty")
    if x_old.size == 1:
        moved = np.moveaxis(values, axis, 0)
        out = np.broadcast_to(moved, (x_new.size,) + moved.shape[1:]).copy()
        return np.moveaxis(out, 0, axis)
    x_clip = np.clip(x_new, x_old.min(), x_old.max())
    order = np.argsort(x_old)
    x_sorted = x_old[order]
    moved = np.moveaxis(values, axis, 0)[order]
    flat = moved.reshape(moved.shape[0], -1)
    out = np.empty((x_new.size, flat.shape[1]), dtype=values.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(x_clip, x_sorted, flat[:, i])
    out = out.reshape((x_new.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def _interp_wrap_angle(values, x_old_deg, x_new_deg, axis: int) -> np.ndarray:
    """Periodic (mod-360) linear interpolation along ``axis``."""
    x_old = np.mod(np.asarray(x_old_deg, dtype=float), 360.0)
    x_new = np.mod(np.asarray(x_new_deg, dtype=float), 360.0)
    order = np.argsort(x_old)
    x_sorted = x_old[order]
    moved = np.moveaxis(values, axis, 0)[order]
    x_ext = np.concatenate([x_sorted - 360.0, x_sorted, x_sorted + 360.0])
    data_ext = np.concatenate([moved, moved, moved], axis=0)
    flat = data_ext.reshape(data_ext.shape[0], -1)
    out = np.empty((x_new.size, flat.shape[1]), dtype=values.dtype)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(x_new, x_ext, flat[:, i])
    out = out.reshape((x_new.size,) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def _interp_wrap_angle_phase(phase, x_old_deg, x_new_deg, axis: int) -> np.ndarray:
    real = _interp_wrap_angle(np.cos(phase), x_old_deg, x_new_deg, axis=axis)
    imag = _interp_wrap_angle(np.sin(phase), x_old_deg, x_new_deg, axis=axis)
    return np.arctan2(imag, real)


# ============================================================================
# Phase 1 -- bulk extrusion expansion
# ============================================================================

@dataclass
class ExpansionMeta:
    length: float
    extrusion_axis: str
    input_domain: str
    broadside_azimuth_deg: float
    output_angle_range: str


def expand_2d_to_3d(
    grid_2d: RcsGrid,
    length: float,
    *,
    azimuths_deg: Optional[Sequence[float]] = None,
    elevations_deg: Optional[Sequence[float]] = None,
    frequencies: Optional[Sequence[float]] = None,
    polarizations: Optional[Sequence] = None,
    extrusion_axis: str = "y",
    input_domain: str = "scattering_width",
    broadside_azimuth_deg: float = 0.0,
    output_angle_range: str = "signed",
    preserve_phase: bool = True,
) -> RcsGrid:
    """Expand a 2D grid to 3D assuming a body extruded by ``length`` along Y.

        sigma_3D(az, el, f) = (2 L^2 / lambda)
                             * sinc^2(k L sin(az - az_broadside))
                             * sigma_2D(el mod 360, f)
    """
    elev_in = np.asarray(grid_2d.elevations, dtype=float)
    if elev_in.size != 1:
        raise ValueError(
            f"expand_2d_to_3d expects a 2D grid (single elevation); got {elev_in.size}."
        )
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError(f"length must be finite and positive, got {length!r}")
    if input_domain not in ("scattering_width", "rcs"):
        raise ValueError("input_domain must be 'scattering_width' or 'rcs'")
    if output_angle_range not in ("signed", "positive"):
        raise ValueError("output_angle_range must be 'signed' or 'positive'")

    az_2d_in = np.asarray(grid_2d.azimuths, dtype=float)
    f_in = np.asarray(grid_2d.frequencies, dtype=float)
    pol_in = np.asarray(grid_2d.polarizations)

    if azimuths_deg is not None:
        az_out = np.asarray(list(azimuths_deg), dtype=float)
    elif output_angle_range == "signed":
        az_out = np.arange(-180.0, 180.0, 1.0)
    else:
        az_out = np.arange(0.0, 360.0, 1.0)
    az_out = _normalize_to_range(az_out, output_angle_range)

    if elevations_deg is not None:
        el_out = np.asarray(list(elevations_deg), dtype=float)
    else:
        el_out = _normalize_to_range(az_2d_in, output_angle_range)
    el_out = _normalize_to_range(el_out, output_angle_range)

    f_out = np.asarray(list(frequencies), dtype=float) if frequencies is not None else f_in

    pol_in_str = np.asarray([str(p) for p in pol_in])
    pol_labels = [str(p) for p in polarizations] if polarizations is not None else list(pol_in_str)
    pol_out = np.asarray(pol_labels, dtype="<U")
    pol_idx = []
    for p in pol_out:
        matches = np.where(pol_in_str == str(p))[0]
        if matches.size == 0:
            raise ValueError(f"requested polarization {p!r} not present in 2D grid")
        pol_idx.append(int(matches[0]))

    power_in = np.asarray(grid_2d.rcs_power, dtype=np.float64)[:, 0, :, :][:, :, pol_idx]
    phase_in = np.asarray(grid_2d.rcs_phase, dtype=np.float64)[:, 0, :, :][:, :, pol_idx]

    power_el = _interp_wrap_angle(power_in, az_2d_in, el_out, axis=0)
    phase_el = _interp_wrap_angle_phase(phase_in, az_2d_in, el_out, axis=0)
    power_elf = _interp_clip_axis(power_el, f_in, f_out, axis=1)
    phase_elf = _interp_clip_axis(phase_el, f_in, f_out, axis=1)

    freq_unit = (grid_2d.units or {}).get("frequency", "GHz")
    lam = _wavelength_m(f_out, freq_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(lam > 0.0, 2.0 * np.pi / lam, np.nan)

    sin_az = np.sin(np.deg2rad(az_out - float(broadside_azimuth_deg)))
    kL_sinaz = np.outer(sin_az, k * length)
    sinc_vals = _sinc(kL_sinaz)
    sinc2_vals = sinc_vals * sinc_vals

    if input_domain == "scattering_width":
        with np.errstate(divide="ignore", invalid="ignore"):
            two_L2_over_lam = np.where(lam > 0.0, (2.0 * length ** 2) / lam, np.nan)
        multiplier = two_L2_over_lam[np.newaxis, :] * sinc2_vals
    else:
        multiplier = sinc2_vals

    power_out = multiplier[:, np.newaxis, :, np.newaxis] * power_elf[np.newaxis, :, :, :]

    if preserve_phase:
        sign_phase = np.where(sinc_vals >= 0.0, 0.0, np.pi)
        phase_out = (
            phase_elf[np.newaxis, :, :, :]
            + sign_phase[:, np.newaxis, :, np.newaxis]
        ).astype(np.float32)
    else:
        phase_out = np.full(power_out.shape, np.nan, dtype=np.float32)

    az_order = np.argsort(az_out)
    el_order = np.argsort(el_out)
    az_out = az_out[az_order]
    el_out = el_out[el_order]
    power_out = power_out[az_order][:, el_order]
    phase_out = phase_out[az_order][:, el_order]

    new_units = dict(grid_2d.units or {})
    new_units.setdefault("azimuth", "deg")
    new_units["elevation"] = "deg"
    tag = (
        f"2D->3D extrusion | L={length:g} m | axis={extrusion_axis} | "
        f"broadside_az={broadside_azimuth_deg} deg | input_domain={input_domain} | "
        f"range={output_angle_range}"
    )
    history = f"{grid_2d.history} | {tag}" if grid_2d.history else tag

    return RcsGrid(
        azimuths=az_out,
        elevations=el_out,
        frequencies=f_out,
        polarizations=pol_out,
        rcs_power=power_out.astype(np.float32),
        rcs_phase=phase_out,
        rcs_domain="power_phase",
        source_path=grid_2d.source_path,
        history=history,
        units=new_units,
    )


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
    """Single-point convenience wrapper for expand_2d_to_3d."""
    sub = expand_2d_to_3d(
        grid_2d, length,
        azimuths_deg=[azimuth_deg],
        elevations_deg=[elevation_deg],
        frequencies=[frequency],
        polarizations=[polarization],
        preserve_phase=True,
        **kwargs,
    )
    p = float(sub.rcs_power[0, 0, 0, 0])
    ph = float(sub.rcs_phase[0, 0, 0, 0])
    if return_complex:
        if not np.isfinite(p):
            return complex("nan")
        if not np.isfinite(ph):
            ph = 0.0
        return complex(np.sqrt(p) * np.exp(1j * ph))
    return {
        "rcs_linear_m2": p,
        "phase_rad": ph,
        "rcs_dbsm": 10.0 * np.log10(max(p, 1e-12)),
    }


# ============================================================================
# Phase 2 -- mesh I/O (pure numpy STL reader)
# ============================================================================

@dataclass
class MeshContext:
    vertices: np.ndarray        # (V, 3) float
    faces: np.ndarray           # (M, 3) int indices into vertices
    face_normals: Optional[np.ndarray] = None

    @property
    def tris(self) -> np.ndarray:
        return self.vertices[self.faces]

    @property
    def centroids(self) -> np.ndarray:
        return self.tris.mean(axis=1)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def _stl_is_binary(path: str) -> bool:
    size = os.path.getsize(path)
    if size < 84:
        return False
    with open(path, "rb") as f:
        f.seek(80)
        (n,) = struct.unpack("<I", f.read(4))
    return size == 84 + n * 50


def _load_stl_binary(path: str) -> MeshContext:
    dt = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    with open(path, "rb") as f:
        f.seek(80)
        (n,) = struct.unpack("<I", f.read(4))
        data = np.frombuffer(f.read(n * 50), dtype=dt, count=n)
    # Interleave vertices as (t0_v0, t0_v1, t0_v2, t1_v0, ...) so faces=arange(3n) works
    vs = np.stack([data["v0"], data["v1"], data["v2"]], axis=1)      # (n, 3, 3)
    verts = vs.reshape(-1, 3).astype(np.float64)                     # (3n, 3)
    faces = np.arange(3 * n, dtype=np.int64).reshape(n, 3)
    normals = np.asarray(data["normal"], dtype=np.float64)
    return MeshContext(vertices=verts, faces=faces, face_normals=normals)


def _load_stl_ascii(path: str) -> MeshContext:
    verts: list[list[float]] = []
    normals: list[list[float]] = []
    cur_normal = None
    with open(path, "r") as f:
        for raw in f:
            t = raw.strip().split()
            if not t:
                continue
            if t[0].lower() == "facet" and len(t) >= 5 and t[1].lower() == "normal":
                cur_normal = [float(t[2]), float(t[3]), float(t[4])]
            elif t[0].lower() == "vertex" and len(t) >= 4:
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0].lower() == "endfacet" and cur_normal is not None:
                normals.append(cur_normal)
                cur_normal = None
    V = np.asarray(verts, dtype=np.float64)
    if V.shape[0] == 0 or V.shape[0] % 3 != 0:
        raise ValueError(f"ASCII STL parse produced {V.shape[0]} vertices; expected multiple of 3")
    n_tri = V.shape[0] // 3
    faces = np.arange(3 * n_tri, dtype=np.int64).reshape(n_tri, 3)
    N = np.asarray(normals, dtype=np.float64) if normals else None
    return MeshContext(vertices=V, faces=faces, face_normals=N)


def load_stl(path: str) -> MeshContext:
    """Load a binary or ASCII STL into a MeshContext (pure numpy)."""
    if _stl_is_binary(path):
        return _load_stl_binary(path)
    return _load_stl_ascii(path)


def save_stl_binary(path: str, vertices: np.ndarray, faces: np.ndarray,
                    normals: Optional[np.ndarray] = None) -> str:
    """Write a binary STL from (V,3) vertices and (M,3) face indices."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if normals is None:
        v = vertices[faces]
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        n = np.cross(e1, e2)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(nn > 0, n / nn, 0.0)
        normals = n.astype(np.float32)
    else:
        normals = np.asarray(normals, dtype=np.float32)
    n_tri = faces.shape[0]
    with open(path, "wb") as f:
        f.write(b" " * 80)
        f.write(struct.pack("<I", n_tri))
        for i in range(n_tri):
            f.write(normals[i].tobytes())
            for j in range(3):
                f.write(vertices[faces[i, j]].tobytes())
            f.write(b"\x00\x00")
    return path


# ============================================================================
# Phase 2 -- vectorized Moller-Trumbore ray/triangle intersection
# ============================================================================

class RayMeshIntersector:
    """Vectorized any-hit ray caster using the Moller-Trumbore algorithm."""

    def __init__(self, mesh: MeshContext):
        tris = mesh.tris
        self.v0 = tris[:, 0, :].astype(np.float64)
        self.e1 = (tris[:, 1, :] - tris[:, 0, :]).astype(np.float64)
        self.e2 = (tris[:, 2, :] - tris[:, 0, :]).astype(np.float64)
        self.n_tri = self.v0.shape[0]

    def any_hit(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        t_min: float = 1.0e-6,
        t_max: float = np.inf,
        ray_chunk: int = 2048,
        parallel_eps: float = 1.0e-12,
    ) -> np.ndarray:
        """Return (N,) bool: True if ray i hits any triangle at t in (t_min, t_max)."""
        origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        if origins.shape != directions.shape:
            raise ValueError("origins and directions must have the same shape")
        n = origins.shape[0]
        hits = np.zeros(n, dtype=bool)
        for i in range(0, n, ray_chunk):
            sl = slice(i, min(n, i + ray_chunk))
            hits[sl] = self._chunk(origins[sl], directions[sl],
                                   t_min=t_min, t_max=t_max,
                                   parallel_eps=parallel_eps)
        return hits

    def _chunk(self, O, D, *, t_min, t_max, parallel_eps) -> np.ndarray:
        O_ = O[:, None, :]
        D_ = D[:, None, :]
        v0 = self.v0[None, :, :]
        e1 = self.e1[None, :, :]
        e2 = self.e2[None, :, :]
        h = np.cross(D_, e2)
        a = np.einsum("nmi,nmi->nm", e1, h)
        parallel = np.abs(a) < parallel_eps
        a_safe = np.where(parallel, 1.0, a)
        f = 1.0 / a_safe
        s = O_ - v0
        u = f * np.einsum("nmi,nmi->nm", s, h)
        q = np.cross(s, e1)
        v = f * np.einsum("nmi,nmi->nm", D_, q)
        t = f * np.einsum("nmi,nmi->nm", e2, q)
        valid = (
            (~parallel)
            & (u >= 0.0) & (u <= 1.0)
            & (v >= 0.0) & ((u + v) <= 1.0)
            & (t > t_min) & (t < t_max)
        )
        return valid.any(axis=1)


# ============================================================================
# Phase 2 -- centerline / local frames
# ============================================================================

@dataclass
class Centerline:
    positions: np.ndarray          # (K, 3)
    tangents: np.ndarray           # (K, 3) unit
    segment_lengths: np.ndarray    # (K,)


def build_centerline(placement_xyz, segment_mode: str = "midpoint") -> Centerline:
    xyz = np.asarray(placement_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"placement_xyz must be shape (K, 3); got {xyz.shape}")
    K = xyz.shape[0]
    if K < 2:
        raise ValueError("centerline needs at least 2 points")

    seg_vecs = np.diff(xyz, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    if np.any(seg_lens <= 0.0):
        raise ValueError("placement_xyz has coincident consecutive points")
    seg_tangents = seg_vecs / seg_lens[:, None]

    tangents = np.zeros_like(xyz)
    tangents[0] = seg_tangents[0]
    tangents[-1] = seg_tangents[-1]
    if K > 2:
        avg = seg_tangents[:-1] + seg_tangents[1:]
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        tangents[1:-1] = avg / norms

    if segment_mode == "midpoint":
        dL = np.zeros(K, dtype=np.float64)
        dL[0] = seg_lens[0] * 0.5
        dL[-1] = seg_lens[-1] * 0.5
        if K > 2:
            dL[1:-1] = 0.5 * (seg_lens[:-1] + seg_lens[1:])
    elif segment_mode == "forward":
        dL = np.empty(K, dtype=np.float64)
        dL[:-1] = seg_lens
        dL[-1] = seg_lens[-1]
    else:
        raise ValueError("segment_mode must be 'midpoint' or 'forward'")

    return Centerline(positions=xyz, tangents=tangents, segment_lengths=dL)


def _incident_unit_vectors(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Unit vectors from target to radar. az CCW from +X in XY; el from XY plane up.

    d = (cos(el) cos(az), cos(el) sin(az), sin(el))
    """
    a = np.deg2rad(np.asarray(az_deg, dtype=float))
    e = np.deg2rad(np.asarray(el_deg, dtype=float))
    return np.stack(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
        axis=-1,
    )


def _local_cs_frames(tangents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each tangent, build (e_east, z_up) forming a right-handed frame with t.

    Canonical: t || +Y  ->  (e_east, z_up) = (+X, +Z).
    Fallback:  if t || +Z, use global +Y to seed instead.
    """
    t = np.asarray(tangents, dtype=np.float64)
    z_global = np.array([0.0, 0.0, 1.0])
    y_global = np.array([0.0, 1.0, 0.0])

    dotz = t @ z_global
    z_proj = z_global[None, :] - dotz[:, None] * t
    nrm = np.linalg.norm(z_proj, axis=1)
    bad = nrm < 1.0e-6

    if np.any(bad):
        doty = t[bad] @ y_global
        y_proj = y_global[None, :] - doty[:, None] * t[bad]
        z_proj[bad] = y_proj
        nrm[bad] = np.linalg.norm(y_proj, axis=1)

    z_local = z_proj / np.where(nrm[:, None] > 0, nrm[:, None], 1.0)
    e_local = np.cross(t, z_local)
    e_nrm = np.linalg.norm(e_local, axis=1, keepdims=True)
    e_local = e_local / np.where(e_nrm > 0, e_nrm, 1.0)
    return e_local, z_local


# ============================================================================
# Phase 2 -- swept expansion along a centerline with shadowing
# ============================================================================

def expand_2d_along_centerline(
    grid_2d: RcsGrid,
    placement_xyz,
    *,
    mesh: Optional[MeshContext] = None,
    azimuths_deg: Optional[Sequence[float]] = None,
    elevations_deg: Optional[Sequence[float]] = None,
    frequencies: Optional[Sequence[float]] = None,
    polarizations: Optional[Sequence] = None,
    coherent: bool = True,
    include_shadowing: bool = True,
    shadow_eps: float = 1.0e-4,
    segment_mode: str = "midpoint",
    output_angle_range: str = "signed",
    ray_chunk: int = 2048,
) -> tuple[RcsGrid, dict]:
    """Apply the 2D RCS template along a centerline inside an STL hull.

    Returns (grid_3d, diagnostics). ``diagnostics`` includes a shadow mask of
    shape (n_az, n_el, K).
    """
    elev_in = np.asarray(grid_2d.elevations, dtype=float)
    if elev_in.size != 1:
        raise ValueError("expand_2d_along_centerline expects a 2D grid (single elevation)")
    if output_angle_range not in ("signed", "positive"):
        raise ValueError("output_angle_range must be 'signed' or 'positive'")

    centerline = build_centerline(placement_xyz, segment_mode=segment_mode)
    K = centerline.positions.shape[0]

    az_2d_in = np.asarray(grid_2d.azimuths, dtype=float)
    f_in = np.asarray(grid_2d.frequencies, dtype=float)
    pol_in = np.asarray(grid_2d.polarizations)

    if azimuths_deg is not None:
        az_out = np.asarray(list(azimuths_deg), dtype=float)
    elif output_angle_range == "signed":
        az_out = np.arange(-180.0, 180.0, 1.0)
    else:
        az_out = np.arange(0.0, 360.0, 1.0)
    az_out = _normalize_to_range(az_out, output_angle_range)

    if elevations_deg is not None:
        el_out = np.asarray(list(elevations_deg), dtype=float)
    else:
        el_out = _normalize_to_range(az_2d_in, output_angle_range)
    el_out = _normalize_to_range(el_out, output_angle_range)

    f_out = np.asarray(list(frequencies), dtype=float) if frequencies is not None else f_in
    pol_in_str = np.asarray([str(p) for p in pol_in])
    pol_labels = [str(p) for p in polarizations] if polarizations is not None else list(pol_in_str)
    pol_out = np.asarray(pol_labels, dtype="<U")
    pol_idx = []
    for p in pol_out:
        matches = np.where(pol_in_str == str(p))[0]
        if matches.size == 0:
            raise ValueError(f"requested polarization {p!r} not present in 2D grid")
        pol_idx.append(int(matches[0]))

    az_out.sort()
    el_out.sort()
    n_az, n_el, n_f, n_pol = az_out.size, el_out.size, f_out.size, pol_out.size

    AZ, EL = np.meshgrid(az_out, el_out, indexing="ij")
    d_hat = _incident_unit_vectors(AZ.ravel(), EL.ravel())   # (n_dir, 3)

    t_hat = centerline.tangents
    e_loc, z_loc = _local_cs_frames(t_hat)
    sin_alpha = np.clip(d_hat @ t_hat.T, -1.0, 1.0)          # (n_dir, K)

    d_perp = d_hat[:, None, :] - sin_alpha[:, :, None] * t_hat[None, :, :]
    phi_e = np.einsum("kj,dkj->dk", e_loc, d_perp)
    phi_z = np.einsum("kj,dkj->dk", z_loc, d_perp)
    phi = np.rad2deg(np.arctan2(phi_z, phi_e))
    phi_mod = np.mod(phi, 360.0)

    n_dir = d_hat.shape[0]
    shadowed = np.zeros((n_dir, K), dtype=bool)
    if include_shadowing and mesh is not None and mesh.faces.shape[0] > 0:
        rmi = RayMeshIntersector(mesh)
        origins = (centerline.positions[None, :, :]
                   + shadow_eps * d_hat[:, None, :]).reshape(-1, 3)
        dirs = np.broadcast_to(d_hat[:, None, :], (n_dir, K, 3)).reshape(-1, 3)
        hit = rmi.any_hit(origins, dirs, t_min=shadow_eps, ray_chunk=ray_chunk)
        shadowed = hit.reshape(n_dir, K)

    freq_unit = (grid_2d.units or {}).get("frequency", "GHz")
    lam = _wavelength_m(f_out, freq_unit)
    with np.errstate(divide="ignore", invalid="ignore"):
        kw = np.where(lam > 0.0, 2.0 * np.pi / lam, np.nan)
    sqrt_two_over_lam = np.where(lam > 0.0, np.sqrt(2.0 / lam), np.nan)

    power_in = np.asarray(grid_2d.rcs_power, dtype=np.float64)[:, 0, :, :][:, :, pol_idx]
    phase_in = np.asarray(grid_2d.rcs_phase, dtype=np.float64)[:, 0, :, :][:, :, pol_idx]
    power_in_f = _interp_clip_axis(power_in, f_in, f_out, axis=1)
    phase_in_f = _interp_clip_axis(phase_in, f_in, f_out, axis=1)

    power_out = np.zeros((n_az, n_el, n_f, n_pol), dtype=np.float64)

    for fi in range(n_f):
        kw_fi = kw[fi]
        sq2ol = sqrt_two_over_lam[fi]

        P_fi = power_in_f[:, fi, :]
        A_fi = phase_in_f[:, fi, :]

        phi_flat = phi_mod.ravel()
        P_at_phi = _interp_wrap_angle(P_fi, az_2d_in, phi_flat, axis=0).reshape(n_dir, K, n_pol)
        A_at_phi = _interp_wrap_angle_phase(A_fi, az_2d_in, phi_flat, axis=0).reshape(n_dir, K, n_pol)

        dL = centerline.segment_lengths
        sinc_k = _sinc(kw_fi * dL[None, :] * sin_alpha)
        d_dot_r = d_hat @ centerline.positions.T
        range_phase = -2.0 * kw_fi * d_dot_r

        amp_rk = sq2ol * dL[None, :] * sinc_k
        amp_rk = np.where(shadowed, 0.0, amp_rk)

        P_safe = np.where(np.isfinite(P_at_phi) & (P_at_phi >= 0.0), P_at_phi, 0.0)
        sqrt_sigma = np.sqrt(P_safe)

        sign_phase = np.where(sinc_k >= 0.0, 0.0, np.pi)
        amp = amp_rk[:, :, None] * sqrt_sigma
        phase = A_at_phi + (range_phase + sign_phase)[:, :, None]
        E = amp * np.exp(1j * phase)

        if coherent:
            E_sum = E.sum(axis=1)
            sigma_dir = (np.abs(E_sum) ** 2).real
        else:
            sigma_dir = (np.abs(E) ** 2).sum(axis=1).real

        power_out[..., fi, :] = sigma_dir.reshape(n_az, n_el, n_pol)

    phase_out = np.full(power_out.shape, np.nan, dtype=np.float32)

    new_units = dict(grid_2d.units or {})
    new_units.setdefault("azimuth", "deg")
    new_units["elevation"] = "deg"
    tag = (
        f"2D->3D swept | K={K} points | coherent={coherent} | "
        f"shadowing={include_shadowing and mesh is not None} | "
        f"segment_mode={segment_mode} | range={output_angle_range}"
    )
    history = f"{grid_2d.history} | {tag}" if grid_2d.history else tag

    grid_3d = RcsGrid(
        azimuths=az_out,
        elevations=el_out,
        frequencies=f_out,
        polarizations=pol_out,
        rcs_power=power_out.astype(np.float32),
        rcs_phase=phase_out,
        rcs_domain="power_phase",
        source_path=grid_2d.source_path,
        history=history,
        units=new_units,
    )
    diagnostics = {
        "shadowed_mask": shadowed.reshape(n_az, n_el, K),
        "centerline": centerline,
        "n_directions": n_dir,
    }
    return grid_3d, diagnostics


__all__ = [
    "ExpansionMeta", "expand_2d_to_3d", "rcs_3d_at",
    "MeshContext", "Centerline", "RayMeshIntersector",
    "load_stl", "save_stl_binary",
    "build_centerline", "expand_2d_along_centerline",
]


# ============================================================================
# Script entry -- edit and run
# ============================================================================

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # PHASE 1 -- simple extrusion
    # -----------------------------------------------------------------------
    P1_INPUT_PATH   = "input_2d.grim"
    P1_OUTPUT_PATH  = "output_3d_phase1.grim"
    P1_LENGTH_M     = 1.5
    P1_BROADSIDE_AZ = 0.0
    P1_AZIMUTHS     = np.arange(-180.0, 180.0, 2.0)
    P1_ELEVATIONS   = np.arange(-180.0, 180.0, 5.0)
    P1_FREQUENCIES  = None
    P1_POLARIZATIONS = None
    P1_INPUT_DOMAIN = "scattering_width"
    P1_PRESERVE_PHASE = True
    RUN_PHASE_1 = True

    # -----------------------------------------------------------------------
    # PHASE 2 -- centerline + STL hull + shadowing
    # -----------------------------------------------------------------------
    P2_INPUT_PATH   = "input_2d.grim"
    P2_STL_PATH     = "hull.stl"
    P2_OUTPUT_PATH  = "output_3d_phase2.grim"
    P2_PLACEMENT_XYZ = np.stack(
        [np.zeros(21), np.linspace(-0.75, 0.75, 21), np.zeros(21)], axis=1
    )
    P2_AZIMUTHS     = np.arange(-180.0, 180.0, 5.0)
    P2_ELEVATIONS   = np.arange(-180.0, 180.0, 10.0)
    P2_FREQUENCIES  = None
    P2_POLARIZATIONS = None
    P2_COHERENT     = True
    P2_SHADOWING    = True
    RUN_PHASE_2 = False

    if RUN_PHASE_1:
        print("=" * 60)
        print("PHASE 1 -- bulk extrusion")
        print("=" * 60)
        grid_2d = RcsGrid.load(P1_INPUT_PATH)
        print(f"Loaded: {P1_INPUT_PATH}")
        print(f"  az(as-el): {len(grid_2d.azimuths)} values")
        print(f"  freqs: {grid_2d.frequencies} "
              f"({(grid_2d.units or {}).get('frequency', 'GHz')})")
        print(f"  pols : {list(grid_2d.polarizations)}")
        grid_3d = expand_2d_to_3d(
            grid_2d,
            length=P1_LENGTH_M,
            azimuths_deg=P1_AZIMUTHS,
            elevations_deg=P1_ELEVATIONS,
            frequencies=P1_FREQUENCIES,
            polarizations=P1_POLARIZATIONS,
            broadside_azimuth_deg=P1_BROADSIDE_AZ,
            input_domain=P1_INPUT_DOMAIN,
            preserve_phase=P1_PRESERVE_PHASE,
            output_angle_range="signed",
        )
        print(f"Expanded: az={len(grid_3d.azimuths)} x el={len(grid_3d.elevations)} "
              f"x f={len(grid_3d.frequencies)} x pol={len(grid_3d.polarizations)}")
        if P1_OUTPUT_PATH:
            print(f"Saved: {grid_3d.save(P1_OUTPUT_PATH)}")

    if RUN_PHASE_2:
        print("=" * 60)
        print("PHASE 2 -- swept expansion with shadowing")
        print("=" * 60)
        grid_2d = RcsGrid.load(P2_INPUT_PATH)
        mesh = load_stl(P2_STL_PATH)
        print(f"Loaded STL: {P2_STL_PATH} ({mesh.faces.shape[0]} triangles)")
        print(f"Centerline: {P2_PLACEMENT_XYZ.shape[0]} points")
        grid_3d, diag = expand_2d_along_centerline(
            grid_2d,
            placement_xyz=P2_PLACEMENT_XYZ,
            mesh=mesh,
            azimuths_deg=P2_AZIMUTHS,
            elevations_deg=P2_ELEVATIONS,
            frequencies=P2_FREQUENCIES,
            polarizations=P2_POLARIZATIONS,
            coherent=P2_COHERENT,
            include_shadowing=P2_SHADOWING,
            output_angle_range="signed",
        )
        sh = diag["shadowed_mask"]
        print(f"Expanded: az={len(grid_3d.azimuths)} x el={len(grid_3d.elevations)} "
              f"x f={len(grid_3d.frequencies)} x pol={len(grid_3d.polarizations)}")
        print(f"Shadowing: {sh.mean()*100:.1f}% of (direction, point) samples shadowed")
        if P2_OUTPUT_PATH:
            print(f"Saved: {grid_3d.save(P2_OUTPUT_PATH)}")
