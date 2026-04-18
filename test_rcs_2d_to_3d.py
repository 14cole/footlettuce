"""Tests for Phase 1 (new convention) + Phase 2 (STL + shadow + swept)."""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/home/claude")
shutil.copy("/mnt/user-data/uploads/grim_dataset.py", "/home/claude/grim_dataset.py")

from grim_dataset import RcsGrid, C0
from rcs_2d_to_3d import (
    expand_2d_to_3d, rcs_3d_at,
    load_stl, save_stl_binary, MeshContext, RayMeshIntersector,
    build_centerline, expand_2d_along_centerline,
    _normalize_to_range, _sinc, _incident_unit_vectors, _local_cs_frames,
)


def approx(a, b, rel=1e-5, abs_=0.0):
    return abs(a - b) <= abs_ + rel * max(1.0, abs(b))


def make_iso_2d(sigma=0.5, freqs_ghz=(10.0,), n_az=361, pols=("VV",)):
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el = np.asarray([0.0])
    f = np.asarray(freqs_ghz, dtype=float)
    p = np.asarray(pols)
    shape = (len(az), 1, len(f), len(p))
    return RcsGrid(
        azimuths=az, elevations=el, frequencies=f, polarizations=p,
        rcs_power=np.full(shape, sigma, dtype=np.float32),
        rcs_phase=np.zeros(shape, dtype=np.float32),
        rcs_domain="power_phase",
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
    )


passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {name}  {detail}")
    if ok:
        passed += 1
    else:
        failed += 1


print("=" * 72)
print("Phase 1 -- new convention (sinc^2 in AZIMUTH, sigma_2D indexed by EL)")
print("=" * 72)

# -----------------------------------------------------------------------
# P1.1: broadside is now at az = 0 (east), not el = 0
# -----------------------------------------------------------------------
sigma = 0.5
L = 1.2
f_ghz = 10.0
lam = C0 / (f_ghz * 1e9)
expected_broadside = (2.0 * L ** 2 / lam) * sigma

g2d = make_iso_2d(sigma=sigma, freqs_ghz=(f_ghz,))
g3d = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0, 45.0, 90.0, -90.0, 180.0, -180.0],
    elevations_deg=[0.0, 30.0, 90.0, 180.0, -90.0],
    frequencies=[f_ghz], polarizations=["VV"],
)
# at az = 0 (east, broadside), any el  -> broadside value
for ei, el in enumerate([0.0, 30.0, 90.0, 180.0, -90.0]):
    # find az=0 index
    ai = int(np.where(np.isclose(g3d.azimuths, 0.0))[0][0])
    got = float(g3d.rcs_power[ai, ei, 0, 0])
    check(f"broadside @ az=0 el={el}", approx(got, expected_broadside, rel=1e-5),
          f"got={got:.4e} exp={expected_broadside:.4e}")

# at az = ±180 (west), also broadside (sin(180)=0, sinc=1)
for az_test in [180.0, -180.0]:
    matches = np.where(np.isclose(g3d.azimuths, az_test))[0]
    if matches.size == 0:
        # -180 gets normalized to -180 (kept) or 180 gets normalized to -180
        continue
    ai = int(matches[0])
    got = float(g3d.rcs_power[ai, 0, 0, 0])
    check(f"broadside @ az={az_test} (west side)", approx(got, expected_broadside, rel=1e-5))

# -----------------------------------------------------------------------
# P1.2: first sinc^2 null is at az = arcsin(lambda/L), near east
# -----------------------------------------------------------------------
el_null_deg = np.rad2deg(np.arcsin(lam / L))
g3d_null = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0, el_null_deg, 0.5 * el_null_deg],
    elevations_deg=[0.0],
    frequencies=[f_ghz], polarizations=["VV"],
)
def idx_of(arr, val):
    return int(np.argmin(np.abs(np.asarray(arr) - val)))
i0 = idx_of(g3d_null.azimuths, 0.0)
in2 = idx_of(g3d_null.azimuths, el_null_deg)
ihalf = idx_of(g3d_null.azimuths, 0.5 * el_null_deg)
check("first sinc^2 null ~ 0",
      g3d_null.rcs_power[in2, 0, 0, 0] < 1e-6 * g3d_null.rcs_power[i0, 0, 0, 0],
      f"null={g3d_null.rcs_power[in2,0,0,0]:.2e} broad={g3d_null.rcs_power[i0,0,0,0]:.2e}")
check("half-way < broadside > null ordering",
      g3d_null.rcs_power[i0, 0, 0, 0] > g3d_null.rcs_power[ihalf, 0, 0, 0]
      > g3d_null.rcs_power[in2, 0, 0, 0])

# -----------------------------------------------------------------------
# P1.3: azimuth symmetry (sinc^2 is even in sin, symmetric about 0 and 180)
# -----------------------------------------------------------------------
g3d_sym = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[-30.0, 30.0, 150.0, 210.0],  # 210 wraps to -150
    elevations_deg=[0.0],
    frequencies=[f_ghz], polarizations=["VV"],
)
p_pos30 = float(g3d_sym.rcs_power[idx_of(g3d_sym.azimuths, 30.0), 0, 0, 0])
p_neg30 = float(g3d_sym.rcs_power[idx_of(g3d_sym.azimuths, -30.0), 0, 0, 0])
check("sigma(30) == sigma(-30)  (sinc^2 even)", approx(p_pos30, p_neg30, rel=1e-6),
      f"{p_pos30:.4e} vs {p_neg30:.4e}")
# 150 and 30 have the same sin -> same sinc^2
p_150 = float(g3d_sym.rcs_power[idx_of(g3d_sym.azimuths, 150.0), 0, 0, 0])
check("sigma(30) == sigma(150)  (sin(x)=sin(180-x))", approx(p_pos30, p_150, rel=1e-6))

# -----------------------------------------------------------------------
# P1.4: elevation wrap (indexing the 2D axis)
# -----------------------------------------------------------------------
# Non-axisymmetric source in the "2D azimuth" axis = elevation axis
az = np.linspace(0.0, 360.0, 361, endpoint=False)
pattern = 1.0 + 0.5 * np.cos(np.deg2rad(az))   # max at az=0, min at az=180
power = pattern.astype(np.float32)[:, None, None, None]
phase = np.zeros_like(power)
g2d_asym = RcsGrid(
    azimuths=az, elevations=np.asarray([0.0]),
    frequencies=np.asarray([f_ghz]), polarizations=np.asarray(["VV"]),
    rcs_power=power, rcs_phase=phase, rcs_domain="power_phase",
    units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
)
g3d_asym = expand_2d_to_3d(
    g2d_asym, length=L,
    azimuths_deg=[0.0],
    elevations_deg=[-5.0, 355.0, 90.0],
    frequencies=[f_ghz], polarizations=["VV"],
)
p_neg5 = float(g3d_asym.rcs_power[0, idx_of(g3d_asym.elevations, -5.0), 0, 0])
# 355 -> normalized to -5 signed -> same slot as -5; test via separate interp
g3d_asym2 = expand_2d_to_3d(
    g2d_asym, length=L,
    azimuths_deg=[0.0], elevations_deg=[355.0],
    frequencies=[f_ghz], polarizations=["VV"],
    output_angle_range="positive",
)
p_355 = float(g3d_asym2.rcs_power[0, 0, 0, 0])
check("elevation wrap: sigma(-5 signed) == sigma(355 positive)",
      approx(p_neg5, p_355, rel=1e-5),
      f"{p_neg5:.4e} vs {p_355:.4e}")

# -----------------------------------------------------------------------
# P1.5: output range honored
# -----------------------------------------------------------------------
g3d_signed = expand_2d_to_3d(g2d, length=L, frequencies=[f_ghz], output_angle_range="signed")
check("signed output azimuths in [-180, 180)",
      (g3d_signed.azimuths.min() >= -180.0) and (g3d_signed.azimuths.max() < 180.0))

g3d_pos = expand_2d_to_3d(g2d, length=L, frequencies=[f_ghz], output_angle_range="positive")
check("positive output azimuths in [0, 360)",
      (g3d_pos.azimuths.min() >= 0.0) and (g3d_pos.azimuths.max() < 360.0))

# -----------------------------------------------------------------------
# P1.6: roundtrip
# -----------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    out = os.path.join(td, "e.grim")
    g = expand_2d_to_3d(g2d, length=L, frequencies=[f_ghz])
    g.save(out)
    g2 = RcsGrid.load(out)
    check(".grim roundtrip identical",
          np.allclose(g.rcs_power, g2.rcs_power, equal_nan=True)
          and np.array_equal(g.azimuths, g2.azimuths)
          and np.array_equal(g.elevations, g2.elevations))

# -----------------------------------------------------------------------
# P1.7: _normalize_to_range helper
# -----------------------------------------------------------------------
check("normalize signed: 270 -> -90", approx(float(_normalize_to_range([270], "signed")[0]), -90.0))
check("normalize signed: -10 -> -10", approx(float(_normalize_to_range([-10], "signed")[0]), -10.0))
check("normalize signed: 180 -> -180 (wraps)", approx(float(_normalize_to_range([180], "signed")[0]), -180.0))
check("normalize positive: -10 -> 350", approx(float(_normalize_to_range([-10], "positive")[0]), 350.0))


print()
print("=" * 72)
print("Phase 2 -- STL + ray caster + centerline + swept expansion")
print("=" * 72)


# -----------------------------------------------------------------------
# P2.1: STL write + read roundtrip
# -----------------------------------------------------------------------
def make_box(cx=0, cy=0, cz=0, sx=0.2, sy=1.0, sz=0.2):
    """Return (V, F) for a closed rectangular box centered at (cx,cy,cz)."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    V = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=float)
    F = np.array([
        [0, 2, 1], [0, 3, 2],  # -Z bottom
        [4, 5, 6], [4, 6, 7],  # +Z top
        [0, 1, 5], [0, 5, 4],  # -Y front
        [2, 3, 7], [2, 7, 6],  # +Y back
        [1, 2, 6], [1, 6, 5],  # +X right
        [0, 4, 7], [0, 7, 3],  # -X left
    ], dtype=np.int64)
    return V, F


V, F = make_box()
with tempfile.TemporaryDirectory() as td:
    stl = os.path.join(td, "box.stl")
    save_stl_binary(stl, V, F)
    mesh = load_stl(stl)
    check("STL binary roundtrip vertex count",
          mesh.vertices.shape[0] == 3 * F.shape[0],
          f"(vertices replicated per-face): {mesh.vertices.shape[0]}")
    check("STL binary roundtrip face count", mesh.faces.shape[0] == F.shape[0])
    aabb_min, aabb_max = mesh.aabb
    check("STL AABB x-extent ~ 0.2",
          approx(float(aabb_max[0] - aabb_min[0]), 0.2, rel=1e-5))
    # Reloaded mesh should have the SAME triangle geometry as what we wrote.
    # Sanity: ray from origin through a face should hit exactly once.
    rmi_reloaded = RayMeshIntersector(mesh)
    hits_plus_x = rmi_reloaded.any_hit(np.array([[0.0, 0.0, 0.0]]),
                                        np.array([[1.0, 0.0, 0.0]]))
    check("reloaded STL: ray from interior hits a wall", bool(hits_plus_x[0]))
    # For a closed box, ANY direction from interior must hit. Pick a handful.
    dirs = np.array([[0.612, 0.612, 0.5],
                     [-0.7, 0.5, 0.5],
                     [0.0, 1.0, 0.0],
                     [0.1, 0.1, 0.99]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.zeros_like(dirs)
    hits = rmi_reloaded.any_hit(origins, dirs)
    check("reloaded STL: all sample rays from interior hit the closed box",
          bool(hits.all()), f"hits={hits.tolist()}")

# -----------------------------------------------------------------------
# P2.2: Ray caster any_hit against a box
# -----------------------------------------------------------------------
V, F = make_box(sx=0.4, sy=0.4, sz=0.4)
mesh = MeshContext(vertices=V, faces=F)
rmi = RayMeshIntersector(mesh)
# Ray from outside pointing at center -> hit
origins = np.array([[-10.0, 0.0, 0.0]])
dirs = np.array([[1.0, 0.0, 0.0]])
check("ray hits the box from outside", bool(rmi.any_hit(origins, dirs)[0]))
# Ray from outside pointing away from box -> miss
check("ray misses when pointing away",
      not bool(rmi.any_hit(np.array([[-10.0, 0.0, 0.0]]),
                           np.array([[-1.0, 0.0, 0.0]]))[0]))
# Ray from inside the box pointing any direction -> hits (box walls)
check("ray from interior hits a wall",
      bool(rmi.any_hit(np.array([[0.0, 0.0, 0.0]]),
                       np.array([[1.0, 0.0, 0.0]]))[0]))

# -----------------------------------------------------------------------
# P2.3: Centerline construction -- tangents, segment lengths
# -----------------------------------------------------------------------
pts = np.stack([np.zeros(5), np.linspace(0, 1, 5), np.zeros(5)], axis=1)  # along +Y
cl = build_centerline(pts, segment_mode="midpoint")
check("tangents align with +Y for Y-axis line",
      np.allclose(cl.tangents, np.tile([0, 1, 0], (5, 1))))
check("midpoint segment lengths sum to total length",
      approx(float(cl.segment_lengths.sum()), 1.0, rel=1e-6))
check("endpoint segments are half of interior",
      approx(float(cl.segment_lengths[0]), 0.5 * float(cl.segment_lengths[2]), rel=1e-6))

# -----------------------------------------------------------------------
# P2.4: Local CS frame: +Y tangent -> east=+X, up=+Z
# -----------------------------------------------------------------------
e, z = _local_cs_frames(np.array([[0.0, 1.0, 0.0]]))
check("t=+Y -> e_local = +X", np.allclose(e[0], [1, 0, 0]))
check("t=+Y -> z_local = +Z", np.allclose(z[0], [0, 0, 1]))

# -----------------------------------------------------------------------
# P2.5: Swept expansion matches Phase 1 in the dense-centerline limit
# -----------------------------------------------------------------------
# For a straight body along +Y with length L, dense placement should reproduce
# the (2 L^2 / lambda) sinc^2(k L sin az) * sigma_2D(el) pattern.
L_test = 1.0
K = 101                                         # dense centerline
placement = np.stack(
    [np.zeros(K), np.linspace(-L_test / 2, L_test / 2, K), np.zeros(K)],
    axis=1,
)
az_test = np.array([0.0, 10.0, 30.0])
el_test = np.array([0.0, 45.0])

g3d_p1 = expand_2d_to_3d(
    g2d, length=L_test,
    azimuths_deg=az_test, elevations_deg=el_test,
    frequencies=[f_ghz], polarizations=["VV"],
    output_angle_range="signed",
)
g3d_p2, diag = expand_2d_along_centerline(
    g2d,
    placement_xyz=placement,
    mesh=None,
    azimuths_deg=az_test,
    elevations_deg=el_test,
    frequencies=[f_ghz],
    polarizations=["VV"],
    coherent=True,
    include_shadowing=False,
    output_angle_range="signed",
)
# Both grids: axes sorted ascending. Compare directly.
ok = True
detail = []
for ai, az in enumerate(np.sort(az_test)):
    for ei, el in enumerate(np.sort(el_test)):
        v1 = float(g3d_p1.rcs_power[ai, ei, 0, 0])
        v2 = float(g3d_p2.rcs_power[ai, ei, 0, 0])
        if not approx(v1, v2, rel=5e-2):    # 5% tolerance (discretization error)
            ok = False
        detail.append(f"(az={az}, el={el}): P1={v1:.3e} P2={v2:.3e}")
check("swept (no shadow) ~= Phase1 extrusion for dense centerline", ok, "\n      " + "\n      ".join(detail))

# -----------------------------------------------------------------------
# P2.6: Shadowing halves the contribution when mesh blocks one half
# -----------------------------------------------------------------------
# Build a centerline along +Y split into two halves; put a blocker wall
# only on the +Y half, so when the radar is "east" (az=0), only the -Y
# half is visible to rays going to +X (wait, shadowing is along the line
# of sight -- we block rays going +X from the +Y half).
#
# Simplest geometry: centerline along Y from -1 to +1, 21 points; wall
# at x = 0.5 covering y in [0, 1]. Rays from points with y > 0 going +X
# hit the wall (shadowed); rays from points with y < 0 going +X miss.
K = 21
pts = np.stack([np.zeros(K), np.linspace(-1.0, 1.0, K), np.zeros(K)], axis=1)

# Wall at x=0.5: a single quad covering y in [0.05, 1.0] (avoid y=0 edge on centerline)
Vw = np.array([
    [0.5, 0.05, -1.0],
    [0.5, 1.0,  -1.0],
    [0.5, 1.0,   1.0],
    [0.5, 0.05,  1.0],
], dtype=float)
Fw = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
wall = MeshContext(vertices=Vw, faces=Fw)

# Request a single direction az=0, el=0 (toward +X from each point)
g3d_shadow, diag = expand_2d_along_centerline(
    g2d,
    placement_xyz=pts,
    mesh=wall,
    azimuths_deg=[0.0],
    elevations_deg=[0.0],
    frequencies=[f_ghz],
    polarizations=["VV"],
    coherent=False,           # incoherent so the math is easy to reason about
    include_shadowing=True,
    output_angle_range="signed",
)
sh = diag["shadowed_mask"][0, 0, :]             # (K,)
# Points with y > 0 should be shadowed (toward wall at x=0.5)
# Points with y <= 0 should not be
n_shadowed_pos_y = int(sh[pts[:, 1] > 0].sum())
n_unshadowed_neg_y = int((~sh[pts[:, 1] <= 0]).sum())
check("all +Y points are shadowed by the wall", n_shadowed_pos_y == int((pts[:, 1] > 0).sum()),
      f"shadowed={n_shadowed_pos_y}")
check("all -Y (and y=0) points are visible", n_unshadowed_neg_y == int((pts[:, 1] <= 0).sum()),
      f"visible={n_unshadowed_neg_y}")

# -----------------------------------------------------------------------
# P2.7: With no shadowing, same request gives >= power
# -----------------------------------------------------------------------
g3d_noshadow, _ = expand_2d_along_centerline(
    g2d, placement_xyz=pts, mesh=wall,
    azimuths_deg=[0.0], elevations_deg=[0.0],
    frequencies=[f_ghz], polarizations=["VV"],
    coherent=False, include_shadowing=False,
    output_angle_range="signed",
)
p_shadow = float(g3d_shadow.rcs_power[0, 0, 0, 0])
p_noshadow = float(g3d_noshadow.rcs_power[0, 0, 0, 0])
check("shadowed RCS < unshadowed RCS", p_shadow < p_noshadow,
      f"shadowed={p_shadow:.3e} vs unshadowed={p_noshadow:.3e}")
# And roughly half (symmetric centerline, symmetric wall from y=0 up)
# 11 points visible out of 21 (y <= 0), so ~= 11/21 of total
expected_ratio = 11.0 / 21.0
got_ratio = p_shadow / p_noshadow
check("shadowed ~ (visible fraction) * unshadowed (incoherent sum)",
      approx(got_ratio, expected_ratio, rel=0.02),
      f"ratio={got_ratio:.3f} vs expected={expected_ratio:.3f}")

# -----------------------------------------------------------------------
# P2.8: incident unit vector at known directions
# -----------------------------------------------------------------------
d = _incident_unit_vectors(np.array([0.0]), np.array([0.0]))
check("d(az=0, el=0) = +X", np.allclose(d[0], [1, 0, 0]))
d = _incident_unit_vectors(np.array([90.0]), np.array([0.0]))
check("d(az=90, el=0) = +Y", np.allclose(d[0], [0, 1, 0]))
d = _incident_unit_vectors(np.array([0.0]), np.array([90.0]))
check("d(az=0, el=90) = +Z", np.allclose(d[0], [0, 0, 1]))


print()
print("=" * 72)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 72)
if failed:
    sys.exit(1)
