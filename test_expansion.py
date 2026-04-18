"""Math + integration sanity checks for rcs_2d_to_3d."""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/home/claude")
# Copy grim_dataset.py alongside so the import works
shutil.copy("/mnt/user-data/uploads/grim_dataset.py", "/home/claude/grim_dataset.py")

from grim_dataset import RcsGrid, C0
from rcs_2d_to_3d import expand_2d_to_3d, rcs_3d_at, _sinc, _sinc_sq


# ---------------------------------------------------------------------------
# Build a synthetic isotropic 2D grid: sigma_2D(phi, f) = constant
# ---------------------------------------------------------------------------

def make_isotropic_2d(sigma2d=0.5, freqs_ghz=(10.0,), n_az=361, pols=("VV",)):
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    el = np.asarray([0.0])
    f = np.asarray(freqs_ghz, dtype=float)
    p = np.asarray(pols, dtype=object)
    shape = (len(az), len(el), len(f), len(p))
    power = np.full(shape, sigma2d, dtype=np.float32)
    phase = np.zeros(shape, dtype=np.float32)
    return RcsGrid(
        azimuths=az, elevations=el, frequencies=f, polarizations=p,
        rcs_power=power, rcs_phase=phase,
        rcs_domain="power_phase",
        units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
        history="synthetic isotropic 2D",
    )


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol * max(1.0, abs(b))


# ---------------------------------------------------------------------------
# TEST 1: Broadside (el = 0) should give sigma_3D = (2 L^2 / lambda) * sigma_2D
# ---------------------------------------------------------------------------
sigma2d = 0.5
L = 1.2
freq_ghz = 10.0
lam = C0 / (freq_ghz * 1e9)

g2d = make_isotropic_2d(sigma2d=sigma2d, freqs_ghz=(freq_ghz,))
g3d = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0, 45.0, 180.0],
    elevations_deg=[0.0],
    frequencies=[freq_ghz],
    polarizations=["VV"],
)
expected_broadside = (2.0 * L ** 2 / lam) * sigma2d
print("TEST 1 — broadside RCS")
for ai, az in enumerate([0.0, 45.0, 180.0]):
    got = float(g3d.rcs_power[ai, 0, 0, 0])
    ok = approx(got, expected_broadside, tol=1e-5)
    print(f"  az={az:5.1f}  got={got:.6f}  expected={expected_broadside:.6f}  {'OK' if ok else 'FAIL'}")
    assert ok

# ---------------------------------------------------------------------------
# TEST 2: First sinc^2 null is at kL*sin(el) = pi, i.e. el = arcsin(lambda / L)
# ---------------------------------------------------------------------------
el_null_rad = np.arcsin(lam / L)
el_null_deg = np.rad2deg(el_null_rad)
g3d_null = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0],
    elevations_deg=[0.0, el_null_deg, 0.5 * el_null_deg],
    frequencies=[freq_ghz],
    polarizations=["VV"],
)
print("TEST 2 — first sinc^2 null")
vals = g3d_null.rcs_power[0, :, 0, 0]
print(f"  broadside (el=0):        {vals[0]:.6e}")
print(f"  half-null (el={0.5*el_null_deg:.3f}):  {vals[2]:.6e}")
print(f"  first null (el={el_null_deg:.3f}):     {vals[1]:.6e}")
assert vals[0] > vals[2] > vals[1]
assert vals[1] < 1e-6 * vals[0], f"null should be ~0, got {vals[1]} vs broadside {vals[0]}"
print("  OK (monotone decrease to null)")

# ---------------------------------------------------------------------------
# TEST 3: Sinc squared shape at a specific elevation — analytic check
# ---------------------------------------------------------------------------
print("TEST 3 — sinc^2 shape matches closed form")
el_test = 15.0  # deg
k = 2.0 * np.pi / lam
x = k * L * np.sin(np.deg2rad(el_test))
expected = (2.0 * L ** 2 / lam) * (np.sin(x) / x) ** 2 * sigma2d

g3d_pt = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0],
    elevations_deg=[el_test],
    frequencies=[freq_ghz],
    polarizations=["VV"],
)
got = float(g3d_pt.rcs_power[0, 0, 0, 0])
print(f"  el={el_test}  got={got:.6e}  expected={expected:.6e}  "
      f"{'OK' if approx(got, expected, 1e-5) else 'FAIL'}")
assert approx(got, expected, 1e-5)

# ---------------------------------------------------------------------------
# TEST 4: Symmetry in elevation (sin is odd, sinc^2 is even) — should be symmetric
# ---------------------------------------------------------------------------
print("TEST 4 — elevation symmetry (sinc^2 is even)")
g3d_sym = expand_2d_to_3d(
    g2d, length=L,
    azimuths_deg=[0.0],
    elevations_deg=[-30.0, 30.0],
    frequencies=[freq_ghz],
    polarizations=["VV"],
)
a, b = g3d_sym.rcs_power[0, 0, 0, 0], g3d_sym.rcs_power[0, 1, 0, 0]
print(f"  el=-30: {a:.6e}   el=+30: {b:.6e}   {'OK' if approx(float(a), float(b), 1e-6) else 'FAIL'}")
assert approx(float(a), float(b), 1e-6)

# ---------------------------------------------------------------------------
# TEST 5: Azimuth wrap-around (request az=-5 should equal az=355)
# ---------------------------------------------------------------------------
print("TEST 5 — azimuth wrap-around")
# Use a non-isotropic 2D pattern so wrap is meaningful
az = np.linspace(0.0, 360.0, 361, endpoint=False)
el = np.asarray([0.0])
f = np.asarray([freq_ghz])
p = np.asarray(["VV"], dtype=object)
power = (1.0 + 0.5 * np.cos(np.deg2rad(az))).astype(np.float32)  # az-dependent
power = power[:, None, None, None] * np.ones((len(az), 1, 1, 1), dtype=np.float32)
phase = np.zeros_like(power)
g2d_nonuniform = RcsGrid(
    azimuths=az, elevations=el, frequencies=f, polarizations=p,
    rcs_power=power, rcs_phase=phase,
    rcs_domain="power_phase",
    units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
)
g3d_wrap = expand_2d_to_3d(
    g2d_nonuniform, length=L,
    azimuths_deg=[-5.0, 355.0],
    elevations_deg=[0.0],
    frequencies=[freq_ghz],
    polarizations=["VV"],
)
v1, v2 = float(g3d_wrap.rcs_power[0, 0, 0, 0]), float(g3d_wrap.rcs_power[1, 0, 0, 0])
print(f"  az=-5 -> {v1:.6e}   az=355 -> {v2:.6e}   {'OK' if approx(v1, v2, 1e-6) else 'FAIL'}")
assert approx(v1, v2, 1e-6)

# ---------------------------------------------------------------------------
# TEST 6: .grim round-trip — expanded grid saves and reloads identically
# ---------------------------------------------------------------------------
print("TEST 6 — .grim save/load round-trip")
with tempfile.TemporaryDirectory() as td:
    out_path = os.path.join(td, "expanded.grim")
    g3d_full = expand_2d_to_3d(g2d, length=L, frequencies=[freq_ghz])
    written = g3d_full.save(out_path)
    g3d_reload = RcsGrid.load(written)
    same_power = np.allclose(g3d_full.rcs_power, g3d_reload.rcs_power, equal_nan=True)
    same_axes = (
        np.array_equal(g3d_full.azimuths, g3d_reload.azimuths)
        and np.array_equal(g3d_full.elevations, g3d_reload.elevations)
        and np.array_equal(g3d_full.frequencies, g3d_reload.frequencies)
        and np.array_equal(g3d_full.polarizations, g3d_reload.polarizations)
    )
    print(f"  power match: {same_power}   axes match: {same_axes}   "
          f"{'OK' if same_power and same_axes else 'FAIL'}")
    assert same_power and same_axes

# ---------------------------------------------------------------------------
# TEST 7: Single-point query helper
# ---------------------------------------------------------------------------
print("TEST 7 — rcs_3d_at single-point query")
c = rcs_3d_at(g2d, L, azimuth_deg=0.0, elevation_deg=0.0,
              frequency=freq_ghz, polarization="VV", return_complex=True)
expected_amp = np.sqrt(expected_broadside)
got_amp = abs(c)
print(f"  |sigma_3D|^{0.5} got={got_amp:.6f}  expected={expected_amp:.6f}  "
      f"{'OK' if approx(got_amp, expected_amp, 1e-5) else 'FAIL'}")
assert approx(got_amp, expected_amp, 1e-5)

# ---------------------------------------------------------------------------
# TEST 8: sinc(0) = 1, sinc^2(0) = 1
# ---------------------------------------------------------------------------
print("TEST 8 — sinc helper behaviour")
assert _sinc(np.array([0.0]))[0] == 1.0
assert _sinc_sq(np.array([0.0]))[0] == 1.0
assert abs(_sinc(np.array([np.pi]))[0]) < 1e-12
print("  OK")

print("\nAll tests passed.")
