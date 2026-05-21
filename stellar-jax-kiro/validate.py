#!/usr/bin/env python3
"""Local copy of validate.py adapted for this workspace."""
import sys
import random
import numpy as np
from pathlib import Path

MIST_DIR = Path("/Users/freschri/code/stellar-jax/data/mist")
TOLERANCE = 0.01
N_DRAWS = 5
Z_SUN = 0.0142857
FEH_TO_DIR = {
    -0.25: "MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.0_EEPS",
    0.00: "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS",
    0.25: "MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.0_EEPS",
}

def feh_to_Z(feh): return Z_SUN * 10**feh

def find_track(mass, feh):
    fehs = sorted(FEH_TO_DIR.keys())
    closest_feh = min(fehs, key=lambda f: abs(f - feh))
    track_dir = MIST_DIR / FEH_TO_DIR[closest_feh]
    mass_int = int(round(mass * 100))
    fname = f"{mass_int:05d}M.track.eep"
    track_file = track_dir / fname
    if not track_file.exists():
        available = sorted(track_dir.glob("*.eep"))
        masses_available = [int(f.stem.replace("M.track", "")) / 100 for f in available]
        closest_mass = min(masses_available, key=lambda m: abs(m - mass))
        mass_int = int(round(closest_mass * 100))
        fname = f"{mass_int:05d}M.track.eep"
        track_file = track_dir / fname
    return track_file, closest_feh

def load_mist_track(track_file):
    with open(track_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("#") and "star_age" in line:
            header_idx = i; break
    cols = lines[header_idx].strip("#").split()
    data = np.loadtxt(lines[header_idx + 1:])
    cm = {name: i for i, name in enumerate(cols)}
    center_h1 = data[:, cm["center_h1"]]
    ms_mask = center_h1 > 0.01
    ms_data = data[ms_mask]
    return {
        "star_age": ms_data[:, cm["star_age"]],
        "log_L": ms_data[:, cm["log_L"]],
        "log_Teff": ms_data[:, cm["log_Teff"]],
        "log_R": ms_data[:, cm["log_R"]],
        "center_h1": ms_data[:, cm["center_h1"]],
    }

def run_jax(mass, Z):
    from stellar import evolve_star
    result = evolve_star(mass, Z=Z)
    return {k: np.array(v) for k, v in result.items()}

def compare(jax_track, mist_track):
    mist_age = mist_track["star_age"]
    errors = {}
    for q in ["log_L", "log_Teff", "log_R", "center_h1"]:
        if q not in jax_track or len(jax_track[q]) == 0:
            errors[q] = float("inf"); continue
        jax_vals = np.interp(mist_age, jax_track["star_age"], jax_track[q])
        mist_vals = mist_track[q]
        if q.startswith("log_"):
            err = float(np.max(np.abs(jax_vals - mist_vals)))
        else:
            mask = mist_vals > 0.02
            if mask.sum() == 0: errors[q] = 0.0; continue
            err = float(np.max(np.abs((jax_vals[mask] - mist_vals[mask]) / mist_vals[mask])))
        errors[q] = err
    return errors

def main():
    random.seed(42)  # reproducible for debugging
    all_pass = True
    for i in range(N_DRAWS):
        mass = round(random.uniform(0.8, 2.0), 2)
        feh = round(random.uniform(-0.25, 0.25), 2)
        Z = feh_to_Z(feh)
        print(f"\n--- Draw {i+1}/{N_DRAWS}: M={mass}, [Fe/H]={feh}, Z={Z:.5f} ---")
        track_file, actual_feh = find_track(mass, feh)
        mist = load_mist_track(track_file)
        print(f"  MIST: {track_file.name}, {len(mist['star_age'])} pts, age={mist['star_age'][0]/1e6:.0f}-{mist['star_age'][-1]/1e6:.0f} Myr")
        jax = run_jax(mass, Z)
        print(f"  JAX:  {len(jax['log_L'])} pts, age={jax['star_age'][0]/1e6:.0f}-{jax['star_age'][-1]/1e6:.0f} Myr")
        errors = compare(jax, mist)
        for q, err in errors.items():
            status = "PASS" if err < TOLERANCE else "FAIL"
            print(f"  {q:12s}: {err:.4f}  [{status}]")
            if status == "FAIL": all_pass = False
        if not all_pass: break
    print("\n" + ("PASS" if all_pass else "FAIL"))
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
