"""
Baku LapSim - build, correlate and calibrate a lap time sim against real telemetry.

Real data (run on your machine, needs internet for FastF1):
    python run_baku.py                       # 2026 Azerbaijan quali, overall fastest lap
    python run_baku.py --driver RUS          # a specific driver
    python run_baku.py --session FP2         # another session

Offline self-test (synthetic track, checks that calibration recovers known params):
    python run_baku.py --offline
"""
import argparse, json, os
from dataclasses import replace
import numpy as np
import pandas as pd

from lapsim import CarParams, simulate
from track import Track, load_fastf1, rebuild_track
from correlate import (compare, calibrate, sensitivity, plot_correlation, plot_geometry,
                       corner_geometry_report, print_geometry_report, data_quality_mask, distance_offset)


def offline_reference(track, seed=0):
    """Fake 'real' telemetry from a sim with hidden params + sensor noise."""
    truth = replace(CarParams(), mu=1.85, cla=4.3, cda=1.15, cda_straight=0.68, brake_g_max=4.8, p_ers=240e3)
    sim = simulate(track, truth)
    rng = np.random.default_rng(seed)
    d = np.arange(0, track.length, 4.0)
    v = np.interp(d, sim.s, sim.v * 3.6) + rng.normal(0, 2.0, len(d))
    ref = pd.DataFrame({"distance": d, "speed_kmh": v})
    return ref, sim.lap_time, truth


def print_report(title, m):
    print(f"\n== {title} ==")
    print(f"Lap time   sim {m['lap_time_sim']:.3f}s | real {m['lap_time_ref']:.3f}s | "
          f"delta {m['lap_time_delta']:+.3f}s   (full lap, includes flagged corners)")
    print(f"Clean time sim {m['clean_time_sim']:.3f}s | real {m['clean_time_ref']:.3f}s | "
          f"delta {m['clean_time_delta']:+.3f}s   (flagged corners excluded)")
    print(f"Speed RMSE {m['speed_rmse_kmh']:.1f} km/h | bias {m['speed_bias_kmh']:+.1f} km/h (clean)")
    print(f"Top speed  sim {m['top_speed_sim']:.1f} | real {m['top_speed_ref']:.1f} km/h")
    if m.get("zones"):
        print("Error by driving phase (sim - real):")
        for name, (bias, rmse, share) in m["zones"].items():
            print(f"  {name:24s} bias {bias:+6.1f} km/h | RMSE {rmse:5.1f} km/h | {share:4.1f}% of lap")
    if m["corners"]:
        print("Corner minimum speeds (distance m: real / sim km/h):")
        for dist, vr, vs, fl in m["corners"]:
            tag = "   [flagged: GPS geometry implausible, excluded from fit]" if fl else ""
            print(f"  {dist:7.0f}: {vr:6.1f} / {vs:6.1f}  ({vs - vr:+.1f}){tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--gp", default="Azerbaijan")
    ap.add_argument("--session", default="Q")
    ap.add_argument("--driver", default=None)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--sigma", type=float, default=6.0, help="GPS smoothing length (m)")
    ap.add_argument("--lat-g-cap", type=float, default=5.5, help="curvature feasibility cap (g)")
    ap.add_argument("--max-laps", type=int, default=80, help="laps pooled for track geometry")
    ap.add_argument("--g-flag", type=float, default=3.6,
                    help="flag slow corners whose geometry implies more lateral g than this")
    ap.add_argument("--geometry-sweep", action="store_true",
                    help="only compare corner geometry across smoothing lengths, then exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.offline:
        track = Track.synthetic_street()
        ref, ref_time, truth = offline_reference(track)
        meta = {"driver": "synthetic truth", "lap_time": ref_time,
                "session": "Offline self-test (synthetic Baku-like track)"}
    else:
        track, ref, meta, gps = load_fastf1(a.year, a.gp, a.session, a.driver,
                                            max_laps=a.max_laps, sigma_m=a.sigma,
                                            lat_g_cap=a.lat_g_cap)
        ref_time = meta["lap_time"]
        print(f"GPS pooled from {meta['gps_laps_pooled']} laps")
        if a.geometry_sweep:
            for reg in (False, True):
                for sg in (4.0, 6.0, 10.0):
                    tr, rf = rebuild_track(gps, sg, register=reg)
                    print_geometry_report(corner_geometry_report(tr, rf),
                                          f"(sigma {sg:g} m, map-matching {'on' if reg else 'off'}) ")
                    m = compare(simulate(tr, CarParams()), rf, ref_time)
                    print(f"  baseline sim: lap {m['lap_time_sim']:.3f}s, "
                          f"speed RMSE {m['speed_rmse_kmh']:.1f} km/h")
            return
        print_geometry_report(corner_geometry_report(track, ref))
        plot_geometry(track, ref, os.path.join(a.out, "geometry_check.png"))
    print(f"Track: {track.name} | {track.length:.0f} m | {len(track.s)} points")
    print(f"Reference: {meta['driver']} {ref_time:.3f}s")

    mask, flagged = data_quality_mask(track, ref, g_flag=a.g_flag)
    if flagged:
        print(f"\nData quality: {len(flagged)} corner(s) flagged "
              f"({', '.join(f'{f:.0f} m' for f in flagged)}); their surroundings "
              f"({mask.mean()*100:.1f}% of lap) are excluded from the fit.")
    base = CarParams()
    sim0 = simulate(track, base)
    print_report("Baseline (uncalibrated)", compare(sim0, ref, ref_time, mask))
    plot_correlation(sim0, ref, meta, os.path.join(a.out, "correlation_baseline.png"))

    p = base
    if not a.no_calibrate:
        print("\nCalibrating...")
        p, _ = calibrate(track, ref, ref_time, base, mask=mask)
        sim1 = simulate(track, p)
        m1 = compare(sim1, ref, ref_time, mask)
        print_report("Calibrated", m1)
        plot_correlation(sim1, ref, meta, os.path.join(a.out, "correlation_calibrated.png"))
        sh, rm = distance_offset(ref["distance"].to_numpy(), ref["speed_kmh"].to_numpy(), sim1)
        print(f"Alignment check: best global shift {sh:+.0f} m would give speed RMSE {rm:.1f} km/h "
              f"(0 m = sim and real already aligned)")
        print(f"Energy: deployed {sim1.energy_deployed/1e6:.2f} MJ, "
              f"harvested {sim1.energy_harvested/1e6:.2f} MJ, "
              f"clipping (battery empty) on {sim1.clipping.mean()*100:.1f}% of lap, "
              f"MGU-K speed taper on {sim1.taper.mean()*100:.1f}%")
        if a.offline:
            print("\nParameter recovery (truth vs fitted):")
            for k in ("mu", "cda", "cda_straight", "brake_g_max", "p_ers"):
                print(f"  {k:14s} {getattr(truth, k):.4g} vs {getattr(p, k):.4g}")
        with open(os.path.join(a.out, "calibrated_params.json"), "w") as f:
            json.dump({"meta": meta, "params": p.to_dict(), "metrics": m1}, f, indent=2)

    base_t, sens = sensitivity(track, p)
    print(f"\n== Sensitivity: lap-time change for +5% of each parameter (base {base_t:.3f}s) ==")
    for k, dt in sorted(sens.items(), key=lambda kv: kv[1]):
        print(f"  {k:14s} {dt:+.3f} s")
    print(f"\nPlots and params saved to ./{a.out}/")


if __name__ == "__main__":
    main()
