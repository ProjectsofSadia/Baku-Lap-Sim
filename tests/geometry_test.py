"""Validates GPS -> track geometry on a closed synthetic circuit with F1-like GPS:
~3.7 Hz sampling, position noise, and per-lap Distance-channel drift."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.signal import find_peaks
from lapsim import CarParams, simulate
from track import Track, track_from_gps


def closed_test_track(step=2.0, L_first=1200.0, L_up=500.0):
    d = np.deg2rad
    segs = [(L_first, None, 1), (d(90)*30, 30, 1), (400, None, 1), (d(90)*60, 60, 1),
            (300, None, 1), (d(45)*25, 25, 1), (d(45)*25, 25, -1), (200, None, 1),
            (d(45)*25, 25, -1), (d(45)*25, 25, 1), (400, None, 1), (d(90)*15, 15, 1),
            (60, None, 1), (d(90)*15, 15, -1), (40, None, 1), (d(90)*15, 15, 1),
            (L_up, None, 1), (d(90)*40, 40, 1)]
    s_l, k_l, s0 = [], [], 0.0
    for L, r, sg in segs:
        m = max(1, int(round(L/step)))
        s_l.append(s0 + np.arange(m)*(L/m)); k_l.append(np.full(m, 0 if r is None else sg/r)); s0 += L
    s, k = np.concatenate(s_l), np.concatenate(k_l)
    ds = np.r_[np.diff(s), s0 - s[-1]]
    h = np.cumsum(k*ds) - k*ds
    x, y = np.cumsum(np.cos(h)*ds), np.cumsum(np.sin(h)*ds)
    return Track(s, ds, k, x, y, "closed test"), (x[-1]-x[0], y[-1]-y[0])


L1, L2 = 1200.0, 500.0
for _ in range(20):
    true, gap = closed_test_track(L_first=L1, L_up=L2)
    L1 -= gap[0]; L2 += gap[1]
    if np.hypot(*gap) < 0.01:
        break
p = CarParams(); sim = simulate(true, p)
rng = np.random.default_rng(1)
mins, _ = find_peaks(-sim.v, prominence=5)


def gps(n_laps, noise, drift=0.0):
    D, X, Y = [], [], []
    for _ in range(n_laps):
        tg = np.arange(rng.uniform(0, .27), sim.lap_time, 0.27)
        sg = np.interp(tg, sim.t, sim.s) % true.length
        X.append((np.interp(sg, true.s, true.x) + rng.normal(0, noise, len(sg))) * 10)
        Y.append((np.interp(sg, true.s, true.y) + rng.normal(0, noise, len(sg))) * 10)
        ph = rng.uniform(0, 2*np.pi, 3); amp = rng.uniform(-1, 1, 3) * drift / 1.7
        err = sum(a*np.sin(2*np.pi*(k+1)*sg/true.length + ph_)
                  for k, (a, ph_) in enumerate(zip(amp, ph)))
        D.append((sg + err) % true.length)
    return np.concatenate(D), np.concatenate(X), np.concatenate(Y)


def score(tr, label):
    s = simulate(tr, p)
    ap = np.interp(sim.s[mins], tr.s, s.v*3.6) - sim.v[mins]*3.6
    vt = np.interp(tr.s, sim.s, sim.v*3.6)
    print(f"{label:34s} lap {s.lap_time:6.2f}s (true {sim.lap_time:.2f})  "
          f"RMSE {np.sqrt(np.mean((s.v*3.6-vt)**2)):5.1f}  apex |err| mean "
          f"{np.mean(np.abs(ap)):5.1f} max {np.max(np.abs(ap)):5.1f} km/h", flush=True)


if __name__ == "__main__":
    for drift in (0.0, 10.0):
        D, X, Y = gps(60, 1.0, drift)
        for reg in (False, True):
            tr = track_from_gps(D, X, Y, lap_length=true.length, sigma_m=6.0, register=reg,
                                wot_distance=sim.s, wot_mask=sim.wot,
                                ref_distance=sim.s, ref_speed_kmh=sim.v*3.6)
            score(tr, f"drift +-{drift:g} m, map-match {reg}")
