"""Sim-to-track correlation: metrics, parameter calibration, sensitivity study, plots."""
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks
from lapsim import simulate

DEFAULT_BOUNDS = {
    "mu": (1.2, 3.0), "cla": (2.0, 6.5), "cda": (0.6, 1.6),
    "cla_straight": (1.0, 4.0), "cda_straight": (0.4, 1.2),
    "soc_start": (1.0e6, 6.0e6), "p_ice": (300e3, 480e3),
    "harvest_limit": (2.0e6, 9.0e6),
    "brake_g_max": (3.0, 6.5),
    "p_ers": (100e3, 350e3),
}

# ClA is held at its prior by default: at a low-speed track like Baku almost every limiting
# corner is slow, so the data cannot separate tyre grip (mu) from downforce (ClA).
# brake_g_max is held at 5 g: tested as a fitted parameter, it was not identifiable separately
# from downforce (ClA), which is also held at its prior.
# p_ers is the EFFECTIVE electrical deployment power: with it fixed at the 350 kW maximum,
# the sim out-accelerates the real car above 200 km/h and the fit hides it with high drag.
DEFAULT_FIT = ("mu", "cda", "cda_straight", "harvest_limit", "p_ers")


def data_quality_mask(track, ref, g_flag=3.6, v_max_kmh=180.0, window_m=60.0):
    """Flag stretches whose reconstructed geometry implies implausible lateral g for the real
    car at its real speed (e.g. GPS multipath on street circuits). Scans the WHOLE lap, not
    only the real car's clear minimum-speed points, so shallow dips (e.g. the castle section)
    are caught too. Returns (mask over reference samples, list of flagged distances)."""
    d = ref["distance"].to_numpy()
    v = np.interp(track.s, d, ref["speed_kmh"].to_numpy())
    g = (v / 3.6) ** 2 * np.abs(track.kappa) / 9.81
    bad = (g > g_flag) & (v < v_max_kmh)
    flagged = []
    if bad.any():                                   # one entry per contiguous bad stretch
        idx = np.flatnonzero(bad)
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for grp in groups:
            flagged.append(float(track.s[grp[np.argmax(g[grp])]]))
    mask = np.zeros(len(d), dtype=bool)
    for f in flagged:
        mask |= np.abs(d - f) <= window_m
    return mask, flagged


def time_over(dist, v_kmh, keep, step=1.0):
    """Time to cover the kept parts of the lap, integrating 1/v over distance."""
    g = np.arange(0.0, dist.max(), step)
    v = np.maximum(np.interp(g, dist, v_kmh) / 3.6, 1.0)
    k = np.interp(g, dist, keep.astype(float)) > 0.5
    return float(np.sum(step / v[k]))


def compare(sim, ref, ref_lap_time, mask=None):
    """Speed-trace and lap-time metrics of a sim result vs reference telemetry.
    `mask` (bool over reference samples) excludes data-quality-flagged regions from
    RMSE, bias and the 'clean' time comparison."""
    d = ref["distance"].to_numpy()
    v_ref = ref["speed_kmh"].to_numpy()
    v_sim = np.interp(d, sim.s, sim.v * 3.6)
    keep = ~mask if mask is not None else np.ones(len(d), dtype=bool)
    err = (v_sim - v_ref)[keep]
    mins, _ = find_peaks(-v_ref, prominence=25, distance=20)
    corners = [(float(d[i]), float(v_ref[i]), float(v_sim[i]), bool(not keep[i])) for i in mins]
    t_ref_clean = time_over(d, v_ref, keep); t_sim_clean = time_over(d, v_sim, keep)
    return {
        "zones": zone_errors(d, v_ref, v_sim, keep),
        "clean_time_sim": t_sim_clean, "clean_time_ref": t_ref_clean,
        "clean_time_delta": t_sim_clean - t_ref_clean,
        "lap_time_sim": sim.lap_time, "lap_time_ref": ref_lap_time,
        "lap_time_delta": sim.lap_time - ref_lap_time,
        "speed_rmse_kmh": float(np.sqrt(np.mean(err ** 2))),
        "speed_bias_kmh": float(np.mean(err)),
        "top_speed_sim": float(v_sim.max()), "top_speed_ref": float(v_ref.max()),
        "corners": corners,  # (distance, min speed real, min speed sim, flagged)
    }


def calibrate(track, ref, ref_lap_time, base, fit=DEFAULT_FIT, mask=None,
              bounds=None, w_time=1.0, w_apex=0.5, maxiter=500, verbose=True):
    """Fit selected CarParams to minimise speed RMSE + lap-time error + corner apex error.
    The apex term stops the fit from trading slow corners against fast straights."""
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    x0 = np.array([getattr(base, k) for k in fit], float)
    d = ref["distance"].to_numpy(); v_ref = ref["speed_kmh"].to_numpy()
    keep = ~mask if mask is not None else np.ones(len(d), dtype=bool)
    apex_idx, _ = find_peaks(-v_ref, prominence=25, distance=20)
    apex_idx = apex_idx[keep[apex_idx]]
    t_ref_clean = time_over(d, v_ref, keep)
    history = []

    def unpack(z):
        vals = x0 * z
        return replace(base, **{k: float(np.clip(v, *bounds[k])) for k, v in zip(fit, vals)})

    def cost(z):
        p = unpack(z)
        sim = simulate(track, p)
        v_sim = np.interp(d, sim.s, sim.v * 3.6)
        rmse = np.sqrt(np.mean((v_sim - v_ref)[keep] ** 2))
        apex = np.mean(np.abs(v_sim[apex_idx] - v_ref[apex_idx])) if len(apex_idx) else 0.0
        t_err = abs(time_over(d, v_sim, keep) - t_ref_clean)
        c = rmse + w_time * t_err + w_apex * apex
        # penalty for leaving bounds (clip keeps the sim valid, penalty keeps optimiser honest)
        for k, v in zip(fit, x0 * z):
            lo, hi = bounds[k]
            c += 50.0 * (max(lo - v, 0) + max(v - hi, 0)) / (hi - lo)
        # physical consistency: straight mode must cut drag and downforce
        c += 20.0 * max(p.cda_straight - 0.95 * p.cda, 0) + 20.0 * max(p.cla_straight - p.cla, 0)
        history.append(c)
        return c

    z = np.ones(len(fit))
    for _ in range(2):  # restart once: Nelder-Mead often stalls on the first simplex
        res = minimize(cost, z, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3})
        z = res.x
    p_fit = unpack(res.x)
    if verbose:
        print(f"Calibration: {len(history)} sims, cost {history[0]:.2f} -> {res.fun:.2f}")
        for k in fit:
            lo, hi = bounds[k]; v = getattr(p_fit, k)
            flag = "   <-- at bound: compensating for another error, or not identifiable from this data" \
                if (v - lo) < 0.02 * (hi - lo) or (hi - v) < 0.02 * (hi - lo) else ""
            print(f"  {k:14s} {getattr(base, k):>12.4g} -> {v:>12.4g}{flag}")
    return p_fit, history


def sensitivity(track, p, params=("mu", "cla", "cda", "cda_straight", "soc_start", "p_ice", "p_ers", "brake_g_max"),
                step=0.05):
    """Lap-time change for a +5% change in each parameter (classic performance-engineering table)."""
    base = simulate(track, p).lap_time
    out = {}
    for k in params:
        p2 = replace(p, **{k: getattr(p, k) * (1 + step)})
        out[k] = simulate(track, p2).lap_time - base
    return base, out


def plot_correlation(sim, ref, meta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = ref["distance"].to_numpy(); v_ref = ref["speed_kmh"].to_numpy()
    v_sim = np.interp(d, sim.s, sim.v * 3.6)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    ax[0].plot(d, v_ref, color="black", lw=1.2, label=f"Real ({meta.get('driver', 'ref')})")
    ax[0].plot(sim.s, sim.v * 3.6, color="tab:red", lw=1.2, label="Sim")
    ax[0].scatter(sim.s[sim.taper], sim.v[sim.taper] * 3.6, s=4, color="tab:purple",
                  label="Sim: MGU-K speed taper", zorder=3)
    ax[0].scatter(sim.s[sim.clipping], sim.v[sim.clipping] * 3.6, s=4, color="tab:orange",
                  label="Sim: clipping (battery empty)", zorder=3)
    ax[0].set_ylabel("Speed (km/h)"); ax[0].legend(loc="lower right")
    ax[0].set_title(f"{meta.get('session', '')}  |  sim {sim.lap_time:.3f}s vs real "
                    f"{meta.get('lap_time', float('nan')):.3f}s")
    ax[1].plot(d, v_sim - v_ref, color="tab:blue", lw=1)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set_ylabel("Sim - real (km/h)")
    ax[2].plot(sim.s, sim.soc / 1e6, color="tab:green", lw=1.2)
    ax[2].set_ylabel("Battery (MJ)"); ax[2].set_xlabel("Distance (m)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_geometry(track, ref, path):
    """Geometry sanity check: curvature and the lateral g it implies at the real speed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = np.interp(track.s, ref["distance"], ref["speed_kmh"]) / 3.6
    fig = plt.figure(figsize=(12, 8))
    a1 = fig.add_subplot(2, 2, (1, 3)); a1.plot(track.x, track.y, lw=1.5, color="black")
    a1.set_aspect("equal"); a1.set_title("Reconstructed track"); a1.axis("off")
    a2 = fig.add_subplot(2, 2, 2); a2.plot(track.s, track.kappa, lw=0.8)
    a2.set_ylabel("Curvature (1/m)"); a2.grid(alpha=.3)
    a3 = fig.add_subplot(2, 2, 4); a3.plot(track.s, v * v * np.abs(track.kappa) / 9.81, lw=0.8, color="tab:red")
    a3.set_ylabel("Implied lateral g at real speed"); a3.set_xlabel("Distance (m)"); a3.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def corner_geometry_report(track, ref, window_m=15.0):
    """For each real corner apex: the tightest radius the reconstructed track has there,
    and the lateral g the REAL car would have needed at its real speed on that radius.
    Slow corners above ~3.5 g mean the geometry is too tight there (not the car model)."""
    d = ref["distance"].to_numpy(); v = ref["speed_kmh"].to_numpy()
    mins, _ = find_peaks(-v, prominence=25, distance=20)
    rows = []
    for i in mins:
        m = np.abs(track.s - d[i]) <= window_m
        k = np.max(np.abs(track.kappa[m])) if m.any() else np.nan
        r = 1.0 / k if k > 0 else np.inf
        g = (v[i] / 3.6) ** 2 * k / 9.81
        rows.append((float(d[i]), float(v[i]), float(r), float(g)))
    return rows


def print_geometry_report(rows, label=""):
    print(f"\n== Corner geometry check {label}==")
    print("  distance  real km/h  radius m  implied lateral g")
    for dist, v, r, g in rows:
        flag = "   <-- too tight?" if g > 3.5 and v < 180 else ""
        print(f"  {dist:8.0f}  {v:9.1f}  {r:8.1f}  {g:8.2f}{flag}")


def zone_errors(d, v_ref, v_sim, keep):
    """Mean speed error (sim - real, km/h) and RMSE by driving phase, classified from the
    REAL trace: braking, low-speed traction (accelerating below 200 km/h), high-speed
    acceleration (above 200 km/h). Shows WHERE the model is wrong, not just how much."""
    order = np.argsort(d, kind="stable")
    du, idx = np.unique(d[order], return_index=True)          # drop repeated distances
    dv_u = np.gradient(v_ref[order][idx], du) if len(du) > 2 else np.zeros(len(du))
    dv = np.interp(d, du, dv_u)
    zones = {
        "braking": dv < -0.05,
        "traction <200 km/h": (dv > 0.02) & (v_ref < 200),
        "acceleration >200 km/h": (dv > 0.02) & (v_ref >= 200),
    }
    out = {}
    for name, m in zones.items():
        m = m & keep
        if m.sum() > 5:
            e = (v_sim - v_ref)[m]
            out[name] = (float(e.mean()), float(np.sqrt(np.mean(e ** 2))), float(m.mean() * 100))
    return out


def distance_offset(d, v_ref, sim, max_shift=40.0, step=1.0):
    """Best single distance shift (m) aligning the sim speed trace to the real one. A large
    value means braking-zone error is partly misalignment rather than physics."""
    shifts = np.arange(-max_shift, max_shift + step, step)
    errs = [np.sqrt(np.mean((np.interp(d + sh, sim.s, sim.v * 3.6, period=sim.s[-1] + (sim.s[1]-sim.s[0])) - v_ref) ** 2))
            for sh in shifts]
    i = int(np.argmin(errs))
    return float(shifts[i]), float(errs[i])
