"""Track geometry: arc length + curvature, from FastF1 telemetry or a synthetic layout."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class Track:
    s: np.ndarray       # distance at start of each segment (m)
    ds: np.ndarray      # segment length (m)
    kappa: np.ndarray   # signed curvature (1/m)
    x: np.ndarray
    y: np.ndarray
    name: str = "track"
    xy_scale: float = 1.0   # factor applied to raw GPS XY (FastF1 uses 1/10 m)

    @property
    def length(self):
        return float(self.s[-1] + self.ds[-1])

    @classmethod
    def from_xy(cls, x, y, step=5.0, smooth_m=60.0, length=None, k_max=1 / 12.0, name="track"):
        """Build a closed track from noisy XY points (e.g. FastF1 position data)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-6]
        x, y = x[keep], y[keep]
        xc, yc = np.r_[x, x[0]], np.r_[y, y[0]]
        seg = np.hypot(np.diff(xc), np.diff(yc))
        raw_len = seg.sum()
        scale = (length / raw_len) if length else 1.0
        xc, yc = xc * scale, yc * scale
        sc = np.r_[0.0, np.cumsum(seg * scale)]
        total = sc[-1]
        n = int(round(total / step))
        s = np.linspace(0.0, total, n, endpoint=False)
        xs, ys = np.interp(s, sc, xc), np.interp(s, sc, yc)
        h = total / n
        win = max(5, int(smooth_m / h) | 1)
        xs = savgol_filter(xs, win, 3, mode="wrap")
        ys = savgol_filter(ys, win, 3, mode="wrap")
        dx = (np.roll(xs, -1) - np.roll(xs, 1)) / (2 * h)
        dy = (np.roll(ys, -1) - np.roll(ys, 1)) / (2 * h)
        ddx = (np.roll(xs, -1) - 2 * xs + np.roll(xs, 1)) / h ** 2
        ddy = (np.roll(ys, -1) - 2 * ys + np.roll(ys, 1)) / h ** 2
        kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
        kappa = np.clip(kappa, -k_max, k_max)
        return cls(s, np.full(n, h), kappa, xs, ys, name)

    @classmethod
    def synthetic_street(cls, step=5.0):
        """Synthetic street circuit with Baku-LIKE features (long flat-out run, tight
        castle-style section, many 90 deg corners). NOT the real Baku layout; use it
        only for offline testing when FastF1 data isn't available."""
        d = np.deg2rad
        segs = [  # (length_m, radius_m or None, sign)
            (350, None, 1), (d(90) * 30, 30, 1), (250, None, 1), (d(90) * 30, 30, -1),
            (550, None, 1), (d(90) * 28, 28, 1), (300, None, 1), (d(90) * 30, 30, 1),
            (200, None, 1), (d(60) * 40, 40, -1), (250, None, 1), (d(70) * 45, 45, 1),
            (120, None, 1), (d(90) * 35, 35, -1), (150, None, 1), (d(90) * 15, 15, 1),
            (80, None, 1), (d(70) * 20, 20, -1), (100, None, 1), (d(60) * 25, 25, 1),
            (250, None, 1), (d(45) * 60, 60, -1), (200, None, 1), (d(90) * 50, 50, 1),
            (300, None, 1), (d(90) * 35, 35, -1), (300, None, 1), (d(90) * 40, 40, 1),
            (250, None, 1), (d(90) * 25, 25, -1), (250, None, 1), (d(40) * 120, 120, 1),
            (500, None, 1), (d(30) * 500, 500, -1), (600, None, 1), (d(25) * 700, 700, 1),
        ]
        segs.append((100, None, 1))
        s_list, k_list = [], []
        s0 = 0.0
        for L, r, sgn in segs:
            m = max(1, int(round(L / step)))
            s_list.append(s0 + np.arange(m) * (L / m))
            k_list.append(np.full(m, 0.0 if r is None else sgn / r))
            s0 += L
        s, kappa = np.concatenate(s_list), np.concatenate(k_list)
        ds = np.r_[np.diff(s), s0 - s[-1]]
        heading = np.cumsum(kappa * ds)
        x, y = np.cumsum(np.cos(heading) * ds), np.cumsum(np.sin(heading) * ds)
        return cls(s, ds, kappa, x, y, "synthetic street (Baku-like)")


def load_fastf1(year=2026, gp="Azerbaijan", session="Q", driver=None, cache_dir="f1_cache",
                max_laps=80, sigma_m=6.0, lat_g_cap=5.5, wot_throttle=98.0):
    """Reference = fastest lap (or a driver's fastest). Geometry = GPS pooled from all
    representative laps in the session (within 107% of the fastest), which averages
    out GPS noise. Returns (Track, reference telemetry DataFrame, meta dict)."""
    import os
    import fastf1
    import pandas as pd

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    ses = fastf1.get_session(year, gp, session)
    ses.load(laps=True, telemetry=True, weather=False, messages=False)
    ref_laps = ses.laps.pick_drivers(driver) if driver else ses.laps
    lap = ref_laps.pick_fastest()
    tel = lap.get_telemetry().dropna(subset=["Speed", "Distance"])
    ref = pd.DataFrame({
        "distance": tel["Distance"].to_numpy(float),
        "speed_kmh": tel["Speed"].to_numpy(float),
        "throttle": tel["Throttle"].to_numpy(float),
        "brake": tel["Brake"].astype(float).to_numpy(),
    })
    L = float(ref["distance"].max())
    ref_raw = ref.copy()

    # pool GPS from representative laps
    fastest = lap["LapTime"].total_seconds()
    pool = ses.laps.pick_wo_box()
    pool = pool[pool["LapTime"].dt.total_seconds() < 1.07 * fastest]
    pool = pool.sort_values("LapTime").head(max_laps)
    D, X, Y, used = [], [], [], 0
    for _, lp in pool.iterlaps():
        try:
            tl = lp.get_telemetry()
        except Exception:
            continue
        if "Source" in tl.columns:
            tl = tl[tl["Source"] == "pos"]
        tl = tl.dropna(subset=["X", "Y", "Distance"])
        if len(tl) < 50:
            continue
        dmax = tl["Distance"].max()
        if not (0.95 * L < dmax < 1.05 * L):
            continue
        D.append(tl["Distance"].to_numpy(float) * L / dmax)
        X.append(tl["X"].to_numpy(float)); Y.append(tl["Y"].to_numpy(float))
        used += 1
    if not D:
        raise RuntimeError("No usable GPS laps found; try --session FP2 or check the data.")
    D, X, Y = np.concatenate(D), np.concatenate(X), np.concatenate(Y)
    track = track_from_gps(D, X, Y, lap_length=L, sigma_m=sigma_m,
                           name=f"{year} {gp} {session}")
    ref = align_reference(track, ref, tel)
    track = segment_track(track, ref, lat_g_cap, wot_throttle)
    meta = {"driver": lap["Driver"], "team": lap["Team"], "lap_time": fastest,
            "session": f"{year} {gp} {session}", "gps_laps_pooled": used}
    gps = {"D": D, "X": X, "Y": Y, "L": L, "lat_g_cap": lat_g_cap,
           "wot_throttle": wot_throttle, "ref_tel": tel, "ref_raw": ref_raw}
    return track, ref, meta, gps


def align_reference(track, ref, tel):
    """Re-express the reference lap's Distance channel in map-matched track distance, so
    sim and real speed traces line up corner by corner."""
    pos = tel[tel["Source"] == "pos"] if "Source" in tel.columns else tel
    pos = pos.dropna(subset=["X", "Y", "Distance"])
    d_old = pos["Distance"].to_numpy(float)
    d_new = project_to_track(track, d_old, pos["X"].to_numpy(), pos["Y"].to_numpy())
    d_new = np.maximum.accumulate(d_new)
    out = ref.copy()
    out["distance"] = np.clip(np.interp(ref["distance"], d_old, d_new), 0.0, track.length - 1e-6)
    return out


def rebuild_track(gps, sigma_m, register=True, name="track"):
    """Rebuild geometry from already-loaded pooled GPS with different settings.
    Returns (track, aligned reference)."""
    tr = track_from_gps(gps["D"], gps["X"], gps["Y"], lap_length=gps["L"], sigma_m=sigma_m,
                        register=register, name=name)
    ref = align_reference(tr, gps["ref_raw"], gps["ref_tel"]) if register else gps["ref_raw"]
    return segment_track(tr, ref, gps["lat_g_cap"], gps["wot_throttle"]), ref


def _bin_smooth(d, x, y, L, bin_m, sigma_m):
    """Average samples into distance bins, Gaussian-smooth, return bin centres and path."""
    from scipy.ndimage import gaussian_filter1d
    nb = int(round(L / bin_m)); hb = L / nb
    idx = np.minimum((np.mod(d, L) / hb).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    xb = np.bincount(idx, x, nb); yb = np.bincount(idx, y, nb)
    ok = cnt > 0
    centres = (np.arange(nb) + 0.5) * hb
    xb = np.interp(centres, centres[ok], xb[ok] / cnt[ok], period=L)
    yb = np.interp(centres, centres[ok], yb[ok] / cnt[ok], period=L)
    sig = max(sigma_m / hb, 0.5)
    return centres, hb, gaussian_filter1d(xb, sig, mode="wrap"), gaussian_filter1d(yb, sig, mode="wrap")


def map_match(d, x, y, L, bin_m=2.0, sigma_m=6.0, iterations=3, search_m=60.0):
    """Re-assign each GPS sample's lap distance by projecting it onto the pooled path
    (nearest point, searched only within +-search_m of its current distance so parallel
    sections of track can't be confused). Fixes per-lap Distance-channel drift, which
    otherwise smears corners when many laps are averaged."""
    from scipy.spatial import cKDTree
    d = np.mod(np.asarray(d, float), L)
    for _ in range(iterations):
        c, hb, xb, yb = _bin_smooth(d, x, y, L, bin_m, sigma_m)
        fine = np.arange(0.0, L, 0.5)
        xf = np.interp(fine, c, xb, period=L); yf = np.interp(fine, c, yb, period=L)
        tree = cKDTree(np.c_[xf, yf])
        _, nn = tree.query(np.c_[x, y], k=32)
        cand = fine[nn]                                   # candidate distances
        gap = np.abs((cand - d[:, None] + L / 2) % L - L / 2)
        cand_ok = np.where(gap <= search_m, cand, np.nan)
        best = cand_ok[:, 0]                               # nearest valid candidate
        for j in range(1, cand_ok.shape[1]):
            best = np.where(np.isnan(best), cand_ok[:, j], best)
        d = np.where(np.isnan(best), d, best)
    return d


def track_from_gps(distance, x, y, lap_length=None, step=5.0, bin_m=2.0, sigma_m=6.0,
                   k_max=1 / 12.0, name="track", wot_distance=None, wot_mask=None,
                   wot_trim=30.0, ref_distance=None, ref_speed_kmh=None, lat_g_cap=5.5,
                   register=True):
    """Build a track from RAW GPS samples pooled over one or many laps.

    distance : each sample's distance along its lap (already normalised to a common
               lap length when pooling several laps)
    Steps: scale XY (FastF1 uses 1/10 m) -> average all samples into `bin_m` bins
    (many laps => noise averages out) -> Gaussian smoothing of `sigma_m` metres ->
    curvature by finite differences -> optional full-throttle segmentation."""
    d = np.asarray(distance, float); x = np.asarray(x, float); y = np.asarray(y, float)
    L = float(lap_length or d.max())
    d = np.mod(d, L)
    # scale XY (FastF1 uses 1/10 m) so the path length matches the lap distance
    centres, hb, xb, yb = _bin_smooth(d, x, y, L, bin_m, sigma_m)
    path = np.hypot(np.diff(np.r_[xb, xb[0]]), np.diff(np.r_[yb, yb[0]])).sum()
    xy_scale = L / path
    x, y = x * xy_scale, y * xy_scale
    if register:
        d = map_match(d, x, y, L, bin_m, sigma_m)
    centres, hb, xb, yb = _bin_smooth(d, x, y, L, bin_m, sigma_m)
    dx = (np.roll(xb, -1) - np.roll(xb, 1)) / (2 * hb)
    dy = (np.roll(yb, -1) - np.roll(yb, 1)) / (2 * hb)
    ddx = (np.roll(xb, -1) - 2 * xb + np.roll(xb, 1)) / hb ** 2
    ddy = (np.roll(yb, -1) - 2 * yb + np.roll(yb, 1)) / hb ** 2
    kb = np.clip((dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5), -k_max, k_max)
    n = int(round(L / step))
    s = np.linspace(0.0, L, n, endpoint=False)
    kappa = np.interp(s, centres, kb, period=L)
    xs = np.interp(s, centres, xb, period=L); ys = np.interp(s, centres, yb, period=L)
    kappa = _segment(s, kappa, ref_distance, ref_speed_kmh, lat_g_cap,
                     wot_distance, wot_mask, wot_trim)
    return Track(s, np.full(n, L / n), kappa, xs, ys, name, xy_scale)


def _segment(s, kappa, ref_distance, ref_speed_kmh, lat_g_cap, wot_distance, wot_mask, wot_trim):
    if ref_distance is not None and lat_g_cap:
        # feasibility cap: the real car cannot have turned tighter than lat_g_cap
        # allows at the speed it was actually doing (removes GPS-noise spikes)
        v = np.interp(s, np.asarray(ref_distance, float), np.asarray(ref_speed_kmh, float)) / 3.6
        k_cap = lat_g_cap * 9.81 / np.maximum(v, 10.0) ** 2
        kappa = np.clip(kappa, -k_cap, k_cap)
    if wot_distance is not None:
        kappa = zero_curvature_at_full_throttle(s, kappa, wot_distance, wot_mask, wot_trim)
    return kappa


def segment_track(track, ref, lat_g_cap=5.5, wot_throttle=98.0, wot_trim=30.0):
    """Apply the lateral-g cap and full-throttle segmentation using reference telemetry."""
    from dataclasses import replace
    k = _segment(track.s, track.kappa, ref["distance"], ref["speed_kmh"], lat_g_cap,
                 ref["distance"], ref["throttle"] >= wot_throttle, wot_trim)
    return replace(track, kappa=k)


def project_to_track(track, d_guess, x_raw, y_raw, search_m=60.0):
    """Lap distance of raw GPS points, by nearest point on the track path (searched only
    within +-search_m of d_guess). Returns distances unwrapped relative to d_guess."""
    from scipy.spatial import cKDTree
    L = track.length
    fine = np.arange(0.0, L, 0.5)
    xf = np.interp(fine, track.s, track.x, period=L); yf = np.interp(fine, track.s, track.y, period=L)
    x = np.asarray(x_raw, float) * track.xy_scale; y = np.asarray(y_raw, float) * track.xy_scale
    d_guess = np.asarray(d_guess, float)
    _, nn = cKDTree(np.c_[xf, yf]).query(np.c_[x, y], k=32)
    cand = fine[nn]
    off = (cand - d_guess[:, None] + L / 2) % L - L / 2
    off = np.where(np.abs(off) <= search_m, off, np.nan)
    best = off[:, 0]
    for j in range(1, off.shape[1]):
        best = np.where(np.isnan(best), off[:, j], best)
    return d_guess + np.nan_to_num(best)


def zero_curvature_at_full_throttle(s, kappa, wot_distance, wot_mask, trim=30.0):
    """Where the driver is flat out, corners are not limiting speed, so GPS-noise
    curvature there is removed. The first `trim` metres of each full-throttle zone
    are kept so corner exits still limit traction."""
    w = np.interp(s, np.asarray(wot_distance, float), np.asarray(wot_mask, float)) > 0.5
    h = s[1] - s[0]
    k_trim = int(round(trim / h))
    starts = np.flatnonzero(w & ~np.roll(w, 1))
    for i in starts:
        w[np.arange(i, i + k_trim) % len(w)] = False
    out = kappa.copy()
    out[w] = 0.0
    return out
