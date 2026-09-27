"""
Quasi-steady-state (QSS) point-mass lap time simulator with a simplified
2026-style energy model (ICE + MGU-K deployment, braking harvest, active aero).

All CarParams defaults are starting assumptions meant to be CALIBRATED against
telemetry (see correlate.py). Regulation-derived values (MGU-K power, speed
taper, harvest limit) are approximations - verify them against the current
FIA Formula 1 Technical Regulations before quoting them anywhere.
"""
from dataclasses import dataclass, asdict
import numpy as np

G = 9.81
RHO = 1.18  # air density (kg/m^3), warm near-sea-level conditions


@dataclass
class CarParams:
    mass: float = 800.0              # kg, car + driver + quali fuel
    mu: float = 1.7                  # effective tyre friction coefficient
    cla: float = 3.8                 # downforce coeff x area, corner mode (m^2)
    cda: float = 1.05                # drag coeff x area, corner mode (m^2)
    cla_straight: float = 2.3        # active aero straight mode
    cda_straight: float = 0.75
    straight_radius: float = 600.0   # m, straight mode allowed where radius > this
    p_ice: float = 400e3             # W
    p_ers: float = 350e3             # W, MGU-K peak deployment
    ers_taper_start: float = 290 / 3.6  # m/s, deployment starts tapering (verify regs)
    ers_taper_end: float = 355 / 3.6    # m/s, deployment reaches zero (verify regs)
    soc_start: float = 4.0e6         # J usable battery energy at lap start
    harvest_power: float = 350e3     # W, max regen under braking
    harvest_limit: float = 8.5e6     # J, max harvest per lap (verify regs)
    crr: float = 0.012               # rolling resistance coefficient
    brake_g_max: float = 5.0         # cap on braking decel (g); typical F1 peak, held fixed
    drive_share: float = 0.55        # share of tyre load on the driven (rear) axle: traction
                                     # limit is drive_share x total grip (F1 cars are RWD)

    def to_dict(self):
        return asdict(self)


def ers_available_power(v, p):
    """MGU-K power available at speed v (linear taper between start and end speed)."""
    if v <= p.ers_taper_start:
        return p.p_ers
    if v >= p.ers_taper_end:
        return 0.0
    return p.p_ers * (p.ers_taper_end - v) / (p.ers_taper_end - p.ers_taper_start)


def corner_speed_limit(kappa, p, v_cap=110.0):
    """Max steady-state speed from lateral grip: m v^2 k = mu (m g + 0.5 rho ClA v^2)."""
    k = np.abs(kappa)
    denom = p.mass * k - 0.5 * RHO * p.cla * p.mu
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = np.where(denom > 0, p.mu * p.mass * G / denom, np.inf)
    return np.minimum(np.sqrt(v2), v_cap)


def _long_grip(v, k, p, cla):
    """Longitudinal accel available from the tyres after lateral demand (friction circle)."""
    a_tot = p.mu * (G + 0.5 * RHO * cla * v * v / p.mass)
    a_lat = v * v * abs(k)
    r = min(a_lat / a_tot, 1.0)
    return a_tot * np.sqrt(1.0 - r * r)


def _backward_pass(ds, kappa, vlim, p):
    """Braking envelope: the fastest speed at each point that can still make the next corner."""
    n = len(vlim)
    v = vlim.copy()
    for _ in range(2):  # two sweeps so the envelope wraps across the start/finish line
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            vn = v[j]
            a_brake = min(_long_grip(vn, kappa[j], p, p.cla), p.brake_g_max * G)
            a_drag = 0.5 * RHO * p.cda * vn * vn / p.mass + p.crr * G
            v[i] = min(v[i], np.sqrt(vn * vn + 2.0 * (a_brake + a_drag) * ds[i]))
    return v


@dataclass
class SimResult:
    s: np.ndarray
    v: np.ndarray            # m/s
    t: np.ndarray            # cumulative time (s)
    soc: np.ndarray          # J
    ers_power: np.ndarray    # W deployed
    harvest_power: np.ndarray  # W harvested
    mode: np.ndarray         # 0 corner-aero throttle, 1 straight-aero throttle, 2 braking/lift
    clipping: np.ndarray     # bool: on throttle, power-limited, battery empty
    taper: np.ndarray        # bool: on throttle, power-limited, MGU-K reduced by speed taper
    wot: np.ndarray          # bool: wide-open throttle (power-limited)
    lap_time: float
    energy_deployed: float
    energy_harvested: float


def simulate(track, p):
    kappa, ds = track.kappa, track.ds
    n = len(kappa)
    vlim = corner_speed_limit(kappa, p)
    vb = _backward_pass(ds, kappa, vlim, p)
    straight = np.abs(kappa) < 1.0 / p.straight_radius

    def run(v0):
        v = np.zeros(n); t = np.zeros(n); soc_a = np.zeros(n)
        ers_a = np.zeros(n); hv_a = np.zeros(n)
        mode = np.zeros(n, dtype=int); clip = np.zeros(n, dtype=bool)
        tap = np.zeros(n, dtype=bool); wot = np.zeros(n, dtype=bool)
        v[0] = min(v0, vb[0])
        soc, deployed, harvested, time = p.soc_start, 0.0, 0.0, 0.0
        v_end = v[0]
        for i in range(n):
            vi = v[i]
            vi_eff = max(vi, 5.0)
            j = (i + 1) % n
            use_straight = straight[i]
            cla = p.cla_straight if use_straight else p.cla
            cda = p.cda_straight if use_straight else p.cda
            f_res = 0.5 * RHO * cda * vi * vi + p.crr * p.mass * G
            p_ers_av = ers_available_power(vi, p) if soc > 0 else 0.0
            a_pow = ((p.p_ice + p_ers_av) / vi_eff - f_res) / p.mass
            # rear-wheel drive: only the driven axle's share of grip is available for traction
            a_grip = p.drive_share * _long_grip(vi, kappa[i], p, cla) - f_res / p.mass
            a = min(a_pow, a_grip)
            v_acc = np.sqrt(max(vi * vi + 2.0 * a * ds[i], 1.0))

            ers_used = 0.0; harvest = 0.0
            if v_acc <= vb[j]:
                v_next = v_acc
                mode[i] = 1 if use_straight else 0
                p_req = (p.mass * a + f_res) * vi_eff
                ers_used = min(max(p_req - p.p_ice, 0.0), p_ers_av)
                power_lim = (a_pow <= a_grip) and vi > 30
                clip[i] = power_lim and soc <= 0
                wot[i] = power_lim
                tap[i] = power_lim and soc > 0 and p_ers_av < 0.99 * p.p_ers
            else:
                v_next = vb[j]
                mode[i] = 2
                a_req = (v_next ** 2 - vi ** 2) / (2.0 * ds[i])
                if a_req > 0:  # partial throttle
                    p_req = (p.mass * a_req + f_res) * vi_eff
                    ers_used = min(max(p_req - p.p_ice, 0.0), p_ers_av)
                else:          # braking: harvest part of the braking power
                    f_res_n = 0.5 * RHO * p.cda * vi * vi + p.crr * p.mass * G
                    p_brake = max(-p.mass * a_req - f_res_n, 0.0) * 0.5 * (vi + v_next)
                    harvest = min(p.harvest_power, p_brake)

            dt = 2.0 * ds[i] / max(vi + v_next, 1e-3)
            e_dep = min(ers_used * dt, soc)
            e_hv = min(harvest * dt, p.harvest_limit - harvested)
            soc = soc - e_dep + e_hv
            deployed += e_dep; harvested += e_hv; time += dt

            ers_a[i] = e_dep / dt; hv_a[i] = e_hv / dt; soc_a[i] = soc
            t[i] = time
            if i + 1 < n:
                v[i + 1] = v_next
            v_end = v_next
        return v, t, soc_a, ers_a, hv_a, mode, clip, tap, wot, deployed, harvested, v_end

    first = run(vb[0])               # settle the flying-lap entry speed
    v, t, soc, ers, hv, mode, clip, tap, wot, dep, har, _ = run(first[-1])
    return SimResult(track.s, v, t, soc, ers, hv, mode, clip, tap, wot, float(t[-1]), dep, har)
