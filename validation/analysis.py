#!/usr/bin/env python3
"""Validation-case analyzer for channelCHNS.

Reads the output directory of a run (pln_*.bin, ycg.txt, diag_v_int.txt,
run_info.txt), computes the validation metrics, checks them against the
oracle in the case JSON, optionally compares against reference metrics, and
writes metrics.json + PNG plots into the data directory.

Usage:
    python analysis.py --case case1_static_cap_th50_Bo0.json --data output/
    python analysis.py --case sanity_th50_Bo0_coarse.json --data output_sanity/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case JSON file")
    ap.add_argument("--data", required=True, help="run output directory")
    ap.add_argument("--reference", default=None,
                    help="metrics JSON with reference values (default: case's 'reference' block)")
    args = ap.parse_args()

    case = json.load(open(args.case))
    DATA = Path(args.data)
    ri = case["runtime_input"]
    ct = case["compile_time"]
    TH_S = float(ri["th_s"])
    NY = int(ct["grid"]["NY"]); NX = int(ct["grid"]["NX"]); NZ = int(ct["grid"]["NZ"])
    NXP = 3 * NX // 2
    LX, LY, LZ = (float(ct["domain"][k]) for k in ("LX", "LY", "LZ"))
    RAD = float(ct["Rad"])

    # ---------- hard checks: the small files analysis needs must exist ----------
    # (the full run output incl. the big .bin restart fields is checked by run_case.py)
    hard = case.get("hard_checks", {})
    need = [f for f in hard.get("files", []) if not f.endswith(".bin")]
    need += ["ycg.txt", "diag_v_int.txt"] if "ycg.txt" not in need else []
    missing = [f for f in need if not (DATA / f).exists()]
    if not list(DATA.glob("pln_*.bin")):
        missing.append("pln_*.bin")
    if missing:
        print("HARD CHECK FAIL: missing files:", missing)
        sys.exit(2)

    # ---------- grid ----------
    y = LY * 0.5 * (1.0 - np.cos(np.pi * np.arange(NY) / (NY - 1)))  # Chebyshev, p_y=0
    x = LX * (np.arange(NXP) / NXP - 0.5)

    # ---------- last plane cut = end of run ----------
    pln_files = sorted(DATA.glob("pln_*.bin"), key=lambda p: float(p.stem.split("_")[1]))
    if not pln_files:
        print("HARD CHECK FAIL: no pln_*.bin in", DATA); sys.exit(2)
    pln_last = pln_files[-1]
    t_end = float(pln_last.stem.split("_")[1])
    c = np.fromfile(pln_last, dtype=np.float32).reshape(NY, NXP)
    print(f"loaded {pln_last.name} (t={t_end}), c range [{c.min():.3f}, {c.max():.3f}]")

    # ---------- theoretical spherical cap (a priori, from the IC formulas) ----------
    an_rd = np.deg2rad(TH_S - 90.0)
    sn = np.sin(an_rd); fc = 1.0 - sn
    Rd = RAD * (2.0 / (fc * fc * (3.0 - fc))) ** (1.0 / 3.0)
    hg = -Rd * sn
    xc = np.sqrt(Rd**2 - hg**2)
    h_max = hg + Rd
    th_c = np.linspace(0, 2 * np.pi, 720)
    x_circ = Rd * np.cos(th_c); y_circ = hg + Rd * np.sin(th_c)
    L = 1.2; ang = np.deg2rad(TH_S)
    x_rayR = np.array([xc, xc + L * np.cos(ang)]);  y_rayR = np.array([0.0, L * np.sin(ang)])
    x_rayL = np.array([-xc, -xc - L * np.cos(ang)]); y_rayL = np.array([0.0, L * np.sin(ang)])
    print(f"theory: Rd={Rd:.6f}, hg={hg:.6f}, x_c={xc:.6f}, h_max={h_max:.6f}")

    # ---------- colormap with theory overlays ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pc = ax.pcolormesh(x, y, c, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(pc, ax=ax, label="phase field c")
    ax.plot(x_circ, y_circ, "k-", lw=2.2, label="theoretical circle (spherical cap)")
    ax.plot(x_rayR, y_rayR, "g--", lw=2.2, label=f"theoretical contact angle = {TH_S:.0f} deg")
    ax.plot(x_rayL, y_rayL, "g--", lw=2.2)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(loc="upper right")
    ax.set_title(f"{case['name']}  (t={t_end})  th_s={TH_S:.0f} deg")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(DATA / "phasefield_colormap.png", dpi=150)
    plt.close(fig)

    # ---------- measured interface (c=0) ----------
    y_if = np.full(NXP, np.nan)
    for j in range(NXP):
        col = c[:, j]
        above = col > 0.0
        cross = np.where(above[:-1] != above[1:])[0]
        if len(cross) == 0:
            continue
        k = cross[-1]
        y0, y1 = y[k], y[k + 1]; c0, c1 = col[k], col[k + 1]
        y_if[j] = y0 + (0.0 - c0) * (y1 - y0) / (c1 - c0)
    valid = ~np.isnan(y_if)

    # ---------- circle fit -> measured contact angle ----------
    m_all = valid & (np.abs(x) <= xc)
    X, Y = x[m_all], y_if[m_all]
    A = np.column_stack([X, Y, np.ones_like(X)])
    sol, *_ = np.linalg.lstsq(A, X**2 + Y**2, rcond=None)
    x0, y0 = sol[0] / 2.0, sol[1] / 2.0
    R_fit = np.sqrt(sol[2] + x0**2 + y0**2)
    xc_fit = np.sqrt(max(R_fit**2 - y0**2, 0.0))
    ang_circ = float(np.rad2deg(np.arctan2(xc_fit, y0)))
    print(f"circle fit: center=({x0:.5f},{y0:.5f}), R={R_fit:.5f} -> angle {ang_circ:.3f} deg "
          f"(theory {TH_S:.3f}, diff {ang_circ - TH_S:+.3f})")

    # ---------- L2 of interface vs circle, normalized by h_max ----------
    in_cap = valid & (np.abs(x) <= xc)
    x_in = x[in_cap]
    diff = y_if[in_cap] - (hg + np.sqrt(Rd**2 - x_in**2))
    L2 = float(np.sqrt(np.trapz(diff**2, x_in)))
    L2_norm = L2 / h_max
    print(f"L2/h_max = {L2_norm:.6e}  (max |diff| = {np.max(np.abs(diff)):.3e})")

    # ---------- plane mass check ----------
    pln0 = DATA / "pln_0.00.bin"
    if pln0.exists():
        c0 = np.fromfile(pln0, dtype=np.float32).reshape(NY, NXP)
        dx = LX / NXP; dy = np.gradient(y)
        m0 = float(np.sum((c0 + 1.0) / 2.0 * dx * dy[:, None]))
        m1 = float(np.sum((c + 1.0) / 2.0 * dx * dy[:, None]))
        print(f"plane mass: t=0 {m0:.6f} | t={t_end} {m1:.6f} (rel {(m1-m0)/m0:+.2e})")

    # ---------- ycg time series ----------
    ycg_file = next(DATA.glob("ycg*.txt"))
    ts, ycg, vol = np.loadtxt(ycg_file, unpack=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ts, ycg, "o-")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$\langle y\rangle$ of drop")
    ax[0].set_title(f"center of gravity (drift {ycg[-1]-ycg[0]:+.2e})")
    ax[1].plot(ts, vol / vol[0], "o-"); ax[1].axhline(1.0, color="k", ls=":")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$V(t)/V(0)$")
    ax[1].set_title(f"drop volume (rel drift {(vol[-1]-vol[0])/vol[0]:+.2e})")
    fig.tight_layout(); fig.savefig(DATA / "ycg_timeseries.png", dpi=150); plt.close(fig)
    print(f"ycg rows: {len(ts)}; cg drift {ycg[-1]-ycg[0]:+.3e}; V rel drift {(vol[-1]-vol[0])/vol[0]:+.3e}")

    # ---------- spurious-current diagnostics ----------
    diag_file = DATA / "diag_v_int.txt"
    dt_, umax_cap, urms_cap, Ifrac = np.loadtxt(diag_file, unpack=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].semilogy(dt_, umax_cap, "o-")
    ax[0].set_xlabel("t"); ax[0].set_ylabel(r"$u_{max}\sqrt{We}$"); ax[0].set_title("umax_cap")
    ax[1].semilogy(dt_, urms_cap, "o-")
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$u_{int,rms}\sqrt{We}$"); ax[1].set_title("u_int_rms_cap")
    ax[2].plot(dt_, Ifrac, "o-"); ax[2].axhline(1.0, color="k", ls=":")
    ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$I_0/I_0(t_0)$"); ax[2].set_title("Ifrac")
    fig.tight_layout(); fig.savefig(DATA / "diag_spurious_currents.png", dpi=150); plt.close(fig)
    print(f"diag rows: {len(dt_)}; max umax_cap {umax_cap.max():.3e}; max u_int_rms_cap {urms_cap.max():.3e}; "
          f"Ifrac in [{Ifrac.min():.6f}, {Ifrac.max():.6f}]")

    # ---------- metrics ----------
    settled = dt_ > case["oracle"].get("Ifrac_abs_drift_settled", {}).get("settled_after_fraction", 0.5) * dt_[-1]
    metrics = {
        "contact_angle_circle_fit_deg": ang_circ,
        "circle_L2_over_hmax": L2_norm,
        "umax_cap": float(umax_cap.max()),
        "u_int_rms_cap": float(urms_cap.max()),
        "Ifrac_abs_max": float(np.max(np.abs(Ifrac - 1.0))),
        "Ifrac_abs_drift_settled": float(np.max(np.abs(Ifrac[settled] - 1.0))),
        "volume_rel_drift_abs": abs(float((vol[-1] - vol[0]) / vol[0])),
        "cg_abs_drift": abs(float(ycg[-1] - ycg[0])),
    }

    # ---------- hard check: all finite ----------
    if hard.get("all_finite", False):
        vals = np.concatenate([c.ravel(), ycg, vol, dt_, umax_cap, urms_cap, Ifrac])
        if not np.all(np.isfinite(vals)):
            print("HARD CHECK FAIL: non-finite values in outputs")
            sys.exit(2)

    # ---------- oracle ----------
    orc = case["oracle"]
    print(f"validation case: {case['name']}")
    allpass = True
    for k, spec in orc.items():
        v = metrics[k]
        if "expected" in spec:
            ok = abs(v - spec["expected"]) <= spec["abs_tol"]
            detail = f"{v:.4f} vs expected {spec['expected']} +/- {spec['abs_tol']}"
        else:
            ok = v <= spec["max"]
            detail = f"{v:.3e} <= {spec['max']:.1e}"
        allpass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {detail}")

    # ---------- reference comparison ----------
    ref = None
    if args.reference:
        ref = json.load(open(args.reference))
    elif "reference" in case:
        ref = case["reference"]
    if ref:
        print("vs reference:")
        for k, v in metrics.items():
            if k in ref:
                print(f"  {k}: {v:.6e}  (ref {ref[k]:.6e}, delta {v - ref[k]:+.3e})")

    metrics_out = {"case": case["name"], "t_end": t_end, "metrics": metrics,
                   "oracle_pass": bool(allpass)}
    with open(DATA / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("ORACLE:", "PASS" if allpass else "FAIL")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
