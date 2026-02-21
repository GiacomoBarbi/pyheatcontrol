#!/usr/bin/env python3
"""
Benchmark script for optimization: run with fixed parameters, save or compare metrics.
Use --save-baseline to record metrics (run once before code changes).
Use --check to run and compare against saved baseline (run after code changes).
Metrics J, energy, violation must match within tolerance; runtime may differ.

Run from repo root with the pyheatcontrol environment active (e.g. conda fenicsx):
  python scripts/benchmark_optimization.py --save-baseline   # record baseline
  python scripts/benchmark_optimization.py --check           # verify after changes
"""
from __future__ import annotations

import json
import os
import sys

# Add src to path so we can run from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Relative to script dir
BASELINE_JSON = os.path.join(SCRIPT_DIR, "benchmark_baseline_metrics.json")

# Tolerances for metric comparison (physics/numerics must be unchanged)
RTOL = 1e-12  # relative tolerance
ATOL = 1e-14  # absolute tolerance


def make_benchmark_args():
    """Build argparse.Namespace with fixed benchmark parameters (reproducible, small run)."""
    from argparse import Namespace

    args = Namespace(
        L=0.1,
        H=None,
        n=3,
        dt=50.0,
        T_final=200.0,
        k=0.15,
        rho=1100.0,
        c=1500.0,
        T_ambient=25.0,
        T_ref=160.0,
        T_ref_func=None,
        T_ref_func_xt=None,
        control_boundary_dirichlet=["yL,0.0,1.0"],
        control_boundary_neumann=[],
        control_distributed=[],
        dirichlet_bc=[],
        dirichlet_disturbance=[],
        robin_boundary=[],
        no_default_controls=False,
        target_zone=["0.3,0.7,0.3,0.7"],
        target_boundary=[],
        alpha_track=1.0,
        alpha_u=1e-4,
        gamma_u=1e-2,
        beta=1e3,
        sc_maxit=2,
        sc_tol=1e-6,
        u_init=180.0,
        u_min=25.0,
        u_max=250.0,
        check_grad=False,
        beta_u=0.0,
        dirichlet_spatial_reg="L2",
        sc_type="lower",
        sc_lower=None,
        sc_upper=None,
        sc_start_time=None,
        sc_end_time=None,
        constraint_zone=None,  # set below
        lr=5.0,
        inner_maxit=8,
        grad_tol=1e-3,
        fd_eps=1e-2,
        output_freq=1,
        output_dir="resu",
        no_vtk=True,
        verbose=False,
        debug=False,
    )
    if args.sc_type in ["lower", "box"]:
        args.sc_lower = args.sc_lower if args.sc_lower is not None else 25.0
    else:
        args.sc_lower = -1e10
    if args.sc_type in ["upper", "box"]:
        args.sc_upper = args.sc_upper if args.sc_upper is not None else 25.0
    else:
        args.sc_upper = 1e10
    args.constraint_zone = args.target_zone.copy() if args.target_zone else ["0.3,0.7,0.3,0.7"]
    return args


def run_optimization():
    from pyheatcontrol.logging_config import setup_logging
    from pyheatcontrol.optimization import optimization_time_dependent

    args = make_benchmark_args()
    setup_logging(verbose=args.verbose, debug=args.debug)
    _, metrics = optimization_time_dependent(args)
    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser(description="Benchmark optimization: save baseline or check against it.")
    p.add_argument("--save-baseline", action="store_true", help="Run and save metrics to baseline JSON")
    p.add_argument("--check", action="store_true", help="Run and compare metrics to baseline")
    p.add_argument("--baseline-file", default=BASELINE_JSON, help="Path to baseline metrics JSON")
    p.add_argument("--rtol", type=float, default=RTOL, help="Relative tolerance for comparison")
    p.add_argument("--atol", type=float, default=ATOL, help="Absolute tolerance for comparison")
    parsed = p.parse_args()

    if not parsed.save_baseline and not parsed.check:
        print("Use --save-baseline to record metrics, or --check to compare.", file=sys.stderr)
        sys.exit(2)

    metrics = run_optimization()

    if parsed.save_baseline:
        out = {
            "J": metrics["J"],
            "energy": metrics["energy"],
            "violation": metrics["violation"],
            "runtime": metrics["runtime"],
        }
        with open(parsed.baseline_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved baseline metrics to {parsed.baseline_file}")
        return

    # --check
    if not os.path.isfile(parsed.baseline_file):
        print(f"Baseline file not found: {parsed.baseline_file}", file=sys.stderr)
        sys.exit(1)
    with open(parsed.baseline_file) as f:
        baseline = json.load(f)

    rtol, atol = parsed.rtol, parsed.atol
    ok = True
    for key in ["J", "energy", "violation"]:
        a, b = baseline[key], metrics[key]
        diff = abs(a - b)
        tol = atol + rtol * (abs(a) + abs(b)) / 2
        if diff > tol:
            print(f"MISMATCH {key}: baseline={a:.16e} current={b:.16e} diff={diff:.4e} tol={tol:.4e}")
            ok = False
        else:
            print(f"OK {key}: baseline={a:.12e} current={b:.12e}")
    print(f"runtime: baseline={baseline['runtime']:.2f}s current={metrics['runtime']:.2f}s (may differ)")

    if not ok:
        sys.exit(1)
    print("All metrics match baseline.")
    sys.exit(0)


if __name__ == "__main__":
    main()
