#!/usr/bin/env python3
"""
Profiling script per identificare colli di bottiglia nella libreria.
Misura i tempi per ogni componente: forward, adjoint, gradient, line search, cost, etc.
"""
import sys
import os
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Enable detailed timing in the optimization module
import pyheatcontrol.optimization as opt_module
opt_module.PROFILE_TIMING = True

from argparse import Namespace
from pyheatcontrol.logging_config import setup_logging
from pyheatcontrol.optimization import optimization_time_dependent


def make_args(**kwargs):
    """Crea args di default con override."""
    defaults = dict(
        L=0.1, H=None, n=3, dt=50.0, T_final=200.0,
        k=0.15, rho=1100.0, c=1500.0, T_ambient=25.0, T_ref=160.0,
        T_ref_func=None, T_ref_func_xt=None,
        control_boundary_dirichlet=[],
        control_boundary_neumann=[],
        control_distributed=[],
        dirichlet_bc=[], dirichlet_disturbance=[], robin_boundary=[],
        no_default_controls=False,
        target_zone=[],
        target_boundary=[],
        alpha_track=1.0, alpha_u=1e-4, gamma_u=1e-2,
        beta=1e3, sc_maxit=2, sc_tol=1e-6,
        u_init=180.0, u_min=25.0, u_max=250.0,
        check_grad=False, beta_u=0.0, dirichlet_spatial_reg='L2',
        sc_type='lower', sc_lower=25.0, sc_upper=1e10,
        sc_start_time=None, sc_end_time=None,
        constraint_zone=[],
        lr=5.0, inner_maxit=8, grad_tol=1e-3, fd_eps=1e-2,
        output_freq=100, output_dir='resu', no_vtk=True,
        verbose=False, debug=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


CASI_TEST = {
    "baseline_dirichlet": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='lower', sc_lower=25.0,
    ),
    "baseline_neumann": make_args(
        control_boundary_neumann=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='lower', sc_lower=25.0,
    ),
    "baseline_distributed": make_args(
        control_distributed=['0.0,0.2,0.0,0.2'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='lower', sc_lower=25.0,
    ),
    "no_constraints": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=[],
        sc_maxit=0,
    ),
    "mesh_n5": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        n=5,
    ),
    "mesh_n8": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        n=8,
    ),
    "dt_10": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        dt=10.0, T_final=200.0, inner_maxit=4,
    ),
    "T_final_600": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        dt=20.0, T_final=600.0,
    ),
    "boundary_target": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_boundary=['x0,0.0,1.0'],
    ),
    "dirichlet_bc": make_args(
        control_boundary_neumann=['yL,0.0,1.0'],
        dirichlet_bc=['y0,0.0,1.0,25.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
    ),
}


def run_case(name, args):
    """Esegue un caso di test e ritorna risultati + metriche."""
    print(f"\n{'='*60}")
    print(f"CASE: {name}")
    print(f"{'='*60}")
    
    # Stampa parametri chiave
    n_ctrl = len(args.control_boundary_dirichlet) + len(args.control_boundary_neumann) + len(args.control_distributed)
    n_targets = len(args.target_zone) + len(args.target_boundary)
    n_constraints = len(args.constraint_zone)
    n_steps = int(args.T_final / args.dt)
    
    print(f"  n={args.n}, dt={args.dt}, T={args.T_final}, steps={n_steps}")
    print(f"  Controls: {n_ctrl} (D:{len(args.control_boundary_dirichlet)}, N:{len(args.control_boundary_neumann)}, Dist:{len(args.control_distributed)})")
    print(f"  Targets: {n_targets}, Constraints: {n_constraints}")
    
    t0 = time.perf_counter()
    try:
        result = optimization_time_dependent(args)
        elapsed = time.perf_counter() - t0
        
        T_final, metrics = result
        metrics['runtime'] = elapsed
        metrics['success'] = True
        
        print(f"  Runtime: {elapsed:.2f}s")
        print(f"  J={metrics['J']:.2e}, violation={metrics['violation']:.2e}")
        
        return metrics
        
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ERROR: {e}")
        return {'runtime': elapsed, 'success': False, 'error': str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run profiling tests')
    parser.add_argument('--case', choices=list(CASI_TEST.keys()) + ['all'], default='all',
                        help='Which case to run')
    parser.add_argument('--output', default='profiling_results.json',
                        help='Output JSON file')
    args_parser = parser.parse_args()
    
    setup_logging(verbose=False, debug=False)
    
    if args_parser.case == 'all':
        cases_to_run = CASI_TEST
    else:
        cases_to_run = {args_parser.case: CASI_TEST[args_parser.case]}
    
    results = {}
    
    for name, case_args in cases_to_run.items():
        print(f"\n{'#'*60}")
        print(f"# Running: {name}")
        print(f"{'#'*60}")
        
        try:
            metrics = run_case(name, case_args)
            results[name] = metrics
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'#'*60}")
    print("# SUMMARY")
    print(f"{'#'*60}")
    
    print(f"\n{'Case':<25} {'Runtime':>10} {'Success':>10} {'J':>15}")
    print("-" * 65)
    
    for name, metrics in results.items():
        runtime = metrics.get('runtime', 0)
        success = 'Yes' if metrics.get('success', False) else 'NO'
        J = metrics.get('J', 0)
        print(f"{name:<25} {runtime:>10.2f}s {success:>10} {J:>15.2e}")
    
    # Timing breakdown
    print(f"\n{'#'*60}")
    print("# TIMING BREAKDOWN (first case with profiling)")
    print(f"{'#'*60}")
    
    for name, metrics in results.items():
        if 'timing' in metrics:
            t = metrics['timing']
            total = t.get('total', 0)
            if total > 0:
                print(f"\n{name}:")
                print(f"  Total:     {total:8.3f}s")
                print(f"  Forward:    {t.get('forward', 0):8.3f}s ({100*t.get('forward', 0)/total:5.1f}%)")
                print(f"  Adjoint:    {t.get('adjoint', 0):8.3f}s ({100*t.get('adjoint', 0)/total:5.1f}%)")
                print(f"  Gradient:   {t.get('gradient', 0):8.3f}s ({100*t.get('gradient', 0)/total:5.1f}%)")
                print(f"  Cost:       {t.get('cost', 0):8.3f}s ({100*t.get('cost', 0)/total:5.1f}%)")
                print(f"  Armijo:     {t.get('armijo', 0):8.3f}s ({100*t.get('armijo', 0)/total:5.1f}%)")
                print(f"  SC update:  {t.get('sc_update', 0):8.3f}s ({100*t.get('sc_update', 0)/total:5.1f}%)")
            break
    
    # Salva risultati
    with open(args_parser.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args_parser.output}")


if __name__ == '__main__':
    main()
