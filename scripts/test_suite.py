#!/usr/bin/env python3
"""
Comprehensive test suite for pyheatcontrol library.
Tests various features to ensure correctness.
"""
import sys
import os
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from argparse import Namespace
from pyheatcontrol.logging_config import setup_logging
from pyheatcontrol.optimization import optimization_time_dependent


def make_args(**kwargs):
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
        output_freq=100, output_dir='resu', no_vtk=True, debug=False,
    )
    defaults.update(kwargs)
    return Namespace(**defaults)


TESTS = {
    # Control types
    "dirichlet_control": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    "neumann_control": make_args(
        control_boundary_neumann=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    "distributed_control": make_args(
        control_distributed=['0.0,0.2,0.0,0.2'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    
    # State constraints
    "lower_constraint": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='lower', sc_lower=25.0,
    ),
    "upper_constraint": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='upper', sc_upper=200.0,
    ),
    "box_constraint": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        sc_type='box', sc_lower=25.0, sc_upper=200.0,
    ),
    
    # No constraints
    "no_constraints": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=[],
        sc_maxit=0,
    ),
    
    # Mesh refinement
    "mesh_n2": make_args(
        n=2,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    "mesh_n4": make_args(
        n=4,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    
    # Time refinement
    "dt_20": make_args(
        dt=20.0,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        inner_maxit=4,
    ),
    "dt_10": make_args(
        dt=10.0,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
        inner_maxit=2,
    ),
    
    # Boundary target
    "boundary_target": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_boundary=['x0,0.0,1.0'],
    ),
    
    # Dirichlet BC
    "dirichlet_bc": make_args(
        control_boundary_neumann=['yL,0.0,1.0'],
        dirichlet_bc=['y0,0.0,1.0,25.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
    ),
    
    # Combined controls
    "dirichlet_plus_neumann": make_args(
        control_boundary_dirichlet=['yL,0.0,1.0'],
        control_boundary_neumann=['y0,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
    ),
    
    # Physical parameters
    "high_k": make_args(
        k=0.5,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
    
    # Learning rate
    "small_lr": make_args(
        lr=1.0,
        control_boundary_dirichlet=['yL,0.0,1.0'],
        target_zone=['0.3,0.7,0.3,0.7'],
        constraint_zone=['0.3,0.7,0.3,0.7'],
    ),
}


def run_test(name, args):
    """Run a single test case."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    try:
        t0 = time.perf_counter()
        result = optimization_time_dependent(args)
        elapsed = time.perf_counter() - t0
        
        metrics = result[1]
        success = True
        
        print(f"  Runtime: {elapsed:.2f}s")
        print(f"  J: {metrics['J']:.6e}")
        print(f"  Violation: {metrics['violation']:.2e}")
        
        return {
            'success': True,
            'runtime': elapsed,
            'J': metrics['J'],
            'violation': metrics['violation'],
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'success': False,
            'error': str(e),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--test', choices=list(TESTS.keys()) + ['all'], default='all',
                        help='Which test to run')
    parser.add_argument('--output', default='test_results.json',
                        help='Output JSON file')
    args_parser = parser.parse_args()
    
    setup_logging(verbose=False, debug=False)
    
    if args_parser.test == 'all':
        tests_to_run = TESTS
    else:
        tests_to_run = {args_parser.test: TESTS[args_parser.test]}
    
    results = {}
    
    for name, test_args in tests_to_run.items():
        result = run_test(name, test_args)
        results[name] = result
    
    # Summary
    print(f"\n{'#'*60}")
    print("# SUMMARY")
    print(f"{'#'*60}")
    
    passed = sum(1 for r in results.values() if r.get('success', False))
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    print(f"\n{'Test':<30} {'Status':>10} {'Runtime':>10} {'J':>15}")
    print("-" * 70)
    
    for name, result in results.items():
        status = 'PASS' if result.get('success', False) else 'FAIL'
        runtime = result.get('runtime', 0)
        J = result.get('J', 0)
        print(f"{name:<30} {status:>10} {runtime:>10.2f}s {J:>15.2e}")
    
    # Save results
    with open(args_parser.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args_parser.output}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
