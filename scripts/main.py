#!/usr/bin/env python3
"""
Time-dependent control con approccio segregato (forward + backward adjoint)

Features:
- Controlli u(t) variabili nel tempo (non più costanti)
- Adjoint equation risolta backward in time
- Gradient esatto via adjoint
- Multiple control zones (boundary Dirichlet + Neumann + distributed)
- State constraints con Moreau-Yosida
- Storage di tutte le soluzioni Y per adjoint
"""

from mpi4py import MPI
import argparse
import time

from pyheatcontrol.io_utils import _import_sanity_check
from pyheatcontrol.cli import build_parser
from pyheatcontrol.optimization import optimization_time_dependent

_import_sanity_check()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    parser = argparse.ArgumentParser(
        description="Heat V4 - Time-Dependent Optimal Control with Dirichlet + Neumann BC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser = build_parser()
    args = parser.parse_args()

    # Default thresholds based on constraint type
    if args.sc_type in ["lower", "box"]:
        if args.sc_lower is None:
            args.sc_lower = args.T_ref
    else:
        args.sc_lower = -1e10  # Very low (no lower constraint)

    if args.sc_type in ["upper", "box"]:
        if args.sc_upper is None:
            args.sc_upper = args.T_ref
    else:
        args.sc_upper = 1e10  # Very high (no upper constraint)

    # sanity check per box
    if args.sc_type == "box" and args.sc_lower > args.sc_upper:
        raise ValueError(
            f"Box constraint invalido: sc_lower={args.sc_lower} > sc_upper={args.sc_upper}"
        )

    import os

    print(f"[DEBUG] running file = {os.path.abspath(__file__)}", flush=True)

    # Default controls (solo se NON disattivati)
    if (
        not args.no_default_controls
        and not args.control_boundary_dirichlet
        and not args.control_boundary_neumann
        and not args.control_distributed
    ):
        args.control_boundary_dirichlet = ["yL,0.0,1.0"]

    if not args.target_zone:
        args.target_zone = ["0.3,0.7,0.3,0.7"]
    if not args.constraint_zone:
        # default: uguale alla target per backward compatibility
        args.constraint_zone = args.target_zone.copy()

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        print("\n" + "#" * 70)
        print("# HEAT PARABOLIC V4 - TIME-DEPENDENT CONTROL (DIRICHLET + NEUMANN)")
        print("#" * 70)

    t0 = time.time()

    optimization_time_dependent(args)

    elapsed = time.time() - t0
    if rank == 0:
        print(f"\n[TIME] Total: {elapsed:.2f}s")
        print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
