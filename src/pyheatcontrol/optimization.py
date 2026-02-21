import logging
import numpy as np
import math
import ufl
import time as time_module
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Function, functionspace
from pyheatcontrol.parsing_utils import parse_box, parse_boundary_segment, parse_robin_segment, parse_dirichlet_bc, parse_dirichlet_disturbance, parse_neumann_bc
from pyheatcontrol.mesh_utils import create_mesh
from pyheatcontrol.io_utils import save_visualization_output
from pyheatcontrol.solver import TimeDepHeatSolver
from pyheatcontrol.gradcheck import check_gradient_fd
from pyheatcontrol.logging_config import logger

PROFILE_TIMING = False  # Set to True to collect detailed timing

def armijo_line_search(
    solver,
    u_distributed_funcs_time,
    u_neumann_funcs_time,
    u_dirichlet_funcs_time,
    dir_distributed,   # search direction (CG or steepest)
    dir_neumann,       # search direction (CG or steepest)
    dir_dirichlet,     # search direction (CG or steepest)
    J_current,
    dir_dot_grad,      # <direction, gradient> for Armijo
    T_ref,
    num_steps,
    alpha_init=10.0,
    rho=0.5,
    c=1e-4,
    max_iter=20,
    comm=None,
    rank=0
):
    """
    Backtracking line search with Armijo condition.

    Returns (alpha, Y_trial_all, J_trial) so the caller can reuse the trajectory
    and avoid a redundant forward solve.
    """

    alpha = alpha_init

    # Make backup of current controls
    u_dist_backup = None
    u_neum_backup = None
    u_dir_backup = None

    if solver.n_ctrl_distributed > 0:
        u_dist_backup = [[u_distributed_funcs_time[m][j].x.array.copy()
                          for j in range(solver.n_ctrl_distributed)]
                         for m in range(num_steps)]

    if solver.n_ctrl_neumann > 0:
        u_neum_backup = [[u_neumann_funcs_time[m][i].x.array.copy()
                          for i in range(solver.n_ctrl_neumann)]
                         for m in range(num_steps)]

    if solver.n_ctrl_dirichlet > 0:
        u_dir_backup = [[u_dirichlet_funcs_time[m][j].x.array.copy()
                         for j in range(solver.n_ctrl_dirichlet)]
                        for m in range(num_steps)]

    # Armijo condition threshold
    armijo_threshold = J_current - c * alpha * dir_dot_grad

    for ls_iter in range(max_iter):
        # Try step with current alpha
        # Update distributed
        if solver.n_ctrl_distributed > 0:
            for j in range(solver.n_ctrl_distributed):
                dofs_j = solver.distributed_dofs[j]
                for m in range(num_steps):
                    u_distributed_funcs_time[m][j].x.array[dofs_j] = \
                        u_dist_backup[m][j][dofs_j] - alpha * dir_distributed[m][j][dofs_j]
                    u_distributed_funcs_time[m][j].x.scatter_forward()

        # Update Neumann
        if solver.n_ctrl_neumann > 0:
            for i in range(solver.n_ctrl_neumann):
                dofs_i = solver.neumann_dofs[i]
                for m in range(num_steps):
                    u_neumann_funcs_time[m][i].x.array[dofs_i] = \
                        u_neum_backup[m][i][dofs_i] + alpha * dir_neumann[m][i][dofs_i]
                    u_neumann_funcs_time[m][i].x.scatter_forward()

        # Update Dirichlet
        if solver.n_ctrl_dirichlet > 0:
            for j in range(solver.n_ctrl_dirichlet):
                dofs_j = solver.dirichlet_dofs[j]
                for m in range(num_steps):
                    u_dirichlet_funcs_time[m][j].x.array[dofs_j] = \
                        u_dir_backup[m][j][dofs_j] - alpha * dir_dirichlet[m][j][dofs_j]
                    u_dirichlet_funcs_time[m][j].x.scatter_forward()

        # Evaluate J at trial point
        Y_trial = solver.solve_forward(
            u_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time
        )
        J_trial, _, _, _, _ = solver.compute_cost(
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            Y_trial,
            T_ref
        )

        # Check Armijo condition
        if J_trial <= armijo_threshold:
            if rank == 0:
                print(f"    [LS] Found alpha={alpha:.3e} after {ls_iter+1} tries, J={J_trial:.6e}")
            return alpha, Y_trial, J_trial

        # Reduce step size
        alpha *= rho
        armijo_threshold = J_current - c * alpha * dir_dot_grad

    # Max iterations reached - return last alpha and last trial state/cost
    if rank == 0:
        print(f"    [LS] Max iter reached, using alpha={alpha:.3e}")
    return alpha, Y_trial, J_trial

def optimization_time_dependent(args):
    """Time-dependent optimal control with adjoint-based gradient."""

    comm = MPI.COMM_WORLD
    rank = comm.rank

    import time
    t0 = time.perf_counter()

    if rank == 0:
        logger.info("\n" + "=" * 70)
        logger.info("TIME-DEPENDENT OPTIMAL CONTROL (Segregated Approach)")
        logger.info("=" * 70)

    # Setup
    L = args.L
    H = args.H if args.H is not None else L
    dt = args.dt
    T_final_time = args.T_final
    num_steps = int(T_final_time / dt)
    Nt = num_steps + 1
    k_val = args.k
    rho = args.rho
    c = args.c
    T_ambient = args.T_ambient
    T_ref_base = args.T_ref
    T_ref = T_ref_base  # initial value for logging
    T_ref_func = None
    if args.T_ref_func is not None:
        parts = args.T_ref_func.split(",")
        if len(parts) != 2:
            raise ValueError("T-ref-func requires: func_type,param")
        func_type = parts[0].strip()
        param = float(parts[1])
        if func_type not in ["const", "sin", "cos", "tanh"]:
            raise ValueError(f"func_type must be one of const/sin/cos/tanh (got: {func_type})")
        T_ref_func = (func_type, param)


    if rank == 0:
        logger.info(f"Domain: {L * 100:.1f}cm x {H * 100:.1f}cm")
        logger.info(f"Time: {T_final_time}s, dt={dt}s, steps={num_steps}, Nt={Nt}")
        logger.info(f"Physical: k={k_val}, rho={rho}, c={c}")
        logger.info(f"T_ambient={T_ambient}°C, T_ref={T_ref}°C")

    # Parse zones
    ctrl_dirichlet = [
        parse_boundary_segment(s) for s in args.control_boundary_dirichlet
    ]
    ctrl_neumann = [parse_boundary_segment(s) for s in args.control_boundary_neumann]
    ctrl_distributed_boxes = [parse_box(s) for s in args.control_distributed]
    robin_segments = [parse_robin_segment(s) for s in args.robin_boundary]
    dirichlet_bcs = [parse_dirichlet_bc(s) for s in args.dirichlet_bc]
    dirichlet_disturbances = [parse_dirichlet_disturbance(s) for s in args.dirichlet_disturbance]
    neumann_bcs = [parse_neumann_bc(s) for s in getattr(args, "neumann_bc", []) or []]
    # Prescribed Neumann must not overlap with control Neumann (same segment)
    ctrl_neumann_set = {(s, t0, t1) for s, t0, t1 in ctrl_neumann}
    for side, tmin, tmax, val in neumann_bcs:
        if (side, tmin, tmax) in ctrl_neumann_set:
            raise ValueError(
                f"Prescribed Neumann segment ({side},{tmin},{tmax}) cannot be the same as a control Neumann segment."
            )
    target_boxes = [parse_box(s) for s in args.target_zone]
    target_boundaries = [parse_boundary_segment(s) for s in args.target_boundary]
    constraint_boxes = [parse_box(s) for s in args.constraint_zone]
    neumann_by_side = {"x0": [], "xL": [], "y0": [], "yL": []}
    for side, tmin, tmax in ctrl_neumann:
        neumann_by_side[side].append((tmin, tmax))
    fully_neumann_count = 0
    for side in ["x0", "xL", "y0", "yL"]:
        segments = neumann_by_side[side]
        total_coverage = sum(tmax - tmin for tmin, tmax in segments)
        if total_coverage >= 0.99:
            fully_neumann_count += 1
    if fully_neumann_count == 4 and len(ctrl_dirichlet) == 0:
        if rank == 0:
            logger.error("ERROR: All 4 sides have only Neumann conditions without any Dirichlet.")
            logger.error("Problem is ill-posed. Add at least one Dirichlet segment (fixed or controlled).")
        raise ValueError("Ill-posed problem: 4 Neumann sides without Dirichlet")
    # ------------------------------------------------------------
    # Safety check: no controls and no zones → pure forward run
    # ------------------------------------------------------------
    if (
        len(ctrl_dirichlet) == 0
        and len(ctrl_neumann) == 0
        and len(ctrl_distributed_boxes) == 0
        and len(target_boxes) == 0
        and len(constraint_boxes) == 0
    ):
        if rank == 0:
            logger.warning("No controls, no target zones, no constraints defined.")
            logger.warning("Running pure forward simulation (J ≡ 0).")
            # args.no_vtk = True

    if rank == 0:
        logger.debug(f"constraint_boxes = {constraint_boxes}")
        logger.info("\nCONTROL ZONES")
        logger.info(f"  Boundary Dirichlet: {len(ctrl_dirichlet)}")
        for i, seg in enumerate(ctrl_dirichlet):
            logger.info(f"    {i + 1}. {seg}")
        logger.info(f"  Boundary Neumann: {len(ctrl_neumann)}")
        for i, seg in enumerate(ctrl_neumann):
            logger.info(f"    {i + 1}. {seg}")
        logger.info(f"  Prescribed Neumann (k∂_n T=g): {len(neumann_bcs)}")
        for i, (side, t0, t1, g) in enumerate(neumann_bcs):
            logger.info(f"    {i + 1}. ({side},{t0},{t1}) g={g}")
        logger.info(f"  Distributed boxes: {len(ctrl_distributed_boxes)}")
        for i, box in enumerate(ctrl_distributed_boxes):
            logger.info(f"    {i + 1}. {box}")
        logger.info("\nTARGET ZONES")
        for i, box in enumerate(target_boxes):
            logger.info(f"  {i + 1}. {box}")
        logger.info("\nCONSTRAINT ZONES")
        for i, box in enumerate(constraint_boxes):
            logger.info(f"  {i + 1}. {box}")

    # Mesh
    domain = create_mesh(args.n, L, H)
    V = functionspace(domain, ("Lagrange", 2))

    # Pre-compute T_ref values for each time step
    if args.T_ref_func_xt is not None:
        # T_ref depends on space and time
        parts = args.T_ref_func_xt.split(",")
        if len(parts) != 2:
            raise ValueError("T-ref-func-xt requires: func_type,param")
        func_type_xt = parts[0].strip()
        param_xt = float(parts[1])

        T_ref_values = []
        for step in range(num_steps + 1):
            t = step * dt
            T_ref_func_step = Function(V)
            if func_type_xt == "sin_x_minus_t":
                # r(x,t) = sin(param * (x - t))
                T_ref_func_step.interpolate(lambda x, t=t, omega=param_xt: np.sin(omega * (x[0] - t)))
            elif func_type_xt == "sin_x_plus_t":
                # r(x,t) = sin(param * (x + t))
                T_ref_func_step.interpolate(lambda x, t=t, omega=param_xt: np.sin(omega * (x[0] + t)))
            elif func_type_xt == "cos_x_minus_t":
                # r(x,t) = cos(param * (x - t))
                T_ref_func_step.interpolate(lambda x, t=t, omega=param_xt: np.cos(omega * (x[0] - t)))
            elif func_type_xt == "one_minus_y_sin_x_minus_t":
            # r(x,y,t) = (1 - y) * sin(x - t)
                T_ref_func_step.interpolate(lambda x, t=t: (1 - x[1]) * np.sin(x[0] - t))
            elif func_type_xt == "x_pi_minus_x_sin_x_minus_t":
            # r(x,t) = x(π-x) * sin(x - t) for Example 6.4
                T_ref_func_step.interpolate(lambda x, t=t: x[0] * (np.pi - x[0]) * np.sin(x[0] - t))
            else:
                raise ValueError(f"Unknown func_type_xt: {func_type_xt}")
            T_ref_values.append(T_ref_func_step)
    elif args.T_ref_func is not None:
        # T_ref depends only on time (scalar per step)
        parts = args.T_ref_func.split(",")
        if len(parts) != 2:
            raise ValueError("T-ref-func requires: func_type,param")
        func_type = parts[0].strip()
        param = float(parts[1])

        T_ref_values = np.zeros(num_steps + 1)
        for step in range(num_steps + 1):
            t = step * dt
            if func_type == "const":
                T_ref_values[step] = param
            elif func_type == "sin":
                T_ref_values[step] = param * math.sin(t)
            elif func_type == "cos":
                T_ref_values[step] = param * math.cos(t)
            elif func_type == "tanh":
                T_ref_values[step] = param * math.tanh(t)
    else:
        # T_ref is constant
        T_ref_values = np.full(num_steps + 1, T_ref_base)

    T_ref = T_ref_base  # for backward compatibility with function calls
    
    if rank == 0:
        logger.debug(f"Domain bounds: x=[0, {L}], y=[0, {L}]")
        logger.debug("Target zones in physical coords:")
        for i, (xmin, xmax, ymin, ymax) in enumerate(target_boxes):
            logger.debug(
                f"  Zone {i + 1}: x=[{xmin * L}, {xmax * L}], y=[{ymin * L}, {ymax * L}]"
            )

    # Solver
    solver = TimeDepHeatSolver(
        domain,
        V,
        dt,
        num_steps,
        k_val,
        rho,
        c,
        T_ambient,
        ctrl_dirichlet,
        ctrl_neumann,
        ctrl_distributed_boxes,
        target_boxes,
        target_boundaries,
        constraint_boxes,
        L,
        H,
        robin_segments,
        dirichlet_bcs,
        dirichlet_disturbances,
        neumann_bcs,
        args.alpha_track,
        args.alpha_u,
        args.gamma_u,
        args.beta_u,
        args.dirichlet_spatial_reg,
        getattr(args, "ksp_type", "gmres"),
        getattr(args, "ksp_rtol", 1e-10),
    )
    solver.T_ref_values = T_ref_values

    # Timing dictionary for profiling
    timing = {
        'forward': 0.0,
        'adjoint': 0.0,
        'gradient': 0.0,
        'cost': 0.0,
        'armijo': 0.0,
        'armijo_fwd': 0.0,
        'sc_update': 0.0,
        'total': 0.0,
    }

    # Neumann control: space-time Function (P2 in space)
    u_neumann_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_neumann):
            q = Function(V)
            q.x.array[:] = 0.0
            dofs_i = solver.neumann_dofs[i]
            q.x.array[dofs_i] = args.u_init
            q.x.scatter_forward()
            row.append(q)
        u_neumann_funcs_time.append(row)

    # Dirichlet control: Function(V) per zone per time step
    u_dirichlet_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_dirichlet):
            uD_dir = Function(V)
            uD_dir.x.array[:] = 0.0
            dofs_i = solver.dirichlet_dofs[i]
            uD_dir.x.array[dofs_i] = args.u_init
            uD_dir.x.scatter_forward()
            row.append(uD_dir)
        u_dirichlet_funcs_time.append(row)

    # Debug check (MPI-safe)
    if solver.n_ctrl_dirichlet > 0:
        uD_test = u_dirichlet_funcs_time[0][0]
        dofs_test = solver.dirichlet_dofs[0]

        if logger.isEnabledFor(logging.DEBUG):
            a_loc = uD_test.x.array
            if a_loc.size > 0:
                local_min = float(a_loc.min())
                local_max = float(a_loc.max())
            else:
                local_min = np.inf
                local_max = -np.inf
            gmin = comm.allreduce(local_min, op=MPI.MIN)
            gmax = comm.allreduce(local_max, op=MPI.MAX)
            b_loc = uD_test.x.array[dofs_test]
            if b_loc.size > 0:
                local_bmin = float(b_loc.min())
                local_bmax = float(b_loc.max())
            else:
                local_bmin = np.inf
                local_bmax = -np.inf
            gbmin = comm.allreduce(local_bmin, op=MPI.MIN)
            gbmax = comm.allreduce(local_bmax, op=MPI.MAX)
            if rank == 0:
                logger.debug(
                    f"uD_Dirichlet(t=0) global min/max = {gmin:.6e}, {gmax:.6e}"
                )
                logger.debug(
                    f"uD_Dirichlet(t=0) on ΓD min/max = {gbmin:.6e}, {gbmax:.6e}"
                )


    # Distributed control: space-time Function (P2 in space)
    u_distributed_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_distributed):
            uD = Function(V)
            uD.x.array[:] = 0.0
            
            dofs_c = solver.distributed_dofs[i]
            uD.x.array[dofs_c] = args.u_init
            uD.x.scatter_forward()
            row.append(uD)
        u_distributed_funcs_time.append(row)

    if solver.n_ctrl_distributed > 0 and logger.isEnabledFor(logging.DEBUG):
        uD_test = u_distributed_funcs_time[0][0]
        dofs_test = solver.distributed_dofs[0]
        a_loc = uD_test.x.array
        if a_loc.size > 0:
            loc_min, loc_max = float(a_loc.min()), float(a_loc.max())
        else:
            loc_min, loc_max = np.inf, -np.inf
        gmin = comm.allreduce(loc_min, op=MPI.MIN)
        gmax = comm.allreduce(loc_max, op=MPI.MAX)
        b_loc = uD_test.x.array[dofs_test]
        if b_loc.size > 0:
            loc_bmin, loc_bmax = float(b_loc.min()), float(b_loc.max())
        else:
            loc_bmin, loc_bmax = np.inf, -np.inf
        gbmin = comm.allreduce(loc_bmin, op=MPI.MIN)
        gbmax = comm.allreduce(loc_bmax, op=MPI.MAX)
        n_dofs_local = len(dofs_test)
        n_dofs_global = comm.allreduce(n_dofs_local, op=MPI.SUM)
        if rank == 0:
            logger.debug(f"uD(t=0) global min/max = {gmin:.6e}, {gmax:.6e}")
            logger.debug(f"uD(t=0) on Ωc min/max = {gbmin:.6e}, {gbmax:.6e}")
            logger.debug(f"Number of DOFs in Ωc (global): {n_dofs_global}")

    has_constraints = len(constraint_boxes) > 0

    # Constraint time window (always defined, even without constraints)
    sc_start_time = args.sc_start_time if args.sc_start_time is not None else 0.0
    sc_end_time = args.sc_end_time if args.sc_end_time is not None else T_final_time
    sc_start_step = int(sc_start_time / dt)
    sc_end_step = int(sc_end_time / dt)
    sc_start_step = max(0, min(sc_start_step, num_steps))
    sc_end_step = max(sc_start_step, min(sc_end_step, num_steps))

    solver.set_constraint_params(
        args.sc_type,
        args.sc_lower,
        args.sc_upper,
        args.beta,
        sc_start_step,
        sc_end_step,
    )

    # Initialize multipliers to zero (always, even without constraints)
    # FIX: Use numpy arrays instead of Function lists to avoid MPI communicator exhaustion
    n_dofs_Vc = solver.Vc.dofmap.index_map.size_local + solver.Vc.dofmap.index_map.num_ghosts
    solver.mu_lower_time = np.zeros((num_steps + 1, n_dofs_Vc), dtype=np.float64)
    solver.mu_upper_time = np.zeros((num_steps + 1, n_dofs_Vc), dtype=np.float64)
    # Working Function for assembly operations
    solver._mu_work = Function(solver.Vc)

    if rank == 0 and has_constraints:
        logger.info(f"\nPATH-CONSTRAINT Window: t ∈ [{sc_start_time:.1f}, {sc_end_time:.1f}]s")
        logger.info(f"PATH-CONSTRAINT Steps: m ∈ [{sc_start_step}, {sc_end_step}] (of {num_steps})")
        if args.sc_lower is not None:
            logger.info(f"PATH-CONSTRAINT Lower bound: T ≥ {args.sc_lower:.1f}°C")
        if args.sc_upper is not None:
            logger.info(f"PATH-CONSTRAINT Upper bound: T ≤ {args.sc_upper:.1f}°C")
        logger.info(f"PATH-CONSTRAINT Initialized {len(solver.mu_lower_time)} multiplier functions")

    if rank == 0:
        x_cells = domain.geometry.x
        cell_dofmap = domain.geometry.dofmap
        cell_centers = x_cells[cell_dofmap].mean(axis=1)

        for i, marker in enumerate(solver.target_markers):
            marked_cells_idx = np.where(marker)[0]
            logger.debug(f"Target zone {i + 1}: {len(marked_cells_idx)} marked cells")
            if len(marked_cells_idx) > 0:
                centers = cell_centers[marked_cells_idx]
                for j, c in enumerate(centers[:5]):
                    logger.debug(f"  Cell {j}: ({c[0]:.4f}, {c[1]:.4f})")

        logger.info("\nOPTIMIZATION")
        logger.info(f"  Boundary controls: Dirichlet={solver.n_ctrl_dirichlet}, Neumann={solver.n_ctrl_neumann}")
        logger.info(f"  Box controls (Function(V) per step): {solver.n_ctrl_distributed}")
        logger.info(f"  Control time steps (intervals): {num_steps}  (State nodes Nt={Nt})")
        logger.info(f"  SC: beta={args.beta}, maxit={args.sc_maxit}")
        logger.info(f"  Gradient descent: lr={args.lr}, inner_maxit={args.inner_maxit}")
        logger.info(f"  Regularization: α_track={args.alpha_track}, α_u={args.alpha_u}, γ_u={args.gamma_u}")
        logger.debug("Target zones cells:")
        for i, marker in enumerate(solver.target_markers):
            n_cells = np.sum(marker)
            logger.debug(f"  Zone {i + 1}: {n_cells} cells marked")

        if has_constraints and args.sc_maxit > 0:
            logger.info("\nSC LOOP Starting...\n")
        else:
            logger.info("\nSC LOOP skipped (no constraints or sc_maxit=0)\n")

    u_controls = np.zeros((0, num_steps), dtype=float)
    sc_iter = -1

    outer_maxit = args.sc_maxit if has_constraints else 1
    delta_mu, feas_inf = float("inf"), float("inf")

    # Moreau-Yosida loop
    for sc_iter in range(outer_maxit):

        # When inner_maxit=0, do one forward so Y_all/J are defined for SC and final output
        if args.inner_maxit == 0:
            Y_all = solver.solve_forward(
                u_neumann_funcs_time,
                u_distributed_funcs_time,
                u_dirichlet_funcs_time,
            )
            J, J_track, J_reg_L2, J_reg_H1, J_penalty = solver.compute_cost(
                u_distributed_funcs_time,
                u_neumann_funcs_time,
                u_dirichlet_funcs_time,
                Y_all,
                T_ref,
            )

        # Inner gradient descent loop
        for inner_iter in range(args.inner_maxit):
            t0_fwd = time_module.perf_counter() if PROFILE_TIMING else 0
            Y_all = solver.solve_forward(
                u_neumann_funcs_time,
                u_distributed_funcs_time,
                u_dirichlet_funcs_time,
            )
            timing['forward'] += time_module.perf_counter() - t0_fwd if PROFILE_TIMING else 0

            t0_cost = time_module.perf_counter() if PROFILE_TIMING else 0
            J, J_track, J_reg_L2, J_reg_H1, J_penalty = solver.compute_cost(
                u_distributed_funcs_time,
                u_neumann_funcs_time,
                u_dirichlet_funcs_time,
                Y_all,
                T_ref,
            )
            timing['cost'] += time_module.perf_counter() - t0_cost if PROFILE_TIMING else 0

            t0_adj = time_module.perf_counter() if PROFILE_TIMING else 0
            P_all = solver.solve_adjoint(Y_all, T_ref)
            timing['adjoint'] += time_module.perf_counter() - t0_adj if PROFILE_TIMING else 0

            t0_grad = time_module.perf_counter() if PROFILE_TIMING else 0
            grad = solver.compute_gradient(
                u_controls,
                Y_all,
                P_all,
                u_distributed_funcs_time,
                u_neumann_funcs_time,
                u_dirichlet_funcs_time,
            )
            timing['gradient'] += time_module.perf_counter() - t0_grad if PROFILE_TIMING else 0

#             # Update Dirichlet controls on ΓD
#             if solver.n_ctrl_dirichlet > 0:
#                 for j in range(solver.n_ctrl_dirichlet):
#                     dofs_j = solver.dirichlet_dofs[j]
#                     for m in range(num_steps):
#                         uD = u_dirichlet_funcs_time[m][j]
#                         gD = solver.grad_u_dirichlet_time[m][j]
#
#                         uD.x.array[dofs_j] -= args.lr * gD.x.array[dofs_j]
#                         uD.x.scatter_forward()
#
#             # Update Neumann controls
#             if solver.n_ctrl_neumann > 0:
#                 for i in range(solver.n_ctrl_neumann):
#                     dofs_i = solver.neumann_dofs[i]
#                     for m in range(num_steps):
#                         q = u_neumann_funcs_time[m][i]
#                         g = solver.grad_q_neumann_time[m][i]
#                         q.x.array[dofs_i] -= args.lr * g.x.array[dofs_i]
#                         q.x.scatter_forward()
#                 if solver.domain.comm.rank == 0:
#                     logger.debug(
#                         f"q(t0) min/max = {u_neumann_funcs_time[0][0].x.array.min()}, {u_neumann_funcs_time[0][0].x.array.max()}"
#                     )
#                     logger.debug(
#                         f"q(tend) min/max = {u_neumann_funcs_time[-1][0].x.array.min()}, {u_neumann_funcs_time[-1][0].x.array.max()}"
#                     )
#
#             # Update distributed controls
#             if solver.n_ctrl_distributed > 0:
#                 for j in range(solver.n_ctrl_distributed):
#                     dofs_j = solver.distributed_dofs[j]
#                     for m in range(num_steps):
#                         uD = u_distributed_funcs_time[m][j]
#                         gD = solver.grad_u_distributed_time[m][j]
#                         uD.x.array[dofs_j] -= args.lr * gD.x.array[dofs_j]
#                         uD.x.scatter_forward()
#                 if rank == 0:
#                     a0 = u_distributed_funcs_time[0][0].x.array
#                     logger.debug(
#                         f"uD0(t0) min/max = {float(a0.min())}, {float(a0.max())}"
#                     )

            if args.check_grad and sc_iter == 0 and inner_iter == 0:
                J0, fd, ad, rel = check_gradient_fd(
                    solver,
                    u_controls,
                    u_distributed_funcs_time,
                    u_neumann_funcs_time,
                    u_dirichlet_funcs_time,
                    T_ref,
                    eps=args.fd_eps,
                    seed=1,
                    m0=0,
                )

                if rank == 0:
                    print(
                        f"[GRAD-CHECK] J={J0:.6e}  FD={fd:.6e}  AD={ad:.6e}  rel_err={rel:.3e}"
                    )
                    print("[GRAD-CHECK] Done. Exiting now.")

                import sys

                sys.exit(0)
            if rank == 0:
                logger.debug(f"u_old = {u_controls.copy()}")
                logger.debug(f"grad = {grad.copy()}")

            # ============================================================
            # Conjugate Gradient direction computation
            # ============================================================
            # Compute gradient norm squared
            grad_sq = 0.0
            nloc = V.dofmap.index_map.size_local

            # Dirichlet gradient norm
            if solver.n_ctrl_dirichlet > 0:
                for j in range(solver.n_ctrl_dirichlet):
                    dofs_j = solver.dirichlet_dofs[j]
                    dofs_j_owned = dofs_j[dofs_j < nloc]
                    for m in range(num_steps):
                        gD = solver.grad_u_dirichlet_time[m][j]
                        grad_sq += float(np.sum(gD.x.array[dofs_j_owned] ** 2))

            # Neumann gradient norm
            if solver.n_ctrl_neumann > 0:
                for i in range(solver.n_ctrl_neumann):
                    dofs_i = solver.neumann_dofs[i]
                    dofs_i_owned = dofs_i[dofs_i < nloc]
                    for m in range(num_steps):
                        g = solver.grad_q_neumann_time[m][i]
                        grad_sq += float(np.sum(g.x.array[dofs_i_owned] ** 2))

            # Distributed gradient norm
            if solver.n_ctrl_distributed > 0:
                for j in range(solver.n_ctrl_distributed):
                    dofs_j = solver.distributed_dofs[j]
                    dofs_j_owned = dofs_j[dofs_j < nloc]
                    for m in range(num_steps):
                        gD = solver.grad_u_distributed_time[m][j]
                        grad_sq += np.sum(gD.x.array[dofs_j_owned] ** 2)

            grad_sq = comm.allreduce(grad_sq, op=MPI.SUM)
            grad_norm = np.sqrt(grad_sq)

            # Conjugate Gradient: compute beta and direction
            # Initialize direction arrays on first iteration
            if inner_iter == 0:
                grad_sq_old = 0.0
                # Initialize direction = -gradient (steepest descent for first iter)
                dir_dirichlet = [[solver.grad_u_dirichlet_time[m][j].x.array.copy() 
                                  for j in range(solver.n_ctrl_dirichlet)]
                                 for m in range(num_steps)] if solver.n_ctrl_dirichlet > 0 else None
                dir_neumann = [[solver.grad_q_neumann_time[m][i].x.array.copy()
                                for i in range(solver.n_ctrl_neumann)]
                               for m in range(num_steps)] if solver.n_ctrl_neumann > 0 else None
                dir_distributed = [[solver.grad_u_distributed_time[m][j].x.array.copy()
                                    for j in range(solver.n_ctrl_distributed)]
                                   for m in range(num_steps)] if solver.n_ctrl_distributed > 0 else None
            else:
                # Fletcher-Reeves beta
                beta = grad_sq / grad_sq_old if grad_sq_old > 1e-16 else 0.0
                
                # Update direction: d = -grad + beta * d_old
                if solver.n_ctrl_dirichlet > 0:
                    for j in range(solver.n_ctrl_dirichlet):
                        for m in range(num_steps):
                            dir_dirichlet[m][j] = solver.grad_u_dirichlet_time[m][j].x.array.copy() + beta * dir_dirichlet[m][j]
                
                if solver.n_ctrl_neumann > 0:
                    for i in range(solver.n_ctrl_neumann):
                        for m in range(num_steps):
                            dir_neumann[m][i] = solver.grad_q_neumann_time[m][i].x.array.copy() + beta * dir_neumann[m][i]
                
                if solver.n_ctrl_distributed > 0:
                    for j in range(solver.n_ctrl_distributed):
                        for m in range(num_steps):
                            dir_distributed[m][j] = solver.grad_u_distributed_time[m][j].x.array.copy() + beta * dir_distributed[m][j]

            # Save grad_sq for next iteration
            grad_sq_old = grad_sq

            # Compute <direction, gradient> for Armijo (= grad_sq for first iter)
            dir_dot_grad = 0.0
            if solver.n_ctrl_dirichlet > 0:
                for j in range(solver.n_ctrl_dirichlet):
                    dofs_j = solver.dirichlet_dofs[j]
                    dofs_j_owned = dofs_j[dofs_j < nloc]
                    for m in range(num_steps):
                        dir_dot_grad += float(np.sum(dir_dirichlet[m][j][dofs_j_owned] * 
                                                     solver.grad_u_dirichlet_time[m][j].x.array[dofs_j_owned]))
            if solver.n_ctrl_neumann > 0:
                for i in range(solver.n_ctrl_neumann):
                    dofs_i = solver.neumann_dofs[i]
                    dofs_i_owned = dofs_i[dofs_i < nloc]
                    for m in range(num_steps):
                        dir_dot_grad += float(np.sum(dir_neumann[m][i][dofs_i_owned] *
                                                     solver.grad_q_neumann_time[m][i].x.array[dofs_i_owned]))
            if solver.n_ctrl_distributed > 0:
                for j in range(solver.n_ctrl_distributed):
                    dofs_j = solver.distributed_dofs[j]
                    dofs_j_owned = dofs_j[dofs_j < nloc]
                    for m in range(num_steps):
                        dir_dot_grad += float(np.sum(dir_distributed[m][j][dofs_j_owned] *
                                                     solver.grad_u_distributed_time[m][j].x.array[dofs_j_owned]))
            dir_dot_grad = comm.allreduce(dir_dot_grad, op=MPI.SUM)

            # Armijo line search with CG direction (returns trajectory to avoid redundant forward)
            t0_armijo = time_module.perf_counter() if PROFILE_TIMING else 0
            alpha, Y_trial_all, J_trial = armijo_line_search(
                solver,
                u_distributed_funcs_time,
                u_neumann_funcs_time,
                u_dirichlet_funcs_time,
                dir_distributed,
                dir_neumann,
                dir_dirichlet,
                J,  # Current objective
                dir_dot_grad,
                T_ref,
                num_steps,
                alpha_init=args.lr,  # Use args.lr as initial guess
                rho=0.5,
                c=1e-4,
                max_iter=5,  # reduced from 8 - usually converges in 1-2 iterations
                comm=comm,
                rank=rank
            )
            timing['armijo'] += time_module.perf_counter() - t0_armijo if PROFILE_TIMING else 0

            # Controls and state: reuse trajectory from line search (no redundant forward)
            Y_all = Y_trial_all
            J, J_track, J_reg_L2, J_reg_H1, J_penalty = solver.compute_cost(
                u_distributed_funcs_time,
                u_neumann_funcs_time,
                u_dirichlet_funcs_time,
                Y_all,
                T_ref,
            )

            if solver.n_ctrl_dirichlet > 0:
                dofs = solver.bc_dofs_list_dirichlet[0]
                u_loc = u_dirichlet_funcs_time[0][0].x.array[dofs]
                tag = "Dir0"
            elif solver.n_ctrl_neumann > 0:
                u_loc = u_neumann_funcs_time[0][0].x.array
                tag = "Neu0"
            elif solver.n_ctrl_distributed > 0:
                u_loc = u_distributed_funcs_time[0][0].x.array
                tag = "Box0"
            else:
                u_loc = None
                tag = None

            if u_loc is None or u_loc.size == 0:
                loc_min = float("inf")
                loc_max = float("-inf")
            else:
                loc_min = float(u_loc.min())
                loc_max = float(u_loc.max())

            gmin = comm.allreduce(loc_min, op=MPI.MIN)
            gmax = comm.allreduce(loc_max, op=MPI.MAX)

            if rank == 0:
                if tag is None or gmin == float("inf") or gmax == float("-inf"):
                    u_info = "no controls"
                else:
                    u_info = f"{tag}=[{gmin:.2f},{gmax:.2f}]"


                print(
                    f"  Inner {inner_iter:2d}: J={J:.3e} (track={J_track:.3e}, "
                    f"L2={J_reg_L2:.3e}, H1={J_reg_H1:.3e}), "
                    f"||∇J||={grad_norm:.3e}, u: {u_info}"
                )

            if grad_norm < args.grad_tol:
                if rank == 0:
                    print(f"  [Inner converged at iter {inner_iter}]")
                break

        # functionspace/Function/interpolate are MPI collective
        Vc = solver.Vc
        Tcell = Function(Vc)
        Tcell.interpolate(Y_all[-1])
        if rank == 0:
            mask = solver.sc_marker.astype(bool)
            n_local = len(mask)
            logger.debug(f"len(sc_marker) = {len(solver.sc_marker)}")
            logger.debug(f"len(Tcell DG0) = {len(Tcell.x.array)}")
            logger.debug(f"len(Y_all[-1] V) = {len(Y_all[-1].x.array)}")
            if np.any(mask):
                logger.debug(
                    f"Y_all[-1] on constraint Tmin/Tmax = {float(Tcell.x.array[:n_local][mask].min())}, {float(Tcell.x.array[:n_local][mask].max())}"
                )
            else:
                logger.debug("No cells in constraint mask (constraint zone empty).")
        if not has_constraints:
            delta_mu, feas_inf = 0.0, 0.0
            if domain.comm.rank == 0:
                logger.debug("SC skipped mu update (no constraint zone).")
        else:
            t0_sc = time_module.perf_counter() if PROFILE_TIMING else 0
            delta_mu, feas_inf = solver.update_multiplier_mu(
                Y_all,
                args.sc_type,
                args.sc_lower,
                args.sc_upper,
                args.beta,
                sc_start_step,
                sc_end_step,
            )
            timing['sc_update'] += time_module.perf_counter() - t0_sc if PROFILE_TIMING else 0
            if rank == 0:
                logger.info(f"SC OUTER k={sc_iter} feas_inf={feas_inf:.6e} delta_mu={delta_mu:.6e}")

        Y_mu = solver.solve_forward(
            u_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time,
        )
        J_mu, Jt_mu, JL2_mu, JH1_mu, Jp_mu = solver.compute_cost(
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            Y_mu,
            T_ref,
        )

        if rank == 0:
            logger.debug(
                f"J-AFTER-MU J={J_mu:.6e} (track={Jt_mu:.6e}, L2={JL2_mu:.6e}, H1={JH1_mu:.6e})"
            )

        if rank == 0:
            # Get max multiplier across all time steps in constraint window
            muL_max = 0.0
            muU_max = 0.0
            for m in range(sc_start_step, sc_end_step + 1):
                muL_max = max(muL_max, np.max(solver.mu_lower_time[m]))
                muU_max = max(muU_max, np.max(solver.mu_upper_time[m]))
            logger.debug(f"μL_max={muL_max:.3e}, μU_max={muU_max:.3e}")

            if u_controls.size > 0:
                logger.debug(
                    f"u_mean={float(np.mean(u_controls)):.1f}, u_std={float(np.std(u_controls)):.1f}"
                )
            else:
                logger.debug("u_mean=n/a, u_std=n/a (no scalar controls)")

        if not has_constraints:
            break

        if feas_inf < args.sc_tol and delta_mu < 1e-4:
            if rank == 0:
                print(f"[CONVERGED] feas_inf < {args.sc_tol}\n")
            break


    if rank == 0:
        if has_constraints:
            logger.debug(
                f"CHECK-MU-FINAL max muL at final = {max(np.max(solver.mu_lower_time[m]) for m in range(sc_start_step, sc_end_step + 1))}"
            )
        else:
            logger.debug("CHECK-MU-FINAL no constraints -> skipping mu check")
    # Final results
    Y_final_all = solver.solve_forward(
        u_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time
    )
    J_final, J_track_final, J_reg_L2_final, J_reg_H1_final, J_penalty_final = solver.compute_cost(
        u_distributed_funcs_time,
        u_neumann_funcs_time,
        u_dirichlet_funcs_time,
        Y_final_all,
        T_ref,
    )
    # ============================================================
    # L2 error vs time on Omega_c (FINAL trajectory only)
    # writes: l2_error_Omegac.dat
    # ============================================================
    from dolfinx.fem import form, assemble_scalar

    dx_err = ufl.Measure("dx", domain=domain)
    X = ufl.SpatialCoordinate(domain)

    # Check if we have target zones or target boundaries
    has_target_zone = len(solver.target_markers) > 0
    has_target_boundary = len(solver.target_boundary_ds) > 0

    if has_target_zone:
        # Usa la prima target zone come Omega_c
        marker_err = solver.target_markers[0]

        # indicatrice cellwise su DG0 (una volta sola)
        V0_err = solver.Vc
        chi_err = Function(V0_err)
        nloc_err = V0_err.dofmap.index_map.size_local
        chi_err.x.array[:nloc_err] = marker_err.astype(PETSc.ScalarType)
        chi_err.x.scatter_forward()

        if rank == 0:
            f_err = open("l2_error_Omegac.dat", "w")
            f_err.write("# t   L2_Omegac\n")
        else:
            f_err = None

        for m, Tm in enumerate(Y_final_all):  # Nt = num_steps+1
            t = m * dt
            r_expr = (1.0 - X[1]) * ufl.sin(X[0] - t)
            e_expr = r_expr - Tm

            val_loc = assemble_scalar(form((e_expr * e_expr) * chi_err * dx_err))
            val = comm.allreduce(val_loc, op=MPI.SUM)
            l2 = float(np.sqrt(val))

            if rank == 0:
                f_err.write(f"{t:.16e} {l2:.16e}\n")

        if rank == 0:
            f_err.close()
            print("Wrote: l2_error_Omegac.dat")

    elif has_target_boundary:
        ds_err = solver.target_boundary_ds[0]
        if rank == 0:
            f_err = open("l2_error_boundary.dat", "w")
            f_err.write("# t   L2_boundary\n")
        else:
            f_err = None
        for m, Tm in enumerate(Y_final_all):
            t = m * dt
            r_expr = X[0] * (np.pi - X[0]) * ufl.sin(X[0] - t)
            e_expr = r_expr - Tm
            val_loc = assemble_scalar(form((e_expr * e_expr) * ds_err))
            val = comm.allreduce(val_loc, op=MPI.SUM)
            l2 = float(np.sqrt(val))
            if rank == 0:
                f_err.write(f"{t:.16e} {l2:.16e}\n")
        if rank == 0:
            f_err.close()
            print("Wrote: l2_error_boundary.dat")
    #========================================================================
    n_ctrl_total = (
        solver.n_ctrl_dirichlet + solver.n_ctrl_neumann + solver.n_ctrl_distributed
    )
    if rank == 0:
        print("[FINAL RESULTS]")
        print(f"  SC iterations: {sc_iter + 1}")
        print(
            f"  J_final = {J_final:.3e} (track={J_track_final:.3e}, "
            f"L2={J_reg_L2_final:.3e}, H1={J_reg_H1_final:.3e})"
        )
    # ============================================================
    # H1 temporal regularization diagnostics (distributed controls)
    # ============================================================
    if solver.n_ctrl_distributed > 0:
        from dolfinx.fem import form, assemble_scalar

        comm = domain.comm
        dx = ufl.Measure("dx", domain=domain)

        if logger.isEnabledFor(logging.DEBUG):
            if rank == 0:
                logger.debug("\n" + "=" * 70)
                logger.debug("H1 TEMPORAL REGULARIZATION CHECK (DISTRIBUTED)")
                logger.debug("=" * 70)
                logger.debug(
                    f"gamma_u = {solver.gamma_u:.6e}, dt = {solver.dt:.6e}, num_steps = {num_steps}"
                )
            for j in range(solver.n_ctrl_distributed):
                chiV = solver.chi_distributed_V[j]
                meas_loc = assemble_scalar(form(chiV * dx))
                meas = comm.allreduce(meas_loc, op=MPI.SUM)
                den_loc = assemble_scalar(form(chiV * dx))
                den = comm.allreduce(den_loc, op=MPI.SUM)
                if rank == 0:
                    logger.debug(f"\nH1t-DIST zone {j}")
                    logger.debug(f"  meas(Ωc) ≈ {float(meas):.6e}")
                rough2 = 0.0
                max_step_L2 = 0.0
                sample_means = []
                stride = max(1, (num_steps - 1) // 10)
                sample_idx = list(range(0, num_steps, stride))
                if (num_steps - 1) not in sample_idx:
                    sample_idx.append(num_steps - 1)
                for m in sample_idx:
                    um = u_distributed_funcs_time[m][j]
                    num_loc = assemble_scalar(form(um * chiV * dx))
                    num = comm.allreduce(num_loc, op=MPI.SUM)
                    mean_um = float(num / den) if den > 1e-14 else float("nan")
                    sample_means.append((m * solver.dt, mean_um))
                for m in range(num_steps - 1):
                    u0 = u_distributed_funcs_time[m][j]
                    u1 = u_distributed_funcs_time[m + 1][j]
                    step_loc = assemble_scalar(form(((u1 - u0) ** 2) * chiV * dx))
                    step = float(comm.allreduce(step_loc, op=MPI.SUM))
                    rough2 += step
                    max_step_L2 = max(max_step_L2, step)
                pred_JH1 = 0.5 * solver.gamma_u / solver.dt * rough2
                rough = math.sqrt(rough2)
                avg_step = math.sqrt(rough2 / max(1, num_steps - 1))
                if rank == 0:
                    logger.debug(f"\nH1t-DIST zone {j}")
                    logger.debug(f"  predicted J_H1 = {pred_JH1:.6e}")
                    logger.debug(f"  roughness sqrt(sum ||Δu||^2) = {rough:.6e}")
                    logger.debug(
                        f"  avg step L2-norm over Ωc (sqrt(mean ||Δu||^2)) = {avg_step:.6e}"
                    )
                    logger.debug(f"  max step ||Δu||^2 over Ωc = {max_step_L2:.6e}")
                    logger.debug("  sampled mean(u) over Ωc:")
                    for t, mu in sample_means:
                        logger.debug(f"    t={t:8.1f}s  mean(u)={mu:+.6e}")

        # Control statistics
        for ic in range(n_ctrl_total):
            if ic < solver.n_ctrl_dirichlet:
                # Dirichlet: real control uD(t) on ΓD
                unit = "°C"
                j = ic
                label = f"Dirichlet {j}"
                dofs = solver.dirichlet_dofs[j]
                # spatial mean on ΓD per time step
                vals_t = np.array(
                    [
                        u_dirichlet_funcs_time[m][j].x.array[dofs].mean()
                        for m in range(num_steps)
                    ]
                )
                logger.info(
                    f"    {label} (REAL uD on ΓD): "
                    f"mean={vals_t.mean():.6e} {unit}, "
                    f"std={vals_t.std():.6e}, "
                    f"min={vals_t.min():.6e}, "
                    f"max={vals_t.max():.6e}"
                )
            elif ic < solver.n_ctrl_dirichlet + solver.n_ctrl_neumann:
                # Neumann: real control q(t) on Γ
                unit = "(flux units)"
                j = ic - solver.n_ctrl_dirichlet
                label = f"Neumann {j}"
                dofs = solver.neumann_dofs[j]
                # spatial mean on Γ per time step
                vals_t = np.array(
                    [
                        u_neumann_funcs_time[m][j].x.array[dofs].mean()
                        for m in range(num_steps)
                    ]
                )
                logger.info(
                    f"    {label} (REAL q on Gamma): "
                    f"mean={vals_t.mean():.6e} {unit}, "
                    f"std={vals_t.std():.6e}, "
                    f"min={vals_t.min():.6e}, "
                    f"max={vals_t.max():.6e}"
                )
            else:
                unit = "(source units)"
                j = ic - solver.n_ctrl_dirichlet - solver.n_ctrl_neumann
                label = f"Distributed {j}"
                # distributed control uD(t) in Ω (mean over all DOFs of V)
                vals_t = np.array(
                    [
                        u_distributed_funcs_time[m][j].x.array.mean()
                        for m in range(num_steps)
                    ]
                )
                logger.info(
                    f"    {label} (REAL uD in Omega): "
                    f"mean={vals_t.mean():.6e} {unit}, "
                    f"std={vals_t.std():.6e}, "
                    f"min={vals_t.min():.6e}, "
                    f"max={vals_t.max():.6e}"
                )

        # Time evolution (subsampled)
        if n_ctrl_total > 0:
            step_stride = max(1, num_steps // 10)
            logger.debug(
                f"T_final_time={T_final_time}, dt={dt}, num_steps={num_steps}, "
                f"u_controls.shape={u_controls.shape}, q_time_len={len(u_neumann_funcs_time)}"
            )
            for ic in range(n_ctrl_total):
                if ic < solver.n_ctrl_dirichlet:
                    j = ic
                    label = f"Dirichlet {j}"
                    unit = "°C"
                    dofs = solver.dirichlet_dofs[j]

                    def get_val(t_idx):
                        # spatial mean on ΓD at time t_idx
                        return float(
                            u_dirichlet_funcs_time[t_idx][j].x.array[dofs].mean()
                        )
                elif ic < solver.n_ctrl_dirichlet + solver.n_ctrl_neumann:
                    j = ic - solver.n_ctrl_dirichlet
                    label = f"Neumann {j}"
                    unit = "(flux units)"
                    dofs = solver.neumann_dofs[j]

                    def get_val(t_idx):
                        # spatial mean on Γ at time t_idx
                        return float(
                            u_neumann_funcs_time[t_idx][j].x.array[dofs].mean()
                        )
                else:
                    j = ic - solver.n_ctrl_dirichlet - solver.n_ctrl_neumann
                    label = f"Distributed {j}"
                    unit = "(source units)"

                    def get_val(t_idx):
                        return float(u_distributed_funcs_time[t_idx][j].x.array.mean())

                logger.info(
                    f"\n  Control {ic} evolution [{label}] (every {step_stride} steps):"
                )
                idxs = list(range(0, num_steps, step_stride))
                if (num_steps - 1) not in idxs:
                    idxs.append(num_steps - 1)
                for t_idx in idxs:
                    t_val = t_idx * dt
                    logger.info(f"    t={t_val:8.1f}s: u={get_val(t_idx):.6e} {unit}")
                # held-last (final control applied)
                logger.info(
                    f"    t={T_final_time:8.1f}s: u={get_val(num_steps - 1):.6e} {unit}  (held-last)"
                )

    T_final_diag = Y_final_all[-1]

    # Postprocessing: mean temperature in target zones
    # functionspace/Function/assemble_scalar are MPI collective
    V0_dg0 = solver.Vc
    from dolfinx.fem import assemble_scalar, form

    dx = ufl.Measure("dx", domain=domain)

    for i, marker in enumerate(solver.target_markers):
        chi_dg0 = Function(V0_dg0)
        n_local = V0_dg0.dofmap.index_map.size_local
        chi_dg0.x.array[:n_local] = marker.astype(PETSc.ScalarType)
        chi_dg0.x.scatter_forward()
        T_cell = Function(V0_dg0)
        T_cell.interpolate(T_final_diag)

        Vc = solver.Vc
        Tcell_sc = Function(Vc)
        Tcell_sc.interpolate(T_final_diag)

        zone_integral_local = assemble_scalar(form(T_cell * chi_dg0 * dx))
        chi_integral_local = assemble_scalar(form(chi_dg0 * dx))

        zone_integral = comm.allreduce(zone_integral_local, op=MPI.SUM)
        chi_integral = comm.allreduce(chi_integral_local, op=MPI.SUM)

        if rank == 0:
            mask = solver.sc_marker.astype(bool)
            n_local_sc = len(mask)
            if np.any(mask):
                logger.debug(
                    f"T_final_diag on constraint Tmin/Tmax = {float(Tcell_sc.x.array[:n_local_sc][mask].min())}, {float(Tcell_sc.x.array[:n_local_sc][mask].max())}"
                )
            else:
                logger.debug("No constraint cells -> skipping Tmin/Tmax check.")

            if chi_integral > 1e-12:
                T_mean = zone_integral / chi_integral
                print(f"\n  Target zone {i + 1}:")
                print(f"    T_mean ≈ {T_mean:.6f}°C")
                # === constraint deficit (diagnostic) using sc_lower/sc_upper ===
                if has_constraints:
                    if args.sc_type == "lower":
                        threshold = args.sc_lower
                        deficit = threshold - T_mean
                        if deficit <= 0:
                            print("    [OK] Lower constraint satisfied ✓")
                        else:
                            print(f"    [INFO] Lower deficit ≈ {deficit:.1f}°C")

                    elif args.sc_type == "upper":
                        threshold = args.sc_upper
                        excess = T_mean - threshold
                        if excess <= 0:
                            print("    [OK] Upper constraint satisfied ✓")
                        else:
                            print(f"    [INFO] Upper excess ≈ {excess:.1f}°C")

                    else:  # "box"
                        low = args.sc_lower
                        up = args.sc_upper
                        deficit_low = low - T_mean
                        excess_up = T_mean - up
                        viol = max(deficit_low, excess_up, 0.0)
                        if viol <= 0:
                            print("    [OK] Box constraint satisfied ✓")
                        else:
                            if deficit_low > 0:
                                print(f"    [INFO] Box violation (below) ≈ {deficit_low:.1f}°C")
                            else:
                                print(f"    [INFO] Box violation (above) ≈ {excess_up:.1f}°C")

            else:
                print(f"\n  Target zone {i + 1}: Empty or zero measure")

    # ============================================================
    # Save visualization output
    # ============================================================
    if rank == 0 and not args.no_vtk:
        logger.info("\n" + "=" * 70)
        logger.info("SAVING VISUALIZATION OUTPUT")
        logger.info("=" * 70 + "\n")

    P_final_all = solver.solve_adjoint(Y_final_all, T_ref)
    if not args.no_vtk:
        save_visualization_output(
            solver,
            Y_final_all,
            P_final_all,
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            sc_start_step,
            sc_end_step,
            args,
            num_steps,
            target_boxes,
        target_boundaries,
            ctrl_distributed_boxes,
        )

    # =======================
    # Metrics for test scripts
    # =======================
    runtime = time.perf_counter() - t0

    energy = J_reg_L2_final + J_reg_H1_final

    # max violation on curing window
    violation = 0.0
    if has_constraints:
        for m in range(sc_start_step, sc_end_step + 1):
            # interpolate nodal T to DG0 (cellwise) to match sc_marker
            Vc = solver.Vc
            Tcell = Function(Vc)
            Tcell.interpolate(Y_final_all[m])

            a_loc = Tcell.x.array
            mask = solver.sc_marker.astype(bool)
            if a_loc.size > 0 and mask.size > 0 and mask.any():
                a_sc = a_loc[:mask.size][mask]  # temperatures on constrained cells

                if args.sc_type == "lower":
                    viol_m = float((args.sc_lower - a_sc).max())
                    if viol_m < 0.0:
                        viol_m = 0.0

                elif args.sc_type == "upper":
                    viol_m = float((a_sc - args.sc_upper).max())
                    if viol_m < 0.0:
                        viol_m = 0.0

                else:  # "box"
                    violL = float((args.sc_lower - a_sc).max())
                    if violL < 0.0:
                        violL = 0.0
                    violU = float((a_sc - args.sc_upper).max())
                    if violU < 0.0:
                        violU = 0.0
                    viol_m = max(violL, violU)

                violation = max(violation, viol_m)


    metrics = {
        "J": float(J_final),
        "J_track": float(J_track_final),
        "J_reg_L2": float(J_reg_L2_final),
        "J_reg_H1": float(J_reg_H1_final),
        "J_penalty": float(J_penalty_final),
        "energy": float(energy),
        "violation": float(violation),
        "runtime": float(runtime),
    }

    # Add detailed timing if profiling enabled
    if PROFILE_TIMING:
        timing['total'] = runtime
        metrics['timing'] = timing

    return T_final_diag, metrics
