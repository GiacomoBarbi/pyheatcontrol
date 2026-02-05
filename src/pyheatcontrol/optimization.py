import numpy as np
import math
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Function, functionspace
from pyheatcontrol.parsing_utils import parse_box, parse_boundary_segment
from pyheatcontrol.mesh_utils import create_mesh
from pyheatcontrol.io_utils import save_visualization_output
from pyheatcontrol.solver import TimeDepHeatSolver
from pyheatcontrol.gradcheck import check_gradient_fd
from pyheatcontrol.logging_config import logger

def optimization_time_dependent(args):
    """Ottimizzazione con time-dependent control e adjoint-based gradient"""

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        logger.info("\n" + "="*70)
        logger.info("TIME-DEPENDENT OPTIMAL CONTROL (Segregated Approach)")
        logger.info("="*70)

    # Setup
    L = args.L
    dt = args.dt
    T_final_time = args.T_final
    num_steps = int(T_final_time / dt)
    Nt = num_steps + 1
    k_val = args.k
    rho = args.rho
    c = args.c
    T_ambient = args.T_ambient
    T_cure = args.T_cure

    if rank == 0:
        logger.info(f"Domain: {L*100:.1f}cm x {L*100:.1f}cm")
        logger.info(f"Time: {T_final_time}s, dt={dt}s, steps={num_steps}, Nt={Nt}")
        logger.info(f"Physical: k={k_val}, rho={rho}, c={c}")
        logger.info(f"T_ambient={T_ambient}°C, T_cure={T_cure}°C")

    # Parse zones
    ctrl_dirichlet = [parse_boundary_segment(s) for s in args.control_boundary_dirichlet]
    ctrl_neumann = [parse_boundary_segment(s) for s in args.control_boundary_neumann]
    ctrl_distributed_boxes = [parse_box(s) for s in args.control_distributed]
    target_boxes = [parse_box(s) for s in args.target_zone]
    constraint_boxes = [parse_box(s) for s in args.constraint_zone]
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
            logger.info(f"    {i+1}. {seg}")
        logger.info(f"  Boundary Neumann: {len(ctrl_neumann)}")
        for i, seg in enumerate(ctrl_neumann):
            logger.info(f"    {i+1}. {seg}")
        logger.info(f"  Distributed boxes: {len(ctrl_distributed_boxes)}")
        for i, box in enumerate(ctrl_distributed_boxes):
            logger.info(f"    {i+1}. {box}")
        logger.info("\nTARGET ZONES")
        for i, box in enumerate(target_boxes):
            logger.info(f"  {i+1}. {box}")
        logger.info("\nCONSTRAINT ZONES")
        for i, box in enumerate(constraint_boxes):
            logger.info(f"  {i+1}. {box}")

    # Mesh
    domain = create_mesh(args.n, L)
    V = functionspace(domain, ("Lagrange", 2))
    # Aggiungi:
    if rank == 0:
        logger.debug(f"Domain bounds: x=[0, {L}], y=[0, {L}]")
        logger.debug("Target zones in physical coords:")
        for i, (xmin, xmax, ymin, ymax) in enumerate(target_boxes):
            logger.debug(f"  Zone {i+1}: x=[{xmin*L}, {xmax*L}], y=[{ymin*L}, {ymax*L}]")

    # Solver
    solver = TimeDepHeatSolver(
        domain, V, dt, num_steps, k_val, rho, c, T_ambient,
        ctrl_dirichlet, ctrl_neumann, ctrl_distributed_boxes, target_boxes, constraint_boxes, L,
        args.alpha_track, args.alpha_u, args.gamma_u, args.beta_u, args.dirichlet_spatial_reg
    )

    # === NEW: Neumann control as space-time Function (P2 in space) ===
    u_neumann_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_neumann):
            q = Function(V)
            q.x.array[:] = 0.0  # Zero everywhere
            dofs_i = solver.neumann_dofs[i]
            q.x.array[dofs_i] = args.u_init  # u_init only on ΓN
            q.x.scatter_forward()
            row.append(q)
        u_neumann_funcs_time.append(row)

    # Dirichlet controls: Function(V) per zone per time step
    u_dirichlet_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_dirichlet):
            uD_dir = Function(V)
            uD_dir.x.array[:] = 0.0  # Zero everywhere
            dofs_i = solver.dirichlet_dofs[i]
            uD_dir.x.array[dofs_i] = args.u_init  # u_init only on ΓD
            uD_dir.x.scatter_forward()
            row.append(uD_dir)
        u_dirichlet_funcs_time.append(row)

    # Debug check
    if rank == 0 and solver.n_ctrl_dirichlet > 0:
        uD_test = u_dirichlet_funcs_time[0][0]
        dofs_test = solver.dirichlet_dofs[0]
        logger.debug(f"uD_Dirichlet(t=0) global min/max = {uD_test.x.array.min():.6e}, {uD_test.x.array.max():.6e}")
        logger.debug(f"uD_Dirichlet(t=0) on ΓD min/max = {uD_test.x.array[dofs_test].min():.6e}, {uD_test.x.array[dofs_test].max():.6e}")

    # === NEW: Distributed control as space-time Function (P2 in space) ===
    u_distributed_funcs_time = []
    for _ in range(num_steps):
        row = []
        for i in range(solver.n_ctrl_distributed):
            uD = Function(V)
            uD.x.array[:] = 0.0  # ✅ Zero everywhere
            # Set u_init only on DOFs in Ωc
            dofs_c = solver.distributed_dofs[i]
            uD.x.array[dofs_c] = args.u_init
            uD.x.scatter_forward()
            row.append(uD)
        u_distributed_funcs_time.append(row)

    # === DEBUG: Verify initialization ===
    if rank == 0 and solver.n_ctrl_distributed > 0:
        uD_test = u_distributed_funcs_time[0][0]
        dofs_test = solver.distributed_dofs[0]
        logger.debug(f"uD(t=0) global min/max = {uD_test.x.array.min():.6e}, {uD_test.x.array.max():.6e}")
        logger.debug(f"uD(t=0) on Ωc min/max = {uD_test.x.array[dofs_test].min():.6e}, {uD_test.x.array[dofs_test].max():.6e}")
        logger.debug(f"Number of DOFs in Ωc: {len(dofs_test)}")
        logger.debug(f"Total DOFs in V: {len(uD_test.x.array)}")

    has_constraints = (len(constraint_boxes) > 0)

    # --- definisci sempre la finestra (serve comunque per range/print) ---
    sc_start_time = args.sc_start_time if args.sc_start_time is not None else 0.0
    sc_end_time   = args.sc_end_time   if args.sc_end_time   is not None else T_final_time
    sc_start_step = int(sc_start_time / dt)
    sc_end_step   = int(sc_end_time   / dt)
    sc_start_step = max(0, min(sc_start_step, num_steps))
    sc_end_step   = max(sc_start_step, min(sc_end_step, num_steps))

    # --- inizializza SEMPRE mu_lower_time / mu_upper_time (tutti zero) ---
    Vc = functionspace(domain, ("DG", 0))
    solver.mu_lower_time = []
    solver.mu_upper_time = []
    for _ in range(num_steps + 1):
        mu_L = Function(Vc)
        mu_L.x.array[:] = 0.0
        mu_U = Function(Vc)
        mu_U.x.array[:] = 0.0
        solver.mu_lower_time.append(mu_L)
        solver.mu_upper_time.append(mu_U)

    if rank == 0 and has_constraints:
        logger.info(f"\nPATH-CONSTRAINT Window: t ∈ [{sc_start_time:.1f}, {sc_end_time:.1f}]s")
        logger.info(f"PATH-CONSTRAINT Steps: m ∈ [{sc_start_step}, {sc_end_step}] (of {num_steps})")
        if args.sc_lower is not None:
            logger.info(f"PATH-CONSTRAINT Lower bound: T ≥ {args.sc_lower:.1f}°C")
        if args.sc_upper is not None:
            logger.info(f"PATH-CONSTRAINT Upper bound: T ≤ {args.sc_upper:.1f}°C")
        logger.info(f"PATH-CONSTRAINT Initialized {len(solver.mu_lower_time)} multiplier functions")

    # Aggiungi:
    if rank == 0:
        x_cells = domain.geometry.x
        cell_dofmap = domain.geometry.dofmap
        cell_centers = x_cells[cell_dofmap].mean(axis=1)

        for i, marker in enumerate(solver.target_markers):
            marked_cells_idx = np.where(marker)[0]
            logger.debug(f"Target zone {i+1}: {len(marked_cells_idx)} marked cells")
            if len(marked_cells_idx) > 0:
                centers = cell_centers[marked_cells_idx]
                for j, c in enumerate(centers[:5]):
                    logger.debug(f"  Cell {j}: ({c[0]:.4f}, {c[1]:.4f})")

    n_ctrl_scalar = solver.n_ctrl_scalar
    n_vars_total = n_ctrl_scalar * num_steps

    if rank == 0:
        logger.info("\nOPTIMIZATION")
        logger.info(f"  Scalar controls (u_controls): {n_ctrl_scalar} (Dirichlet={solver.n_ctrl_dirichlet}, Neumann={solver.n_ctrl_neumann})")
        logger.info(f"  Distributed controls (Function(V) per step): {solver.n_ctrl_distributed}")
        logger.info(f"  Control time steps (intervals): {num_steps}  (State nodes Nt={Nt})")
        logger.info(f"  Total variables: {n_vars_total}")
        logger.info(f"  SC: beta={args.beta}, maxit={args.sc_maxit}")
        logger.info(f"  Gradient descent: lr={args.lr}, inner_maxit={args.inner_maxit}")
        logger.info(f"  Regularization: α_track={args.alpha_track}, α_u={args.alpha_u}, γ_u={args.gamma_u}")
        logger.debug("Target zones cells:")
        for i, marker in enumerate(solver.target_markers):
            n_cells = np.sum(marker)
            logger.debug(f"  Zone {i+1}: {n_cells} cells marked")

    # Initial guess
    if n_ctrl_scalar > 0:
        u_controls = np.full((n_ctrl_scalar, num_steps), args.u_init, dtype=float)
    else:
        u_controls = np.zeros((0, num_steps), dtype=float)

    if rank == 0:
        if has_constraints and args.sc_maxit > 0:
            logger.info("\nSC LOOP Starting...\n")
        else:
            logger.info("\nSC LOOP skipped (no constraints or sc_maxit=0)\n")

    sc_iter = -1

    # Moreau-Yosida loop
    for sc_iter in range(args.sc_maxit):
        # Y_all = solver.solve_forward(u_neumann_funcs_time, u_distributed_funcs_time, T_cure)

        # Inner optimization
        for inner_iter in range(args.inner_maxit):

            Y_all = solver.solve_forward(u_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure)
            J, J_track, J_reg_L2, J_reg_H1 = solver.compute_cost(u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time, Y_all, T_cure)
            P_all = solver.solve_adjoint(Y_all, T_cure)
            grad = solver.compute_gradient(u_controls, Y_all, P_all, u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time)

            # Update Dirichlet controls (Function(V) on ΓD)
            if solver.n_ctrl_dirichlet > 0:
                for j in range(solver.n_ctrl_dirichlet):
                    dofs_j = solver.dirichlet_dofs[j]
                    for m in range(num_steps):
                        uD = u_dirichlet_funcs_time[m][j]
                        gD = solver.grad_u_dirichlet_time[m][j]
                        # Update only on ΓD
                        uD.x.array[dofs_j] -= args.lr * gD.x.array[dofs_j]
                        uD.x.scatter_forward()

            # UPDATE Neumann (time-dependent, space-dependent)
            if solver.n_ctrl_neumann > 0:
                for i in range(solver.n_ctrl_neumann):
                    dofs_i = solver.neumann_dofs[i]
                    for m in range(num_steps):
                        q = u_neumann_funcs_time[m][i]
                        g = solver.grad_q_neumann_time[m][i]
                        q.x.array[dofs_i] -= args.lr * g.x.array[dofs_i]
                        q.x.scatter_forward()
                if solver.domain.comm.rank == 0:
                    logger.debug(f"q(t0) min/max = {u_neumann_funcs_time[0][0].x.array.min()}, {u_neumann_funcs_time[0][0].x.array.max()}")
                    logger.debug(f"q(tend) min/max = {u_neumann_funcs_time[-1][0].x.array.min()}, {u_neumann_funcs_time[-1][0].x.array.max()}")

            # UPDATE distributed (time-dependent, space-dependent)
            if solver.n_ctrl_distributed > 0:
                for j in range(solver.n_ctrl_distributed):
                    dofs_j = solver.distributed_dofs[j]  # ✅ DOF in Ωc
                    for m in range(num_steps):
                        uD = u_distributed_funcs_time[m][j]
                        gD = solver.grad_u_distributed_time[m][j]
                        uD.x.array[dofs_j] -= args.lr * gD.x.array[dofs_j]  # ✅ Solo Ωc
                        uD.x.scatter_forward()
                if rank == 0:
                    a0 = u_distributed_funcs_time[0][0].x.array
                    logger.debug(f"uD0(t0) min/max = {float(a0.min())}, {float(a0.max())}")

            if args.check_grad and sc_iter == 0 and inner_iter == 0:
                J0, fd, ad, rel = check_gradient_fd(solver, u_controls, u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time,
                    T_cure, eps=args.fd_eps, seed=1, m0=0)

                if rank == 0:
                    print(f"[GRAD-CHECK] J={J0:.6e}  FD={fd:.6e}  AD={ad:.6e}  rel_err={rel:.3e}")
                    print("[GRAD-CHECK] Done. Exiting now.")

                import sys
                sys.exit(0)

            if rank == 0:
                logger.debug(f"u_old = {u_controls.copy()}")
                logger.debug(f"grad = {grad.copy()}")

            # norma del gradiente: include anche controlli Function(P2)
            grad_sq = 0.0

            # scalari (se esistono)
            if grad.size > 0:
                grad_sq += float(np.sum(grad**2))

            # Dirichlet Function(P2) su ΓD (solo dofs del bordo)
            if solver.n_ctrl_dirichlet > 0:
                for j in range(solver.n_ctrl_dirichlet):
                    dofs_j = solver.dirichlet_dofs[j]
                    for m in range(num_steps):
                        gD = solver.grad_u_dirichlet_time[m][j]
                        grad_sq += float(np.sum(gD.x.array[dofs_j]**2))

            # Neumann Function(P2) su Gamma (solo dofs del bordo)
            if solver.n_ctrl_neumann > 0:
                for i in range(solver.n_ctrl_neumann):
                    dofs_i = solver.neumann_dofs[i]
                    for m in range(num_steps):
                        g = solver.grad_q_neumann_time[m][i]
                        grad_sq += float(np.sum(g.x.array[dofs_i]**2))

            # Distributed Function(P2) in Omega (tutti i dofs)
            if solver.n_ctrl_distributed > 0:
                for j in range(solver.n_ctrl_distributed):
                    for m in range(num_steps):
                        gD = solver.grad_u_distributed_time[m][j]
                        dofs_j = solver.distributed_dofs[j]
                        grad_sq += np.sum(gD.x.array[dofs_j]**2)

            grad_norm = np.sqrt(grad_sq)

            # ============================================================
            # Recompute STATE and COST after updating Function controls
            # (so the printed J matches the updated controls)
            # ============================================================
            Y_all = solver.solve_forward(u_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure)
            J, J_track, J_reg_L2, J_reg_H1 = solver.compute_cost(u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time, Y_all, T_cure)

            # if rank == 0 and inner_iter % 5 == 0:
            if rank == 0:
                if grad.size > 0:
                    u_mean = np.mean(u_controls)
                    u_std = np.std(u_controls)
                    u_info = f"{u_mean:.1f}±{u_std:.1f}"
                else:
                    u_info = "n/a (no scalar controls)"

                print(f"  Inner {inner_iter:2d}: J={J:.3e} (track={J_track:.3e}, "f"L2={J_reg_L2:.3e}, H1={J_reg_H1:.3e}), "f"||∇J||={grad_norm:.3e}, u: {u_info}")

            if grad_norm < args.grad_tol:
                if rank == 0:
                    print(f"  [Inner converged at iter {inner_iter}]")
                break

        if rank == 0:
            Vc = functionspace(domain, ("DG", 0))
            Tcell = Function(Vc)
            Tcell.interpolate(Y_all[-1])
            mask = solver.sc_marker.astype(bool)
            logger.debug(f"len(sc_marker) = {len(solver.sc_marker)}")
            logger.debug(f"len(Tcell DG0) = {len(Tcell.x.array)}")
            logger.debug(f"len(Y_all[-1] V) = {len(Y_all[-1].x.array)}")

            if np.any(mask):
                logger.debug(f"Y_all[-1] on constraint Tmin/Tmax = {float(Tcell.x.array[mask].min())}, {float(Tcell.x.array[mask].max())}")
            else:
                logger.debug("No cells in constraint mask (constraint zone empty).")

        if not has_constraints:
            delta_mu, feas_inf = 0.0, 0.0
            if domain.comm.rank == 0:
                logger.debug("SC skipped mu update (no constraint zone).")
        else:
            delta_mu, feas_inf = solver.update_multiplier_mu(Y_all, args.sc_type, args.sc_lower, args.sc_upper, args.beta, sc_start_step, sc_end_step)

        Y_mu = solver.solve_forward(u_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure)
        J_mu, Jt_mu, JL2_mu, JH1_mu = solver.compute_cost(u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time, Y_mu, T_cure)

        if rank == 0:
            logger.debug(f"J-AFTER-MU J={J_mu:.6e} (track={Jt_mu:.6e}, L2={JL2_mu:.6e}, H1={JH1_mu:.6e})")

        if rank == 0:
            # Get max multiplier across all time steps in constraint window
            muL_max = 0.0
            muU_max = 0.0
            for m in range(sc_start_step, sc_end_step + 1):
                muL_max = max(muL_max, np.max(solver.mu_lower_time[m].x.array))
                muU_max = max(muU_max, np.max(solver.mu_upper_time[m].x.array))
            logger.debug(f"μL_max={muL_max:.3e}, μU_max={muU_max:.3e}")

            if u_controls.size > 0:
                logger.debug(f"u_mean={float(np.mean(u_controls)):.1f}, u_std={float(np.std(u_controls)):.1f}")
            else:
                logger.debug("u_mean=n/a, u_std=n/a (no scalar controls)")

        if feas_inf < args.sc_tol and delta_mu < 1e-4:
            if rank == 0:
                print(f"[CONVERGED] feas_inf < {args.sc_tol}\n")
            break

    if rank == 0:
        if has_constraints:
            logger.debug(f"CHECK-MU-FINAL max muL at final = {max(np.max(solver.mu_lower_time[m].x.array) for m in range(sc_start_step, sc_end_step + 1))}")
        else:
            logger.debug("CHECK-MU-FINAL no constraints -> skipping mu check")

    # Final results
    Y_final_all = solver.solve_forward(u_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure)
    J_final, J_track_final, J_reg_L2_final, J_reg_H1_final = solver.compute_cost(u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time, Y_final_all, T_cure)

    if rank == 0:
        n_ctrl_total = solver.n_ctrl_dirichlet + solver.n_ctrl_neumann + solver.n_ctrl_distributed

        print("[FINAL RESULTS]")
        print(f"  SC iterations: {sc_iter + 1}")
        print(f"  J_final = {J_final:.3e} (track={J_track_final:.3e}, "f"L2={J_reg_L2_final:.3e}, H1={J_reg_H1_final:.3e})")

        # ============================================================
        # DIAGNOSTICA H1 TEMPORALE (Distributed controls)
        # ============================================================
        if solver.n_ctrl_distributed > 0:
            from dolfinx.fem import form, assemble_scalar
            comm = domain.comm
            dx = ufl.Measure("dx", domain=domain)

            if rank == 0:
                logger.debug("\n" + "="*70)
                logger.debug("H1 TEMPORAL REGULARIZATION CHECK (DISTRIBUTED)")
                logger.debug("="*70)
                logger.debug(f"gamma_u = {solver.gamma_u:.6e}, dt = {solver.dt:.6e}, num_steps = {num_steps}")

            for j in range(solver.n_ctrl_distributed):
                chiV = solver.chi_distributed_V[j]

                # --- misura globale di Ωc (serve MPI allreduce) ---
                meas_loc = assemble_scalar(form(chiV * dx))
                meas = comm.allreduce(meas_loc, op=MPI.SUM)

                # --- den globale (uguale a meas, ma lo teniamo separato per chiarezza) ---
                den_loc = assemble_scalar(form(chiV * dx))
                den = comm.allreduce(den_loc, op=MPI.SUM)

                if rank == 0:
                    logger.debug(f"\nH1t-DIST zone {j}")
                    logger.debug(f"  meas(Ωc) ≈ {float(meas):.6e}")

                rough2 = 0.0
                max_step_L2 = 0.0

                # Campioni della media spaziale su Ωc (10 punti)
                sample_means = []
                stride = max(1, (num_steps - 1)//10)
                sample_idx = list(range(0, num_steps, stride))
                if (num_steps - 1) not in sample_idx:
                    sample_idx.append(num_steps - 1)

                # serie media (con numeratore globale)
                for m in sample_idx:
                    um = u_distributed_funcs_time[m][j]
                    num_loc = assemble_scalar(form(um * chiV * dx))
                    num = comm.allreduce(num_loc, op=MPI.SUM)
                    mean_um = float(num / den) if den > 1e-14 else float("nan")
                    sample_means.append((m * solver.dt, mean_um))

                # roughness vera (L2 su Ωc) — step per step con allreduce
                for m in range(num_steps - 1):
                    u0 = u_distributed_funcs_time[m][j]
                    u1 = u_distributed_funcs_time[m+1][j]
                    step_loc = assemble_scalar(form(((u1 - u0)**2) * chiV * dx))
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
                    logger.debug(f"  avg step L2-norm over Ωc (sqrt(mean ||Δu||^2)) = {avg_step:.6e}")
                    logger.debug(f"  max step ||Δu||^2 over Ωc = {max_step_L2:.6e}")
                    logger.debug("  sampled mean(u) over Ωc:")
                    for (t, mu) in sample_means:
                        logger.debug(f"    t={t:8.1f}s  mean(u)={mu:+.6e}")

        # ---------- STATISTICHE ----------
        for ic in range(n_ctrl_total):
            if ic < solver.n_ctrl_dirichlet:
                # Dirichlet: STAMPA IL CONTROLLO REALE uD(t) SU ΓD
                unit = "°C"
                j = ic
                label = f"Dirichlet {j}"
                dofs = solver.dirichlet_dofs[j]
                # media spaziale su ΓD, per ogni time step
                vals_t = np.array([u_dirichlet_funcs_time[m][j].x.array[dofs].mean() for m in range(num_steps)])
                logger.info(
                    f"    {label} (REAL uD on ΓD): "
                    f"mean={vals_t.mean():.6e} {unit}, "
                    f"std={vals_t.std():.6e}, "
                    f"min={vals_t.min():.6e}, "
                    f"max={vals_t.max():.6e}"
                )
            elif ic < solver.n_ctrl_dirichlet + solver.n_ctrl_neumann:
                # Neumann: STAMPA IL CONTROLLO REALE q(t) SU Γ
                unit = "(flux units)"
                j = ic - solver.n_ctrl_dirichlet
                label = f"Neumann {j}"
                dofs = solver.neumann_dofs[j]
                # media spaziale su Gamma, per ogni time step
                vals_t = np.array([u_neumann_funcs_time[m][j].x.array[dofs].mean() for m in range(num_steps)])
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
                # statistiche sul controllo REALE uD(t) in Ω (qui: media su tutti i dofs di V)
                vals_t = np.array([u_distributed_funcs_time[m][j].x.array.mean() for m in range(num_steps)])
                logger.info(
                    f"    {label} (REAL uD in Omega): "
                    f"mean={vals_t.mean():.6e} {unit}, "
                    f"std={vals_t.std():.6e}, "
                    f"min={vals_t.min():.6e}, "
                    f"max={vals_t.max():.6e}"
                )

        # ---------- EVOLUZIONE TEMPORALE (sottocampionata) ----------
        if n_ctrl_total > 0:
            step_stride = max(1, num_steps // 10)
            logger.debug(f"T_final_time={T_final_time}, dt={dt}, num_steps={num_steps}, "
                f"u_controls.shape={u_controls.shape}, q_time_len={len(u_neumann_funcs_time)}")
            for ic in range(n_ctrl_total):
                if ic < solver.n_ctrl_dirichlet:
                    j = ic
                    label = f"Dirichlet {j}"
                    unit = "°C"
                    dofs = solver.dirichlet_dofs[j]
                    def get_val(t_idx):
                        # media spaziale su ΓD al tempo t_idx
                        return float(u_dirichlet_funcs_time[t_idx][j].x.array[dofs].mean())
                elif ic < solver.n_ctrl_dirichlet + solver.n_ctrl_neumann:
                    j = ic - solver.n_ctrl_dirichlet
                    label = f"Neumann {j}"
                    unit = "(flux units)"
                    dofs = solver.neumann_dofs[j]
                    def get_val(t_idx):
                        # media spaziale su Gamma al tempo t_idx
                        return float(u_neumann_funcs_time[t_idx][j].x.array[dofs].mean())
                else:
                    j = ic - solver.n_ctrl_dirichlet - solver.n_ctrl_neumann
                    label = f"Distributed {j}"
                    unit = "(source units)"
                    def get_val(t_idx):
                        return float(u_distributed_funcs_time[t_idx][j].x.array.mean())

                logger.info(f"\n  Control {ic} evolution [{label}] (every {step_stride} steps):")
                idxs = list(range(0, num_steps, step_stride))
                if (num_steps - 1) not in idxs:
                    idxs.append(num_steps - 1)
                for t_idx in idxs:
                    t_val = t_idx * dt
                    logger.info(f"    t={t_val:8.1f}s: u={get_val(t_idx):.6e} {unit}")
                # held-last (ultimo controllo applicato)
                logger.info(f"    t={T_final_time:8.1f}s: u={get_val(num_steps-1):.6e} {unit}  (held-last)")

        T_final_diag = Y_final_all[-1]

        # --- POSTPROCESSING CORRETTO DELLA TEMPERATURA MEDIA IN TARGET ZONE ---
        V0_dg0 = functionspace(domain, ("DG", 0))
        from dolfinx.fem import assemble_scalar, form
        dx = ufl.Measure("dx", domain=domain)

        for i, marker in enumerate(solver.target_markers):
            chi_dg0 = Function(V0_dg0)
            chi_dg0.x.array[:] = marker.astype(PETSc.ScalarType)
            T_cell = Function(V0_dg0)
            T_cell.interpolate(T_final_diag)

            if rank == 0:
                Vc = functionspace(domain, ("DG", 0))
                Tcell_sc = Function(Vc)
                Tcell_sc.interpolate(T_final_diag)
                mask = solver.sc_marker.astype(bool)
                if np.any(mask):
                    logger.debug(f"T_final_diag on constraint Tmin/Tmax = {float(Tcell_sc.x.array[mask].min())}, {float(Tcell_sc.x.array[mask].max())}")
                else:
                    logger.debug("No constraint cells -> skipping Tmin/Tmax check.")

            zone_integral = assemble_scalar(form(T_cell * chi_dg0 * dx))
            chi_integral  = assemble_scalar(form(chi_dg0 * dx))

            if chi_integral > 1e-12:
                T_mean = zone_integral / chi_integral
                print(f"\n  Target zone {i+1}:")
                print(f"    T_mean ≈ {T_mean:.6f}°C")

                deficit = T_cure - T_mean
                if deficit <= 0:
                    print("    [OK] Constraint satisfied ✓")
                else:
                    print(f"    [INFO] Deficit ≈ {deficit:.1f}°C")
            else:
                print(f"\n  Target zone {i+1}: Empty or zero measure")

    # ============================================================
    # Save visualization output
    # ============================================================
    if rank == 0 and not args.no_vtk:
        logger.info("\n" + "="*70)
        logger.info("SAVING VISUALIZATION OUTPUT")
        logger.info("="*70 + "\n")

    P_final_all = solver.solve_adjoint(Y_final_all, T_cure)
    if not args.no_vtk:
        save_visualization_output(
        solver, Y_final_all, P_final_all,  # ← Usa Y_final_all invece di Y_all!
        u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time,
        sc_start_step, sc_end_step, args, num_steps, target_boxes, ctrl_distributed_boxes
        )

