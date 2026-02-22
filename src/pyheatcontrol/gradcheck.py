import numpy as np
import ufl
from dolfinx.fem import Function, assemble_scalar, form


def check_gradient_fd(
    solver,
    u_controls,
    u_distributed_funcs_time,
    u_neumann_funcs_time,
    u_dirichlet_funcs_time,
    T_cure,
    eps=1e-3,
    seed=0,
    m0=0,
    amp_base=1e-1,
):
    """
    Finite-difference vs adjoint gradient check (central differences).

    Computes directional derivative in a random direction r:
      FD: (J(u+eps*r) - J(u-eps*r)) / (2*eps)
      AD: <g, r> (boundary or volume integral depending on control type)

    Consistent with the Riesz representant used in compute_gradient.
    For Dirichlet controls, optionally applies a non-constant base perturbation
    to excite spatial H1(Γ) terms (removed after the test).
    """

    rank = solver.domain.comm.rank
    assert 0 <= m0 < solver.num_steps, "m0 must be in [0, num_steps-1]"

    dx = ufl.Measure("dx", domain=solver.domain)

    # Helper for optional non-constant base (Dirichlet only)
    base_applied = False

    # ============================================================
    # 1) scegli cosa perturbare (priorità: Dirichlet, Neumann, Distributed)
    # ============================================================
    if solver.n_ctrl_dirichlet > 0:
        ctrl_type = "Dirichlet"
        j0 = 0
        dofs = solver.dirichlet_dofs[j0]
        ndofs = len(dofs)
        if ndofs == 0:
            raise RuntimeError("Dirichlet DOFs empty: cannot perform FD check.")

        ds_j = solver.dirichlet_measures[j0]
        mid = solver.dirichlet_marker_ids[j0]

        rng = np.random.default_rng(seed)
        r = rng.standard_normal(ndofs)
        r /= max(1e-14, np.linalg.norm(r))

        def apply_base():
            nonlocal base_applied
            if amp_base is None or amp_base == 0.0:
                return
            u = u_dirichlet_funcs_time[m0][j0]
            u.x.array[dofs] += float(amp_base) * r
            u.x.scatter_forward()
            base_applied = True

        def undo_base():
            nonlocal base_applied
            if not base_applied:
                return
            u = u_dirichlet_funcs_time[m0][j0]
            u.x.array[dofs] -= float(amp_base) * r
            u.x.scatter_forward()
            base_applied = False

        def apply_perturb(sign):
            u = u_dirichlet_funcs_time[m0][j0]
            u.x.array[dofs] += float(sign) * eps * r
            u.x.scatter_forward()

        def undo_perturb(sign):
            u = u_dirichlet_funcs_time[m0][j0]
            u.x.array[dofs] -= float(sign) * eps * r
            u.x.scatter_forward()

        def get_ad(g):
            rf = Function(solver.V)
            rf.x.array[:] = 0.0
            rf.x.array[dofs] = r
            rf.x.scatter_forward()

            # L2 pairing always present
            ad_val = assemble_scalar(form(g * rf * ds_j(mid)))

            # If H1 boundary Riesz, add tangential term
            if (
                getattr(solver, "dirichlet_spatial_reg", "L2") == "H1"
                and getattr(solver, "beta_u", 0.0) > 1e-16
            ):
                tg_g = solver.tgrad(g)
                tg_r = solver.tgrad(rf)
                ad_val += solver.beta_u * assemble_scalar(
                    form(ufl.inner(tg_g, tg_r) * ds_j(mid))
                )

            return float(ad_val)

        # Apply base perturbation (to excite spatial H1 on ΓD)
        apply_base()

    elif solver.n_ctrl_neumann > 0:
        ctrl_type = "Neumann"
        j0 = 0
        dofs = solver.neumann_dofs[j0]
        ndofs = len(dofs)
        if ndofs == 0:
            raise RuntimeError("Neumann DOFs empty: cannot perform FD check.")

        mid = solver.neumann_marker_ids[j0]

        rng = np.random.default_rng(seed)
        r = rng.standard_normal(ndofs)
        r /= max(1e-14, np.linalg.norm(r))

        def undo_base():
            return  # no-op

        def apply_perturb(sign):
            q = u_neumann_funcs_time[m0][j0]
            q.x.array[dofs] += float(sign) * eps * r
            q.x.scatter_forward()

        def undo_perturb(sign):
            q = u_neumann_funcs_time[m0][j0]
            q.x.array[dofs] -= float(sign) * eps * r
            q.x.scatter_forward()

        def get_ad(g):
            # <g, r>_{L2(ΓN)} = ∫_ΓN g * r ds
            rf = Function(solver.V)
            rf.x.array[:] = 0.0
            rf.x.array[dofs] = r
            rf.x.scatter_forward()
            return float(assemble_scalar(form(g * rf * solver.ds_neumann(mid))))

    elif solver.n_ctrl_distributed > 0:
        ctrl_type = "Distributed"
        j0 = 0
        dofs = solver.distributed_dofs[j0]
        ndofs = len(dofs)
        if ndofs == 0:
            raise RuntimeError("Distributed DOFs empty: cannot perform FD check.")

        chiV = solver.chi_distributed_V[j0]

        rng = np.random.default_rng(seed)
        r = rng.standard_normal(ndofs)
        r /= max(1e-14, np.linalg.norm(r))

        def undo_base():
            return  # no-op

        def apply_perturb(sign):
            u = u_distributed_funcs_time[m0][j0]
            u.x.array[dofs] += float(sign) * eps * r
            u.x.scatter_forward()

        def undo_perturb(sign):
            u = u_distributed_funcs_time[m0][j0]
            u.x.array[dofs] -= float(sign) * eps * r
            u.x.scatter_forward()

        def get_ad(g):
            # <g, r>_{L2(Ωc)} = ∫ g * r * χ dx
            rf = Function(solver.V)
            rf.x.array[:] = 0.0
            rf.x.array[dofs] = r
            rf.x.scatter_forward()
            return float(assemble_scalar(form(g * rf * chiV * dx)))

    else:
        raise RuntimeError(
            "No controls available (Dirichlet/Neumann/Distributed)."
        )

    if rank == 0:
        print(
            f"[FD-CHECK] ctrl_type={ctrl_type}, zone={j0}, m0={m0}, ndofs={ndofs}, eps={eps}, seed={seed}",
            flush=True,
        )
        print(
            f"[FD-CHECK] alpha_u={solver.alpha_u:.6e}, gamma_u={solver.gamma_u:.6e}, beta_u={getattr(solver, 'beta_u', 0.0):.6e}, amp_base={amp_base}",
            flush=True,
        )

    try:
        # ============================================================
        # 2) BASE: forward, cost, adjoint, grad
        # ============================================================
        Y0 = solver.solve_forward(
            u_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time,
        )
        J0, *_ = solver.compute_cost(
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            Y0,
            T_cure,
        )
        P0 = solver.solve_adjoint(Y0, T_cure)
        solver.compute_gradient(
            u_controls,
            Y0,
            P0,
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
        )

        if ctrl_type == "Dirichlet":
            g = solver.grad_u_dirichlet_time[m0][j0]
        elif ctrl_type == "Neumann":
            g = solver.grad_q_neumann_time[m0][j0]
        else:
            g = solver.grad_u_distributed_time[m0][j0]

        ad = get_ad(g)

        # ============================================================
        # 3) FD centrale: J(u+eps*r) e J(u-eps*r)
        # ============================================================
        apply_perturb(+1.0)
        Yp = solver.solve_forward(
            u_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time,
        )
        Jp, *_ = solver.compute_cost(
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            Yp,
            T_cure,
        )
        undo_perturb(+1.0)

        apply_perturb(-1.0)
        Ym = solver.solve_forward(
            u_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time,
        )
        Jm, *_ = solver.compute_cost(
            u_distributed_funcs_time,
            u_neumann_funcs_time,
            u_dirichlet_funcs_time,
            Ym,
            T_cure,
        )
        undo_perturb(-1.0)

        fd = float((Jp - Jm) / (2.0 * eps))
        rel = abs(fd - ad) / max(1.0, abs(fd), abs(ad))

        if rank == 0:
            print(f"[FD-CHECK] J0={J0:.12e}  Jp={Jp:.12e}  Jm={Jm:.12e}", flush=True)
            print(
                f"[GRAD-CHECK] FD={fd:+.6e}  AD={ad:+.6e}  rel_err={rel:.3e}",
                flush=True,
            )

        return J0, fd, ad, rel

    finally:
        # Remove non-constant base perturbation (if applied) to keep state clean
        undo_base()
