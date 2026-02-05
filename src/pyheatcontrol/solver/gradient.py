import numpy as np
from petsc4py import PETSc
import ufl
from ufl import dx, grad as ufl_grad, inner, TestFunction, FacetNormal

from dolfinx.fem import form, Function, functionspace
from dolfinx.fem.petsc import assemble_vector
from pyheatcontrol.logging_config import logger


def _init_gradient_forms_impl(self):
    """Inizializza form pre-compilate per compute_gradient"""
    n = FacetNormal(self.domain)
    v = TestFunction(self.V)
    dx = ufl.Measure("dx", domain=self.domain)

    # Placeholder per adjoint p (viene aggiornato nel loop)
    self._p_placeholder = Function(self.V)

    # Placeholder per controlli (per regularizzazione)
    self._uD_placeholder = Function(self.V)
    self._q_placeholder = Function(self.V)

    # --- Dirichlet forms ---
    self._grad_dirichlet_adj_forms = []
    self._grad_dirichlet_reg_forms = []
    self._grad_dirichlet_regH1_forms = []

    for i in range(self.n_ctrl_dirichlet):
        ds_i = self.dirichlet_measures[i]
        mid = self.dirichlet_marker_ids[i]

        # Adjoint contribution: (-k ∇p·n) * v on ΓD
        flux_form = (
            (-self.k_therm * inner(ufl_grad(self._p_placeholder), n)) * v * ds_i(mid)
        )
        self._grad_dirichlet_adj_forms.append(form(flux_form))

        # L2 regularization: alpha_u * dt * uD * v on ΓD
        if self.alpha_u > 1e-16:
            reg_form = self.alpha_u * self.dt * self._uD_placeholder * v * ds_i(mid)
            self._grad_dirichlet_reg_forms.append(form(reg_form))
        else:
            self._grad_dirichlet_reg_forms.append(None)

        # H1 spatial regularization on ΓD (tangential gradient): beta_u * dt * <∇Γ uD, ∇Γ v>_ΓD
        # Always build it (or set None), so compute_gradient never assembles forms inside loops.
        if self.beta_u > 1e-16:
            tg_u = self.tgrad(self._uD_placeholder)
            tg_v = self.tgrad(v)
            regH1_form = self.beta_u * self.dt * inner(tg_u, tg_v) * ds_i(mid)
            self._grad_dirichlet_regH1_forms.append(form(regH1_form))
        else:
            self._grad_dirichlet_regH1_forms.append(None)

    # --- Neumann forms ---
    self._grad_neumann_adj_forms = []
    self._grad_neumann_reg_forms = []
    for i in range(self.n_ctrl_neumann):
        mid = self.neumann_marker_ids[i]

        # Adjoint: -p * v on Γ
        adj_form = -self._p_placeholder * v * self.ds_neumann(mid)
        self._grad_neumann_adj_forms.append(form(adj_form))

        # L2 reg: alpha_u * dt * q * v on Γ
        if self.alpha_u > 1e-16:
            reg_form = (
                self.alpha_u * self.dt * self._q_placeholder * v * self.ds_neumann(mid)
            )
            self._grad_neumann_reg_forms.append(form(reg_form))
        else:
            self._grad_neumann_reg_forms.append(None)

    # --- Distributed forms ---
    self._grad_distributed_adj_forms = []
    self._grad_distributed_reg_forms = []
    for i in range(self.n_ctrl_distributed):
        chiV = self.chi_distributed_V[i]

        # Adjoint: p * chi * v dx
        adj_form = self._p_placeholder * chiV * v * dx
        self._grad_distributed_adj_forms.append(form(adj_form))

        # L2 reg: alpha_u * dt * uD * chi * v dx
        if self.alpha_u > 1e-16:
            reg_form = self.alpha_u * self.dt * self._uD_placeholder * chiV * v * dx
            self._grad_distributed_reg_forms.append(form(reg_form))
        else:
            self._grad_distributed_reg_forms.append(None)

    self._gradient_forms_initialized = True


def compute_gradient_impl(
    self,
    u_controls,
    Y_all,
    P_all,
    u_distributed_funcs_time,
    q_neumann_funcs_time,
    u_dirichlet_funcs_time,
):
    """
    Compute gradient via adjoint.
    - Neumann control è P2 in spazio e time-dependent: self.grad_q_neumann_time[m][i]
    - Distributed control è P2 in spazio e time-dependent: self.grad_u_distributed_time[m][i]
    (NOTA: non entra in u_controls, quindi NON incrementare idx per lui)
    """
    grad = np.zeros((self.n_ctrl_spatial, self.num_steps))
    # Inizializza forms se non già fatto
    if not hasattr(self, "_gradient_forms_initialized"):
        self._init_gradient_forms()

    # ============================================================
    # Neumann gradient containers: [m][i] Function(V)
    # ============================================================
    if (
        self.grad_q_neumann_time is None
        or len(self.grad_q_neumann_time) != self.num_steps
    ):
        self.grad_q_neumann_time = [
            [Function(self.V) for _ in range(self.n_ctrl_neumann)]
            for _ in range(self.num_steps)
        ]
    for m in range(self.num_steps):
        for i in range(self.n_ctrl_neumann):
            self.grad_q_neumann_time[m][i].x.array[:] = 0.0
            self.grad_q_neumann_time[m][i].x.scatter_forward()

    # ============================================================
    # Distributed gradient containers: [m][i] Function(V)
    # ============================================================
    if (
        self.grad_u_distributed_time is None
        or len(self.grad_u_distributed_time) != self.num_steps
    ):
        self.grad_u_distributed_time = [
            [Function(self.V) for _ in range(self.n_ctrl_distributed)]
            for _ in range(self.num_steps)
        ]
    for m in range(self.num_steps):
        for i in range(self.n_ctrl_distributed):
            self.grad_u_distributed_time[m][i].x.array[:] = 0.0
            self.grad_u_distributed_time[m][i].x.scatter_forward()

    # ============================================================
    # Dirichlet gradient containers: [m][i] Function(V)
    # ============================================================
    if (
        self.grad_u_dirichlet_time is None
        or len(self.grad_u_dirichlet_time) != self.num_steps
    ):
        self.grad_u_dirichlet_time = [
            [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
            for _ in range(self.num_steps)
        ]
    for m in range(self.num_steps):
        for i in range(self.n_ctrl_dirichlet):
            self.grad_u_dirichlet_time[m][i].x.array[:] = 0.0
            self.grad_u_dirichlet_time[m][i].x.scatter_forward()
    # ============================================================
    # Adjoint contribution
    # ============================================================
    for m in range(self.num_steps):
        p_next = P_all[m + 1]
        idx = 0

        # ---- Dirichlet controls (P2), gradient with L2(Γ) or H1(Γ) ----
        for i in range(self.n_ctrl_dirichlet):
            dofs_i = self.dirichlet_dofs[i]
            ds_i = self.dirichlet_measures[i]
            mid = self.dirichlet_marker_ids[i]
            uD_current = u_dirichlet_funcs_time[m][i]

            # Aggiorna placeholder per p
            self._p_placeholder.x.array[:] = p_next.x.array[:]
            self._p_placeholder.x.scatter_forward()

            # Adjoint contribution (usa form pre-compilata)
            b_adj = assemble_vector(self._grad_dirichlet_adj_forms[i])
            b_adj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Total RHS (start with adjoint)
            b_total = b_adj.copy()

            # Add L2(Γ) regularization contribution
            if self.alpha_u > 1e-16:
                self._uD_placeholder.x.array[:] = uD_current.x.array[:]
                self._uD_placeholder.x.scatter_forward()
                b_reg_L2 = assemble_vector(self._grad_dirichlet_reg_forms[i])
                b_reg_L2.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )
                b_total.axpy(1.0, b_reg_L2)

            # Add H1(Γ) regularization contribution (tangential gradient)
            # NOTA: H1 spaziale è raro, lasciamo form() qui per ora
            if self.dirichlet_spatial_reg == "H1" and self.beta_u > 1e-16:
                # Use precompiled H1 form (tangential gradient on ΓD)
                self._uD_placeholder.x.array[:] = uD_current.x.array[:]
                self._uD_placeholder.x.scatter_forward()

                formH1 = self._grad_dirichlet_regH1_forms[i]
                if formH1 is not None:
                    b_reg_H1 = assemble_vector(formH1)
                    b_reg_H1.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )
                    b_total.axpy(1.0, b_reg_H1)

                if self.domain.comm.rank == 0 and m == 0:
                    h1_norm = b_reg_H1.norm()
                    l2_norm = b_reg_L2.norm() if self.alpha_u > 1e-16 else 0.0
                    logger.debug(
                        f"H1: m={m}, i={i}: ||b_L2||={l2_norm:.3e}, ||b_H1||={h1_norm:.3e}, ratio={h1_norm / max(l2_norm, 1e-16):.3e}"
                    )

            # Solve Riesz map: M_dirichlet * gD = b_total
            gD = self.grad_u_dirichlet_time[m][i]
            gD.x.petsc_vec.set(0.0)
            # Ensure RHS is supported only on ΓD dofs
            imap = self.V.dofmap.index_map
            nloc = imap.size_local + imap.num_ghosts
            all_dofs = np.arange(nloc, dtype=np.int32)
            off_dofs = np.setdiff1d(
                all_dofs, dofs_i.astype(np.int32), assume_unique=False
            ).astype(np.int32)

            # Zero RHS outside ΓD
            b_total.setValues(off_dofs, np.zeros(len(off_dofs), dtype=PETSc.ScalarType))
            b_total.assemble()

            self.ksp_dirichlet[i].solve(b_total, gD.x.petsc_vec)
            gD.x.scatter_forward()

            if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                reg_type = (
                    "L2"
                    if self.dirichlet_spatial_reg == "L2"
                    else f"H1(α={self.alpha_u:.1e},β={self.beta_u:.1e})"
                )
                logger.debug(
                    f"GRAD-DIRICHLET-{reg_type} m={m} on ΓD min/max: {float(gD.x.array[dofs_i].min())}, {float(gD.x.array[dofs_i].max())}"
                )
            # b_total.norm() is MPI collective - all ranks must call
            b_norm = float(b_total.norm()) if m == 0 else 0.0
            if self.domain.comm.rank == 0 and m == 0:
                expected_raw = (
                    self.alpha_u * self.dt * uD_current.x.array[dofs_i].mean()
                )
                actual_riesz = gD.x.array[dofs_i].mean()
                logger.debug(
                    f"DIRICHLET-GRAD m={m}: uD mean={uD_current.x.array[dofs_i].mean():.6e}, "
                    f"expected={expected_raw:.6e}, actual={actual_riesz:.6e}, ||b||={b_norm:.6e}"
                )

        # ---- Neumann controls (P2), gradient with proper L2(Γ) integral ----
        for i in range(self.n_ctrl_neumann):
            mid = self.neumann_marker_ids[i]
            dofs_i = self.neumann_dofs[i]
            q_current = q_neumann_funcs_time[m][i]

            # Aggiorna placeholder per p
            self._p_placeholder.x.array[:] = p_next.x.array[:]
            self._p_placeholder.x.scatter_forward()

            # Adjoint contribution (usa form pre-compilata)
            b_adj = assemble_vector(self._grad_neumann_adj_forms[i])
            b_adj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Total RHS (start with adjoint)
            b_total = b_adj.copy()

            # Add L2(Γ) regularization contribution
            if self.alpha_u > 1e-16:
                self._q_placeholder.x.array[:] = q_current.x.array[:]
                self._q_placeholder.x.scatter_forward()
                b_reg = assemble_vector(self._grad_neumann_reg_forms[i])
                b_reg.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )
                b_total.axpy(1.0, b_reg)

            # Solve Riesz map: M_neumann * gq = b_total
            gq = self.grad_q_neumann_time[m][i]
            gq.x.petsc_vec.set(0.0)
            self.ksp_neumann[i].solve(b_total, gq.x.petsc_vec)
            gq.x.scatter_forward()

            if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                logger.debug(
                    f"GRAD-NEUMANN m={m} on Γ min/max: {float(gq.x.array[dofs_i].min())}, {float(gq.x.array[dofs_i].max())}"
                )

            idx += 1

        # ---- Distributed controls (P2), robust assembly ----
        for i in range(self.n_ctrl_distributed):
            chiV = self.chi_distributed_V[i]
            uD_current = u_distributed_funcs_time[m][i]

            # Aggiorna placeholder per p
            self._p_placeholder.x.array[:] = p_next.x.array[:]
            self._p_placeholder.x.scatter_forward()

            # Adjoint contribution (usa form pre-compilata)
            b_adj = assemble_vector(self._grad_distributed_adj_forms[i])
            b_adj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

            # Total RHS (start with adjoint)
            b_total = b_adj.copy()

            # Add L2 regularization contribution
            if self.alpha_u > 1e-16:
                self._uD_placeholder.x.array[:] = uD_current.x.array[:]
                self._uD_placeholder.x.scatter_forward()
                b_reg = assemble_vector(self._grad_distributed_reg_forms[i])
                b_reg.ghostUpdate(
                    addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                )
                b_total.axpy(1.0, b_reg)

            # Solve Riesz map: M * gD = b_total
            gD = self.grad_u_distributed_time[m][i]
            gD.x.petsc_vec.set(0.0)
            self.ksp_distributed[i].solve(b_total, gD.x.petsc_vec)
            gD.x.scatter_forward()

            if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                logger.debug(
                    f"GRAD-UD m={m} min/max: {float(gD.x.array.min())}, {float(gD.x.array.max())}"
                )
    # ============================================================
    # H1 temporal regularization for spatial controls
    # ============================================================
    if self.gamma_u > 1e-16 and self.num_steps >= 2:
        temp = Function(self.V)

        # Distributed controls
        for i in range(self.n_ctrl_distributed):
            for m in range(self.num_steps):
                gD = self.grad_u_distributed_time[m][i]
                u_curr = u_distributed_funcs_time[m][i]

                # Interior points: 2*u^m - u^{m-1} - u^{m+1}
                if m > 0 and m < self.num_steps - 1:
                    u_prev = u_distributed_funcs_time[m - 1][i]
                    u_next = u_distributed_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    chiV = self.chi_distributed_V[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (2 * u_curr - u_prev - u_next)
                            * chiV
                            * v
                            * dx
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    # Solve Riesz map
                    temp.x.petsc_vec.set(0.0)
                    self.ksp_distributed[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()

                # First step: u^0 - u^1
                elif m == 0:
                    u_next = u_distributed_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    chiV = self.chi_distributed_V[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt) * (u_curr - u_next) * chiV * v * dx
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_distributed[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()

                # Last step: u^M - u^{M-1}
                else:  # m == self.num_steps - 1
                    u_prev = u_distributed_funcs_time[m - 1][i]

                    v = TestFunction(self.V)
                    chiV = self.chi_distributed_V[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt) * (u_curr - u_prev) * chiV * v * dx
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_distributed[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()

        # Neumann controls H1 in time
        for i in range(self.n_ctrl_neumann):
            for m in range(self.num_steps):
                gq = self.grad_q_neumann_time[m][i]
                q_curr = q_neumann_funcs_time[m][i]

                # Interior points
                if m > 0 and m < self.num_steps - 1:
                    q_prev = q_neumann_funcs_time[m - 1][i]
                    q_next = q_neumann_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    mid = self.neumann_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (2 * q_curr - q_prev - q_next)
                            * v
                            * self.ds_neumann(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_neumann[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gq.x.array[:] += temp.x.array[:]
                    gq.x.scatter_forward()

                # First step
                elif m == 0:
                    q_next = q_neumann_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    mid = self.neumann_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (q_curr - q_next)
                            * v
                            * self.ds_neumann(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_neumann[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gq.x.array[:] += temp.x.array[:]
                    gq.x.scatter_forward()

                # Last step
                else:
                    q_prev = q_neumann_funcs_time[m - 1][i]

                    v = TestFunction(self.V)
                    mid = self.neumann_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (q_curr - q_prev)
                            * v
                            * self.ds_neumann(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_neumann[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gq.x.array[:] += temp.x.array[:]
                    gq.x.scatter_forward()

        # Dirichlet controls
        for i in range(self.n_ctrl_dirichlet):
            for m in range(self.num_steps):
                gD = self.grad_u_dirichlet_time[m][i]
                uD_curr = u_dirichlet_funcs_time[m][i]

                # Interior points
                if m > 0 and m < self.num_steps - 1:
                    uD_prev = u_dirichlet_funcs_time[m - 1][i]
                    uD_next = u_dirichlet_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    ds_i = self.dirichlet_measures[i]
                    mid = self.dirichlet_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (2 * uD_curr - uD_prev - uD_next)
                            * v
                            * ds_i(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()

                # First step
                elif m == 0:
                    uD_next = u_dirichlet_funcs_time[m + 1][i]

                    v = TestFunction(self.V)
                    ds_i = self.dirichlet_measures[i]
                    mid = self.dirichlet_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (uD_curr - uD_next)
                            * v
                            * ds_i(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()

                # Last step
                else:
                    uD_prev = u_dirichlet_funcs_time[m - 1][i]

                    v = TestFunction(self.V)
                    ds_i = self.dirichlet_measures[i]
                    mid = self.dirichlet_marker_ids[i]
                    b_h1t = assemble_vector(
                        form(
                            (self.gamma_u / self.dt)
                            * (uD_curr - uD_prev)
                            * v
                            * ds_i(mid)
                        )
                    )
                    b_h1t.ghostUpdate(
                        addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
                    )

                    temp.x.petsc_vec.set(0.0)
                    self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                    temp.x.scatter_forward()

                    gD.x.array[:] += temp.x.array[:]
                    gD.x.scatter_forward()
    return grad


def tgrad_impl(self, u):
    """Tangential gradient on the boundary: (I - n⊗n) ∇u."""
    n = ufl.FacetNormal(self.domain)
    Id = ufl.Identity(self.domain.geometry.dim)
    P = Id - ufl.outer(n, n)
    return P * ufl.grad(u)


def update_multiplier_mu_impl(
    self, Y_all, sc_type, sc_lower, sc_upper, beta, sc_start_step, sc_end_step
):
    """
    Moreau–Yosida update for PATH constraints over time window [sc_start_step, sc_end_step].
    Works in DG0 (cell-wise) using self.sc_marker.

    Args:
        Y_all: State trajectory [y^0, y^1, ..., y^N]
        sc_type: "lower" | "upper" | "box"
        sc_lower, sc_upper: Temperature bounds
        beta: Penalty parameter
        sc_start_step, sc_end_step: Time window indices

    Returns:
        delta_mu: Max change in multipliers
        feas_inf: Max constraint violation
    """
    Vc = functionspace(self.domain, ("DG", 0))

    # Mask of constrained cells in DG0
    chi_sc_cell = Function(Vc)
    n_local = Vc.dofmap.index_map.size_local
    chi_sc_cell.x.array[:n_local] = self.sc_marker.astype(PETSc.ScalarType)
    chi_sc_cell.x.scatter_forward()
    mask = chi_sc_cell.x.array

    delta_mu_max = 0.0
    feas_inf_max = 0.0

    # Update multipliers for each time step in constraint window
    for m in range(sc_start_step, sc_end_step + 1):
        T_m = Y_all[m]

        # Project temperature to DG0 (cell-wise)
        self._T_cell.interpolate(T_m)
        T_cell = self._T_cell

        # Get old multipliers in DG0
        # Get old multipliers (already DG0)
        muL_old = self.mu_lower_time[m].x.array.copy()
        muU_old = self.mu_upper_time[m].x.array.copy()

        # Compute violations (DG0 arrays)
        violL = np.zeros_like(T_cell.x.array)
        violU = np.zeros_like(T_cell.x.array)

        if sc_type in ["lower", "box"]:
            violL = np.maximum(0.0, sc_lower - T_cell.x.array)
        if sc_type in ["upper", "box"]:
            violU = np.maximum(0.0, T_cell.x.array - sc_upper)

        # Apply mask
        violL *= mask
        violU *= mask

        # Moreau–Yosida update
        muL_new = muL_old + beta * violL
        muU_new = muU_old + beta * violU
        muL_new = np.maximum(0.0, muL_new) * mask
        muU_new = np.maximum(0.0, muU_new) * mask

        # Write back (DG0)
        self.mu_lower_time[m].x.array[:] = muL_new
        self.mu_upper_time[m].x.array[:] = muU_new
        self.mu_lower_time[m].x.scatter_forward()
        self.mu_upper_time[m].x.scatter_forward()

        # Track convergence measures
        dL = float(np.max(np.abs(muL_new - muL_old))) if muL_new.size else 0.0
        dU = float(np.max(np.abs(muU_new - muU_old))) if muU_new.size else 0.0
        delta_mu_max = max(delta_mu_max, dL, dU)

        vL = float(np.max(violL)) if violL.size else 0.0
        vU = float(np.max(violU)) if violU.size else 0.0
        feas_inf_max = max(feas_inf_max, vL, vU)

        # Debug output for first and last step in window
        if self.domain.comm.rank == 0 and (m == sc_start_step or m == sc_end_step):
            active = mask > 0.5
            n_active = int(np.sum(active))
            if n_active > 0:
                Tmin_sc = float(np.min(T_cell.x.array[active]))
                Tmax_sc = float(np.max(T_cell.x.array[active]))
            else:
                Tmin_sc = Tmax_sc = float("nan")

            muL_max = float(np.max(muL_new)) if muL_new.size else 0.0
            muU_max = float(np.max(muU_new)) if muU_new.size else 0.0
            t_m = m * self.dt
            logger.debug(
                f"SC-PATH t={t_m:.1f}s (m={m}): T∈[{Tmin_sc:.2f}, {Tmax_sc:.2f}]°C, violL={vL:.2e}, violU={vU:.2e}, μL_max={muL_max:.2e}, μU_max={muU_max:.2e}"
            )

    # Summary
    if self.domain.comm.rank == 0:
        logger.debug(
            f"SC-PATH-SUMMARY Window steps=[{sc_start_step},{sc_end_step}], max_violation={feas_inf_max:.2e}, max_Δμ={delta_mu_max:.2e}"
        )

    # MPI collective operations - all ranks must call
    Vc_summary = functionspace(self.domain, ("DG", 0))
    mask = self.sc_marker.astype(bool)
    n_local = Vc_summary.dofmap.index_map.size_local
    max_out_L = 0.0
    max_out_U = 0.0
    max_in_L = 0.0
    max_in_U = 0.0
    for m in range(sc_start_step, sc_end_step + 1):
        muL = Function(Vc_summary)
        muL.interpolate(self.mu_lower_time[m])
        muU = Function(Vc_summary)
        muU.interpolate(self.mu_upper_time[m])
        aL = muL.x.array[:n_local]
        aU = muU.x.array[:n_local]
        if aL.size:
            max_in_L = max(
                max_in_L, float(np.max(aL[mask])) if np.any(mask) else 0.0
            )
            max_in_U = max(
                max_in_U, float(np.max(aU[mask])) if np.any(mask) else 0.0
            )
            max_out_L = max(
                max_out_L, float(np.max(aL[~mask])) if np.any(~mask) else 0.0
            )
            max_out_U = max(
                max_out_U, float(np.max(aU[~mask])) if np.any(~mask) else 0.0
            )
    if self.domain.comm.rank == 0:
        logger.debug(
            f"TEST-MU max_in: muL={max_in_L:.3e} muU={max_in_U:.3e} | max_out: muL={max_out_L:.3e} muU={max_out_U:.3e}"
        )

    return delta_mu_max, feas_inf_max
