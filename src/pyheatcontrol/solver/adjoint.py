import numpy as np
from petsc4py import PETSc
import ufl
from ufl import dx, grad as ufl_grad, inner, TestFunction, TrialFunction, FacetNormal

from dolfinx.fem import form, Function, Constant, functionspace, assemble_scalar, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

def solve_adjoint_impl(self, Y_all, T_cure):
    """Backward solve for adjoint equation (tracking + (eventuale) vincolo di stato), trapezoidal in time.
    Tutto coerente in V (P2): niente .x.array per costruire il forcing.
    """
    v = TestFunction(self.V)
    dx = ufl.Measure("dx", domain=self.domain)
    n = FacetNormal(self.domain)
    dt_const = Constant(self.domain, PETSc.ScalarType(self.dt))

    # P_all[m] = p^m for m=0..N (N = num_steps). self.Nt = N+1
    P_all = [None] * self.Nt

    # Adjoint BC: p=0 on Dirichlet-controlled DOFs (se presenti)
    bc_list_adj = [dirichletbc(self._zero_p, dofs) for dofs in self.bc_dofs_list_dirichlet]

    # Assemble adjoint matrix once
    A_adj = assemble_matrix(self.a_adjoint_compiled, bcs=bc_list_adj)
    A_adj.assemble()
    self.ksp.setOperators(A_adj)

    # p_next = p^{m+1}; start with p^{N+1} = 0
    p_next = Function(self.V)
    p_next.x.array[:] = 0.0
    p_next.x.scatter_forward()

    # Work vector for solution p (avoid re-alloc each loop)
    p = Function(self.V)

    # -------------------------------------------------
    # Placeholders + precompiled adjoint RHS forms (ONE-TIME)
    # -------------------------------------------------
    y_ph = Function(self.V)      # placeholder for Y_all[m]
    muL_ph = Function(self.Vc)   # DG0
    muU_ph = Function(self.Vc)   # DG0

    # Build RHS UFL for two weights: 1.0 (interior) and 0.5 (endpoints)
    def build_L_adj_ufl(weight_value: float):
        w = PETSc.ScalarType(weight_value)
        tracking = 0

        if abs(self.alpha_track) > 1e-30:
            for chi_t in self.chi_targets:
                tracking += self.alpha_track * w * self.dt * (y_ph - T_cure) * chi_t * v * dx

        # state-constraint forcing (same as your signs, with weight)
        tracking += (-w) * muL_ph * self.chi_sc * v * dx
        tracking += (+w) * muU_ph * self.chi_sc * v * dx

        return (self.rho_c / dt_const) * p_next * v * dx + tracking

    L_adj_ufl_w1  = build_L_adj_ufl(1.0)
    L_adj_ufl_w05 = build_L_adj_ufl(0.5)
    L_adj_form_w1  = form(L_adj_ufl_w1)
    L_adj_form_w05 = form(L_adj_ufl_w05)

    # Backward: compute p^m for m = N, ..., 0
    for m in reversed(range(self.Nt)):
        y_current = Y_all[m]

        # Update placeholders
        y_current.x.petsc_vec.copy(y_ph.x.petsc_vec)
        y_ph.x.scatter_forward()

        self.mu_lower_time[m].x.petsc_vec.copy(muL_ph.x.petsc_vec)
        muL_ph.x.scatter_forward()

        self.mu_upper_time[m].x.petsc_vec.copy(muU_ph.x.petsc_vec)
        muU_ph.x.scatter_forward()

        # Select correct precompiled form (trapezoidal weight)
        L_form = L_adj_form_w05 if (m == 0 or m == self.num_steps) else L_adj_form_w1

        # trapezoidal weight
        weight = 0.5 if (m == 0 or m == self.num_steps) else 1.0

        # Tracking forcing (UFL, in V) — costruiscilo SOLO se alpha_track != 0
        tracking_form = 0
        if abs(self.alpha_track) > 1e-30:
            for chi_t in self.chi_targets:
                tracking_form += self.alpha_track * weight * self.dt * (y_current - T_cure) * chi_t * v * dx

        # -------------------------
        # State-constraint forcing (NO interpolation, keep DG0)
        # -------------------------
        muL_m = self.mu_lower_time[m]   # DG0 function
        muU_m = self.mu_upper_time[m]   # DG0 function

        # --- TEST 3: check SC forcing is only on chi_sc (not targets) ---
        if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps):
            sc_int = assemble_scalar(form((muL_m + muU_m) * self.chi_sc * dx))
            tgt_ints = [
                assemble_scalar(form((muL_m + muU_m) * chi_t * dx))
                for chi_t in self.chi_targets
            ]

        tracking_form += (-weight) * muL_m * self.chi_sc * v * dx
        tracking_form += (+weight) * muU_m * self.chi_sc * v * dx

        # Adjoint RHS: (rho_c/dt) * p^{m+1} + forcing
        L_adj = (self.rho_c / dt_const) * p_next * v * dx + tracking_form

        # Assemble adjoint RHS into a reused vector (allocate once on first iteration)
        if m == self.num_steps:
            b_adj = assemble_vector(L_form)   # allocate Vec once (first assembly)
        else:
            with b_adj.localForm() as b_loc:
                b_loc.set(0.0)
            assemble_vector(b_adj, L_form)    # refill existing Vec

        apply_lifting(b_adj, [self.a_adjoint_compiled], bcs=[bc_list_adj])
        b_adj.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_adj, bc_list_adj)

        # solve for p^m
        self.ksp.solve(b_adj, p.x.petsc_vec)
        p.x.scatter_forward()
        P_all[m] = p.copy()

        # update p_next <- p
        p.x.petsc_vec.copy(p_next.x.petsc_vec)
        p_next.x.scatter_forward()

    self.P_all = P_all
    return P_all
