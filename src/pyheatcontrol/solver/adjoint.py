from petsc4py import PETSc
import ufl
from ufl import TestFunction
import numpy as np
from dolfinx.fem import form, Function, Constant, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc


def solve_adjoint_impl(self, Y_all, T_ref):
    """Backward solve for adjoint equation (tracking + state constraint), trapezoidal in time."""
    v = TestFunction(self.V)
    dx = ufl.Measure("dx", domain=self.domain)
    dt_const = Constant(self.domain, PETSc.ScalarType(self.dt))

    # P_all[m] = p^m for m=0..N (N = num_steps). self.Nt = N+1
    P_all = [None] * self.Nt

    # Adjoint BC: p=0 on Dirichlet-controlled DOFs
    bc_list_adj = [
        dirichletbc(self._zero_p, dofs) for dofs in self.bc_dofs_list_dirichlet
    ]
    # Add fixed Dirichlet BCs (p=0 on Γ where y is prescribed)
    for dofs in self.dirichlet_bc_dofs:
        bc_list_adj.append(dirichletbc(self._zero_p, dofs))
    # Add Dirichlet disturbance BCs (p=0 on Γ where d(t) is prescribed)
    for dofs in self.dirichlet_dist_dofs:
        bc_list_adj.append(dirichletbc(self._zero_p, dofs))

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
    y_ph = Function(self.V)  # placeholder for Y_all[m]
    muL_ph = Function(self.Vc)  # DG0
    muU_ph = Function(self.Vc)  # DG0
    T_ref_ph = Function(self.V)  # placeholder for T_ref (can vary in space and time)
    T_ref_ph.x.array[:] = T_ref if isinstance(T_ref, (int, float)) else 0.0
    T_ref_ph.x.scatter_forward()

    # Build RHS UFL with trapezoidal weights: 1.0 (interior), 0.5 (endpoints)
    def build_L_adj_ufl(weight_value: float):
        w = PETSc.ScalarType(weight_value)
        tracking = 0

        if abs(self.alpha_track) > 1e-30:
            for chi_t in self.chi_targets:
                tracking += (
                    self.alpha_track * w * self.dt * (y_ph - T_ref_ph) * chi_t * v * dx
                )
            for ds_b in self.target_boundary_ds:
                tracking += (
                    self.alpha_track * w * self.dt * (y_ph - T_ref_ph) * v * ds_b
                )

        # state-constraint forcing
        tracking += (-w) * muL_ph * self.chi_sc_cell * v * dx
        tracking += (+w) * muU_ph * self.chi_sc_cell * v * dx

        return (self.rho_c / dt_const) * p_next * v * dx + tracking

    L_adj_ufl_w1 = build_L_adj_ufl(1.0)
    L_adj_ufl_w05 = build_L_adj_ufl(0.5)
    L_adj_form_w1 = form(L_adj_ufl_w1)
    L_adj_form_w05 = form(L_adj_ufl_w05)

    # Backward: compute p^m for m = N, ..., 0
    for m in reversed(range(self.Nt)):
        y_current = Y_all[m]

        # Update placeholders
        y_current.x.petsc_vec.copy(y_ph.x.petsc_vec)
        y_ph.x.scatter_forward()
        # Update T_ref placeholder for this time step
        if hasattr(self, 'T_ref_values'):
            T_ref_step = self.T_ref_values[m]
            if isinstance(T_ref_step, (int, float, np.floating)):
                T_ref_ph.x.array[:] = T_ref_step
            else:
                T_ref_ph.x.array[:] = T_ref_step.x.array[:]
            T_ref_ph.x.scatter_forward()

        muL_ph.x.array[:] = self.mu_lower_time[m]
        muL_ph.x.scatter_forward()

        muU_ph.x.array[:] = self.mu_upper_time[m]
        muU_ph.x.scatter_forward()

        # Select correct precompiled form (trapezoidal weight)
        L_form = L_adj_form_w05 if (m == 0 or m == self.num_steps) else L_adj_form_w1

        # Assemble adjoint RHS (allocate on first iteration, reuse thereafter)
        if m == self.num_steps:
            b_adj = assemble_vector(L_form)
        else:
            with b_adj.localForm() as b_loc:
                b_loc.set(0.0)
            assemble_vector(b_adj, L_form)

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
