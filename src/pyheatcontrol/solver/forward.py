from petsc4py import PETSc
from ufl import dx, TestFunction

from dolfinx.fem import form, Function, Constant, assemble_scalar, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

from pyheatcontrol.logging_config import logger

def solve_forward_impl(self, q_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure):

        T_old = Function(self.V)
        v = TestFunction(self.V)
        dt_c = Constant(self.domain, PETSc.ScalarType(self.dt))
        Y_all = []
        T = Function(self.V)
        T.x.array[:] = self.T_ambient
        T_old.x.array[:] = self.T_ambient
        Y_all.append(T.copy())

        # -------------------------------------------------
        # Dirichlet BC placeholders (ONE-TIME) + assemble A once
        # -------------------------------------------------
        bc_list = []
        uD_bc = []

        for i in range(self.n_ctrl_dirichlet):
            uD_i = Function(self.V)
            uD_i.x.array[:] = 0.0
            uD_i.x.scatter_forward()
            uD_bc.append(uD_i)

            dofs_i = self.dirichlet_dofs[i]
            bc_list.append(dirichletbc(uD_i, dofs_i))

        # Assemble state matrix ONCE (Dirichlet DOFs are the same for all time steps)
        A = assemble_matrix(self.a_state_compiled, bcs=bc_list)
        A.assemble()
        self.ksp.setOperators(A)

        # -------------------------------------------------
        # RHS placeholders + precompiled RHS form (ONE-TIME)
        # -------------------------------------------------
        # Placeholder per il termine dei moltiplicatori (DG0)
        mu_sum_cell = Function(self.Vc)
        mu_sum_cell.x.array[:] = 0.0
        mu_sum_cell.x.scatter_forward()

        # Placeholder per i controlli distribuiti (P2)
        u_dist_cur = []
        for i in range(self.n_ctrl_distributed):
            f = Function(self.V)
            f.x.array[:] = 0.0
            f.x.scatter_forward()
            u_dist_cur.append(f)

        # Nota: self.q_neumann_funcs[i] esiste già (P2) e tu lo aggiorni nel loop quindi possiamo usarlo direttamente nel form RHS.
        # Costruisci RHS UFL UNA VOLTA SOLA (usa placeholders e funzioni che cambiano solo nei valori)
        L_rhs_ufl = (self.rho_c / dt_c) * T_old * v * dx

        # Distributed term: usa placeholder u_dist_cur[i]
        for i in range(self.n_ctrl_distributed):
            L_rhs_ufl += u_dist_cur[i] * self.chi_distributed_V[i] * v * dx

        # State-constraint multipliers: usa mu_sum_cell placeholder (DG0)
        L_rhs_ufl += mu_sum_cell * self.chi_sc_cell * v * dx

        # Neumann term: usa self.q_neumann_funcs aggiornato nel loop
        for i, mid in enumerate(self.neumann_marker_ids):
            L_rhs_ufl += self.q_neumann_funcs[i] * v * self.ds_neumann(mid)

        # Precompila form RHS
        rhs_form = form(L_rhs_ufl)

        for step in range(self.num_steps):
            # -------------------------
            # Update Dirichlet BC values (NO re-assembly of matrix)
            # -------------------------
            for i in range(self.n_ctrl_dirichlet):
                uD_time = u_dirichlet_funcs_time[step][i]
                dofs_i = self.dirichlet_dofs[i]

                # copy only on ΓD dofs (keep zeros elsewhere)
                uD_bc[i].x.array[:] = 0.0
                uD_bc[i].x.array[dofs_i] = uD_time.x.array[dofs_i]
                uD_bc[i].x.scatter_forward()

                if step == 0 and self.domain.comm.rank == 0 and i == 0:
                    logger.debug(f"uD_time.x.array[dofs_i] = {uD_time.x.array[dofs_i][:5]}")
                if step == 0 and self.domain.comm.rank == 0 and self.n_ctrl_dirichlet > 0:
                    dofs0 = self.dirichlet_dofs[0]
                    logger.debug(f"uD_bc[0] on ΓD min/max = {float(uD_bc[0].x.array[dofs0].min())}, {float(uD_bc[0].x.array[dofs0].max())}")

            # -------------------------
            # Neumann control (P2) - keep only boundary DOFs of each segment
            # -------------------------
            # Handle both [step] (single zone) and [step][i] (multiple zones) formats
            for i in range(self.n_ctrl_neumann):
                dofs_i = self.neumann_dofs[i]

                # start from zero everywhere
                self.q_neumann_funcs[i].x.array[:] = 0.0

                # Get the Function for this zone and time step
                q_time = q_neumann_funcs_time[step][i]  # ✅ [step][i]

                # copy only on the boundary dofs of this Neumann segment
                self.q_neumann_funcs[i].x.array[dofs_i] = q_time.x.array[dofs_i]
                self.q_neumann_funcs[i].x.scatter_forward()
                # DEBUG
                if step == 0 and i == 0 and self.domain.comm.rank == 0:
                    logger.debug(f"step={step}, q_func min/max (global) = "
                        f"{self.q_neumann_funcs[i].x.array.min():.6e}, "
                        f"{self.q_neumann_funcs[i].x.array.max():.6e}")
                    logger.debug(f"step={step}, q_func min/max (on Γ) = "
                        f"{self.q_neumann_funcs[i].x.array[dofs_i].min():.6e}, "
                        f"{self.q_neumann_funcs[i].x.array[dofs_i].max():.6e}")

            # (optional debug: only first step)
            if step == 0 and self.domain.comm.rank == 0 and self.n_ctrl_neumann > 0:
                dofs0 = self.neumann_dofs[0]
                q0 = self.q_neumann_funcs[0].x.array
                logger.debug(
                    f"step=0 q_on_Gamma min/max = {float(q0[dofs0].min())}, {float(q0[dofs0].max())} "
                    f"| q_global min/max = {float(q0.min())}, {float(q0.max())}"
                )
                mid0 = self.neumann_marker_ids[0]
                q_int = assemble_scalar(form(self.q_neumann_funcs[0] * self.ds_neumann(mid0)))
                logger.debug(f"int_Gamma q ds = {float(q_int)}")

            # -------------------------
            # RHS (placeholders update + assemble precompiled form)
            # -------------------------
            # Update distributed control placeholders for this step
            for i in range(self.n_ctrl_distributed):
                u_distributed_funcs_time[step][i].x.petsc_vec.copy(u_dist_cur[i].x.petsc_vec)
                u_dist_cur[i].x.scatter_forward()

            # Update mu_sum_cell (DG0) for this step (+1 because state is at end of interval)
            mu_sum_cell.x.array[:] = (self.mu_lower_time[step + 1].x.array + self.mu_upper_time[step + 1].x.array)
            mu_sum_cell.x.scatter_forward()

            # -------------------------
            # solve
            # -------------------------
            # Assemble RHS into a reused vector (allocate once on first step)
            if step == 0:
                b = assemble_vector(rhs_form)   # allocates Vec
            else:
                with b.localForm() as b_loc:
                    b_loc.set(0.0)
                assemble_vector(b, rhs_form)    # fills existing Vec

            apply_lifting(b, [self.a_state_compiled], bcs=[bc_list])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bc_list)

            self.ksp.solve(b, T.x.petsc_vec)
            T.x.scatter_forward()

            if step == 0 and self.domain.comm.rank == 0 and self.n_ctrl_dirichlet > 0:
                dofs_check = self.dirichlet_dofs[0]
                logger.debug(f"T after solve on ΓD = {T.x.array[dofs_check][:5]}")

            if step == self.num_steps - 1 and self.domain.comm.rank == 0:
                logger.debug(f"T global min/max: {float(T.x.array.min()):.12e} {float(T.x.array.max()):.12e}")
                if self.n_ctrl_dirichlet > 0:
                    dofsD = self.dirichlet_dofs[0]
                    logger.debug(f"T on ΓD min/max: {float(T.x.array[dofsD].min()):.12e} {float(T.x.array[dofsD].max()):.12e}")

            T.x.petsc_vec.copy(T_old.x.petsc_vec)
            T_old.x.scatter_forward()
            Y_all.append(T.copy())

        self.Y_all = Y_all
        return Y_all
