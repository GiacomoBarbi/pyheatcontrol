import logging
from petsc4py import PETSc
from ufl import dx, TestFunction
import numpy as np
from mpi4py import MPI
import math

from dolfinx.fem import form, Function, Constant, assemble_scalar, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

from pyheatcontrol.logging_config import logger

def _global_minmax(comm, arr):
    # arr è un numpy array locale (può essere vuoto)
    if arr.size > 0:
        lmin = float(arr.min())
        lmax = float(arr.max())
    else:
        lmin = np.inf
        lmax = -np.inf
    gmin = comm.allreduce(lmin, op=MPI.MIN)
    gmax = comm.allreduce(lmax, op=MPI.MAX)
    return gmin, gmax, arr.size


def solve_forward_impl(
    self, q_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time
):
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

    # Add fixed Dirichlet BCs
    for i in range(self.n_dirichlet_bc):
        bc_list.append(dirichletbc(self.dirichlet_bc_funcs[i], self.dirichlet_bc_dofs[i]))
    # Add Dirichlet disturbance BCs
    for i in range(self.n_dirichlet_dist):
        bc_list.append(dirichletbc(self.dirichlet_dist_funcs[i], self.dirichlet_dist_dofs[i]))

    # Assemble state matrix ONCE (Dirichlet DOFs are the same for all time steps)
    A = assemble_matrix(self.a_state_compiled, bcs=bc_list)
    A.assemble()
    self.ksp.setOperators(A)

    # -------------------------------------------------
    # RHS placeholders + precompiled RHS form (ONE-TIME)
    # -------------------------------------------------
    # Distributed control placeholders (P2)
    u_dist_cur = []
    for i in range(self.n_ctrl_distributed):
        f = Function(self.V)
        f.x.array[:] = 0.0
        f.x.scatter_forward()
        u_dist_cur.append(f)

    # self.q_neumann_funcs[i] already exists (P2) and is updated in the loop
    # Build RHS UFL once (placeholders are updated by value in the time loop)
    L_rhs_ufl = (self.rho_c / dt_c) * T_old * v * dx

    # Distributed source term
    for i in range(self.n_ctrl_distributed):
        L_rhs_ufl += u_dist_cur[i] * self.chi_distributed_V[i] * v * dx

    # Neumann boundary term (control)
    for i, mid in enumerate(self.neumann_marker_ids):
        L_rhs_ufl += self.q_neumann_funcs[i] * v * self.ds_neumann(mid)

    # Prescribed (non-homogeneous) Neumann: k ∂_n T = g
    if getattr(self, "n_neumann_bc", 0) > 0:
        for i in range(self.n_neumann_bc):
            mid = self.neumann_prescribed_marker_ids[i]
            g_i = self.neumann_prescribed_constants[i]
            L_rhs_ufl += g_i * v * self.ds_neumann_prescribed(mid)

    # Precompile RHS form
    rhs_form = form(L_rhs_ufl)

    for step in range(self.num_steps):

        # Update Dirichlet disturbance values for current time
        t_current = (step + 1) * self.dt
        for i in range(self.n_dirichlet_dist):
            func_type, param = self.dirichlet_dist_params[i]
            dofs_i = self.dirichlet_dist_dofs[i]
            if func_type == "tanh":
                val = math.tanh(param * t_current)
                self.dirichlet_dist_funcs[i].x.array[dofs_i] = val
            elif func_type == "sin":
                val = math.sin(param * t_current)
                self.dirichlet_dist_funcs[i].x.array[dofs_i] = val
            elif func_type == "cos":
                val = math.cos(param * t_current)
                self.dirichlet_dist_funcs[i].x.array[dofs_i] = val
            elif func_type == "sin_y_cos_t":
                # d(y,t) = sin(param * y) * cos(t)
                self.dirichlet_dist_funcs[i].interpolate(
                    lambda x, t=t_current, p=param: np.sin(p * x[1]) * np.cos(t)
                )
            elif func_type == "sin_y_sin_t":
                # d(y,t) = sin(param * y) * sin(t)
                self.dirichlet_dist_funcs[i].interpolate(
                    lambda x, t=t_current, p=param: np.sin(p * x[1]) * np.sin(t)
                )
            else:  # const
                val = param
                self.dirichlet_dist_funcs[i].x.array[dofs_i] = val
            self.dirichlet_dist_funcs[i].x.scatter_forward()

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

            if step == 0 and i == 0 and logger.isEnabledFor(logging.DEBUG):
                comm = self.domain.comm
                dofs0 = self.dirichlet_dofs[0]
                vals0 = uD_bc[0].x.array[dofs0]
                gmin, gmax, nloc = _global_minmax(comm, vals0)
                nglob = comm.allreduce(int(nloc), op=MPI.SUM)
                if comm.rank == 0:
                    logger.debug(f"Dirichlet ΓD dofs: local(rank0)={nloc}, global={nglob}")
                    logger.debug(f"uD_bc[0] on ΓD global min/max = {gmin:.6e}, {gmax:.6e}")

        # -------------------------
        # Neumann control (P2) - keep only boundary DOFs of each segment
        # -------------------------
        
        for i in range(self.n_ctrl_neumann):
            dofs_i = self.neumann_dofs[i]

            # zero everywhere, then copy boundary DOFs
            self.q_neumann_funcs[i].x.array[:] = 0.0

            
            q_time = q_neumann_funcs_time[step][i]

            
            self.q_neumann_funcs[i].x.array[dofs_i] = q_time.x.array[dofs_i]
            self.q_neumann_funcs[i].x.scatter_forward()

            if step == 0 and i == 0 and logger.isEnabledFor(logging.DEBUG):
                comm = self.domain.comm
                a_loc = self.q_neumann_funcs[i].x.array
                if a_loc.size > 0:
                    local_min = float(a_loc.min())
                    local_max = float(a_loc.max())
                else:
                    local_min = np.inf
                    local_max = -np.inf
                gmin = comm.allreduce(local_min, op=MPI.MIN)
                gmax = comm.allreduce(local_max, op=MPI.MAX)
                b_loc = self.q_neumann_funcs[i].x.array[dofs_i]
                if b_loc.size > 0:
                    local_bmin = float(b_loc.min())
                    local_bmax = float(b_loc.max())
                else:
                    local_bmin = np.inf
                    local_bmax = -np.inf
                gbmin = comm.allreduce(local_bmin, op=MPI.MIN)
                gbmax = comm.allreduce(local_bmax, op=MPI.MAX)
                if comm.rank == 0:
                    logger.debug(
                        f"step={step}, q_func min/max (global) = "
                        f"{gmin:.6e}, {gmax:.6e}"
                    )
                    logger.debug(
                        f"step={step}, q_func min/max (on Γ) = "
                        f"{gbmin:.6e}, {gbmax:.6e}"
                    )
        
        if step == 0 and self.n_ctrl_neumann > 0 and logger.isEnabledFor(logging.DEBUG):
            comm = self.domain.comm

            dofs0 = self.neumann_dofs[0]
            q0 = self.q_neumann_funcs[0].x.array
            mid0 = self.neumann_marker_ids[0]

            # assemble_scalar is MPI collective -> all ranks must call
            q_int = assemble_scalar(form(self.q_neumann_funcs[0] * self.ds_neumann(mid0)))

            # --- q on Gamma (local slice + allreduce) ---
            vals_g = q0[dofs0]
            if vals_g.size > 0:
                local_gmin = float(vals_g.min())
                local_gmax = float(vals_g.max())
            else:
                local_gmin = np.inf
                local_gmax = -np.inf

            gmin = comm.allreduce(local_gmin, op=MPI.MIN)
            gmax = comm.allreduce(local_gmax, op=MPI.MAX)

            # --- q global (local slice + allreduce) ---
            if q0.size > 0:
                local_min = float(q0.min())
                local_max = float(q0.max())
            else:
                local_min = np.inf
                local_max = -np.inf

            qmin = comm.allreduce(local_min, op=MPI.MIN)
            qmax = comm.allreduce(local_max, op=MPI.MAX)

            # q_int is already global from assemble_scalar, but printing only on rank 0
            if comm.rank == 0:
                logger.debug(
                    f"step=0 q_on_Gamma min/max = {gmin:.6e}, {gmax:.6e} "
                    f"| q_global min/max = {qmin:.6e}, {qmax:.6e}"
                )
                logger.debug(f"int_Gamma q ds = {float(q_int):.6e}")

        # -------------------------
        # RHS (placeholders update + assemble precompiled form)
        # -------------------------
        # Update distributed control placeholders
        for i in range(self.n_ctrl_distributed):
            u_distributed_funcs_time[step][i].x.petsc_vec.copy(
                u_dist_cur[i].x.petsc_vec
            )
            u_dist_cur[i].x.scatter_forward()

        # -------------------------
        # solve
        # -------------------------
        # Assemble RHS (allocate on first step, reuse thereafter)
        if step == 0:
            b = assemble_vector(rhs_form)
        else:
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            assemble_vector(b, rhs_form)

        apply_lifting(b, [self.a_state_compiled], bcs=[bc_list])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bc_list)

        self.ksp.solve(b, T.x.petsc_vec)
        T.x.scatter_forward()

        if step == self.num_steps - 1 and logger.isEnabledFor(logging.DEBUG):
            comm = self.domain.comm
            gminT, gmaxT, _ = _global_minmax(comm, T.x.array)
            if comm.rank == 0:
                logger.debug(f"T global min/max: {gminT:.12e} {gmaxT:.12e}")
            if self.n_ctrl_dirichlet > 0:
                dofsD = self.dirichlet_dofs[0]
                nloc = self.V.dofmap.index_map.size_local
                dofsD_owned = dofsD[dofsD < nloc]
                vals = T.x.array[dofsD_owned]
                gmin, gmax, _ = _global_minmax(comm, vals)
                if comm.rank == 0:
                    logger.debug(f"T_final on ΓD min/max: {gmin:.12e} {gmax:.12e}")

        T.x.petsc_vec.copy(T_old.x.petsc_vec)
        T_old.x.scatter_forward()
        Y_all.append(T.copy())

    self.Y_all = Y_all
    return Y_all
