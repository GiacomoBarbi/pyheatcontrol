# solver.py

import numpy as np
from petsc4py import PETSc
import ufl
from ufl import dx, grad as ufl_grad, inner, TestFunction, TrialFunction, FacetNormal
from dolfinx.fem import form, Function, Constant, functionspace, assemble_scalar, dirichletbc
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from mesh_utils import (create_boundary_condition_function, create_boundary_facet_tags, mark_cells_in_boxes)


class TimeDepHeatSolver:
    """
    Heat solver con TIME-DEPENDENT controls
    Supporta: Dirichlet boundary, Neumann boundary, Distributed
    u_controls shape: (n_ctrl_spatial, Nt)
    """

    def __init__(self, domain, V, dt, num_steps, k_val, rho, c, T_ambient,
             control_boundary_dirichlet, control_boundary_neumann,
             control_distributed_boxes, target_boxes, constraint_boxes, L,
             alpha_track, alpha_u, gamma_u, beta_u, dirichlet_spatial_reg):

        self.domain = domain
        self.V = V
        self.V0 = functionspace(domain, ("Lagrange", 1))
        self.Vc = functionspace(domain, ("DG", 0))
        self.dt = dt
        self.num_steps = num_steps
        self.Nt = num_steps + 1
        self.T_ambient = T_ambient
        self.L = L

        # Physical parameters
        self.k_therm = Constant(domain, PETSc.ScalarType(k_val))
        self.rho_c = Constant(domain, PETSc.ScalarType(rho * c))
        self.k_val = k_val

        # Optimization parameters
        self.alpha_track = alpha_track
        self.alpha_u = alpha_u
        self.gamma_u = gamma_u
        self.beta_u = beta_u
        self.dirichlet_spatial_reg = dirichlet_spatial_reg

        # Controls count
        self.n_ctrl_dirichlet = len(control_boundary_dirichlet)
        self.n_ctrl_neumann = len(control_boundary_neumann)
        self.n_ctrl_distributed = len(control_distributed_boxes)
        self.n_ctrl_scalar = 0  # Dirichlet non è più scalare, è Function(V) come Neumann
        self.n_ctrl_spatial = self.n_ctrl_scalar  # compatibilità (se lo usi altrove)

        self.control_boundary_dirichlet = control_boundary_dirichlet
        self.control_boundary_neumann = control_boundary_neumann

        self._T_cell = Function(self.Vc)  # DG0

        if self.domain.comm.rank == 0:
            print("[DEBUG-SOLVER] constraint_boxes received =", constraint_boxes, flush=True)

        # -------------------------
        # Dirichlet controls
        # -------------------------
        from dolfinx.mesh import meshtags

        tdim = domain.topology.dim
        fdim = tdim - 1

        self.bc_funcs_dirichlet = []
        self.bc_dofs_list_dirichlet = []
        self.dirichlet_marker_ids = []
        self.dirichlet_measures = []

        self._zero_p = Function(self.V)
        self._zero_p.x.array[:] = 0.0
        self._zero_p.x.scatter_forward()

        # per costruire un UNICO facet_tags complessivo (solo per visualizzazione omega_qd)
        all_facets_D = []
        all_values_D = []

        for i, seg in enumerate(control_boundary_dirichlet):
            # dofs + funzione bc (per applicare bc distribuito su ΓD)
            dofs, bc_f = create_boundary_condition_function(domain, V, [seg], L)
            self.bc_dofs_list_dirichlet.append(dofs)
            self.bc_funcs_dirichlet.append(bc_f)

            # marker per questo segmento
            marker_id = i + 1
            facet_tags_i, _ = create_boundary_facet_tags(domain, [seg], L, marker_id)

            # misura ds “per questo segmento”
            self.dirichlet_marker_ids.append(marker_id)
            self.dirichlet_measures.append(ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_i))

            # accumula facets/values per costruire una meshtags unica
            all_facets_D.append(facet_tags_i.indices.astype(np.int32))
            all_values_D.append(facet_tags_i.values.astype(np.int32))

        # ---- build ONE combined meshtags for all Dirichlet segments (for visualization) ----
        self.dirichlet_facet_tags = None
        if len(all_facets_D) > 0:
            facets = np.hstack(all_facets_D).astype(np.int32)
            values = np.hstack(all_values_D).astype(np.int32)

            # ordina per facet id
            order = np.argsort(facets, kind="mergesort")
            facets = facets[order]
            values = values[order]

            # rimuovi duplicati (se esistono): tieni l’ULTIMO marker per facet
            # (così se due segmenti si sovrappongono, uno “vince” e non crasha)
            unique_facets, first_idx = np.unique(facets, return_index=True)
            # trick: per tenere lanciato l’ultimo valore, usiamo return_index sul reversed
            unique_facets_rev, first_idx_rev = np.unique(facets[::-1], return_index=True)
            last_idx = (len(facets) - 1) - first_idx_rev
            last_idx_sorted = np.sort(last_idx)

            facets_u = facets[last_idx_sorted]
            values_u = values[last_idx_sorted]

            self.dirichlet_facet_tags = meshtags(domain, fdim, facets_u, values_u)

        # -------------------------
        # Neumann controls (P2 on Γ)
        # -------------------------
        tdim = domain.topology.dim
        fdim = tdim - 1

        self.neumann_marker_ids = []
        self.ds_neumann = ufl.Measure("ds", domain=domain)

        all_facets = []
        all_values = []

        for i, seg in enumerate(control_boundary_neumann):
            marker_id = i + 1
            facet_tags_i, _ = create_boundary_facet_tags(domain, [seg], L, marker_id)
            self.neumann_marker_ids.append(marker_id)
            all_facets.append(facet_tags_i.indices)
            all_values.append(facet_tags_i.values)

        if all_facets:
            facets = np.hstack(all_facets).astype(np.int32)
            values = np.hstack(all_values).astype(np.int32)

            # ordina per facet id
            order = np.argsort(facets, kind="mergesort")
            facets = facets[order]
            values = values[order]

            # rimuovi duplicati: tieni l’ULTIMO marker per facet
            unique_facets_rev, first_idx_rev = np.unique(facets[::-1], return_index=True)
            last_idx = (len(facets) - 1) - first_idx_rev
            last_idx_sorted = np.sort(last_idx)

            facets_u = facets[last_idx_sorted]
            values_u = values[last_idx_sorted]

            self.neumann_facet_tags = meshtags(domain, fdim, facets_u, values_u)
            self.ds_neumann = ufl.Measure("ds", domain=domain, subdomain_data=self.neumann_facet_tags)

        # Neumann DOFs restricted to Γ
        self.neumann_dofs = []
        for seg in control_boundary_neumann:
            dofs, _ = create_boundary_condition_function(domain, V, [seg], L)
            self.neumann_dofs.append(dofs)
        # -------------------------------------------------
        # Riesz maps for Neumann gradients (CORRECT PLACE)
        # -------------------------------------------------
        self.M_neumann = []
        self.ksp_neumann = []

        u_mass = TrialFunction(self.V)
        v_mass = TestFunction(self.V)

        for i, mid in enumerate(self.neumann_marker_ids):
            eps = PETSc.ScalarType(1e-12)
            M_i = assemble_matrix(form(u_mass * v_mass * self.ds_neumann(mid) + eps * u_mass * v_mass * dx))
            M_i.assemble()
            # Make invertible outside ΓN by identity rows/cols (same idea as Dirichlet)
            dofs_i = self.neumann_dofs[i]   # dofs on this Neumann segment (ΓN_i)

            imap = self.V.dofmap.index_map
            nloc = imap.size_local + imap.num_ghosts
            all_dofs = np.arange(nloc, dtype=np.int32)
            off_dofs = np.setdiff1d(all_dofs, dofs_i.astype(np.int32), assume_unique=False).astype(np.int32)

            M_i.zeroRowsColumns(off_dofs, diag=1.0)
            M_i.assemble()

            self.M_neumann.append(M_i)

            ksp_i = PETSc.KSP().create(self.domain.comm)
            ksp_i.setType("cg")
            ksp_i.getPC().setType("jacobi")
            ksp_i.setTolerances(rtol=1e-12)
            ksp_i.setOperators(M_i)
            self.ksp_neumann.append(ksp_i)

        # Neumann control functions (P2)
        self.q_neumann_funcs = []
        for _ in range(self.n_ctrl_neumann):
            q = Function(self.V)
            q.x.array[:] = 0.0
            q.x.scatter_forward()
            self.q_neumann_funcs.append(q)

        # Subito dopo il loop che crea M_neumann
        if self.domain.comm.rank == 0 and self.n_ctrl_neumann > 0:
            print(f"[DEBUG-M_NEUMANN] Number of Neumann zones: {len(self.M_neumann)}")
            for i, M_i in enumerate(self.M_neumann):
                # Get matrix norm
                M_norm = M_i.norm()
                print(f"  Zone {i}: Matrix norm = {M_norm:.6e}", flush=True)
        # -------------------------------------------------
        # Dirichlet control functions (P2) - DISTRIBUTED ON BOUNDARY
        # -------------------------------------------------
        self.u_dirichlet_funcs = []
        for _ in range(self.n_ctrl_dirichlet):
            uD = Function(self.V)
            uD.x.array[:] = 0.0
            uD.x.scatter_forward()
            self.u_dirichlet_funcs.append(uD)

        # Dirichlet DOFs restricted to ΓD (già esistono in bc_dofs_list_dirichlet)
        self.dirichlet_dofs = self.bc_dofs_list_dirichlet  # Riusa quelli già calcolati!

        if self.domain.comm.rank == 0 and self.n_ctrl_dirichlet > 0:
            for i, dofs in enumerate(self.dirichlet_dofs):
                print(f"[INIT] Dirichlet control zone {i+1}: {len(dofs)} DOFs on boundary", flush=True)

        # -------------------------------------------------
        # Mass matrices for Dirichlet control gradients (restricted to ΓD)
        # -------------------------------------------------
        self.M_dirichlet = []
        self.ksp_dirichlet = []

        if self.n_ctrl_dirichlet > 0:
            u_trial = TrialFunction(V)
            v_test  = TestFunction(V)
            n = ufl.FacetNormal(domain)
            I = ufl.Identity(domain.geometry.dim)
            P = I - ufl.outer(n, n)  # projector on tangent space

            for i in range(self.n_ctrl_dirichlet):
                ds_i = self.dirichlet_measures[i]
                mid  = self.dirichlet_marker_ids[i]

                # Always integrate only on the marked segment
                mass_form = u_trial * v_test * ds_i(mid)

                if self.dirichlet_spatial_reg == "H1":
                    # tangential gradient term on boundary
                    gt_u = P * ufl.grad(u_trial)
                    gt_v = P * ufl.grad(v_test)
                    stiff_form = ufl.inner(gt_u, gt_v) * ds_i(mid)
                    a_form = mass_form + PETSc.ScalarType(self.beta_u) * stiff_form
                else:
                    a_form = mass_form  # pure L2

                eps = PETSc.ScalarType(1e-12)
                M_i = assemble_matrix(form(a_form + eps * u_trial * v_test * dx))
                M_i.assemble()

                # Make invertible outside ΓD by identity rows/cols
                dofs_i = self.dirichlet_dofs[i]
                imap = self.V.dofmap.index_map
                nloc = imap.size_local  # SOLO owned dofs
                all_dofs = np.arange(nloc, dtype=np.int32)

                dofs_i = dofs_i.astype(np.int32)
                dofs_i_owned = dofs_i[dofs_i < nloc]  # filtra eventuali ghost

                off_dofs = np.setdiff1d(all_dofs, dofs_i_owned, assume_unique=False).astype(np.int32)
                M_i.zeroRowsColumns(off_dofs, diag=1.0)
                M_i.assemble()

                self.M_dirichlet.append(M_i)

                ksp_i = PETSc.KSP().create(self.domain.comm)
                ksp_i.setType("cg")
                ksp_i.getPC().setType("jacobi")
                ksp_i.setTolerances(rtol=1e-12)
                ksp_i.setOperators(M_i)
                self.ksp_dirichlet.append(ksp_i)

        # -------------------------
        # Distributed controls
        # -------------------------
        self.distributed_markers = []
        for box in control_distributed_boxes:
            self.distributed_markers.append(mark_cells_in_boxes(domain, [box], L))

        self.chi_distributed_V = []
        if self.n_ctrl_distributed > 0:
            V_dg0 = functionspace(domain, ("DG", 0))
            for marker in self.distributed_markers:
                chi_dg0 = Function(V_dg0)
                chi_dg0.x.array[:] = 0.0
                chi_dg0.x.array[marker] = 1.0
                chi = Function(self.V)
                chi.interpolate(chi_dg0)
                self.chi_distributed_V.append(chi)

        # ========== STEP 1: Calcola DOF restriction PRIMA ==========
        self.distributed_dofs = []
        for marker in self.distributed_markers:
            dofs_in_cells = []
            for cell_idx in np.where(marker)[0]:
                cell_dofs = self.V.dofmap.cell_dofs(cell_idx)
                dofs_in_cells.extend(cell_dofs)
            dofs_unique = np.unique(np.array(dofs_in_cells, dtype=np.int32))
            self.distributed_dofs.append(dofs_unique)

            if self.domain.comm.rank == 0:
                print(f"[INIT] Distributed control zone {len(self.distributed_dofs)}: {len(dofs_unique)} DOFs")

        # ========== STEP 2: Crea mass matrices DOPO (ora distributed_dofs è pieno) ==========
        self.M_distributed = []
        self.ksp_distributed = []

        if self.n_ctrl_distributed > 0:
            u_trial = TrialFunction(V)
            v_test = TestFunction(V)

            for i in range(self.n_ctrl_distributed):
                chiV = self.chi_distributed_V[i]

                # Mass matrix: ∫_Ωc u * v * χ dx
                M_i = assemble_matrix(form(u_trial * v_test * chiV * dx))
                M_i.assemble()

                # Make invertible outside Ωc by identity rows/cols
                dofs_i = self.distributed_dofs[i]  # dofs in this control box

                imap = self.V.dofmap.index_map
                nloc = imap.size_local + imap.num_ghosts
                all_dofs = np.arange(nloc, dtype=np.int32)
                off_dofs = np.setdiff1d(all_dofs, dofs_i.astype(np.int32), assume_unique=False).astype(np.int32)

                M_i.zeroRowsColumns(off_dofs, diag=1.0)
                M_i.assemble()

                self.M_distributed.append(M_i)

                ksp_i = PETSc.KSP().create(self.domain.comm)
                ksp_i.setType("cg")
                ksp_i.getPC().setType("jacobi")
                ksp_i.setTolerances(rtol=1e-12)
                ksp_i.setOperators(M_i)
                self.ksp_distributed.append(ksp_i)
        # -------------------------
        # Targets
        # -------------------------
        self.target_markers = []
        for box in target_boxes:
            self.target_markers.append(mark_cells_in_boxes(domain, [box], L))

        self.chi_targets = []
        for marker in self.target_markers:
            chi_dg0 = Function(functionspace(domain, ("DG", 0)))
            chi_dg0.x.array[:] = marker.astype(PETSc.ScalarType)
            chi = Function(self.V0)
            chi.interpolate(chi_dg0)
            self.chi_targets.append(chi)

        # -------------------------
        # State constraints (SEPARATE from targets)
        # -------------------------
        self.constraint_markers = []
        for box in constraint_boxes:
            self.constraint_markers.append(mark_cells_in_boxes(domain, [box], L))

        # unione di tutte le constraint boxes
        if len(self.constraint_markers) == 0:
            self.sc_marker = np.zeros(functionspace(domain, ("DG", 0)).dofmap.index_map.size_local, dtype=bool)
        else:
            self.sc_marker = np.zeros_like(self.constraint_markers[0])

        for m in self.constraint_markers:
            self.sc_marker |= m

        self.chi_sc_cell = Function(self.Vc)
        self.chi_sc_cell.x.array[:] = self.sc_marker.astype(PETSc.ScalarType)
        self.chi_sc_cell.x.scatter_forward()
        chi_sc_dg0 = Function(functionspace(domain, ("DG", 0)))
        chi_sc_dg0.x.array[:] = self.sc_marker.astype(PETSc.ScalarType)
        self.chi_sc = Function(self.V0)
        self.chi_sc.interpolate(chi_sc_dg0)

        if self.domain.comm.rank == 0:
            print(f"[INIT] Constraint zone: {int(np.sum(self.sc_marker))} cells marked", flush=True)
            mt = [assemble_scalar(form(chi*dx)) for chi in self.chi_targets]
            msc = assemble_scalar(form(self.chi_sc*dx))
            print("[DEBUG-MEAS] meas(targets) =", mt, flush=True)
            print("[DEBUG-MEAS] meas(constraint) =", float(msc), flush=True)
            tgt_union = np.zeros_like(self.sc_marker, dtype=bool)
            for m in self.target_markers:
                tgt_union |= m
            overlap = int(np.sum(tgt_union & self.sc_marker))
            print(
                f"[TEST-ZONES] cells(target_union)={int(np.sum(tgt_union))} "
                f"cells(sc)={int(np.sum(self.sc_marker))} overlap={overlap}",
                flush=True
            )

        # Multipliers are now time-dependent (one per time step)
        # Will be initialized properly when sc_start_time/sc_end_time are known
        self.mu_lower_time = []
        self.mu_upper_time = []

        # -------------------------
        # Forms
        # -------------------------
        v = TestFunction(V)
        T_trial = TrialFunction(V)
        dt_c = Constant(domain, PETSc.ScalarType(dt))

        self.a_state = ((self.rho_c / dt_c) * T_trial * v * dx + self.k_therm * inner(ufl_grad(T_trial), ufl_grad(v)) * dx)
        self.a_adjoint = self.a_state
        # Pre-compile forms (evita JIT nel loop temporale)
        self.a_state_compiled = form(self.a_state)
        self.a_adjoint_compiled = form(self.a_adjoint)
        self.ksp = PETSc.KSP().create(domain.comm)
        self.ksp.setType("gmres")
        self.ksp.getPC().setType("hypre")
        self.ksp.getPC().setHYPREType("boomeramg")
        self.ksp.setTolerances(rtol=1e-10)

        # -------------------------------------------------
        # Riesz map for Neumann gradient: M g = b
        # (mass matrix in V)
        # -------------------------------------------------
        u_mass = TrialFunction(V)
        v_mass = TestFunction(V)
        self.M_mass = assemble_matrix(form(u_mass * v_mass * dx))
        self.M_mass.assemble()

        self.ksp_mass = PETSc.KSP().create(domain.comm)
        self.ksp_mass.setType("cg")
        self.ksp_mass.getPC().setType("jacobi")
        self.ksp_mass.setTolerances(rtol=1e-12)
        self.ksp_mass.setOperators(self.M_mass)

        self.Y_all = []
        self.P_all = []

        # Pre-allocate gradient containers (evita ri-allocazione in compute_gradient)
        self.grad_q_neumann_time = None
        self.grad_u_distributed_time = None
        self.grad_u_dirichlet_time = None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def solve_forward(self, q_neumann_funcs_time, u_distributed_funcs_time, u_dirichlet_funcs_time, T_cure):

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
                    print(f"[BC-DEBUG] uD_time.x.array[dofs_i] = {uD_time.x.array[dofs_i][:5]}")
                if step == 0 and self.domain.comm.rank == 0 and self.n_ctrl_dirichlet > 0:
                    dofs0 = self.dirichlet_dofs[0]
                    print("[BC-DEBUG new] uD_bc[0] on ΓD min/max =",
                      float(uD_bc[0].x.array[dofs0].min()),
                      float(uD_bc[0].x.array[dofs0].max()),
                      flush=True)

            # if step == 0 and self.domain.comm.rank == 0:
            #     print(f"[FWD-BC-DEBUG] bc_f on ΓD = {bc_f.x.array[dofs_i]}", flush=True)
            #     print(f"[FWD-BC-DEBUG] T after solve, T on ΓD = {T.x.array[dofs_i]}", flush=True)
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
                    print(f"[FWD-DEBUG] step={step}, q_func min/max (global) = "
                        f"{self.q_neumann_funcs[i].x.array.min():.6e}, "
                        f"{self.q_neumann_funcs[i].x.array.max():.6e}", flush=True)
                    print(f"[FWD-DEBUG] step={step}, q_func min/max (on Γ) = "
                        f"{self.q_neumann_funcs[i].x.array[dofs_i].min():.6e}, "
                        f"{self.q_neumann_funcs[i].x.array[dofs_i].max():.6e}", flush=True)

            # (optional debug: only first step)
            if step == 0 and self.domain.comm.rank == 0 and self.n_ctrl_neumann > 0:
                dofs0 = self.neumann_dofs[0]
                q0 = self.q_neumann_funcs[0].x.array
                print(
                    "[CHECK-NEUMANN] step=0 q_on_Gamma min/max =",
                    float(q0[dofs0].min()), float(q0[dofs0].max()),
                    " | q_global min/max =",
                    float(q0.min()), float(q0.max()),
                    flush=True
                )
                mid0 = self.neumann_marker_ids[0]
                q_int = assemble_scalar(form(self.q_neumann_funcs[0] * self.ds_neumann(mid0)))
                print("[CHECK-NEUMANN] int_Gamma q ds =", float(q_int), flush=True)

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
                print(f"[SOLVE-DEBUG] T after solve on ΓD = {T.x.array[dofs_check][:5]}")

            if step == self.num_steps - 1 and self.domain.comm.rank == 0:
                print(f"[DEBUG-T] global min/max: {float(T.x.array.min()):.12e} {float(T.x.array.max()):.12e}", flush=True)
                if self.n_ctrl_dirichlet > 0:
                    dofsD = self.dirichlet_dofs[0]
                    print(f"[DEBUG-T] on ΓD min/max: {float(T.x.array[dofsD].min()):.12e} {float(T.x.array[dofsD].max()):.12e}", flush=True)

            T.x.petsc_vec.copy(T_old.x.petsc_vec)
            T_old.x.scatter_forward()
            Y_all.append(T.copy())

        self.Y_all = Y_all
        return Y_all

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def solve_adjoint(self, Y_all, T_cure):
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

            # if self.domain.comm.rank == 0:
            #     print(f"[ADJ-DEBUG] m={m}, weight={weight}", flush=True)

            # Tracking forcing (UFL, in V) — costruiscilo SOLO se alpha_track != 0
            tracking_form = 0
            if abs(self.alpha_track) > 1e-30:
                for chi_t in self.chi_targets:
                    tracking_form += self.alpha_track * weight * self.dt * (y_current - T_cure) * chi_t * v * dx

            # if self.domain.comm.rank == 0:
            #     print(f"[ADJ-DEBUG pre state constr] m={m}, weight={weight}, y_mean={np.mean(y_current.x.array):.2f}", flush=True)

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
                # print(f"[TEST-ADJ-SC] m={m}: int_SC={float(sc_int):.6e} | int_targets={[float(x) for x in tgt_ints]}", flush=True)

            tracking_form += (-weight) * muL_m * self.chi_sc * v * dx
            tracking_form += (+weight) * muU_m * self.chi_sc * v * dx

            # if self.domain.comm.rank == 0:
            #     print(f"[ADJ-DEBUG post state constr] m={m}, weight={weight}, y_mean={np.mean(y_current.x.array):.2f}", flush=True)

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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_gradient_forms(self):
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
            flux_form = (-self.k_therm * inner(ufl_grad(self._p_placeholder), n)) * v * ds_i(mid)
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
                reg_form = self.alpha_u * self.dt * self._q_placeholder * v * self.ds_neumann(mid)
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_gradient(self, u_controls, Y_all, P_all, u_distributed_funcs_time, q_neumann_funcs_time, u_dirichlet_funcs_time):
        """
        Compute gradient via adjoint.
        - Neumann control è P2 in spazio e time-dependent: self.grad_q_neumann_time[m][i]
        - Distributed control è P2 in spazio e time-dependent: self.grad_u_distributed_time[m][i]
        (NOTA: non entra in u_controls, quindi NON incrementare idx per lui)
        """

        grad = np.zeros((self.n_ctrl_spatial, self.num_steps))
        # Inizializza forms se non già fatto
        if not hasattr(self, '_gradient_forms_initialized'):
            self._init_gradient_forms()

        n = FacetNormal(self.domain)

        # ============================================================
        # Neumann gradient containers: [m][i] Function(V)
        # ============================================================
        if self.grad_q_neumann_time is None or len(self.grad_q_neumann_time) != self.num_steps:
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
        if self.grad_u_distributed_time is None or len(self.grad_u_distributed_time) != self.num_steps:
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
        if self.grad_u_dirichlet_time is None or len(self.grad_u_dirichlet_time) != self.num_steps:
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
                    b_reg_L2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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
                        b_reg_H1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                        b_total.axpy(1.0, b_reg_H1)

                    if self.domain.comm.rank == 0 and m == 0:
                        h1_norm = b_reg_H1.norm()
                        l2_norm = b_reg_L2.norm() if self.alpha_u > 1e-16 else 0.0
                        print(f"[H1-DEBUG] m={m}, i={i}: ||b_L2||={l2_norm:.3e}, ||b_H1||={h1_norm:.3e}, ratio={h1_norm/max(l2_norm,1e-16):.3e}", flush=True)

                # Solve Riesz map: M_dirichlet * gD = b_total
                gD = self.grad_u_dirichlet_time[m][i]
                gD.x.petsc_vec.set(0.0)
                # Ensure RHS is supported only on ΓD dofs
                imap = self.V.dofmap.index_map
                nloc = imap.size_local + imap.num_ghosts
                all_dofs = np.arange(nloc, dtype=np.int32)
                off_dofs = np.setdiff1d(all_dofs, dofs_i.astype(np.int32), assume_unique=False).astype(np.int32)

                # Zero RHS outside ΓD
                b_total.setValues(off_dofs, np.zeros(len(off_dofs), dtype=PETSc.ScalarType))
                b_total.assemble()

                self.ksp_dirichlet[i].solve(b_total, gD.x.petsc_vec)
                gD.x.scatter_forward()

                if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                    reg_type = "L2" if self.dirichlet_spatial_reg == "L2" else f"H1(α={self.alpha_u:.1e},β={self.beta_u:.1e})"
                    print(f"[GRAD-DIRICHLET-{reg_type}][m={m}] on ΓD min/max:",
                        float(gD.x.array[dofs_i].min()), float(gD.x.array[dofs_i].max()),
                        flush=True)
                if self.domain.comm.rank == 0 and m == 0:
                    expected_raw = self.alpha_u * self.dt * uD_current.x.array[dofs_i].mean()
                    actual_riesz = gD.x.array[dofs_i].mean()
                    b_norm = float(b_total.norm())
                    print(f"[DIRICHLET-GRAD-DEBUG] m={m}")
                    print(f"  uD mean on ΓD = {uD_current.x.array[dofs_i].mean():.6e}")
                    print(f"  Expected raw gradient (α*dt*uD) = {expected_raw:.6e}")
                    print(f"  Actual Riesz gradient mean = {actual_riesz:.6e}")
                    print(f"  ||b_total|| = {b_norm:.6e}")

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
                    b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    b_total.axpy(1.0, b_reg)

                # Solve Riesz map: M_neumann * gq = b_total
                gq = self.grad_q_neumann_time[m][i]
                gq.x.petsc_vec.set(0.0)
                self.ksp_neumann[i].solve(b_total, gq.x.petsc_vec)
                gq.x.scatter_forward()

                if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                    print(f"[GRAD-NEUMANN][m={m}] on Γ min/max:", float(gq.x.array[dofs_i].min()), float(gq.x.array[dofs_i].max()), flush=True)

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
                    b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                    b_total.axpy(1.0, b_reg)

                # Solve Riesz map: M * gD = b_total
                gD = self.grad_u_distributed_time[m][i]
                gD.x.petsc_vec.set(0.0)
                self.ksp_distributed[i].solve(b_total, gD.x.petsc_vec)
                gD.x.scatter_forward()

                if self.domain.comm.rank == 0 and (m == 0 or m == self.num_steps - 1):
                    print(f"[GRAD-UD][m={m}] min/max:", float(gD.x.array.min()), float(gD.x.array.max()), flush=True)
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
                        u_prev = u_distributed_funcs_time[m-1][i]
                        u_next = u_distributed_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        chiV = self.chi_distributed_V[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (2*u_curr - u_prev - u_next) * chiV * v * dx))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        # Solve Riesz map
                        temp.x.petsc_vec.set(0.0)
                        self.ksp_distributed[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gD.x.array[:] += temp.x.array[:]
                        gD.x.scatter_forward()

                    # First step: u^0 - u^1
                    elif m == 0:
                        u_next = u_distributed_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        chiV = self.chi_distributed_V[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (u_curr - u_next) * chiV * v * dx))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_distributed[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gD.x.array[:] += temp.x.array[:]
                        gD.x.scatter_forward()

                    # Last step: u^M - u^{M-1}
                    else:  # m == self.num_steps - 1
                        u_prev = u_distributed_funcs_time[m-1][i]

                        v = TestFunction(self.V)
                        chiV = self.chi_distributed_V[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (u_curr - u_prev) * chiV * v * dx))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

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
                        q_prev = q_neumann_funcs_time[m-1][i]
                        q_next = q_neumann_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        mid = self.neumann_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (2*q_curr - q_prev - q_next) * v * self.ds_neumann(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_neumann[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gq.x.array[:] += temp.x.array[:]
                        gq.x.scatter_forward()

                    # First step
                    elif m == 0:
                        q_next = q_neumann_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        mid = self.neumann_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (q_curr - q_next) * v * self.ds_neumann(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_neumann[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gq.x.array[:] += temp.x.array[:]
                        gq.x.scatter_forward()

                    # Last step
                    else:
                        q_prev = q_neumann_funcs_time[m-1][i]

                        v = TestFunction(self.V)
                        mid = self.neumann_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (q_curr - q_prev) * v * self.ds_neumann(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

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
                        uD_prev = u_dirichlet_funcs_time[m-1][i]
                        uD_next = u_dirichlet_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        ds_i = self.dirichlet_measures[i]
                        mid = self.dirichlet_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (2*uD_curr - uD_prev - uD_next) * v * ds_i(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gD.x.array[:] += temp.x.array[:]
                        gD.x.scatter_forward()

                    # First step
                    elif m == 0:
                        uD_next = u_dirichlet_funcs_time[m+1][i]

                        v = TestFunction(self.V)
                        ds_i = self.dirichlet_measures[i]
                        mid = self.dirichlet_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (uD_curr - uD_next) * v * ds_i(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gD.x.array[:] += temp.x.array[:]
                        gD.x.scatter_forward()

                    # Last step
                    else:
                        uD_prev = u_dirichlet_funcs_time[m-1][i]

                        v = TestFunction(self.V)
                        ds_i = self.dirichlet_measures[i]
                        mid = self.dirichlet_marker_ids[i]
                        b_h1t = assemble_vector(form((self.gamma_u / self.dt) * (uD_curr - uD_prev) * v * ds_i(mid)))
                        b_h1t.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

                        temp.x.petsc_vec.set(0.0)
                        self.ksp_dirichlet[i].solve(b_h1t, temp.x.petsc_vec)
                        temp.x.scatter_forward()

                        gD.x.array[:] += temp.x.array[:]
                        gD.x.scatter_forward()

        return grad
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def tgrad(self, u):
        """Tangential gradient on the boundary: (I - n⊗n) ∇u."""
        n = ufl.FacetNormal(self.domain)
        I = ufl.Identity(self.domain.geometry.dim)
        P = I - ufl.outer(n, n)
        return P * ufl.grad(u)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update_multiplier_mu(self, Y_all, sc_type, sc_lower, sc_upper, beta, sc_start_step, sc_end_step):
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
        chi_sc_cell.x.array[:] = self.sc_marker.astype(PETSc.ScalarType)
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
                    Tmean_sc = float(np.mean(T_cell.x.array[active]))
                else:
                    Tmin_sc = Tmax_sc = Tmean_sc = float("nan")

                muL_max = float(np.max(muL_new)) if muL_new.size else 0.0
                muU_max = float(np.max(muU_new)) if muU_new.size else 0.0
                t_m = m * self.dt
                print(
                    f"[SC-PATH] t={t_m:.1f}s (m={m}): T∈[{Tmin_sc:.2f}, {Tmax_sc:.2f}]°C, "
                    f"violL={vL:.2e}, violU={vU:.2e}, μL_max={muL_max:.2e}, μU_max={muU_max:.2e}",
                    flush=True
                )

        # Summary
        if self.domain.comm.rank == 0:
            print(
                f"[SC-PATH-SUMMARY] Window steps=[{sc_start_step},{sc_end_step}], "
                f"max_violation={feas_inf_max:.2e}, max_Δμ={delta_mu_max:.2e}",
                flush=True
            )
            Vc = functionspace(self.domain, ("DG", 0))
            mask = self.sc_marker.astype(bool)

            max_out_L = 0.0
            max_out_U = 0.0
            max_in_L  = 0.0
            max_in_U  = 0.0

            for m in range(sc_start_step, sc_end_step + 1):
                muL = Function(Vc); muL.interpolate(self.mu_lower_time[m])
                muU = Function(Vc); muU.interpolate(self.mu_upper_time[m])

                aL = muL.x.array
                aU = muU.x.array

                if aL.size:
                    max_in_L  = max(max_in_L,  float(np.max(aL[mask])) if np.any(mask) else 0.0)
                    max_in_U  = max(max_in_U,  float(np.max(aU[mask])) if np.any(mask) else 0.0)
                    max_out_L = max(max_out_L, float(np.max(aL[~mask])) if np.any(~mask) else 0.0)
                    max_out_U = max(max_out_U, float(np.max(aU[~mask])) if np.any(~mask) else 0.0)

            print(
                f"[TEST-MU] max_in:  muL={max_in_L:.3e} muU={max_in_U:.3e} | "
                f"max_out: muL={max_out_L:.3e} muU={max_out_U:.3e}",
                flush=True
            )

        return delta_mu_max, feas_inf_max
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_cost_forms(self, T_cure):
        """Inizializza form pre-compilate per compute_cost"""
        dx = ufl.Measure("dx", domain=self.domain)
        n = FacetNormal(self.domain)

        # Placeholder per temperatura (verrà aggiornato)
        self._T_placeholder = Function(self.V)

        # Tracking forms (una per ogni target zone)
        self._tracking_forms = []
        for chi_t in self.chi_targets:
            ufl_form = 0.5 * self.alpha_track * (self._T_placeholder - T_cure)**2 * chi_t * dx
            self._tracking_forms.append(form(ufl_form))

        self._cost_forms_initialized = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_control_cost_forms(self):
        """Precompila le forms per la regolarizzazione dei controlli (partiamo dal distributed L2)."""
        dx = ufl.Measure("dx", domain=self.domain)

        # Placeholder distributed (uno per controllo)
        self._u_dist_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        for f in self._u_dist_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        # Forms precompilate: ∫ u^2 * chi dx
        self._dist_L2_forms = []
        for j in range(self.n_ctrl_distributed):
            chiV = self.chi_distributed_V[j]
            self._dist_L2_forms.append(form((self._u_dist_L2_ph[j]**2) * chiV * dx))

        self._control_cost_forms_initialized = True

        # Placeholders per H1 temporale (u1 - u0)
        self._u_dist_H1_ph0 = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        self._u_dist_H1_ph1 = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        for f0, f1 in zip(self._u_dist_H1_ph0, self._u_dist_H1_ph1):
            f0.x.array[:] = 0.0; f0.x.scatter_forward()
            f1.x.array[:] = 0.0; f1.x.scatter_forward()

        self._dist_H1t_forms = []
        for j in range(self.n_ctrl_distributed):
            chiV = self.chi_distributed_V[j]
            self._dist_H1t_forms.append(form(((self._u_dist_H1_ph1[j] - self._u_dist_H1_ph0[j])**2) * chiV * dx))

        # --- Neumann L2 placeholders/forms
        self._q_neu_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_neumann)]
        for f in self._q_neu_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        self._neu_L2_forms = []
        for j in range(self.n_ctrl_neumann):
            mid = self.neumann_marker_ids[j]
            self._neu_L2_forms.append(form((self._q_neu_L2_ph[j]**2) * self.ds_neumann(mid)))

        # --- Dirichlet L2 placeholders/forms
        self._u_dir_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        for f in self._u_dir_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        self._dir_L2_forms = []
        for j in range(self.n_ctrl_dirichlet):
            ds_j = self.dirichlet_measures[j]
            mid  = self.dirichlet_marker_ids[j]
            self._dir_L2_forms.append(form((self._u_dir_L2_ph[j]**2) * ds_j(mid)))

        # --- Dirichlet H1 temporal placeholders/forms
        self._u_dir_H1_ph0 = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        self._u_dir_H1_ph1 = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        for f0, f1 in zip(self._u_dir_H1_ph0, self._u_dir_H1_ph1):
            f0.x.array[:] = 0.0; f0.x.scatter_forward()
            f1.x.array[:] = 0.0; f1.x.scatter_forward()

        self._dir_H1t_forms = []
        for j in range(self.n_ctrl_dirichlet):
            ds_j = self.dirichlet_measures[j]
            mid  = self.dirichlet_marker_ids[j]
            self._dir_H1t_forms.append(form(((self._u_dir_H1_ph1[j] - self._u_dir_H1_ph0[j])**2) * ds_j(mid)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_cost(self, u_distributed_funcs_time, u_neumann_funcs_time, u_dirichlet_funcs_time, Y_all, T_cure):
        """Compute cost functional (tracking + L2 space + H1 time)."""

        if not hasattr(self, '_cost_forms_initialized'):
            self._init_cost_forms(T_cure)
        if not hasattr(self, '_control_cost_forms_initialized'):
            self._init_control_cost_forms()
        dx = ufl.Measure("dx", domain=self.domain)

        # ============================================================
        # Tracking (trapezoidal in time)
        # ============================================================
        J_track = 0.0
        J_track_steps = np.zeros(self.num_steps + 1, dtype=float)

        for step in range(self.num_steps + 1):
            T_current = Y_all[step]
            weight = 0.5 if (step == 0 or step == self.num_steps) else 1.0
            val_step = 0.0
            if abs(self.alpha_track) > 1e-30:
                # Aggiorna placeholder
                self._T_placeholder.x.array[:] = T_current.x.array[:]
                self._T_placeholder.x.scatter_forward()

                for compiled_form in self._tracking_forms:
                    val_step += assemble_scalar(compiled_form)

            val_step *= (weight * self.dt)
            J_track += val_step
            J_track_steps[step] = val_step

        self.last_J_track_steps = J_track_steps.copy()

        if self.domain.comm.rank == 0:
            print("[J-DEBUG] J_track_sum_check:",
                f"{np.sum(J_track_steps):.6e}  vs  J_track={J_track:.6e}", flush=True)

        # ============================================================
        # L2 spatial regularization
        # ============================================================
        J_reg_L2 = 0.0

        # ---- Distributed (precompiled L2)
        if self.n_ctrl_distributed > 0 and self.alpha_u > 1e-16:
            coef_L2 = 0.5 * self.alpha_u * self.dt
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_distributed):
                    # update placeholder
                    u_distributed_funcs_time[m][j].x.petsc_vec.copy(self._u_dist_L2_ph[j].x.petsc_vec)
                    self._u_dist_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._dist_L2_forms[j])

        # ---- Dirichlet (precompiled L2)
        if self.n_ctrl_dirichlet > 0 and self.alpha_u > 1e-16:
            coef_L2 = 0.5 * self.alpha_u * self.dt
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_dirichlet):
                    u_dirichlet_funcs_time[m][j].x.petsc_vec.copy(self._u_dir_L2_ph[j].x.petsc_vec)
                    self._u_dir_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._dir_L2_forms[j])

        # ---- Neumann (precompiled L2)
        if self.n_ctrl_neumann > 0 and self.alpha_u > 1e-16:
            coef_L2 = 0.5 * self.alpha_u * self.dt
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_neumann):
                    u_neumann_funcs_time[m][j].x.petsc_vec.copy(self._q_neu_L2_ph[j].x.petsc_vec)
                    self._q_neu_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._neu_L2_forms[j])

        # ============================================================
        # H1 spatial regularization on Dirichlet boundary (tangential)
        # J = (beta_u/2) * dt * sum_m ∫_{ΓD} |∇_Γ uD^m|^2 ds
        # ============================================================
        if (self.n_ctrl_dirichlet > 0 and self.dirichlet_spatial_reg == "H1"
                and self.beta_u > 1e-16):
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_dirichlet):
                    uD = u_dirichlet_funcs_time[m][j]
                    ds_j = self.dirichlet_measures[j]
                    mid  = self.dirichlet_marker_ids[j]
                    tg = self.tgrad(uD)  # same definition as in gradient forms
                    J_reg_L2 += 0.5 * self.beta_u * self.dt * assemble_scalar(form(inner(tg, tg) * ds_j(mid)))

        # ============================================================
        # H1 temporal regularization  ⭐ CORRETTA ⭐
        # J = γ/(2 dt) Σ ||u^{m+1} − u^m||²
        # ============================================================
        J_reg_H1 = 0.0
        if self.gamma_u > 1e-16:
            coef = 0.5 * self.gamma_u / self.dt

            # ---- Distributed (precompiled H1 in time)
            for j in range(self.n_ctrl_distributed):
                for m in range(self.num_steps - 1):
                    u_distributed_funcs_time[m][j].x.petsc_vec.copy(self._u_dist_H1_ph0[j].x.petsc_vec)
                    self._u_dist_H1_ph0[j].x.scatter_forward()
                    u_distributed_funcs_time[m+1][j].x.petsc_vec.copy(self._u_dist_H1_ph1[j].x.petsc_vec)
                    self._u_dist_H1_ph1[j].x.scatter_forward()
                    J_reg_H1 += coef * assemble_scalar(self._dist_H1t_forms[j])


            # ---- Neumann
            for j in range(self.n_ctrl_neumann):
                mid = self.neumann_marker_ids[j]
                for m in range(self.num_steps - 1):
                    q0 = u_neumann_funcs_time[m][j]
                    q1 = u_neumann_funcs_time[m+1][j]
                    J_reg_H1 += coef * assemble_scalar(
                        form(((q1 - q0)**2) * self.ds_neumann(mid))
                    )

            # ---- Dirichlet (precompiled H1 in time)
            for j in range(self.n_ctrl_dirichlet):
                for m in range(self.num_steps - 1):
                    u_dirichlet_funcs_time[m][j].x.petsc_vec.copy(self._u_dir_H1_ph0[j].x.petsc_vec)
                    self._u_dir_H1_ph0[j].x.scatter_forward()
                    u_dirichlet_funcs_time[m+1][j].x.petsc_vec.copy(self._u_dir_H1_ph1[j].x.petsc_vec)
                    self._u_dir_H1_ph1[j].x.scatter_forward()
                    J_reg_H1 += coef * assemble_scalar(self._dir_H1t_forms[j])

        # ============================================================
        J_total = J_track + J_reg_L2 + J_reg_H1
        if self.domain.comm.rank == 0:
            print(f"[COST-DEBUG] J_total={J_total:.12e} J_track={J_track:.12e} J_regL2={J_reg_L2:.12e} J_regH1={J_reg_H1:.12e}", flush=True)

        return J_total, J_track, J_reg_L2, J_reg_H1
