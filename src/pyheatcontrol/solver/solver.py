# solver.py

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from ufl import dx, grad as ufl_grad, inner, TestFunction, TrialFunction
from dolfinx.fem import form, Function, Constant, functionspace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix
from .forward import solve_forward_impl
from .adjoint import solve_adjoint_impl
from .gradient import (
    _init_gradient_forms_impl,
    compute_gradient_impl,
    tgrad_impl,
    update_multiplier_mu_impl,
)
from pyheatcontrol.mesh_utils import (
    create_boundary_condition_function,
    create_boundary_facet_tags,
    mark_cells_in_boxes,
)
from pyheatcontrol.logging_config import logger


class TimeDepHeatSolver:
    """
    Heat solver con TIME-DEPENDENT controls
    Supporta: Dirichlet boundary, Neumann boundary, Distributed
    u_controls shape: (n_ctrl_spatial, Nt)
    """

    def __init__(
        self,
        domain,
        V,
        dt,
        num_steps,
        k_val,
        rho,
        c,
        T_ambient,
        control_boundary_dirichlet,
        control_boundary_neumann,
        control_distributed_boxes,
        target_boxes,
        constraint_boxes,
        L,
        alpha_track,
        alpha_u,
        gamma_u,
        beta_u,
        dirichlet_spatial_reg,
    ):

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
        self.n_ctrl_boxes = len(control_distributed_boxes)

        self.n_ctrl_distributed = self.n_ctrl_boxes

        self.n_ctrl_spatial = 0

        self.control_boundary_dirichlet = control_boundary_dirichlet
        self.control_boundary_neumann = control_boundary_neumann

        self._T_cell = Function(self.Vc)  # DG0

        if self.domain.comm.rank == 0:
            logger.debug(f"constraint_boxes received = {constraint_boxes}")

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

        # accumulate facets/values for a single combined meshtags (for visualization)
        all_facets_D = []
        all_values_D = []

        for i, seg in enumerate(control_boundary_dirichlet):
            # DOFs + BC function for applying distributed BC on ΓD
            dofs, bc_f = create_boundary_condition_function(domain, V, [seg], L)
            self.bc_dofs_list_dirichlet.append(dofs)
            self.bc_funcs_dirichlet.append(bc_f)

            # facet marker for this segment
            marker_id = i + 1
            facet_tags_i, _ = create_boundary_facet_tags(domain, [seg], L, marker_id)

            # ds measure for this segment
            self.dirichlet_marker_ids.append(marker_id)
            self.dirichlet_measures.append(
                ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_i)
            )

            # accumulate facets/values for combined meshtags
            all_facets_D.append(facet_tags_i.indices.astype(np.int32))
            all_values_D.append(facet_tags_i.values.astype(np.int32))

        # ---- build ONE combined meshtags for all Dirichlet segments (for visualization) ----
        self.dirichlet_facet_tags = None
        if len(all_facets_D) > 0:
            facets = np.hstack(all_facets_D).astype(np.int32)
            values = np.hstack(all_values_D).astype(np.int32)

            # sort by facet id
            order = np.argsort(facets, kind="mergesort")
            facets = facets[order]
            values = values[order]

            # deduplicate: keep the last marker per facet
            # (così se due segmenti si sovrappongono, uno “vince” e non crasha)
            unique_facets, first_idx = np.unique(facets, return_index=True)
            # trick: per tenere lanciato l’ultimo valore, usiamo return_index sul reversed
            unique_facets_rev, first_idx_rev = np.unique(
                facets[::-1], return_index=True
            )
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

            # sort by facet id
            order = np.argsort(facets, kind="mergesort")
            facets = facets[order]
            values = values[order]

            # deduplicate: keep the last marker per facet
            unique_facets_rev, first_idx_rev = np.unique(
                facets[::-1], return_index=True
            )
            last_idx = (len(facets) - 1) - first_idx_rev
            last_idx_sorted = np.sort(last_idx)

            facets_u = facets[last_idx_sorted]
            values_u = values[last_idx_sorted]

            self.neumann_facet_tags = meshtags(domain, fdim, facets_u, values_u)
            self.ds_neumann = ufl.Measure(
                "ds", domain=domain, subdomain_data=self.neumann_facet_tags
            )

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
            M_i = assemble_matrix(
                form(
                    u_mass * v_mass * self.ds_neumann(mid) + eps * u_mass * v_mass * dx
                )
            )
            M_i.assemble()
            # Make invertible outside ΓN by identity rows/cols (same idea as Dirichlet)
            dofs_i = self.neumann_dofs[i]  # dofs on this Neumann segment (ΓN_i)

            imap = self.V.dofmap.index_map
            nloc = imap.size_local  # owned DOFs only
            all_dofs = np.arange(nloc, dtype=np.int32)
            dofs_i_owned = dofs_i[dofs_i < nloc].astype(np.int32)
            off_dofs = np.setdiff1d(
                all_dofs, dofs_i_owned, assume_unique=False
            ).astype(np.int32)

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

        # Debug: matrix norms (norm() is MPI collective)
        
        if self.n_ctrl_neumann > 0:
            neumann_norms = [M_i.norm() for M_i in self.M_neumann]
            if self.domain.comm.rank == 0:
                logger.debug(f"Number of Neumann zones: {len(self.M_neumann)}")
                for i, M_norm in enumerate(neumann_norms):
                    logger.debug(f"  Zone {i}: Matrix norm = {M_norm:.6e}")
        # -------------------------------------------------
        # Dirichlet control functions (P2) - DISTRIBUTED ON BOUNDARY
        # -------------------------------------------------
        self.u_dirichlet_funcs = []
        for _ in range(self.n_ctrl_dirichlet):
            uD = Function(self.V)
            uD.x.array[:] = 0.0
            uD.x.scatter_forward()
            self.u_dirichlet_funcs.append(uD)

        # Dirichlet DOFs restricted to ΓD (reuse from bc_dofs_list_dirichlet)
        self.dirichlet_dofs = self.bc_dofs_list_dirichlet

        if self.n_ctrl_dirichlet > 0:
            comm = self.domain.comm
            for i, dofs in enumerate(self.dirichlet_dofs):
                nloc = int(len(dofs))
                nglob = comm.allreduce(nloc, op=MPI.SUM)
                if comm.rank == 0:
                    logger.debug(f"Dirichlet control zone {i+1}: local={nloc}, global={nglob} DOFs on boundary")

        # -------------------------------------------------
        # Mass matrices for Dirichlet control gradients (restricted to ΓD)
        # -------------------------------------------------
        self.M_dirichlet = []
        self.ksp_dirichlet = []

        if self.n_ctrl_dirichlet > 0:
            u_trial = TrialFunction(V)
            v_test = TestFunction(V)
            n = ufl.FacetNormal(domain)
            Id = ufl.Identity(domain.geometry.dim)
            P = Id - ufl.outer(n, n)  # projector on tangent space

            for i in range(self.n_ctrl_dirichlet):
                ds_i = self.dirichlet_measures[i]
                mid = self.dirichlet_marker_ids[i]

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
                nloc = imap.size_local  # owned DOFs only
                all_dofs = np.arange(nloc, dtype=np.int32)

                dofs_i = dofs_i.astype(np.int32)
                dofs_i_owned = dofs_i[dofs_i < nloc]  # filter out ghost DOFs

                off_dofs = np.setdiff1d(
                    all_dofs, dofs_i_owned, assume_unique=False
                ).astype(np.int32)
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
            n_local = V_dg0.dofmap.index_map.size_local
            for marker in self.distributed_markers:
                chi_dg0 = Function(V_dg0)
                chi_dg0.x.array[:] = 0.0
                chi_dg0.x.array[:n_local][marker] = 1.0
                chi_dg0.x.scatter_forward()
                chi = Function(self.V)
                chi.interpolate(chi_dg0)
                self.chi_distributed_V.append(chi)

        # Compute DOF restriction for each distributed control zone
        self.distributed_dofs = []
        for marker in self.distributed_markers:
            dofs_in_cells = []
            for cell_idx in np.where(marker)[0]:
                cell_dofs = self.V.dofmap.cell_dofs(cell_idx)
                dofs_in_cells.extend(cell_dofs)
            dofs_unique = np.unique(np.array(dofs_in_cells, dtype=np.int32))
            self.distributed_dofs.append(dofs_unique)

            if self.domain.comm.rank == 0:
                logger.debug(
                    f"Distributed control zone {len(self.distributed_dofs)}: {len(dofs_unique)} DOFs"
                )

        # Build mass matrices (distributed_dofs is now populated)
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
                nloc = imap.size_local  # owned DOFs only
                all_dofs = np.arange(nloc, dtype=np.int32)
                dofs_i_owned = dofs_i[dofs_i < nloc].astype(np.int32)
                off_dofs = np.setdiff1d(
                    all_dofs, dofs_i_owned, assume_unique=False
                ).astype(np.int32)

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
            V_dg0 = functionspace(domain, ("DG", 0))
            chi_dg0 = Function(V_dg0)
            n_local = V_dg0.dofmap.index_map.size_local
            chi_dg0.x.array[:n_local] = marker.astype(PETSc.ScalarType)
            chi_dg0.x.scatter_forward()
            chi = Function(self.V0)
            chi.interpolate(chi_dg0)
            self.chi_targets.append(chi)

        # -------------------------
        # State constraints (SEPARATE from targets)
        # -------------------------
        self.constraint_markers = []
        for box in constraint_boxes:
            self.constraint_markers.append(mark_cells_in_boxes(domain, [box], L))

        # union of all constraint boxes
        if len(self.constraint_markers) == 0:
            self.sc_marker = np.zeros(
                functionspace(domain, ("DG", 0)).dofmap.index_map.size_local, dtype=bool
            )
        else:
            self.sc_marker = np.zeros_like(self.constraint_markers[0])

        for m in self.constraint_markers:
            self.sc_marker |= m

        self.chi_sc_cell = Function(self.Vc)
        n_local = self.Vc.dofmap.index_map.size_local
        self.chi_sc_cell.x.array[:n_local] = self.sc_marker.astype(PETSc.ScalarType)
        self.chi_sc_cell.x.scatter_forward()
        chi_sc_dg0 = Function(functionspace(domain, ("DG", 0)))
        chi_sc_dg0.x.array[:n_local] = self.sc_marker.astype(PETSc.ScalarType)
        chi_sc_dg0.x.scatter_forward()
        self.chi_sc = Function(self.V0)
        self.chi_sc.interpolate(chi_sc_dg0)

        # assemble_scalar returns LOCAL contribution -> must allreduce to get GLOBAL
        mt_loc = [assemble_scalar(form(chi * dx)) for chi in self.chi_targets]
        mt = [self.domain.comm.allreduce(v, op=MPI.SUM) for v in mt_loc]

        msc_loc = assemble_scalar(form(self.chi_sc * dx))
        msc = self.domain.comm.allreduce(msc_loc, op=MPI.SUM)

        if self.domain.comm.rank == 0:
            logger.debug(f"Constraint zone: {int(np.sum(self.sc_marker))} cells marked")
            logger.debug(f"meas(targets) = {mt}")
            logger.debug(f"meas(constraint) = {float(msc)}")

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

        self.a_state = (self.rho_c / dt_c) * T_trial * v * dx + self.k_therm * inner(
            ufl_grad(T_trial), ufl_grad(v)
        ) * dx
        self.a_adjoint = self.a_state
        # Pre-compile forms (avoid JIT in the time loop)
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

        # Pre-allocate gradient containers
        self.grad_q_neumann_time = None
        self.grad_u_distributed_time = None
        self.grad_u_dirichlet_time = None

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def solve_forward(
        self,
        q_neumann_funcs_time,
        u_distributed_funcs_time,
        u_dirichlet_funcs_time,
    ):
        return solve_forward_impl(
            self,
            q_neumann_funcs_time,
            u_distributed_funcs_time,
            u_dirichlet_funcs_time,
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def solve_adjoint(self, Y_all, T_ref):
        return solve_adjoint_impl(self, Y_all, T_ref)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_gradient_forms(self):
        return _init_gradient_forms_impl(self)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_gradient(
        self,
        u_controls,
        Y_all,
        P_all,
        u_distributed_funcs_time,
        q_neumann_funcs_time,
        u_dirichlet_funcs_time,
    ):
        return compute_gradient_impl(
            self,
            u_controls,
            Y_all,
            P_all,
            u_distributed_funcs_time,
            q_neumann_funcs_time,
            u_dirichlet_funcs_time,
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def tgrad(self, u):
        return tgrad_impl(self, u)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update_multiplier_mu(
        self, Y_all, sc_type, sc_lower, sc_upper, beta, sc_start_step, sc_end_step
    ):
        return update_multiplier_mu_impl(
            self, Y_all, sc_type, sc_lower, sc_upper, beta, sc_start_step, sc_end_step
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_cost_forms(self, T_ref):
        """Initialize precompiled forms for compute_cost."""
        dx = ufl.Measure("dx", domain=self.domain)

        # Temperature placeholder (updated in the loop)
        self._T_placeholder = Function(self.V)

        # Tracking forms (one per target zone)
        self._tracking_forms = []
        for chi_t in self.chi_targets:
            ufl_form = (
                0.5
                * self.alpha_track
                * (self._T_placeholder - T_ref) ** 2
                * chi_t
                * dx
            )
            self._tracking_forms.append(form(ufl_form))

        self._cost_forms_initialized = True

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _init_control_cost_forms(self):
        """Precompile forms for control regularization (L2 + H1 temporal)."""
        dx = ufl.Measure("dx", domain=self.domain)

        # Distributed placeholders (one per control zone)
        self._u_dist_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        for f in self._u_dist_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        # Precompiled forms: ∫ u² χ dx
        self._dist_L2_forms = []
        for j in range(self.n_ctrl_distributed):
            chiV = self.chi_distributed_V[j]
            self._dist_L2_forms.append(form((self._u_dist_L2_ph[j] ** 2) * chiV * dx))

        self._control_cost_forms_initialized = True

        # H1 temporal placeholders (u1 - u0)
        self._u_dist_H1_ph0 = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        self._u_dist_H1_ph1 = [Function(self.V) for _ in range(self.n_ctrl_distributed)]
        for f0, f1 in zip(self._u_dist_H1_ph0, self._u_dist_H1_ph1):
            f0.x.array[:] = 0.0
            f0.x.scatter_forward()
            f1.x.array[:] = 0.0
            f1.x.scatter_forward()

        self._dist_H1t_forms = []
        for j in range(self.n_ctrl_distributed):
            chiV = self.chi_distributed_V[j]
            self._dist_H1t_forms.append(
                form(
                    ((self._u_dist_H1_ph1[j] - self._u_dist_H1_ph0[j]) ** 2) * chiV * dx
                )
            )

        # --- Neumann L2 placeholders/forms
        self._q_neu_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_neumann)]
        for f in self._q_neu_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        self._neu_L2_forms = []
        for j in range(self.n_ctrl_neumann):
            mid = self.neumann_marker_ids[j]
            self._neu_L2_forms.append(
                form((self._q_neu_L2_ph[j] ** 2) * self.ds_neumann(mid))
            )

        # --- Dirichlet L2 placeholders/forms
        self._u_dir_L2_ph = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        for f in self._u_dir_L2_ph:
            f.x.array[:] = 0.0
            f.x.scatter_forward()

        self._dir_L2_forms = []
        for j in range(self.n_ctrl_dirichlet):
            ds_j = self.dirichlet_measures[j]
            mid = self.dirichlet_marker_ids[j]
            self._dir_L2_forms.append(form((self._u_dir_L2_ph[j] ** 2) * ds_j(mid)))

        # --- Dirichlet H1 temporal placeholders/forms
        self._u_dir_H1_ph0 = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        self._u_dir_H1_ph1 = [Function(self.V) for _ in range(self.n_ctrl_dirichlet)]
        for f0, f1 in zip(self._u_dir_H1_ph0, self._u_dir_H1_ph1):
            f0.x.array[:] = 0.0
            f0.x.scatter_forward()
            f1.x.array[:] = 0.0
            f1.x.scatter_forward()

        self._dir_H1t_forms = []
        for j in range(self.n_ctrl_dirichlet):
            ds_j = self.dirichlet_measures[j]
            mid = self.dirichlet_marker_ids[j]
            self._dir_H1t_forms.append(
                form(((self._u_dir_H1_ph1[j] - self._u_dir_H1_ph0[j]) ** 2) * ds_j(mid))
            )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_cost(
        self,
        u_distributed_funcs_time,
        u_neumann_funcs_time,
        u_dirichlet_funcs_time,
        Y_all,
        T_ref,
    ):
        """Compute cost functional J = J_track + J_reg_L2 + J_reg_H1."""

        if not hasattr(self, "_cost_forms_initialized"):
            self._init_cost_forms(T_ref)
        if not hasattr(self, "_control_cost_forms_initialized"):
            self._init_control_cost_forms()

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
                # Update placeholder
                self._T_placeholder.x.array[:] = T_current.x.array[:]
                self._T_placeholder.x.scatter_forward()

                for compiled_form in self._tracking_forms:
                    val_step += assemble_scalar(compiled_form)

            val_step *= weight * self.dt
            J_track += val_step
            J_track_steps[step] = val_step

        self.last_J_track_steps = J_track_steps.copy()

        if self.domain.comm.rank == 0:
            logger.debug(
                f"J_track_sum_check: {np.sum(J_track_steps):.6e}  vs  J_track={J_track:.6e}"
            )

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
                    u_distributed_funcs_time[m][j].x.petsc_vec.copy(
                        self._u_dist_L2_ph[j].x.petsc_vec
                    )
                    self._u_dist_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._dist_L2_forms[j])

        # ---- Dirichlet (precompiled L2)
        if self.n_ctrl_dirichlet > 0 and self.alpha_u > 1e-16:
            coef_L2 = 0.5 * self.alpha_u * self.dt
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_dirichlet):
                    u_dirichlet_funcs_time[m][j].x.petsc_vec.copy(
                        self._u_dir_L2_ph[j].x.petsc_vec
                    )
                    self._u_dir_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._dir_L2_forms[j])

        # ---- Neumann (precompiled L2)
        if self.n_ctrl_neumann > 0 and self.alpha_u > 1e-16:
            coef_L2 = 0.5 * self.alpha_u * self.dt
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_neumann):
                    u_neumann_funcs_time[m][j].x.petsc_vec.copy(
                        self._q_neu_L2_ph[j].x.petsc_vec
                    )
                    self._q_neu_L2_ph[j].x.scatter_forward()
                    J_reg_L2 += coef_L2 * assemble_scalar(self._neu_L2_forms[j])

        # ============================================================
        # H1 spatial regularization on Dirichlet boundary (tangential)
        # J = (beta_u/2) * dt * sum_m ∫_{ΓD} |∇_Γ uD^m|^2 ds
        # ============================================================
        if (
            self.n_ctrl_dirichlet > 0
            and self.dirichlet_spatial_reg == "H1"
            and self.beta_u > 1e-16
        ):
            for m in range(self.num_steps):
                for j in range(self.n_ctrl_dirichlet):
                    uD = u_dirichlet_funcs_time[m][j]
                    ds_j = self.dirichlet_measures[j]
                    mid = self.dirichlet_marker_ids[j]
                    tg = self.tgrad(uD)  # same definition as in gradient forms
                    J_reg_L2 += (
                        0.5
                        * self.beta_u
                        * self.dt
                        * assemble_scalar(form(inner(tg, tg) * ds_j(mid)))
                    )

        # ============================================================
        # H1 temporal regularization
        # J = γ/(2 dt) Σ ||u^{m+1} − u^m||²
        # ============================================================
        J_reg_H1 = 0.0
        if self.gamma_u > 1e-16:
            coef = 0.5 * self.gamma_u / self.dt

            # ---- Distributed (precompiled H1 in time)
            for j in range(self.n_ctrl_distributed):
                for m in range(self.num_steps - 1):
                    u_distributed_funcs_time[m][j].x.petsc_vec.copy(
                        self._u_dist_H1_ph0[j].x.petsc_vec
                    )
                    self._u_dist_H1_ph0[j].x.scatter_forward()
                    u_distributed_funcs_time[m + 1][j].x.petsc_vec.copy(
                        self._u_dist_H1_ph1[j].x.petsc_vec
                    )
                    self._u_dist_H1_ph1[j].x.scatter_forward()
                    J_reg_H1 += coef * assemble_scalar(self._dist_H1t_forms[j])

            # ---- Neumann
            for j in range(self.n_ctrl_neumann):
                mid = self.neumann_marker_ids[j]
                for m in range(self.num_steps - 1):
                    q0 = u_neumann_funcs_time[m][j]
                    q1 = u_neumann_funcs_time[m + 1][j]
                    J_reg_H1 += coef * assemble_scalar(
                        form(((q1 - q0) ** 2) * self.ds_neumann(mid))
                    )

            # ---- Dirichlet (precompiled H1 in time)
            for j in range(self.n_ctrl_dirichlet):
                for m in range(self.num_steps - 1):
                    u_dirichlet_funcs_time[m][j].x.petsc_vec.copy(
                        self._u_dir_H1_ph0[j].x.petsc_vec
                    )
                    self._u_dir_H1_ph0[j].x.scatter_forward()
                    u_dirichlet_funcs_time[m + 1][j].x.petsc_vec.copy(
                        self._u_dir_H1_ph1[j].x.petsc_vec
                    )
                    self._u_dir_H1_ph1[j].x.scatter_forward()
                    J_reg_H1 += coef * assemble_scalar(self._dir_H1t_forms[j])

        # ============================================================
        # MPI reduction: assemble_scalar returns local contributions only
        comm = self.domain.comm
        J_track = comm.allreduce(J_track, op=MPI.SUM)
        J_reg_L2 = comm.allreduce(J_reg_L2, op=MPI.SUM)
        J_reg_H1 = comm.allreduce(J_reg_H1, op=MPI.SUM)

        J_total = J_track + J_reg_L2 + J_reg_H1
        if self.domain.comm.rank == 0:
            logger.debug(
                f"J_total={J_total:.12e} J_track={J_track:.12e} J_regL2={J_reg_L2:.12e} J_regH1={J_reg_H1:.12e}"
            )

        return J_total, J_track, J_reg_L2, J_reg_H1
