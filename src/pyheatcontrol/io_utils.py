# io_utils.py
import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import compute_midpoints
from dolfinx.io import VTKFile
from pyheatcontrol.logging_config import logger


def _import_sanity_check():
    # minimal things we expect io_utils to have available
    _ = (np, MPI, mesh, Function, VTKFile)  # aggiungi/togli qui se serve


def save_visualization_output(
    solver,
    Y_all,
    P_all,
    u_distributed_funcs_time,
    u_neumann_funcs_time,
    u_dirichlet_funcs_time,
    sc_start_step,
    sc_end_step,
    args,
    num_steps,
    target_boxes,
    control_distributed_boxes,
):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    domain = solver.domain
    V = solver.V
    comm = domain.comm
    rank = comm.rank

    # --- Spazi output ---
    V_out = functionspace(domain, ("Lagrange", 1))  # Point Data (P1)
    V0_out = functionspace(domain, ("DG", 0))  # Cell Data (DG0 robusto)

    # --- Lista timesteps da salvare ---
    timesteps_to_save = list(range(0, num_steps, args.output_freq))
    if num_steps not in timesteps_to_save:
        timesteps_to_save.append(num_steps)  # salvo stato finale

    # --- Celle locali (NO ghost) + midpoints locali ---
    tdim = domain.topology.dim
    fdim = tdim - 1
    imap = domain.topology.index_map(tdim)
    num_cells_local = imap.size_local

    cell_ids_local = np.arange(num_cells_local, dtype=np.int32)
    cell_mid_local = compute_midpoints(
        domain, tdim, cell_ids_local
    )  # (num_cells_local, gdim)

    def mark_boxes_DG0(name, boxes):
        """Mark DG0 cell field =1 inside given boxes using cell midpoints (local-only)."""
        f = Function(V0_out, name=name)
        f.x.array[:] = 0.0
        a = f.x.array[:num_cells_local]
        for x0, x1, y0, y1 in boxes:
            inside = np.logical_and.reduce(
                [
                    cell_mid_local[:, 0] >= x0,
                    cell_mid_local[:, 0] <= x1,
                    cell_mid_local[:, 1] >= y0,
                    cell_mid_local[:, 1] <= y1,
                ]
            )
            a[inside] = 1.0
        f.x.scatter_forward()
        return f

    # --------- Helper: zona celle adiacenti a facets taggati ----------
    def mark_boundary_cells_DG0(name, facet_tags, marker_ids=None):
        """
        Crea un campo DG0=1 sulle celle che toccano facets marcati.
        facet_tags: dolfinx.mesh.MeshTags (fdim)
        marker_ids: None -> usa TUTTI i facets in facet_tags
                    int o lista -> usa facet_tags.find(id)
        """
        zone = Function(V0_out, name=name)
        zone.x.array[:] = 0.0
        a = zone.x.array[:num_cells_local]

        if facet_tags is None:
            zone.x.scatter_forward()
            return zone

        # facets selezionati
        if marker_ids is None:
            facets = np.array(facet_tags.indices, dtype=np.int32)
        else:
            if isinstance(marker_ids, (int, np.integer)):
                marker_ids = [int(marker_ids)]
            facets_list = []
            for mid in marker_ids:
                try:
                    facets_list.append(facet_tags.find(int(mid)))
                except Exception:
                    pass
            if len(facets_list) == 0:
                facets = np.array([], dtype=np.int32)
            else:
                facets = np.unique(np.concatenate(facets_list).astype(np.int32))

        if facets.size == 0:
            zone.x.scatter_forward()
            return zone

        # connettività facet -> cell
        domain.topology.create_connectivity(fdim, tdim)
        f2c = domain.topology.connectivity(fdim, tdim)

        cells = []
        for f in facets:
            cells.extend(list(f2c.links(int(f))))
        if len(cells) == 0:
            zone.x.scatter_forward()
            return zone

        cells = np.unique(np.array(cells, dtype=np.int32))

        # marca solo celle locali (no ghost)
        cells_local = cells[cells < num_cells_local]
        a[cells_local] = 1.0
        zone.x.scatter_forward()
        return zone

    # --- Convert box [0,1] -> coordinate fisiche della mesh ---
    X = domain.geometry.x
    xmin, ymin = float(X[:, 0].min()), float(X[:, 1].min())
    xmax, ymax = float(X[:, 0].max()), float(X[:, 1].max())

    def box01_to_phys(box):
        x0n, x1n, y0n, y1n = box
        x0n, x1n = min(x0n, x1n), max(x0n, x1n)
        y0n, y1n = min(y0n, y1n), max(y0n, y1n)
        x0 = xmin + x0n * (xmax - xmin)
        x1 = xmin + x1n * (xmax - xmin)
        y0 = ymin + y0n * (ymax - ymin)
        y1 = ymin + y1n * (ymax - ymin)
        return (x0, x1, y0, y1)

    target_boxes_phys = [box01_to_phys(b) for b in target_boxes]
    control_boxes_phys = [box01_to_phys(b) for b in control_distributed_boxes]

    # --- Zone volumetriche (Cell Data) ---
    target_zone = mark_boxes_DG0("omega_t", target_boxes_phys)
    control_zone_dist = mark_boxes_DG0("omega_c", control_boxes_phys)

    # Constraint zone
    constraint_zone = Function(V0_out, name="omega_sc")
    constraint_zone.x.array[:] = 0.0
    a = constraint_zone.x.array[:num_cells_local]
    if hasattr(solver, "sc_marker") and len(solver.sc_marker) >= num_cells_local:
        a[:] = np.array(solver.sc_marker[:num_cells_local], dtype=PETSc.ScalarType)
    else:
        if rank == 0:
            logger.warning("sc_marker missing/mismatch: omega_sc set to 0 (not target)")
        a[:] = 0.0
    constraint_zone.x.scatter_forward()

    # --- TRY GET TAGS (adatta questi nomi se nel tuo solver sono diversi) ---
    neumann_facet_tags = getattr(solver, "neumann_facet_tags", None)
    dirichlet_facet_tags = getattr(solver, "dirichlet_facet_tags", None)

    neumann_marker_ids = getattr(solver, "neumann_marker_ids", None)
    if neumann_marker_ids is None:
        neumann_marker_ids = getattr(solver, "neumann_marker_id", None)

    dirichlet_marker_ids = getattr(solver, "dirichlet_marker_ids", None)
    if dirichlet_marker_ids is None:
        dirichlet_marker_ids = getattr(solver, "dirichlet_marker_id", None)

    # --- Zone boundary control (Cell Data): fascia di celle adiacenti al bordo controllato ---
    omega_qn = mark_boundary_cells_DG0(
        "omega_qn", neumann_facet_tags, neumann_marker_ids
    )
    omega_qd = mark_boundary_cells_DG0(
        "omega_qd", dirichlet_facet_tags, dirichlet_marker_ids
    )

    # --- Debug utile ---
    if rank == 0:
        n_tgt = int(np.round(target_zone.x.array[:num_cells_local].sum()))
        n_ctrl = int(np.round(control_zone_dist.x.array[:num_cells_local].sum()))
        n_cons = int(np.round(constraint_zone.x.array[:num_cells_local].sum()))
        n_qn = int(np.round(omega_qn.x.array[:num_cells_local].sum()))
        n_qd = int(np.round(omega_qd.x.array[:num_cells_local].sum()))
        logger.debug(
            f"marked cells: target={n_tgt}, control_dist={n_ctrl}, constraint={n_cons}, omega_qn={n_qn}, omega_qd={n_qd}"
        )

    # --- Funzioni nodali (riusate) ---
    y_out = Function(V_out, name="T")
    p_out = Function(V_out, name="Ta")
    uD_out = Function(V_out, name="Q")
    qN_out = Function(V_out, name="q_n")
    uDir_out = Function(V_out, name="q_d")

    # --- Funzioni cella (riusate) ---
    muL_out = Function(V0_out, name="multiplier_lower")
    muU_out = Function(V0_out, name="multiplier_upper")
    vL_out = Function(V0_out, name="violation_lower")
    vU_out = Function(V0_out, name="violation_upper")
    T_cell = Function(V0_out, name="T_cell_tmp")

    tmp = Function(V)  # P2 riusata per sommare controlli

    # --- File VTK/PVD (timeline) ---
    pvd_path = os.path.join(output_dir, "solution.pvd")
    with VTKFile(comm, pvd_path, "w") as vtk:
        for m in timesteps_to_save:
            t = float(m * solver.dt)

            # --- Stato / Aggiunto (P2 -> P1) ---
            y_out.interpolate(Y_all[m])
            p_out.interpolate(P_all[m])
            y_out.x.scatter_forward()
            p_out.x.scatter_forward()

            # --- Controlli: hold-last sul tempo finale ---
            idx = min(m, num_steps - 1)

            # Distributed
            uD_out.x.array[:] = 0.0
            if getattr(solver, "n_ctrl_distributed", 0) > 0:
                tmp.x.array[:] = 0.0
                for i in range(solver.n_ctrl_distributed):
                    tmp.x.array[:] += u_distributed_funcs_time[idx][i].x.array[:]
                tmp.x.scatter_forward()
                uD_out.interpolate(tmp)
            uD_out.x.scatter_forward()

            # Neumann
            qN_out.x.array[:] = 0.0
            if getattr(solver, "n_ctrl_neumann", 0) > 0:
                tmp.x.array[:] = 0.0
                for i in range(solver.n_ctrl_neumann):
                    tmp.x.array[:] += u_neumann_funcs_time[idx][i].x.array[:]
                tmp.x.scatter_forward()
                qN_out.interpolate(tmp)
            qN_out.x.scatter_forward()

            # Dirichlet
            uDir_out.x.array[:] = 0.0
            if getattr(solver, "n_ctrl_dirichlet", 0) > 0:
                tmp.x.array[:] = 0.0
                for i in range(solver.n_ctrl_dirichlet):
                    tmp.x.array[:] += u_dirichlet_funcs_time[idx][i].x.array[:]
                tmp.x.scatter_forward()
                uDir_out.interpolate(tmp)
            uDir_out.x.scatter_forward()

            # --- Moltiplicatori (DG0) ---
            if hasattr(solver, "mu_lower_time") and m < len(solver.mu_lower_time):
                muL_out.interpolate(solver.mu_lower_time[m])
            else:
                muL_out.x.array[:] = 0.0
            muL_out.x.scatter_forward()

            if hasattr(solver, "mu_upper_time") and m < len(solver.mu_upper_time):
                muU_out.interpolate(solver.mu_upper_time[m])
            else:
                muU_out.x.array[:] = 0.0
            muU_out.x.scatter_forward()

            # --- Violazioni (DG0) ---
            T_cell.interpolate(y_out)
            T_cell.x.scatter_forward()

            sc_lower = args.sc_lower if args.sc_lower is not None else args.T_cure
            sc_upper = args.sc_upper if args.sc_upper is not None else 1e10

            viol_L = np.maximum(0.0, sc_lower - T_cell.x.array)
            viol_U = np.maximum(0.0, T_cell.x.array - sc_upper)

            vL_out.x.array[:] = 0.0
            vU_out.x.array[:] = 0.0
            mask = constraint_zone.x.array[:num_cells_local]
            vL_out.x.array[:num_cells_local] = viol_L[:num_cells_local] * mask
            vU_out.x.array[:num_cells_local] = viol_U[:num_cells_local] * mask
            vL_out.x.scatter_forward()
            vU_out.x.scatter_forward()

            # --- Scrittura campi a questo timestep ---
            vtk.write_function(y_out, t)
            vtk.write_function(p_out, t)
            vtk.write_function(uD_out, t)
            vtk.write_function(qN_out, t)
            vtk.write_function(uDir_out, t)

            vtk.write_function(muL_out, t)
            vtk.write_function(muU_out, t)
            vtk.write_function(vL_out, t)
            vtk.write_function(vU_out, t)

            # Zone ad OGNI timestep
            vtk.write_function(target_zone, t)
            vtk.write_function(control_zone_dist, t)
            vtk.write_function(constraint_zone, t)
            vtk.write_function(omega_qn, t)
            vtk.write_function(omega_qd, t)

    if rank == 0:
        logger.info(f"Saved: {pvd_path}")
        logger.info("Open solution.pvd in ParaView (time series + all fields).")
