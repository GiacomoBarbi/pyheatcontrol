# mesh_utils.py
from mpi4py import MPI
import numpy as np
from dolfinx import mesh
from dolfinx.fem import locate_dofs_geometrical, Function, functionspace


def create_mesh(n, L):
    """Mesh quadrata [0,L]x[0,L]"""
    Nx = 2**n
    Ny = 2**n
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [L, L]],
        [Nx, Ny],
        cell_type=mesh.CellType.quadrilateral,
    )
    return domain


def mark_cells_in_boxes(domain, boxes, L):
    """Marca celle in boxes (solo celle owned, non ghost)"""
    V0 = functionspace(domain, ("DG", 0))
    num_cells = V0.dofmap.index_map.size_local
    x_cells = domain.geometry.x
    cell_dofmap = domain.geometry.dofmap
    # Calcola centri solo per le celle owned (prime num_cells righe)
    cell_centers = x_cells[cell_dofmap[:num_cells]].mean(axis=1)
    marker = np.zeros(num_cells, dtype=bool)

    for xmin, xmax, ymin, ymax in boxes:
        xmin_phys = xmin * L
        xmax_phys = xmax * L
        ymin_phys = ymin * L
        ymax_phys = ymax * L

        mask = np.logical_and.reduce(
            [
                cell_centers[:, 0] >= xmin_phys,
                cell_centers[:, 0] <= xmax_phys,
                cell_centers[:, 1] >= ymin_phys,
                cell_centers[:, 1] <= ymax_phys,
            ]
        )
        marker |= mask

    return marker


def create_boundary_condition_function(domain, V, segments, L):
    """Crea funzione boundary per segmenti specificati (Dirichlet)"""

    def boundary_predicate(x):
        eps = 1e-14
        mask = np.zeros(x.shape[1], dtype=bool)

        for side, tmin, tmax in segments:
            tmin_L = tmin * L
            tmax_L = tmax * L

            if side == "yL":
                on_side = np.isclose(x[1], L, atol=eps)
                in_range = (x[0] >= tmin_L - eps) & (x[0] <= tmax_L + eps)
            elif side == "y0":
                on_side = np.isclose(x[1], 0.0, atol=eps)
                in_range = (x[0] >= tmin_L - eps) & (x[0] <= tmax_L + eps)
            elif side == "x0":
                on_side = np.isclose(x[0], 0.0, atol=eps)
                in_range = (x[1] >= tmin_L - eps) & (x[1] <= tmax_L + eps)
            elif side == "xL":
                on_side = np.isclose(x[0], L, atol=eps)
                in_range = (x[1] >= tmin_L - eps) & (x[1] <= tmax_L + eps)
            else:
                continue

            mask |= on_side & in_range

        return mask

    dofs = locate_dofs_geometrical(V, boundary_predicate)
    bc_func = Function(V)

    return dofs, bc_func


def create_boundary_facet_tags(domain, segments, L, marker_id):
    """
    Crea facet tags per boundary (Neumann/Dirichlet) su segmenti specificati.
    Returns: (facet_tags, marker_id)
    """
    from dolfinx.mesh import locate_entities_boundary, meshtags

    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_predicate(x):
        eps = 1e-14
        mask = np.zeros(x.shape[1], dtype=bool)

        for side, tmin, tmax in segments:
            tmin_L = tmin * L
            tmax_L = tmax * L

            if side == "yL":
                on_side = np.isclose(x[1], L, atol=eps)
                in_range = (x[0] >= tmin_L - eps) & (x[0] <= tmax_L + eps)
            elif side == "y0":
                on_side = np.isclose(x[1], 0.0, atol=eps)
                in_range = (x[0] >= tmin_L - eps) & (x[0] <= tmax_L + eps)
            elif side == "x0":
                on_side = np.isclose(x[0], 0.0, atol=eps)
                in_range = (x[1] >= tmin_L - eps) & (x[1] <= tmax_L + eps)
            elif side == "xL":
                on_side = np.isclose(x[0], L, atol=eps)
                in_range = (x[1] >= tmin_L - eps) & (x[1] <= tmax_L + eps)
            else:
                continue

            mask |= on_side & in_range

        return mask

    boundary_facets = locate_entities_boundary(domain, fdim, boundary_predicate).astype(
        np.int32
    )

    # meshtags richiede indici ordinati
    order = np.argsort(boundary_facets)
    boundary_facets = boundary_facets[order]

    values = np.full(boundary_facets.shape, marker_id, dtype=np.int32)
    # values = values[order]  # (ridondante ma coerente)

    facet_tags = meshtags(domain, fdim, boundary_facets, values)
    return facet_tags, marker_id
