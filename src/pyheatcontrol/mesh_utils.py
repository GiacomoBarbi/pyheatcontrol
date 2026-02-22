# mesh_utils.py
from mpi4py import MPI
import numpy as np
from dolfinx import mesh
from dolfinx.fem import locate_dofs_geometrical, Function, functionspace
from pyheatcontrol.constants import EPS_MACHINE

def create_mesh(n, L, H=None, mesh_family="quadrilateral"):
    """Create rectangular mesh [0,L]x[0,H].
    
    Args:
        n: refinement level (2^n cells per direction)
        L: length in x direction
        H: height in y direction (default: L)
        mesh_family: "quadrilateral" (Q) or "triangle" (P)
    """
    if H is None:
        H = L  # backward compatible
    
    Nx = 2**n
    Ny = max(1, int(2**n * H / L))  # scala con aspect ratio
    
    if mesh_family == "triangle":
        cell_type = mesh.CellType.triangle
    else:
        cell_type = mesh.CellType.quadrilateral
    
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [L, H]],
        [Nx, Ny],
        cell_type=cell_type,
    )
    return domain


def mark_cells_in_boxes(domain, boxes, L, H=None, Vc=None):
    """Mark cells inside boxes (owned cells only, no ghosts)."""
    if H is None:
        H = L
    V0 = Vc if Vc is not None else functionspace(domain, ("DG", 0))
    num_cells = V0.dofmap.index_map.size_local
    x_cells = domain.geometry.x
    cell_dofmap = domain.geometry.dofmap
    # Compute centers for owned cells only
    cell_centers = x_cells[cell_dofmap[:num_cells]].mean(axis=1)
    marker = np.zeros(num_cells, dtype=bool)
    for xmin, xmax, ymin, ymax in boxes:
        xmin_phys = xmin * L
        xmax_phys = xmax * L
        ymin_phys = ymin * H
        ymax_phys = ymax * H
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


def create_boundary_condition_function(domain, V, segments, L, H):

    def boundary_predicate(x):
        eps = EPS_MACHINE
        mask = np.zeros(x.shape[1], dtype=bool)

        for side, tmin, tmax in segments:

            if side in ("y0", "yL"):
                scale = L
            else:
                scale = H

            tmin_s = tmin * scale
            tmax_s = tmax * scale

            if side == "yL":
                on_side = np.isclose(x[1], H, atol=eps)
                in_range = (x[0] >= tmin_s - eps) & (x[0] <= tmax_s + eps)

            elif side == "y0":
                on_side = np.isclose(x[1], 0.0, atol=eps)
                in_range = (x[0] >= tmin_s - eps) & (x[0] <= tmax_s + eps)

            elif side == "x0":
                on_side = np.isclose(x[0], 0.0, atol=eps)
                in_range = (x[1] >= tmin_s - eps) & (x[1] <= tmax_s + eps)

            elif side == "xL":
                on_side = np.isclose(x[0], L, atol=eps)
                in_range = (x[1] >= tmin_s - eps) & (x[1] <= tmax_s + eps)

            else:
                continue

            mask |= on_side & in_range

        return mask

    dofs = locate_dofs_geometrical(V, boundary_predicate)
    bc_func = Function(V)

    return dofs, bc_func



def create_boundary_facet_tags(domain, segments, L, H, marker_id):
    from dolfinx.mesh import locate_entities_boundary, meshtags

    tdim = domain.topology.dim
    fdim = tdim - 1

    def boundary_predicate(x):
        eps = EPS_MACHINE
        mask = np.zeros(x.shape[1], dtype=bool)

        for side, tmin, tmax in segments:

            # scala giusta per direzione tangenziale
            if side in ("y0", "yL"):
                scale = L
            else:
                scale = H

            tmin_s = tmin * scale
            tmax_s = tmax * scale

            if side == "yL":  # top
                on_side = np.isclose(x[1], H, atol=eps)
                in_range = (x[0] >= tmin_s - eps) & (x[0] <= tmax_s + eps)

            elif side == "y0":  # bottom
                on_side = np.isclose(x[1], 0.0, atol=eps)
                in_range = (x[0] >= tmin_s - eps) & (x[0] <= tmax_s + eps)

            elif side == "x0":  # left
                on_side = np.isclose(x[0], 0.0, atol=eps)
                in_range = (x[1] >= tmin_s - eps) & (x[1] <= tmax_s + eps)

            elif side == "xL":  # right
                on_side = np.isclose(x[0], L, atol=eps)
                in_range = (x[1] >= tmin_s - eps) & (x[1] <= tmax_s + eps)

            else:
                continue

            mask |= on_side & in_range

        return mask

    boundary_facets = locate_entities_boundary(domain, fdim, boundary_predicate).astype(np.int32)

    order = np.argsort(boundary_facets)
    boundary_facets = boundary_facets[order]

    values = np.full(boundary_facets.shape, marker_id, dtype=np.int32)

    facet_tags = meshtags(domain, fdim, boundary_facets, values)
    return facet_tags, marker_id

