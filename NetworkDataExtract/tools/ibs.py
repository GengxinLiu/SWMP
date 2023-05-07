import numpy as np
# import minpy.numpy as np
import pyvista as pv
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree as KDTree
import trimesh


def compute_point_normals(mesh, points, triids):
    barys = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[triids], points=points)
    normals = trimesh.unitize(
        (mesh.vertex_normals[mesh.faces[triids]] * trimesh.unitize(barys).reshape((-1, 3, 1))).sum(axis=1))
    return normals


def create_ibs(x0, x1, bounds=None):
    """create ibs from two point cloud

    Args:
        x0 ([type]): point cloud 0
        x1 ([type]): point cloud 2
        bounds ([type], optional): (xmin,xmax,ymin,ymax,zmin,zmax) Defaults to None.

    Returns:
        Trimesh: ibs surface
    """
    n0, n1 = len(x0), len(x1)
    x01 = np.concatenate([x0, x1])

    vor = Voronoi(x01)
    # find ridges
    ridge_idx = np.where((vor.ridge_points < n0).sum(-1) == 1)[0]
    # remove points at infinity
    ridge_idx = [i for i in ridge_idx if -1 not in vor.ridge_vertices[i]]
    # create ridge polygons
    polys = np.asarray(vor.ridge_vertices, dtype=object)[ridge_idx]
    polys = np.concatenate(list(map(lambda x: [len(x), ] + x, polys)))

    if bounds is None:
        bounds = np.stack([x01.min(0), x01.max(0)], axis=-1).reshape(-1)
    ibs = pv.PolyData(vor.vertices, polys)
    ibs = ibs.clip_box(bounds.tolist(), invert=False)
    ibs = ibs.triangulate()
    ibs = trimesh.Trimesh(ibs.points, ibs.cells.reshape(-1, 4)[:, 1:4])
    return ibs


def sample_ibs(x0, x1, n=500, adaptive=False, return_norm=False, return_ir=False):
    """sample ibs points from two point cloud

    Args:
        x0 ([type]): center object
        x1 ([type]): other object
        n (int, optional): [description]. Defaults to 500.

    Returns:
        dict
    """
    ibs_dict = {}

    x01 = np.concatenate([x0, x1])
    bounds = np.stack([x01.min(0), x01.max(0)], axis=-1).reshape(-1)
    l = 0.5 * np.linalg.norm(bounds[1::2] - bounds[::2])

    # create ibs mesh
    ibs = create_ibs(x0, x1, bounds=bounds)
    ibs.fix_normals()
    ibs_dict['mesh'] = ibs

    # sample ibs points
    kdtree = None
    if not adaptive:
        ibs_pts, ibs_fid = ibs.sample(n, return_index=True)
    else:
        kdtree = KDTree(x0)
        tri_area = ibs.area_faces
        tri_cent = ibs.triangles_center
        tri_norm = ibs.face_normals
        tri_dist, tri_nn = kdtree.query(tri_cent, workers=-1)
        tri_alpha = np.arccos(
            np.clip(((x0[tri_nn] - tri_cent) * tri_norm / (tri_dist[:, None] + 1e-6)).sum(axis=-1), -1.0, 1.0))
        mask = tri_alpha > np.pi / 2
        tri_alpha[mask] = np.pi - tri_alpha[mask]
        tri_w = tri_area * np.clip(1 - 4 * tri_alpha / np.pi, 0, 1) * np.power(1 - tri_dist / l, 20)
        tri_w = tri_w / tri_w.sum()
        ibs_pts, ibs_fid = ibs.sample(n, return_index=True, face_weight=tri_w)
    ibs_dict['pts'] = ibs_pts

    if return_norm:
        # compute normal
        ibs_norms = compute_point_normals(ibs, ibs_pts, ibs_fid)
        ibs_dict['norm'] = ibs_norms

    if return_ir:
        # compute ir of x0
        if kdtree is None: kdtree = KDTree(x0)
        ir_dist, ir_idx = kdtree.query(ibs_pts)
        ibs_dict['ir_idx'] = ir_idx
        ibs_dict['ir_dist'] = ir_dist
        # ir = x0[ir_idx]

    return ibs_dict
