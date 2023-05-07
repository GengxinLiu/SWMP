import argparse
from configs import GRAPHICS_RESULT_ROOT, MOBILITY_RESULT_GNN_ROOT, DATA_ROOT_DICT, mobility2partnet, \
    MERGE_OBJ_RESULT_ROOT, MERGE_OBB_RESULT_ROOT
import numpy as np
from tools.mobility_tool import PCA, read_json, add_thickness, pc2mesh
import open3d as o3d
import os
from utils import bar
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='laptop')
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    npoints = 1024
    SOURCE_DATA_ROOT = os.path.join(DATA_ROOT_DICT[args.dataset], "dataset")
    GRAPHICS_RESULT_ROOT = os.path.join(f"result_{args.dataset}", GRAPHICS_RESULT_ROOT)
    MOBILITY_RESULT_GNN_ROOT = os.path.join(f"result_{args.dataset}", MOBILITY_RESULT_GNN_ROOT)
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    graphics_result_root = os.path.join(GRAPHICS_RESULT_ROOT, CATEGORY)
    mobility_result_gnn_root = os.path.join(MOBILITY_RESULT_GNN_ROOT, CATEGORY)
    merge_obb_result_root = MERGE_OBJ_RESULT_ROOT
    os.makedirs(MERGE_OBB_RESULT_ROOT, exist_ok=True)
    os.makedirs(MERGE_OBJ_RESULT_ROOT, exist_ok=True)
    files = os.listdir(mobility_result_gnn_root)
    for i, file in enumerate(files):
        mobility_data = read_json(os.path.join(mobility_result_gnn_root, file))
        graphics_data = read_json(os.path.join(graphics_result_root, file))
        for node in mobility_data:
            if node["joint"] == "free":
                continue
            mesh_names = []
            for part in node["parts"]:
                objs = graphics_data[str(part["id"])]["objs"] 
                mesh_names.extend(objs)
            # read mesh_names
            mesh_list = []
            for name in mesh_names:
                if args.dataset == "partnet_mobility":
                    mesh_path = os.path.join(SOURCE_DATA_ROOT, file.split('.')[0], "textured_objs", name + ".obj")
                elif args.dataset == "partnet":
                    mesh_path = os.path.join(SOURCE_DATA_ROOT, mobility2partnet[args.category], "data_v0",
                                             file.split('.')[0], "objs", name + ".obj")
                mesh_list.append(trimesh.load(mesh_path, force='mesh'))
            merged_meshes = trimesh.util.concatenate(mesh_list)
            # o3d.io.write_triangle_mesh(
            #     os.path.join(merge_obb_result_root, "{}_{}.obj".format(file.split('.')[0], node["id"])), merged_meshes.as_open3d,
            #     write_vertex_normals=False, write_vertex_colors=False
            # )

            pc = merged_meshes.sample(npoints)
            mesh = merged_meshes.as_open3d
            pca_val, pca_vec = PCA(pc)
            if (np.abs(pca_val) < 1e-5).sum():
                noise_pc = pc.copy()
                for direction in pca_vec:
                    noise_pc = add_thickness(noise_pc, direction, 0.008)
                np.random.shuffle(noise_pc)
                mesh = pc2mesh(noise_pc[:npoints])
            if len(mesh.vertices) > npoints:
                mesh_pc = merged_meshes.sample(npoints)
                mesh = pc2mesh(mesh_pc)
            mesh.triangle_normals = o3d.utility.Vector3dVector([])
            o3d.io.write_triangle_mesh(
                os.path.join(merge_obb_result_root, "{}_{}.obj".format(file.split('.')[0], node["id"])), mesh,
                write_vertex_normals=False, write_vertex_colors=False
            )
        bar(f"4_MergeObj/{CATEGORY}", i + 1, len(files))
