import numpy as np
import os, json
from tools.mobility_tool import PCA, load_mesh, merge_mesh, read_json, write_mesh, add_thickness, pc2mesh
from configs import DATA_ROOT_DICT, npoints
from tools.utils import bar
import open3d as o3d
import argparse


def merge_obj(obj_folder, hier_tree):
    all_node_mesh = {}

    for node in hier_tree:
        id_ = node['id']

        if 'children' in node.keys():
            sub_mesh = merge_obj(obj_folder, node['children'])
            all_node_mesh = {**all_node_mesh, **sub_mesh}
            child_mesh = [sub_mesh[me['id']] for me in node['children']]
            node_mesh = merge_mesh(child_mesh)
            all_node_mesh[id_] = node_mesh
        else:
            meshs = []
            for obj_name in node['objs']:
                obj_path = os.path.join(obj_folder, obj_name + '.obj')
                mesh = load_mesh(obj_path)
                meshs.append(mesh)
            if len(meshs) > 1:
                meshs = merge_mesh(meshs)
            else:
                meshs = meshs[0]

            all_node_mesh[id_] = meshs

    return all_node_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()

    data_root = os.path.join(DATA_ROOT_DICT[args.dataset], "dataset")

    if args.dataset == "partnet_mobility":
        model_json = json.load(open("model.json", "r"))
        categories = model_json.keys()
        files_dict = {category: model_json[category] for category in categories}
    elif args.dataset == "partnet":
        categories = os.listdir(data_root)
        files_dict = {category: os.listdir(os.path.join(data_root, category, "data_v0")) for category in categories}
    else:
        raise ValueError(f"Not support dataset `{args.dataset}`")

    matlab_input_folder = 'OBBcalculation/Input'
    matlab_output_folder = 'OBBcalculation/Output'
    os.makedirs(matlab_input_folder, exist_ok=True)
    os.makedirs(matlab_output_folder, exist_ok=True)
    for category in categories:
        files = files_dict[category]
        for i, fname in enumerate(files):
            if args.dataset == "partnet_mobility":
                base_path = os.path.join(data_root, fname)
                obj_dir = "textured_objs"
            elif args.dataset == "partnet":
                base_path = os.path.join(data_root, category, "data_v0", fname)
                obj_dir = "objs"
            if not os.path.exists(base_path):
                print(f"\n{base_path} not exist, skip")
                continue
            if category in ["table", "Table"]:
                hier_tree_path = os.path.join(base_path, 'result_complete.json')
            else:
                hier_tree_path = os.path.join(base_path, 'result.json')

            obj_folder = os.path.join(base_path, obj_dir)
            hier_tree = read_json(hier_tree_path)
            all_node_mesh = merge_obj(obj_folder, hier_tree)
            for key in all_node_mesh.keys():
                pc = all_node_mesh[key].sample(npoints)
                mesh = all_node_mesh[key].as_open3d
                pca_val, pca_vec = PCA(pc)
                if (np.abs(pca_val) < 1e-5).sum():
                    noise_pc = pc.copy()
                    for direction in pca_vec:
                        noise_pc = add_thickness(noise_pc, direction, 0.008)
                    np.random.shuffle(noise_pc)
                    mesh = pc2mesh(noise_pc[:npoints])

                if len(mesh.vertices) > npoints:
                    mesh_pc = all_node_mesh[key].sample(npoints)
                    mesh = pc2mesh(mesh_pc)
                mesh.triangle_normals = o3d.utility.Vector3dVector([])
                write_mesh(mesh, matlab_input_folder + '/%s_%s.obj' % (fname, str(key)))
            bar(f"objPreprocess {category}", i + 1, len(files))
