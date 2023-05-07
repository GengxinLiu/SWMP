import os
import numpy as np
import h5py
from tools.mobility_tool import load_mesh, merge_mesh, read_json
from tools.utils import bar
from configs import DATA_ROOT_DICT, npoints
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
    graphics_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graphics")
    pcseg_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "pc_seg")
    categories = os.listdir(graphics_root)
    for category in categories:
        files = os.listdir(os.path.join(graphics_root, category))
        os.makedirs(os.path.join(pcseg_root, category), exist_ok=True)
        for fi, fname in enumerate(files):
            fname = fname.split('.')[0]
            graphics_path = os.path.join(graphics_root, category, fname + ".json")
            if args.dataset == "partnet_mobility":
                base_path = os.path.join(data_root, fname)
                obj_dir = "textured_objs"
            elif args.dataset == "partnet":
                base_path = os.path.join(data_root, category, "data_v0", fname)
                obj_dir = "objs"

            obj_folder = os.path.join(base_path, obj_dir)
            graphics_data = read_json(graphics_path)

            all_pc, all_seg = [], []
            for node in graphics_data.keys():
                if len(graphics_data[node]["objs"]):
                    meshs = []
                    for obj_name in graphics_data[node]["objs"]:
                        obj_path = os.path.join(obj_folder, obj_name + '.obj')
                        mesh = load_mesh(obj_path)
                        meshs.append(mesh)
                    if len(meshs) > 1:
                        meshs = merge_mesh(meshs)
                    else:
                        meshs = meshs[0]
                    all_pc.extend(np.asarray(meshs.sample(npoints)))
                    all_seg.extend(np.array([int(node)] * npoints))
            all_pc = np.vstack(all_pc)
            all_seg = np.hstack(all_seg)
            with h5py.File(os.path.join(pcseg_root, category, fname + ".h5"), 'w') as f:
                f.create_dataset(
                    'pc', data=all_pc, dtype='float', compression=True)
                f.create_dataset(
                    'seg', data=all_seg, dtype='int', compression=True)
            bar(f'5_PcsegDataExtract/{category}', fi + 1, len(files))
