import json
import numpy as np
import math
import os
import tools.mobility_tool as mt
from configs import DATA_ROOT_DICT
from tools.utils import bar
import argparse


def generate_data_partnet(data_root, graphics_root, fname, category):
    if category == "Table":
        result_path = os.path.join(
            data_root, category, 'data_v0', fname, 'result_complete.json')
    else:
        result_path = os.path.join(
            data_root, category, 'data_v0', fname, 'result.json')
    result_data = json.load(open(result_path, 'r'))[0]

    graphics = {}
    root = result_data
    depth = 0
    last_level = [{"id": -1}]
    cur_level = [[root]]
    count_node = 1

    ok_flag = True
    while count_node != 0:
        _last_level, _cur_level = [], []
        for i in range(len(cur_level)):
            brother_nodes = cur_level[i]
            parent_node = last_level[i]
            brother_id = [brother_node["id"]
                          for brother_node in brother_nodes]
            for node in brother_nodes:
                _last_level.append(node)
                csv_path = os.path.join(
                    matlab_output_folder, '{}_{}_box.csv'.format(fname, node['id']))

                if not os.path.exists(csv_path):
                    # if '3535' in csv_path:
                    #     print(csv_path, 'has not exist!')
                    f = open('network_data_extract_log.txt', 'a+')
                    f.write(f'{csv_path} not exist!\n')
                    f.close()
                    ok_flag = False
                    break

                box, dirs = mt.csv2box(csv_path)
                children_id = []
                children = {}
                if "children" in node:
                    for child in node["children"]:
                        children_id.append(child["id"])
                        children[child["id"]] = ""
                _brother_id = brother_id.copy()
                _brother_id.remove(node['id'])
                space = {}
                for _id in _brother_id:
                    space[_id] = "motion"

                graphics[node["id"]] = {"name": node['name'],
                                        "objs": node["objs"] if "objs" in node else [],
                                        "parent": parent_node["id"],
                                        "depth": depth,
                                        "box": box.tolist(),
                                        "brother": _brother_id,
                                        "children_id": children_id,
                                        "leaves": [],
                                        "jointData": {
                                            "axis": {
                                                "origin": [0, 0, 0],
                                                "direction": [1, 0, 0]
                                            },
                                            "limit": {
                                                "a": 0,
                                                "b": 0,
                                                "noLimit": True
                                            }
                                        },  # to be predicted
                                        "joint": "hinge",  # to be predicted
                                        "motype": "R_V_S",  # to be predicted
                                        "edges": {
                                            "children": children,
                                            "space": space}
                                        }
                if "children" in node:
                    _cur_level.append(node["children"])
                else:
                    _cur_level.append([])
                    parent = node["id"]
                    while parent != -1:
                        graphics[parent]["leaves"].append(node["id"])
                        parent = graphics[parent]["parent"]
            if not ok_flag:
                break
        if not ok_flag:
            break
        last_level = _last_level
        cur_level = _cur_level
        depth += 1
        count_node = 0
        for l in cur_level:
            count_node += len(l)
    if ok_flag:
        os.makedirs(os.path.join(graphics_root, category), exist_ok=True)
        json.dump(graphics, open(os.path.join(
            graphics_root, category, fname + ".json"), 'w'))


def generate_data_partnet_mobility(fname, save_dir):
    base_path = os.path.join(data_root, fname)
    if not os.path.exists(base_path):
        print(f"\n{base_path} not exist, skip")
        return
    hier_tree_path = os.path.join(base_path, 'result.json')
    mobi_path = os.path.join(base_path, 'mobility_v2.json')
    if os.path.exists(os.path.join(base_path, 'result_complete.json')):
        hier_tree_path = os.path.join(base_path, 'result_complete.json')
    if os.path.exists(os.path.join(base_path, 'mobility_v2_complete.json')):
        mobi_path = os.path.join(base_path, 'mobility_v2_complete.json')
    data = {
        'hier_tree': mt.read_json(hier_tree_path),
        'mobi': mt.read_json(mobi_path),
    }

    graph, _ = mt.gen_graph(**data)

    for id_ in graph.keys():
        node = graph[id_]

        csv_path = matlab_output_folder + '/%s_%s_box.csv' % (fname, id_)
        box, dirs = mt.csv2box(csv_path)
        graph[id_]['box'] = box.tolist()

        if node['joint'] in ['free', 'static', 'junk', 'heavy', '']:
            graph[id_]['motype'] = ''
            continue
        elif node['jointData'] and 'rotates' in node['jointData']['limit'].keys() \
                and node['jointData']['limit']['rotates']:
            motion_type = 'TR'
        elif node['joint'] == 'slider':
            motion_type = 'T'
        else:
            motion_type = 'R'

        axis_pos = np.asarray(node['jointData']['axis']['origin'])
        axis_dir = np.asarray(node['jointData']['axis']['direction'])
        angle = np.abs(axis_dir[2] / np.linalg.norm(axis_dir))
        angle = math.acos(angle)
        angle = np.rad2deg(angle)
        if angle < 45:
            motion_type += '_V'
        else:
            motion_type += '_H'
        graph[id_]['motype'] = motion_type

        if motion_type[0] == 'R':
            box_corners = np.asarray(graph[id_]['box'])
            pos_idx, distance = mt.motion_pos(
                axis_dir, axis_pos, np.vstack([
                    box_corners, np.mean(box_corners, axis=0).reshape(-1, 3)
                ])
            )
            graph[id_]['motype'] += '_S' if pos_idx < 8 else '_C'

    save_doc = '%s/%s' % (save_dir, graph[0]['name'])
    os.makedirs(save_doc, exist_ok=True)
    json.dump(graph, open('%s/%s.json' % (save_doc, fname), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()

    matlab_output_folder = 'OBBcalculation/Output'
    data_root = os.path.join(DATA_ROOT_DICT[args.dataset], "dataset")
    graphics_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graphics")

    if args.dataset == "partnet_mobility":
        model_json = json.load(open("model.json", "r"))
        categories = model_json.keys()
        files_dict = {category: model_json[category] for category in categories}
    elif args.dataset == "partnet":
        categories = os.listdir(data_root)
        files_dict = {category: os.listdir(os.path.join(data_root, category, "data_v0")) for category in categories}
    else:
        raise ValueError(f"Not support dataset `{args.dataset}`")

    for category in categories:
        files = files_dict[category]
        for i, fname in enumerate(files):
            if args.dataset == "partnet_mobility":
                generate_data_partnet_mobility(fname, graphics_root)
            elif args.dataset == "partnet":
                generate_data_partnet(data_root, graphics_root, fname, category)
            bar("3_GraphicsDataExtract", i + 1, len(files))
