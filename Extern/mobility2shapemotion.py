import os
import json

import tools.mobility_tool as mt
import copy
import shutil
from configs import DATA_ROOT_DICT, SAVE_ROOT_DICT, PARTNET_MOBILITY_ROOT
import argparse

read_json = lambda x: json.load(open(x, 'r'))


def process_mobility(mobility_json):
    ids = []
    objs = []
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['parent'] == -1:
            ids.append(move_part['id'])
            objs += move_part['parts']
    if len(ids) <= 1:
        return mobility_json
    remove_ids = ids[1:]
    remove_json = []
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['parent'] in remove_ids:
            mobility_json[i]['parent'] = ids[0]
        if move_part['id'] in remove_ids:
            remove_json.append(move_part)
    for part in remove_json:
        mobility_json.remove(part)
    mobility_json[ids[0]]['parts'] = objs
    return mobility_json


def merge_part(mobility_json, joint_types):
    ids = []
    remove_json = []
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['joint'] in joint_types or move_part['joint'].split('_')[0] in joint_types:
            ids.append(move_part['id'])
            remove_json.append(move_part)
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['id'] not in ids and move_part['parent'] in ids:
            parent = move_part['parent']
            while (parent in ids):
                for j in range(len(mobility_json)):
                    if mobility_json[j]['id'] == parent:
                        parent = mobility_json[j]['parent']
                        break
            mobility_json[i]['parent'] = parent
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['id'] in ids:
            parent = move_part['parent']
            while (parent in ids):
                for j in range(len(mobility_json)):
                    if mobility_json[j]['id'] == parent:
                        parent = mobility_json[j]['parent']
                        break
            mobility_json[parent]['parts'] += move_part['parts']
    for part in remove_json:
        mobility_json.remove(part)
    return mobility_json


def mobility2shapemotion(mobility_json, result_json, obj_path, part_count, save_folder, joint_types=[]):
    mobility_json = process_mobility(mobility_json)
    mobility_json = merge_part(mobility_json, joint_types)

    if part_count != -1 and len(mobility_json) != part_count:
        shutil.rmtree(save_folder)
        return

    joint_types = []
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        joint = move_part['joint'].split('_')[0]
        if (joint == 'T' or joint == 'R') and joint not in joint_types:
            joint_types.append(joint)
        objs = []
        for part in move_part['parts']:
            part_id = part['id']
            part_objs = result_json[part_id]['objs']
            for obj in part_objs:
                if not os.path.exists(os.path.join(obj_path, '%s.obj' % (obj))):
                    shutil.rmtree(save_folder)
                    return

    if len(joint_types) > 1:
        shutil.rmtree(save_folder)
        return

    template_json = {}
    template_json['dof_name'] = 'dof_rootd'
    template_json['motion_type'] = 'none'
    template_json['center'] = [0, 0, 0]
    template_json['direction'] = [1, 0, 0]
    template_json['children'] = []
    template_json['mobilityIndex'] = -1

    # root
    rootId = -1
    record = [False] * len(mobility_json) 
    for i in range(len(mobility_json)):
        move_part = mobility_json[i]
        if move_part['parent'] == -1:
            rootId = i
            record[i] = True
            objs = []
            for part in move_part['parts']:
                part_id = part['id']
                part_objs = result_json[part_id]['objs']
                objs += part_objs
            meshs = []
            for obj in objs:
                meshs.append(mt.load_mesh(os.path.join(obj_path, '%s.obj' % (obj))))
            new_obj = mt.merge_mesh(meshs).as_open3d
            mt.write_mesh(new_obj, os.path.join(save_folder, 'part_objs', 'none_motion.obj'))
    motions_json = copy.deepcopy(template_json)

    # children
    root_json = motions_json
    globla_partId = {'id': 0}

    def dfs_set_motion(json_data, parentId):
        for i in range(len(mobility_json)):
            if record[i] == True:
                continue
            move_part = mobility_json[i]
            if move_part['parent'] == parentId:
                record[i] = True
                globla_partId['id'] += 1

                child_json = copy.deepcopy(template_json)
                if move_part['joint'] == 'slider' or move_part['joint'].split('_')[0] == 'T':
                    child_json['motion_type'] = 'translation'
                elif move_part['joint'] == 'hinge' or move_part['joint'].split('_')[0] == 'R':
                    child_json['motion_type'] = 'rotation'
                child_json['dof_name'] = 'dof_rootd_%03d_%s' % (globla_partId['id'], child_json['motion_type'][0])
                child_json['center'] = move_part['jointData']['axis']['origin']
                child_json['direction'] = move_part['jointData']['axis']['direction']
                child_json['mobilityIndex'] = i

                objs = []
                for part in move_part['parts']:
                    part_id = part['id']
                    part_objs = result_json[part_id]['objs']
                    objs += part_objs
                meshs = []
                for obj in objs:
                    meshs.append(mt.load_mesh(os.path.join(obj_path, '%s.obj' % (obj))))
                new_obj = mt.merge_mesh(meshs).as_open3d
                mt.write_mesh(new_obj, os.path.join(save_folder, 'part_objs', '%s.obj' % child_json['dof_name']))

                json_data['children'].append(child_json)
        for child in json_data['children']:
            dfs_set_motion(child, mobility_json[child['mobilityIndex']]['id'])

    dfs_set_motion(root_json, mobility_json[rootId]['id'])

    def dfs_delet_key(json_data):
        for child in json_data['children']:
            dfs_delet_key(child)
            child.pop('mobilityIndex')

    root_json.pop('mobilityIndex')
    dfs_delet_key(root_json)
    print("saving to ", os.path.join(save_folder, 'motion_attributes.json'))
    json.dump(motions_json, open(os.path.join(save_folder, 'motion_attributes.json'), 'w'))


def output_parnet2shapemotion(obj_cls, obj_count, dataset_dir, mobility_json_dir, save_dir):
    for tname in obj_cls:
        cls_path = mobility_json_dir + '/%s' % tname
        fnames = os.listdir(cls_path)

        save_cls = save_dir + '/%s' % tname
        if not os.path.exists(save_cls):
            os.makedirs(save_cls)

        for fname in fnames:
            print(f"process {tname}/{fname}")
            fname = fname.split('.')[0]

            save_folder = save_cls + '/%s' % fname
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            save_obj_folder = os.path.join(save_folder, 'part_objs')
            if not os.path.exists(save_obj_folder):
                os.mkdir(save_obj_folder)

            mobility_json = read_json(os.path.join(cls_path, fname + '.json'))
            result_json = read_json(os.path.join(dataset_dir, tname, 'data_v0', fname, 'result.json'))
            result_json = mt.hier2graphic(result_json)

            obj_path = os.path.join(dataset_dir, tname, 'data_v0', fname, 'objs')

            merge_joint_types = []
            if tname == 'table' or tname == 'Table':
                merge_joint_types = ['hinge', 'R']
                result_path = os.path.join(dataset_dir, tname, 'data_v0', fname, 'result_complete.json')
                if os.path.exists(result_path):
                    result_json = read_json(result_path)
                    result_json = mt.hier2graphic(result_json)

            mobility2shapemotion(mobility_json, result_json, obj_path, obj_count[tname], save_folder, merge_joint_types)


def gt_partnetmoblity2shapemotion(obj_cls, obj_count, dataset_dir, save_dir):
    fnames = os.listdir(dataset_dir)
    for fname in fnames:
        fname = fname.split('.')[0]

        result_json = read_json(os.path.join(dataset_dir, fname, 'result.json'))
        result_json = mt.hier2graphic(result_json)
        cls_name = result_json[0]['name']
        if cls_name not in obj_cls:
            continue

        print(fname)
        save_folder = save_dir + '/%s/%s' % (cls_name, fname)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_obj_folder = os.path.join(save_folder, 'part_objs')
        if not os.path.exists(save_obj_folder):
            os.mkdir(save_obj_folder)

        mobility_json = read_json(os.path.join(dataset_dir, fname, 'mobility_v2.json'))
        obj_path = os.path.join(dataset_dir, fname, 'textured_objs')

        merge_joint_types = []
        if cls_name == 'table' or cls_name == 'Table':
            merge_joint_types = ['hinge', 'R']
            result_path = os.path.join(dataset_dir, fname, 'result_complete.json')
            mobility_path = os.path.join(dataset_dir, fname, 'mobility_v2_complete.json')
            if os.path.exists(result_path):
                result_json = read_json(result_path)
                result_json = mt.hier2graphic(result_json)
            if os.path.exists(mobility_path):
                mobility_json = read_json(mobility_path)
        mobility2shapemotion(mobility_json, result_json, obj_path, obj_count[cls_name], save_folder, merge_joint_types)


def main(args):
    print(args.obj_cls, args.obj_count, args.dataset)
    obj_cls = args.obj_cls
    obj_count = {obj_cls[i]: args.obj_count[i] for i in range(len(obj_cls))}
    if args.dataset == "partnet_mobility":
        # output_partnetmoblity2shapemotion(obj_cls, obj_count)
        dataset_dir = os.path.join(DATA_ROOT_DICT["partnet_mobility"], "dataset")
        save_dir = os.path.join(SAVE_ROOT_DICT["partnet_mobility"], "objects")
        gt_partnetmoblity2shapemotion(obj_cls, obj_count, dataset_dir, save_dir)
    elif args.dataset == "partnet":
        dataset_dir = os.path.join(DATA_ROOT_DICT["partnet"], "dataset")
        mobility_json_dir = PARTNET_MOBILITY_ROOT
        save_dir = os.path.join(SAVE_ROOT_DICT["partnet"], "objects")
        output_parnet2shapemotion(obj_cls, obj_count, dataset_dir, mobility_json_dir, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_cls', nargs="+", type=str)
    parser.add_argument('--obj_count', nargs="+", type=int)
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    main(args)
