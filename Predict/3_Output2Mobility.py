from operator import truediv
import os
import json
import numpy as np
from configs import GRAPHICS_RESULT_ROOT, MOBILITY_RESULT_GNN_ROOT, mobility2partnet
import argparse
from utils import bar

read_json = lambda x: json.load(open(x, 'r'))


def dict_squeeze(listofdict):
    dict_ = {}
    for d in listofdict:  # [{key1:value1}, {key2:value2}]
        key = list(d.keys())[0]
        d = d[key]
        dict_[key] = d
    return dict_


def preprocess(output_json):
    for part_id in output_json.keys():
        output_json[part_id]['predict'] = dict_squeeze(output_json[part_id]['predict'])
    return output_json


def output2motion(output_json, save_path):
    '''process output'''
    output_json = preprocess(output_json)

    val = [int(i) for i in output_json.keys()]
    record = [False] * (max(val) + 1)

    move_groups = []
    for part_id in output_json.keys():
        if record[int(part_id)] == True:
            continue
        record[int(part_id)] = True

        nodes = output_json[part_id]['predict']
        if len(nodes) == 0: continue

        move_group = []
        move_group.append(part_id)
        for tail_Id in nodes.keys():
            motype = output_json[part_id]['predict'][tail_Id]['motype']
            reverse_motype = output_json[tail_Id]['predict'][part_id]['motype']
            # fixed 边
            if motype == 'fixed' and reverse_motype == 'fixed':
                move_group.append(tail_Id)
                record[int(tail_Id)] = True

        for bro in output_json[part_id]['brother']:
            if str(bro) not in move_group:
                move_groups.append(move_group)
                group_id = len(move_groups) - 1
                for id in move_group:
                    output_json[id]['move_part'] = group_id
                break
    if len(move_groups) == 0:
        move_groups.append(['0'])

    group_edges = [dict(parent=-1, child=[]) for i in range(len(move_groups))]
    for group_id in range(len(move_groups)):
        group = move_groups[group_id]

        groupId2relations = {}
        for part_id in group:
            nodes = output_json[part_id]['predict']
            for node_id in nodes.keys():
                if node_id not in group:
                    other_group = output_json[node_id]['move_part']
                    motion_inf = nodes[node_id]
                    if other_group not in groupId2relations.keys():
                        groupId2relations[other_group] = []
                    groupId2relations[other_group].append(motion_inf)

        ref_motion_count_max = 0
        for other_groupId in groupId2relations.keys():
            relations = groupId2relations[other_groupId]
            motype2motions = {}
            motion_type_name = ''
            motion_count_max = 0
            for relat in relations:
                motype = relat['motype']
                if motype == 'fixed' or motype == 'none':
                    continue
                if motype not in motype2motions.keys():
                    motype2motions[motype] = []
                motype2motions[motype].append(relat)
                if len(motype2motions[motype]) > motion_count_max:
                    motion_count_max = len(motype2motions[motype])
                    motion_type_name = motype
            if motion_count_max > 0:
                motions = motype2motions[motion_type_name]
                param = {'origin': [], 'direction': []}
                for motion in motions:
                    param['origin'].append(motion['origin'])
                    param['direction'].append(motion['direction'])
                param['origin'] = np.mean(np.array(param['origin']), axis=0)
                param['direction'] = np.mean(np.array(param['direction']), axis=0)

                if ref_motion_count_max > motion_count_max:
                    continue
                ref_motion_count_max = motion_count_max
                group_edges[group_id]['parent'] = other_groupId
                group_edges[group_id]['joint'] = motion_type_name
                group_edges[group_id]['jointData'] = {}
                group_edges[group_id]['jointData']['axis'] = {}
                group_edges[group_id]['jointData']['axis']['origin'] = param['origin'].tolist()
                group_edges[group_id]['jointData']['axis']['direction'] = param['direction'].tolist()

    for i in range(len(group_edges)):
        edge = group_edges[i]
        if edge['parent'] == -1:
            continue

        fast = edge
        slow = edge
        node = None
        while (fast['parent'] != -1 and group_edges[fast['parent']]['parent'] != -1):
            slow = group_edges[slow['parent']]
            fast = group_edges[group_edges[fast['parent']]['parent']]
            if (slow == fast):
                node = slow
                break
        if node is not None:
            group_edges[node['parent']]['parent'] = -1

    tailGroupIds = [] 
    record_edge = [False] * len(group_edges)  
    for i in range(len(group_edges)):
        if record_edge[i] == True:
            continue
        record_edge[i] = True

        tail = group_edges[i]
        tailGroupId = i
        while (tail['parent'] != -1):
            tailGroupId = tail['parent']
            tail = group_edges[tail['parent']]
        if tailGroupId not in tailGroupIds:
            tailGroupIds.append(tailGroupId)

    for i in range(len(group_edges)):
        edge = group_edges[i]
        if edge['parent'] != -1:
            group_edges[edge['parent']]['child'].append(i)

    for i in range(len(tailGroupIds)):
        if (len(move_groups[tailGroupIds[i]]) == 0):
            continue
        part1_id = move_groups[tailGroupIds[i]][0]
        for j in range(i, len(tailGroupIds)):
            if (len(move_groups[tailGroupIds[j]]) == 0):
                continue
            part2_id = move_groups[tailGroupIds[j]][0]
            if int(part1_id) not in output_json[part2_id]['brother']:
                continue
            move_groups[tailGroupIds[i]] += move_groups[tailGroupIds[j]]
            move_groups[tailGroupIds[j]] = []
            childGroupIds = group_edges[tailGroupIds[j]]['child']
            for childGroupId in childGroupIds:
                group_edges[childGroupId]['parent'] = tailGroupIds[i]
                group_edges[tailGroupIds[i]]['child'].append(childGroupId)

    tailGroupIds = []
    record_edge = [False] * len(group_edges)  
    for i in range(len(group_edges)):
        if record_edge[i] == True:
            continue
        record_edge[i] = True

        if len(move_groups[i]) == 0:
            continue
        tail = group_edges[i]
        tailGroupId = i
        while (tail['parent'] != -1):
            tailGroupId = tail['parent']
            tail = group_edges[tail['parent']]
        if tailGroupId not in tailGroupIds:
            tailGroupIds.append(tailGroupId)

    group_hiers = [dict(parent=-1, child={}) for i in range(len(move_groups))]
    for tailGroupId in tailGroupIds:
        firstpart = move_groups[tailGroupId][0]
        # parent = output_json[part0Id]['parent']
        finded = {}

        def find_parent_group(partId):
            if partId == -1 or 'True' in finded.keys():
                return
            for tail in range(len(move_groups)):
                if str(partId) in move_groups[tail]:
                    # finded = True
                    finded['True'] = 1
                    group_hiers[tailGroupId]['parent'] = tail
                    if partId not in group_hiers[tail]['child'].keys():
                        group_hiers[tail]['child'][partId] = []
                    group_hiers[tail]['child'][partId].append(tailGroupId)
                if 'True' in finded.keys():
                    return
            find_parent_group(output_json[str(partId)]['parent'])

        find_parent_group(output_json[firstpart]['parent'])

    for i in range(len(group_hiers)):
        def merge(cur_groupId):
            child = group_hiers[cur_groupId]['child']
            if len(move_groups[cur_groupId]) == 0:
                return
            for partId in child.keys():
                new_parts = []
                remove_groupIds = []
                for groupId in child[partId]:
                    merge(groupId)
                    new_parts += move_groups[groupId]
                    move_groups[groupId] = []
                    remove_groupIds.append(groupId)
                move_groups[cur_groupId] += new_parts
                if len(new_parts) > 0:
                    move_groups[cur_groupId].remove(str(partId))
                    for groupId in remove_groupIds:
                        for children in group_edges[groupId]['child']:
                            group_edges[children]['parent'] = cur_groupId
                            group_edges[cur_groupId]['child'].append(children)

        merge(i)

    root = -1
    part_count = 0
    for i in range(len(move_groups)):
        if len(move_groups[i]) > 0:
            part_count += 1
            if group_edges[i]['parent'] == -1:
                root = i
    for i in range(len(move_groups)):
        leaves = []
        for partId in move_groups[i]:
            for leave in output_json[partId]['leaves']:
                if leave not in leaves:
                    leaves.append(leave)
        for j in range(len(leaves)):
            leave = leaves[j]
            leaves[j] = {'id': leave, 'name': output_json[str(leave)]['name']}
        move_groups[i] = leaves

    motion_json = {}
    motion_json['id'] = 0
    motion_json['parent'] = -1
    motion_json['joint'] = 'free'
    motion_json['name'] = ''
    motion_json['parts'] = []
    motion_json['jointData'] = {}
    motions_json = [dict(motion_json) for i in range(part_count)]

    motions_json[0]['parts'] = move_groups[root]

    next_parts = group_edges[root]['child']
    move_id = {'id': 0}
    parent_id = 0

    def dfs_set_json(next_parts, parent_id):
        for part in next_parts:
            move_id['id'] += 1
            move_partid = move_id['id']
            motions_json[move_partid]['id'] = move_partid
            motions_json[move_partid]['parent'] = parent_id
            motions_json[move_partid]['joint'] = group_edges[part]['joint']
            motions_json[move_partid]['parts'] = move_groups[part]
            motions_json[move_partid]['jointData'] = group_edges[part]['jointData']
            dfs_set_json(group_edges[part]['child'], move_partid)

    dfs_set_json(next_parts, parent_id)

    json.dump(motions_json, open(save_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='laptop')
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    GRAPHICS_RESULT_ROOT = os.path.join(f"result_{args.dataset}", GRAPHICS_RESULT_ROOT)
    MOBILITY_RESULT_GNN_ROOT = os.path.join(f"result_{args.dataset}", MOBILITY_RESULT_GNN_ROOT)
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    graphics_result_root = os.path.join(GRAPHICS_RESULT_ROOT, CATEGORY)
    mobility_result_gnn_root = os.path.join(MOBILITY_RESULT_GNN_ROOT, CATEGORY)  # save
    os.makedirs(mobility_result_gnn_root, exist_ok=True)
    fnames = os.listdir(graphics_result_root)

    for i, fname in enumerate(fnames):
        fname = fname.split('.')[0]

        output_json = read_json(os.path.join(graphics_result_root, fname + '.json'))

        '''set data'''
        for key in output_json.keys():
            output_json[key]['move_part'] = ''

        # save_path = os.path.join(save_folder, 'mobility_v2.json')
        save_path = os.path.join(mobility_result_gnn_root, fname + '.json')
        output2motion(output_json, save_path)
        bar(f"3_Output2Mobility/{args.category}", i + 1, len(fnames))
