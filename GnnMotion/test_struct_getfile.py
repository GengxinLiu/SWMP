import argparse
from ast import arg
from fileinput import filename
import logging
import os
import datetime
from pathlib import Path
from dataset_loader.structure_graph import PartNetDataset, collate_feats, Tree
from graph.Space_edge import Space_Category
from graph.graph_extraction import RelationshipGraph, motion
import torch
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math
import json
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='structure_net.model_struct',
                        help='model name [default: pointnet2 + graph model]')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch Size during training [default: 5]')
    parser.add_argument('--epoch', default=400, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(model, loader, cuda, category):
    correct = {}
    correct['cls_correct'] = 0
    correct['pos_error'] = 0
    correct['drt_error'] = 0
    correct['drt_correct'] = 0

    save_dir = "result/" + category + "/"
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    motion_dict = {}

    for k, data in tqdm(enumerate(loader), total=len(loader)):
        objects = data[0]

        cor = {}
        cor['cls_correct'] = 0
        cor['pos_error'] = 0
        cor['drt_error'] = 0
        cor['drt_correct'] = 0

        for obj in objects:
            if cuda:
                obj.to_cuda()
            # pred, target, e_mask, _, _ = model(obj=obj)
            pred_m_type, pred_pos, pred_drt, target, e_mask, _, _ = model(obj=obj)

            file_name = obj.root.graph_name
            file_name = file_name.replace('\\', '/').split('/')[1]
            node_name = obj.root.node_name
            node_child = obj.root.node_children
            edge_indices = obj.root.edge_indices
            node_isleaf = obj.root.node_isleaf

            # pred_m_type = pred[:, :10]
            # pred_pos = pred[:, 10:13]
            # pred_drt = pred[:, 13:16]

            m_type = target[:, 0].long()
            gt_pos = target[:, 1:4]
            gt_drt = target[:, 4:7]

            # classify motion type， 去除child边
            pred_probability = F.softmax(pred_m_type, dim=1)
            pred_m_type = torch.argmax(pred_m_type, dim=1)
            pred_m_type = pred_m_type.cpu().data.numpy()
            m_type = m_type.cpu().data.numpy()

            inf_dict = {}
            inf_dict[obj.root.part_id] = {}
            inf_dict[obj.root.part_id]["node_type"] = ""
            inf_dict[obj.root.part_id]["predict"] = []
            inf_dict[obj.root.part_id]["GroundTruth"] = []
            for name in node_name:
                inf_dict[name] = {}
                inf_dict[name]["node_type"] = ""
                inf_dict[name]["predict"] = []
                inf_dict[name]["GroundTruth"] = []

            inf_dict[obj.root.part_id]["node_type"] = "root"
            for name in node_isleaf.keys():
                if node_isleaf[name] == True:
                    inf_dict[name]["node_type"] = "leaf"

            inf_dict[obj.root.part_id]["children"] = []
            for child in obj.root.children:
                inf_dict[obj.root.part_id]["children"].append(child.part_id)
            for name in node_child.keys():
                inf_dict[name]["children"] = node_child[name]

            for i in range(len(edge_indices)):
                start_id = edge_indices[i][0]
                end_id = edge_indices[i][1]
                # print(start_id)
                # print(end_id)

                start_node_name = node_name[int(start_id)]
                end_node_name = node_name[int(end_id)]

                p_motion_inf = {}
                p_motion_inf[end_node_name] = {}
                p_motion_inf[end_node_name]["motion_type"] = motion().motion_type_8[pred_m_type[i]]
                p_motion_inf[end_node_name]["motion_probability"] = pred_probability[i].cpu().data.numpy().tolist()
                p_motion_inf[end_node_name]["motion_param"] = {}
                p_motion_inf[end_node_name]["motion_param"]["origin"] = pred_pos[i].cpu().data.numpy().tolist()
                p_motion_inf[end_node_name]["motion_param"]["direction"] = pred_drt[i].cpu().data.numpy().tolist()

                inf_dict[start_node_name]["predict"].append(p_motion_inf)

                t_motion_inf = {}
                t_motion_inf[end_node_name] = {}
                t_motion_inf[end_node_name]["motion_type"] = motion().motion_type_8[m_type[i]]
                t_motion_inf[end_node_name]["motion_param"] = {}
                t_motion_inf[end_node_name]["motion_param"]["origin"] = gt_pos[i].cpu().data.numpy().tolist()
                t_motion_inf[end_node_name]["motion_param"]["direction"] = gt_drt[i].cpu().data.numpy().tolist()

                inf_dict[start_node_name]["GroundTruth"].append(t_motion_inf)

            with open(save_dir + file_name + ".json", 'w') as f:
                json.dump(inf_dict, f)

            cor['cls_correct'] += np.sum(pred_m_type == m_type) / len(m_type)

            # motion position, 仅对R和TR计算
            m_pred_pos = pred_pos[e_mask == 2]
            m_gt_pos = gt_pos[e_mask == 2]
            m_gt_dir = gt_drt[e_mask == 2]
            pos_error = 0
            if len(m_pred_pos) > 0:
                pos_error = torch.linalg.norm(
                    torch.cross(m_pred_pos - m_gt_pos, m_gt_dir) / torch.linalg.norm(m_gt_dir, dim=1).view(-1, 1),
                    dim=1)
                pos_error = torch.sum(pos_error) / len(pos_error)
                cor['pos_error'] += pos_error.cpu().data.numpy()

            # motion direction
            p_drt = pred_drt[e_mask != 0].cpu().data.numpy()
            t_drt = gt_drt[e_mask != 0].cpu().data.numpy()
            drt_cos = np.sum(p_drt * t_drt, axis=1) / (np.linalg.norm(p_drt, axis=1) * np.linalg.norm(t_drt, axis=1))
            drt_error = 0
            if len(p_drt) > 0:
                drt_error = np.rad2deg(np.arccos(drt_cos))
                drt_error[drt_error > 90] = 180 - drt_error[drt_error > 90]
                cor['drt_correct'] += np.sum(drt_error < 15) / len(drt_error)
                cor['drt_error'] += np.sum(drt_error) / len(drt_error)

        for key in correct.keys():
            correct[key] += (cor[key] / len(objects))

    for key in correct.keys():
        correct[key] /= len(loader)
        print(key, " : ", correct[key])

    return correct


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_name = args.log_name
    CATEGORY = log_name[17:]

    # CATEGORY = 'Box'
    DATA_PATH = '../dataset'
    Tree.load_semantic(DATA_PATH, CATEGORY)

    TEST_DATASET = PartNetDataset(phase='test', data_root=DATA_PATH,
                                  class_name=CATEGORY, points_batch_size=args.num_point)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
                                                 num_workers=4, collate_fn=collate_feats)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # classifier = MODEL.RecursiveEncoder(geo_feat_size=100, attribute_size = Tree.num_sem, node_feature_size=256, hidden_size=256,
    #                                  node_symmetric_type='max', edge_symmetric_type='avg', num_gnn_iterations=2,
    #                                  edge_types_size = Space_Category.st_num + Space_Category.dr_num,
    #                                  max_child_num = 10)
    classifier = MODEL.RecursiveEncoder(geo_feat_size=100, attribute_size=256, node_feature_size=256, hidden_size=256,
                                        node_symmetric_type='max', edge_symmetric_type='avg', num_gnn_iterations=2,
                                        edge_types_size=256,
                                        max_child_num=1000)
    classifier.cuda()

    # checkpoint = torch.load('proj_log/StructureNet/2022-04-19_11-51_chair_embedding/model/latest.pth')
    checkpoint = torch.load(f'proj_log/{CATEGORY}/{log_name}/model/latest.pth')
    classifier.load_state_dict(checkpoint)

    with torch.no_grad():
        correct = test(classifier.eval(), testDataLoader, True, CATEGORY)


if __name__ == '__main__':
    args = parse_args()
    args.log_name = '2022-07-31_18-58_Safe'
    main(args)
