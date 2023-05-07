import argparse
from ast import arg
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

CLASS_LABELS = motion().motion_type_8  # 运动类型列表
N_CLASSES = len(CLASS_LABELS)


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids >= 0
    return np.bincount(pred_ids[idxs] * N_CLASSES + gt_ids[idxs], minlength=N_CLASSES ** 2).reshape(
        (N_CLASSES, N_CLASSES)).astype(
        np.ulonglong)


def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        # print(f'Not exist {CLASS_LABELS[label_id]}!!')
        return None
    return (float(tp) / denom, tp, denom)


def evaluate(pred_ids, gt_ids, verbose=True, save_log=True):
    """
    :params pred_ids: numpy array，预测标签
    :params gt_ids: numpy array，真实标签
    """
    if verbose:
        print(f'evaluating num {gt_ids.size} ...')
    confusion = confusion_matrix(pred_ids, gt_ids)
    class_ious = {}
    mean_iou = 0
    count_class = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        if class_ious[label_name] is not None:
            mean_iou += class_ious[label_name][0]
            count_class += 1
        else:
            class_ious[label_name] = (-1, -1, -1)

    mean_iou = mean_iou / count_class
    if verbose:
        print('classes          IoU\n----------------------------')
    res = {}
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if class_ious[label_name][0] == -1:
            continue
        if verbose:
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0],
                                                                   class_ious[label_name][1],
                                                                   class_ious[label_name][2]))
        res[label_name] = [class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]]
    if verbose:
        print('mean iou', mean_iou)
    return mean_iou, res


def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='structure_net.model_struct',
                        help='model name [default: pointnet2 + graph model]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 5]')
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


def test(model, loader, cuda):
    correct = {}
    correct['cls_correct'] = 0
    correct['drt_correct'] = 0
    correct['pos_error'] = 0
    correct['drt_error_mean'] = 0
    correct['drt_error_median'] = []
    y_pred, y_true = [], []
    for k, data in tqdm(enumerate(loader), total=len(loader)):
        objects = data[0]

        cor = {}
        cor['cls_correct'] = 0
        cor['drt_correct'] = 0
        cor['pos_error'] = 0
        cor['drt_error_mean'] = 0

        for obj in objects:
            if cuda:
                obj.to_cuda()
            # pred, target, e_mask, _, _ = model(obj=obj)
            # pred_m_type = pred[:, :10]
            # pred_pos = pred[:, 10:13]
            # pred_drt = pred[:, 13:16]
            pred_m_type, pred_pos, pred_drt, target, e_mask, _, _ = model(obj=obj)

            m_type = target[:, 0].long()
            gt_pos = target[:, 1:4]
            gt_drt = target[:, 4:7]

            # classify motion type， 去除child边
            pred_m_type = torch.argmax(pred_m_type, dim=1)
            pred_m_type = pred_m_type.cpu().data.numpy()
            m_type = m_type.cpu().data.numpy()
            # print("------------------")
            # print("predict: ", pred_m_type)
            # print("truth: ", m_type)
            cor['cls_correct'] += np.sum(pred_m_type == m_type) / len(m_type)
            y_pred.extend(pred_m_type)
            y_true.extend(m_type)
            # print("cls correct: ", np.sum(pred_m_type == m_type) / len(m_type))

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
                # print(drt_error)
                cor['drt_correct'] += np.sum(drt_error < 5) / len(drt_error)
                cor['drt_error_mean'] += np.sum(drt_error) / len(drt_error)

        for key in correct.keys():
            if key == 'drt_error_median':
                correct[key].append(cor['drt_error_mean'])
            else:
                correct[key] += (cor[key] / len(objects))

    for key in correct.keys():
        if key == 'drt_error_median':
            correct[key] = np.median(correct[key])
        else:
            correct[key] /= len(loader)
        print(key, correct[key])
    evaluate(np.array(y_pred), np.array(y_true))
    return correct


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_name = '2022-07-06_10-19_chair'

    # CATEGORY = 'Box'
    CATEGORY = log_name.split('_')[-1]
    # DATA_PATH = '/mnt/disk2/sunqian/GNN_motion/dataset'
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
                                        max_child_num=11)
    classifier.cuda()

    checkpoint = torch.load(f'proj_log/{CATEGORY}/{log_name}/model/latest.pth')
    classifier.load_state_dict(checkpoint)

    with torch.no_grad():
        correct = test(classifier.eval(), testDataLoader, True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
