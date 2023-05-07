import argparse
import logging
import datetime
from pathlib import Path
from dataset_loader.structure_graph import PartNetDataset, collate_feats, Tree
from graph.Space_edge import Space_Category
from graph.graph_extraction import RelationshipGraph
import torch
import importlib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import json
from configs import DATA_ROOT_DICT

torch.backends.cudnn.enabled = False

# 解决num work过多， 原因是pytorch多线程共享tensor是通过打开文件的方式实现的，而打开文件的数量是有限制的
torch.multiprocessing.set_sharing_strategy('file_system')


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def test(model, loader, cuda, experiment_dir, epoch):
    correct = {}
    correct['cls_correct'] = 0
    correct['pos_error'] = 0
    correct['drt_error'] = 0
    correct['drt_correct'] = 0

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
                cor['drt_correct'] += np.sum(drt_error < 5) / len(drt_error)
                cor['drt_error'] += np.sum(drt_error) / len(drt_error)

        for key in correct.keys():
            correct[key] += (cor[key] / len(objects))

    for key in correct.keys():
        correct[key] /= len(loader)

    return correct


def main(args, category, data_root):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')) + '_' + category
    experiment_dir = Path(args.log_root)
    experiment_dir.mkdir(exist_ok=True)
    # experiment_dir = experiment_dir.joinpath('StructureNet')
    experiment_dir = experiment_dir.joinpath(category)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    model_dir = experiment_dir.joinpath('model/')
    model_dir.mkdir(exist_ok=True)
    picture_dir = experiment_dir.joinpath('picture/')
    picture_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # CATEGORY = 'chair'
    CATEGORY = category
    # DATA_PATH = '/mnt/disk2/sunqian/GNN_motion/dataset'

    '''Semantic'''
    Tree.load_semantic(data_root, CATEGORY)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    TRAIN_DATASET = PartNetDataset(phase='train', data_root=data_root,
                                   class_name=CATEGORY, points_batch_size=args.num_point, normalize=args.normalize)
    TEST_DATASET = PartNetDataset(phase='test', data_root=data_root,
                                  class_name=CATEGORY, points_batch_size=args.num_point, normalize=args.normalize)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=4, collate_fn=collate_feats)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
                                                 num_workers=4, collate_fn=collate_feats)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    cuda = True

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # shutil.copy('%s.py' % args.model, str(experiment_dir))
    # shutil.copy('pointnet2\\pointnet_util.py', str(experiment_dir))

    # classifier = MODEL.RecursiveEncoder(geo_feat_size=100, attribute_size = Tree.num_sem, node_feature_size=256, hidden_size=256,
    #                                  node_symmetric_type='max', edge_symmetric_type='avg', num_gnn_iterations=2,
    #                                  edge_types_size = Space_Category.st_num + Space_Category.dr_num,
    #                                  max_child_num = 11)
    classifier = MODEL.RecursiveEncoder(geo_feat_size=100, attribute_size=256, node_feature_size=256, hidden_size=256,
                                        node_symmetric_type='max', edge_symmetric_type='avg', num_gnn_iterations=2,
                                        edge_types_size=256,
                                        max_child_num=1000)
    cls_criterion = MODEL.get_classify_loss(args.weight)
    pos_criterion = MODEL.get_pos_loss()
    drt_criterion = MODEL.get_drt_loss()
    cls_consistancy = MODEL.get_consistency_loss()

    if len(str(args.gpu).split(',')) > 1:
        classifier = nn.DataParallel(classifier)

    if cuda:
        classifier = classifier.cuda()
        cls_criterion = cls_criterion.cuda()
        pos_criterion = pos_criterion.cuda()
        drt_criterion = drt_criterion.cuda()
        cls_consistancy = cls_consistancy.cuda()

    load_pre_model = False
    if load_pre_model == True:
        start_epoch = 0
        log_string('Use pretrain model')
        checkpoint = torch.load('proj_log/StructureNet/2022-04-08_12-31/model/300.pth')
        classifier.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, classifier.parameters()), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    global_step = 0
    accuracy = 0
    # best_instance_acc = 0.0
    # best_class_acc = 0.0
    # mean_correct = []
    draw_losses = {}
    draw_losses['cls_loss'] = []
    draw_losses['pos_loss'] = []
    draw_losses['drt_loss'] = []
    draw_losses['consistancy_loss'] = []

    draw_train_correct = {}
    draw_train_correct['cls_correct'] = []
    draw_train_correct['pos_error'] = []
    draw_train_correct['drt_error'] = []
    draw_train_correct['drt_correct'] = []

    draw_test_correct = {}
    draw_test_correct['cls_correct'] = []
    draw_test_correct['pos_error'] = []
    draw_test_correct['drt_error'] = []
    draw_test_correct['drt_correct'] = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        total_losses = {}
        total_losses['cls_loss'] = torch.zeros(1)
        total_losses['pos_loss'] = torch.zeros(1)
        total_losses['drt_loss'] = torch.zeros(1)
        total_losses['consistancy_loss'] = torch.zeros(1)
        if cuda:
            for key in total_losses.keys():
                total_losses[key] = total_losses[key].cuda()

        classifier = classifier.eval()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # data 是 list[(object1, object2, ......)]
            objects = data[0]

            total_loss = torch.zeros(1)
            loss = {}
            loss['cls_loss'] = torch.zeros(1)
            loss['pos_loss'] = torch.zeros(1)
            loss['drt_loss'] = torch.zeros(1)
            loss['consistancy_loss'] = torch.zeros(1)
            if cuda:
                total_loss = total_loss.cuda()
                for key in loss:
                    loss[key] = loss[key].cuda()

            # print(objects)
            for obj in objects:
                if cuda:
                    obj.to_cuda()
                # pred, target, e_mask, edge_forward, edge_backward = classifier(obj=obj)
                # pred_m_type = pred[:, :10]
                # pred_pos = pred[:, 10:13]
                # pred_drt = pred[:, 13:16]

                pred_m_type, pred_pos, pred_drt, target, e_mask, edge_forward, edge_backward = classifier(obj=obj)
                m_type = target[:, 0].long()
                gt_pos = target[:, 1:4]
                gt_drt = target[:, 4:7]

                cls_loss = cls_criterion(pred_m_type, m_type)
                pos_loss = pos_criterion(pred_pos[e_mask == 2], gt_pos[e_mask == 2], gt_drt[e_mask == 2])
                drt_loss = drt_criterion(pred_drt[e_mask != 0], gt_drt[e_mask != 0])

                # consistancy_loss = 100 * cls_consistancy(pred_m_type[edge_forward], pred_m_type[edge_backward])

                total_loss += cls_loss + pos_loss + drt_loss
                # total_loss += cls_loss + pos_loss + drt_loss + consistancy_loss

                loss['cls_loss'] += cls_loss.item()
                loss['pos_loss'] += pos_loss.item()
                loss['drt_loss'] += drt_loss.item()
                # loss['consistancy_loss'] += consistancy_loss.item()

            total_loss /= len(objects)

            total_loss.backward()

            optimizer.step()

            for key in total_losses:
                total_losses[key] += (loss[key] / len(objects))

            global_step += 1
        log_string("Losses:")
        for (key, value) in total_losses.items():
            loss = (value / len(trainDataLoader))[0].data.cpu().numpy()
            log_string(f"    {key}: {loss}")

            draw_losses[key].append(loss)

        scheduler.step()

        torch.save(classifier.state_dict(), f"{experiment_dir}/model/latest.pth")

        with torch.no_grad():
            correct = test(classifier.eval(), trainDataLoader, cuda, experiment_dir, epoch + 1)
            for key in draw_train_correct.keys():
                draw_train_correct[key].append(correct[key])

        with torch.no_grad():
            correct = test(classifier.eval(), testDataLoader, cuda, experiment_dir, epoch + 1)

            with open(f"{experiment_dir}/predict.txt", 'a') as f:
                f.write("epoch:{}".format(epoch))
                f.write('\n')
                for key in correct.keys():
                    f.write(str(key) + " : " + str(correct[key]))
                    f.write('\n')
                f.write('-------------------------------------------------------------------------')
                f.write('\n')
                f.write('\n')

            for key in draw_test_correct.keys():
                draw_test_correct[key].append(correct[key])
                log_string(f"{key} : {correct[key]}")
            if correct['cls_correct'] >= accuracy:
                accuracy = correct['cls_correct']
                torch.save(classifier.state_dict(), f"{experiment_dir}/model/best.pth")
                log_string(f"save:{epoch + 1}")
        global_epoch += 1

    logger.info('End of training...')


def save_fig(data, save_path, name):
    x = [i for i in range(len(data))]
    y = [data[i] for i in range(len(data))]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel(name)
    save_path = save_path + "/{}.png".format(name)
    plt.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='structure_net.model_struct',
                        help='model name [default: pointnet2 + graph model]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 5]')
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
    parser.add_argument('--normalize', default=True, help='Whether to normalize the point clouds.')
    parser.add_argument('--category', default="laptop", help='category to train.')
    return parser.parse_args()


if __name__ == '__main__':
    data_root = os.path.join(DATA_ROOT_DICT["partnet_mobility"], "network_data")
    args = parse_args()
    category = args.category
    args.log_root = "../models/gnn/"
    os.makedirs(args.log_root, exist_ok=True)
    with open(data_root + "/semantic_merge.json", 'r') as f:
        semantic_dict = json.load(f)
    weights = {
        # 'T_H', 'T_V', 'R_H_C', 'R_H_S', 'R_V_C', 'R_V_S', 'TR_H', 'TR_V', 'fixed', 'none'
        'faucet': torch.Tensor([10, 1, 5, 5, 10, 10, 1, 1, 1, 1]),
        'chair': torch.Tensor([1, 1, 10, 2, 10, 5, 5, 1, 1, 1]),
    }  # 类别权重
    args.weight = weights.get(category, torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])).cuda()
    print(category)
    main(args, category, data_root)
