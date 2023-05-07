import json
from dataloader import PartNetMobilityDataset
import numpy as np
import torch
from network import PointNet, SiameseNet, DGCNN_cls
import argparse
import torch.nn.functional as F
import math
import sys
import os
from datetime import datetime
from configs import DATA_ROOT_DICT


def bar(message, now: int, total: int):
    """
    :param message: string to print.
    :param now: the i-th iteration.
    :param total: total iteration num.
    :return:
    """
    rate = now / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t\t%d/%d' % (message, "=" * rate_num,
                                      " " * (40 - rate_num), rate_nums, now, total)
    if now == total:
        r += "\n"
    sys.stdout.write(r)
    sys.stdout.flush()


def train(dataloader, model, optimizer, cur_epoch, total_epoch):
    model.train()
    res = {'acc': [], 'loss': []}
    for batch_id, data in enumerate(dataloader):
        optimizer.zero_grad()
        shape_pc, positive_shape_pc, negative_shape_pc, shape_pc_ids, pointIndicator = data
        shape_pc = torch.cat([shape_pc, pointIndicator], dim=-1)
        positive_shape_pc = torch.cat([positive_shape_pc, pointIndicator], dim=-1)
        negative_shape_pc = torch.cat([negative_shape_pc, pointIndicator], dim=-1)

        pts1 = torch.cat([shape_pc, shape_pc],
                         dim=0).cuda().permute(0, 2, 1).float()
        pts2 = torch.cat([positive_shape_pc, negative_shape_pc],
                         dim=0).cuda().permute(0, 2, 1).float()
        target = torch.zeros(len(pts1)).float().cuda()
        target[:len(shape_pc)] = 1
        predict = model(pts1, pts2)
        loss = F.binary_cross_entropy(predict, target)
        loss.backward()
        optimizer.step()
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        acc = np.mean(predict.detach().cpu().numpy() == target.cpu().numpy())
        res['loss'].append(loss.cpu().item())
        res['acc'].append(acc)
        bar('epoch {} / {} | loss {:.4f} acc {:.3f}'.format(cur_epoch, total_epoch, np.mean(res['loss']),
                                                            np.mean(res['acc'])),
            batch_id + 1, len(dataloader))
    res['loss'] = np.mean(res['loss'])
    res['acc'] = np.mean(res['acc'])
    return res


def test(dataloader, model):
    model.eval()
    res = {'acc': [], 'loss': []}
    for batch_id, data in enumerate(dataloader):
        shape_pc, positive_shape_pc, negative_shape_pc, shape_pc_ids, pointIndicator = data
        shape_pc = torch.cat([shape_pc, pointIndicator], dim=-1)
        positive_shape_pc = torch.cat([positive_shape_pc, pointIndicator], dim=-1)
        negative_shape_pc = torch.cat([negative_shape_pc, pointIndicator], dim=-1)
        pts1 = torch.cat([shape_pc, shape_pc],
                         dim=0).cuda().permute(0, 2, 1).float()
        pts2 = torch.cat([positive_shape_pc, negative_shape_pc],
                         dim=0).cuda().permute(0, 2, 1).float()
        target = torch.zeros(len(pts1)).float().cuda()
        target[:len(shape_pc)] = 1
        predict = model(pts1, pts2)
        loss = F.binary_cross_entropy(predict, target)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        acc = np.mean(predict.detach().cpu().numpy() == target.cpu().numpy())
        res['loss'].append(loss.cpu().item())
        res['acc'].append(acc)
        bar('eval | loss {:.4f} acc {:.3f}'.format(np.mean(res['loss']), np.mean(res['acc'])),
            batch_id + 1, len(dataloader))
    res['loss'] = np.mean(res['loss'])
    res['acc'] = np.mean(res['acc'])
    print()
    return res


def main(args):
    train_dataset = PartNetMobilityDataset(
        'train', args.data_root, args.category, args.num_point, pair_positive=args.pair_positive)
    test_dataset = PartNetMobilityDataset(
        'test', args.data_root, args.category, args.num_point)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=4)

    # define model
    if args.model == 'pointnet':
        net = SiameseNet(PointNet(args.embed_dim), args.embed_dim).cuda()
    if args.model == 'dgcnn':
        net = SiameseNet(DGCNN_cls(args.embed_dim), args.embed_dim * 2).cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    # create log file
    path_log = os.path.join(
        args.log_root, args.category, datetime.now().strftime('%Y_%m_%d_%H_%M'))
    os.makedirs(path_log, exist_ok=True)
    path_checkpoint = os.path.join(path_log, 'checkpoint.pth')
    path_trainlog = os.path.join(path_log, 'train.txt')
    path_testlog = os.path.join(path_log, 'test.txt')
    path_bestmodel = os.path.join(path_log, 'best.pth')
    path_lastmodel = os.path.join(path_log, 'latest.pth')
    path_args = os.path.join(path_log, 'args.json')
    json.dump(args.__dict__, open(path_args, 'w'))
    best_acc = 0.
    start_epoch = 1
    for epoch in range(start_epoch, args.epoch + 1):
        train_res = train(trainDataLoader, net, optimizer, epoch, args.epoch)
        scheduler.step()
        test_res = test(testDataLoader, net)
        print()
        torch.save({
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc
        }, path_checkpoint)
        if np.mean(test_res['acc']) > best_acc:
            best_acc = np.mean(test_res['acc'])
            torch.save(net.state_dict(), path_bestmodel)
        torch.save(net.state_dict(), path_lastmodel)
        with open(path_trainlog, 'a+') as f_train, open(path_testlog, 'a+') as f_test:
            f_train.write('epoch {} loss {:.4f} acc {:.3f}\n'.format(
                epoch, np.mean(train_res['loss']), np.mean(train_res['acc'])))
            f_test.write('epoch {} loss {:.4f} acc {:.3f}\n'.format(
                epoch, np.mean(test_res['loss']), np.mean(test_res['acc'])))
            f_train.close()
            f_test.close()


def parse_args():
    parser = argparse.ArgumentParser('Models')
    parser.add_argument('--model', type=str, default='dgcnn',
                        help='model name [dgcnn, pointnet]')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='backbone feature dimension')
    parser.add_argument('--category', type=str, default='laptop',
                        help='model name [default: pointnet2 + graph model]')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training [default: 5]')
    parser.add_argument('--epoch', default=4000, type=int,
                        help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=4096,
                        help='Point Number [default: 4096]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer for training [default: Adam]')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='load checkpoint')
    parser.add_argument('--pair_positive', type=float, default=0.5,
                        help="probability to use positive sample as pair input")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.data_root = os.path.join(DATA_ROOT_DICT["partnet_mobility"], "network_data")
    args.log_root = "../models/axis_selection"
    main(args)
