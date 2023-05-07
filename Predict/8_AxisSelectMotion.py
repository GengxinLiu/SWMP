import json
import argparse
import os
import numpy as np
import h5py as h5
from utils import rotate_part, translate_part
import matplotlib.pyplot as plt
import open3d as o3d
from network.axis_selection_network import PointNet, DGCNN_cls, SiameseNet
import torch
import shutil
from itertools import product
import os
from configs import OBB_CANDIDATE_ROOT, SELECT_MOTION_ROOT, DATA_ROOT_DICT, MOBILITY_RESULT_GNN_ROOT, \
    AXIS_SELECTION_MODELS, mobility2partnet, MOBILITY_RESULT_AXIS_SELECT_ROOT
from utils import bar
import json


def show_points(coords, colors=None, labels=None, normals=None, window_name='base'):
    """
    Draw point clods
        :param coords: [N, 3]
        :param colors: [N, 3]
        :param labels: [N]
    """

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(coords)
    if colors is not None:
        if np.max(colors) > 20:  # 0-255
            colors /= 255.
        pc.colors = o3d.utility.Vector3dVector(colors)
    if labels is not None:
        label_set = np.unique(labels)
        cmap = plt.cm.get_cmap('jet', len(label_set))
        label_colors = np.zeros((len(labels), 3), dtype=float)
        for i, l in enumerate(label_set):
            idx = np.where(labels == l)
            label_colors[idx, :] = cmap(i)[:3]
        pc.colors = o3d.utility.Vector3dVector(label_colors)
    show_normals = False
    if normals is not None:
        show_normals = True
        pc.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries(
        [pc], point_show_normal=show_normals, width=500, height=400, window_name=window_name)


def main(args):
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    candidate_motion_root = os.path.join(OBB_CANDIDATE_ROOT, CATEGORY)
    h5_data_root = os.path.join(DATA_PATH, 'pc_seg', CATEGORY)
    files = os.listdir(candidate_motion_root)
    mobility_result_axis_select_root = os.path.join(MOBILITY_RESULT_AXIS_SELECT_ROOT, CATEGORY)
    os.makedirs(mobility_result_axis_select_root, exist_ok=True)
    # load axis selection model
    model_configs = json.load(open(os.path.join(args.model_root, 'args.json'), 'r'))
    if model_configs["model"] == 'pointnet':
        net = SiameseNet(PointNet(model_configs["embed_dim"]), model_configs["embed_dim"]).cuda()
    elif model_configs["model"] == 'dgcnn':
        net = SiameseNet(DGCNN_cls(model_configs["embed_dim"]), model_configs["embed_dim"] * 2).cuda()
    else:
        raise ValueError("Unknown model `{}`".format(model_configs["model"]))
    net.load_state_dict(torch.load(os.path.join(args.model_root, 'best.pth')))
    net.eval()

    for n, file in enumerate(files):
        # load point clouds
        with h5.File(os.path.join(h5_data_root, file + '.h5'), 'r') as f:
            shape_pc = f['pc'][:]
            shape_pc_ids = f['seg'][:]
        joint_root = os.path.join(OBB_CANDIDATE_ROOT, CATEGORY, file)
        bestjoint = {}  
        joint2score = {}  
        mobility_data = json.load(open(os.path.join(MOBILITY_RESULT_GNN_ROOT, CATEGORY, file + ".json"), "r"))
        for joint in os.listdir(joint_root):
            jointDataRoot = os.path.join(joint_root, joint)
            _joint2score = {}

            for path in os.listdir(jointDataRoot):
                # print(file, joint, path)
                if args.debug:
                    print(file, joint, path)
                predict_path = os.path.join(jointDataRoot, path)
                if 'json' not in predict_path:
                    continue
                joint_data = json.load(open(predict_path, 'r'))
                joint_type = joint_data["motype"]
                joint_direction = np.asarray(joint_data["direction"])
                joint_position = np.asarray(joint_data["origin"])
                score_list = []

                movePointIds = []
                for node in mobility_data:
                    if node["id"] == int(joint):
                        for part in node["parts"]:
                            movePointIds.append(part["id"])

                pointIndicator = np.zeros(len(shape_pc_ids)).astype(int)
                for pid in movePointIds:
                    pointIndicator[shape_pc_ids == int(pid)] = 1
                if 'TR' in joint_type:
                    angles = [90]
                    amounts = np.linspace(-0.023, 0.0257, 3)  
                    limits = product(angles, amounts)
                    for angle, amount in limits:
                        move_shape_pc = rotate_part(shape_pc.copy(), pointIndicator,
                                                    joint_direction, joint_position, angle)
                        move_shape_pc = translate_part(move_shape_pc, pointIndicator,
                                                       joint_direction, amount)
                        if len(shape_pc) > model_configs["num_point"]:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=False)
                        else:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=True)
                        input1 = np.hstack([shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        input2 = np.hstack([move_shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        # print(input1.shape, input2.shape)
                        score = net(
                            torch.from_numpy(input1).unsqueeze(0).permute(0, 2, 1).cuda().float(),
                            torch.from_numpy(input2).unsqueeze(0).permute(0, 2, 1).cuda().float())
                        ## debug
                        if args.debug:
                            print("TR", score[0])
                            show_points(move_shape_pc, labels=shape_pc_ids)
                        score_list.append(score[0].detach().cpu().numpy())

                elif 'R' in joint_type:
                    for angle in [-30, -45, -60, -80, 30, 45, 60, 80]:
                        move_shape_pc = rotate_part(shape_pc.copy(), pointIndicator,
                                                    joint_direction, joint_position, angle)
                        if len(shape_pc) > model_configs["num_point"]:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=False)
                        else:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=True)
                        input1 = np.hstack([shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        input2 = np.hstack([move_shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        score = net(
                            torch.from_numpy(input1).unsqueeze(0).permute(0, 2, 1).cuda().float(),
                            torch.from_numpy(input2).unsqueeze(0).permute(0, 2, 1).cuda().float())
                        if args.debug:
                            print("R", score[0])
                            show_points(move_shape_pc, labels=shape_pc_ids)
                        score_list.append(score[0].detach().cpu().numpy())
                elif 'T' in joint_type:
                    for amount in np.linspace(-0.5, 0.5, 10):
                        move_shape_pc = translate_part(shape_pc.copy(), pointIndicator,
                                                       joint_direction, amount)
                        if len(shape_pc) > model_configs["num_point"]:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=False)
                        else:
                            sample_inds = np.random.choice(
                                len(shape_pc), size=model_configs["num_point"], replace=True)
                        input1 = np.hstack([shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        input2 = np.hstack([move_shape_pc[sample_inds], pointIndicator[sample_inds].reshape(-1, 1)])
                        score = net(
                            torch.from_numpy(input1).unsqueeze(0).permute(0, 2, 1).cuda().float(),
                            torch.from_numpy(input2).unsqueeze(0).permute(0, 2, 1).cuda().float())
                        ### debug
                        if args.debug:
                            print("T", score[0])
                            show_points(move_shape_pc, labels=shape_pc_ids)
                        score_list.append(score[0].detach().cpu().numpy())
                else:
                    raise ValueError(f'Unknown motion type `{joint_type}`')
                _joint2score[path] = np.mean(score_list).tolist()
            best_path = sorted(_joint2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            bestjoint[joint] = best_path
            joint2score[joint] = _joint2score
        write_root = os.path.join(SELECT_MOTION_ROOT, CATEGORY, file)
        os.makedirs(write_root, exist_ok=True)
        json.dump(joint2score, open(os.path.join(write_root, 'score.json'), 'w'))
        motion_data = {}  

        for joint in bestjoint:
            shutil.copy(os.path.join(joint_root, joint, bestjoint[joint]),
                        os.path.join(write_root, joint + '.json'))
            motion_data[joint] = json.load(open(os.path.join(write_root, joint + '.json'), 'r'))
        json.dump(motion_data, open(os.path.join(write_root, 'motion.json'), 'w'))
        for node in mobility_data:
            if node["joint"] != "free":
                node["jointData"]["axis"]["direction"] = motion_data[str(node["id"])]["direction"]
                node["jointData"]["axis"]["origin"] = motion_data[str(node["id"])]["origin"]
        json.dump(mobility_data, open(os.path.join(mobility_result_axis_select_root, file + ".json"), "w"))
        bar(f'8_AxisSelectMotion/{CATEGORY}', n + 1, len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='laptop')
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    args.model_root = AXIS_SELECTION_MODELS[args.category]
    DATA_PATH = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    OBB_CANDIDATE_ROOT = os.path.join(f"result_{args.dataset}", OBB_CANDIDATE_ROOT)
    SELECT_MOTION_ROOT = os.path.join(f"result_{args.dataset}", SELECT_MOTION_ROOT)
    MOBILITY_RESULT_GNN_ROOT = os.path.join(f"result_{args.dataset}", MOBILITY_RESULT_GNN_ROOT)
    MOBILITY_RESULT_AXIS_SELECT_ROOT = os.path.join(f"result_{args.dataset}", MOBILITY_RESULT_AXIS_SELECT_ROOT)
    args.debug = False
    main(args)
