from torch.utils.data import Dataset
import torch
import numpy as np
import json
import os
import h5py as h5
from mobility_transform import rotate_module
import matplotlib.pyplot as plt
import open3d as o3d


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


def rotate_part(pts, pointIndicator, joint_dir, joint_pos, angle):
    """
    rotate the moving part along (joint_dir, joint_pos) with angle.
    """
    part_pts = pts[pointIndicator == 1]
    part_pts -= joint_pos
    motion_quat = np.hstack(
        [np.cos(angle / 360 * 2 * 3.14 / 2), np.sin(angle / 360 * 2 * 3.14 / 2) * joint_dir])
    part_pts = \
        rotate_module(torch.from_numpy(part_pts).view(
            1, -1, 3), torch.from_numpy(motion_quat).view(1, 1, 4)).numpy()[0]
    part_pts += joint_pos
    pts[pointIndicator == 1] = part_pts
    return pts


def translate_part(pts, pointIndicator, joint_dir, amount):
    """
    translate the moving part along the translation axis with amount.
    """
    part_pts = pts[pointIndicator == 1]
    part_pts += (joint_dir * amount)
    pts[pointIndicator == 1] = part_pts
    return pts


def dist_between_3d_lines(p1, e1, p2, e2):
    """
    # p1,p2: pivot point
    # e1,e2: axis direction
    """
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    if np.linalg.norm(orth_vect) == 0:
        p22 = p2 + e2
        a = np.cross((p22 - p2), (p2 - p1))
        b = p22 - p2
        return np.linalg.norm(a) / np.linalg.norm(b)

    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)
    return np.abs(dist)


class PartNetMobilityDataset(Dataset):
    def __init__(self, phase, data_root, class_name='all', points_batch_size=4096, pair_positive=1.):
        super(PartNetMobilityDataset, self).__init__()
        self.phase = phase
        self.data_root = data_root
        self.class_name = class_name
        self.points_batch_size = points_batch_size
        with open(data_root + "/" + phase + ".struct.json", 'r') as load_f:
            json_data = json.load(load_f)
        self.data_list = []
        if class_name == 'all':
            for category in json_data.keys():
                self.data_list += [os.path.join(category, data_id)
                                   for data_id in json_data[category]]
        else:
            for data_id in json_data[class_name]:
                graphics_path = os.path.join(
                    self.data_root, "graphics", class_name, data_id + ".json")
                motion_data = json.load(open(graphics_path, 'r'))
                moveable_joint, moveable_obb = [], []
                for joint_id in motion_data:
                    for edge in motion_data[joint_id]["edges"]["space"]:
                        if motion_data[joint_id]["edges"]["space"][edge] == "motion":
                            moveable_joint.append(joint_id)
                            moveable_obb.append(motion_data[joint_id]['box'])
                            break
                if len(moveable_joint) == 0:
                    print(f"filter no motion instace {data_id}")
                    continue
                self.data_list += [os.path.join(class_name, data_id)]
        self.pair_positive = pair_positive
        self.part_frequency = {}

    def __getitem__(self, index):
        fn = self.data_list[index]
        # print(fn)
        # load point clouds
        pc_path = os.path.join(self.data_root, "pc_seg", fn + ".h5")
        graphics_path = os.path.join(
            self.data_root, "graphics", fn + ".json")
        with h5.File(pc_path, 'r') as f:
            shape_pc = f['pc'][:]
            shape_pc_ids = f['seg'][:]
        if len(shape_pc) > self.points_batch_size:
            sample_inds = np.random.choice(
                len(shape_pc), size=self.points_batch_size, replace=False)
        else:
            sample_inds = np.random.choice(
                len(shape_pc), size=self.points_batch_size, replace=True)
        shape_pc = shape_pc[sample_inds]
        shape_pc_ids = shape_pc_ids[sample_inds]
        # load motion parameters; load OBB, select edge to generate negative motion parameters
        motion_data = json.load(open(graphics_path, 'r'))
        moveable_joint, moveable_obb = [], []

        for joint_id in motion_data:
            for edge in motion_data[joint_id]["edges"]["space"]:
                if motion_data[joint_id]["edges"]["space"][edge] == "motion":
                    moveable_joint.append(joint_id)
                    moveable_obb.append(motion_data[joint_id]['box'])
                    break
        moveable_obb = np.asarray(moveable_obb)
        # random sample one joint
        if fn not in self.part_frequency:
            part_names = [motion_data[joint_id]['name'] for joint_id in moveable_joint]
            count = dict()
            for part in part_names:
                count[part] = count.get(part, 0) + 1
            self.part_frequency[fn] = {}
            for joint_id in moveable_joint:
                self.part_frequency[fn][joint_id] = count[motion_data[joint_id]["name"]]
        inv_freq = [1 / self.part_frequency[fn][joint_id] for joint_id in moveable_joint]
        probas = [inv / sum(inv_freq) for inv in inv_freq]
        ind = np.random.choice(len(moveable_joint), size=1, p=probas)[0]
        obb = moveable_obb[ind]
        joint_id = moveable_joint[ind]
        joint_type = motion_data[joint_id]['motype']
        joint_direction = np.asarray(
            motion_data[joint_id]['jointData']['axis']['direction'])
        joint_position = np.asarray(
            motion_data[joint_id]['jointData']['axis']['origin'])
        joint_limit = motion_data[joint_id]['jointData']['limit']

        A, B, C, D, E, F, G, H = obb

        def addLeaf(hierTree: dict, startNode: str, pointId: list):
            if len(hierTree[startNode]["children_id"]) == 0:
                pointId.append(startNode)
            else:
                for child in hierTree[startNode]["children_id"]:
                    addLeaf(hierTree, str(child), pointId)

        movePointIds = []
        addLeaf(motion_data, str(joint_id), movePointIds)
        for edge in motion_data[joint_id]["edges"]["space"]:
            if motion_data[joint_id]["edges"]["space"][edge] in ["fixed"]:
                addLeaf(motion_data, str(edge), movePointIds)

        pointIndicator = np.zeros(len(shape_pc_ids)).astype(int)
        for n in movePointIds:
            pointIndicator[shape_pc_ids == int(n)] = 1
        if sum(pointIndicator) == 0:
            print(f"empty node at {fn} when move {joint_id}")
        if 'TR' in joint_type:
            tAmount = np.random.uniform(
                low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
            # rAngle = np.random.uniform(low=80, high=180)
            rAngle = 90
            ##################################
            # generate positive point clouds #
            ##################################
            positive_shape_pc = rotate_part(shape_pc.copy(), pointIndicator, joint_direction, joint_position, rAngle)
            positive_shape_pc = translate_part(positive_shape_pc, pointIndicator, joint_direction, tAmount)
            ##################################
            # generate negative point clouds #
            ##################################
            candidate_direction = np.array([
                (A - B) / np.linalg.norm(A - B),
                (A - C) / np.linalg.norm(A - C),
                (A - D) / np.linalg.norm(A - D)
            ])
            candidate_position = np.array([
                (A + H) / 2, (A + F) / 2, (A + E) / 2
            ])
            drt_cos = abs(np.sum(joint_direction * candidate_direction /
                                 (np.linalg.norm(joint_direction)), axis=-1))
            min_ind = np.argsort(drt_cos)[-1]
            candidate_inds = list(range(len(candidate_direction)))
            candidate_inds.remove(min_ind)
            select_ind = np.random.choice(candidate_inds, size=1)

            negative_direction = candidate_direction[select_ind].reshape(-1, )
            negative_position = candidate_position[select_ind].reshape(-1, )
            negative_shape_pc = rotate_part(shape_pc.copy(), pointIndicator, negative_direction, negative_position,
                                            rAngle)
            negative_shape_pc = translate_part(negative_shape_pc, pointIndicator, negative_direction, tAmount)

            ### for debug
            # from debug import getObbEdgePoints
            #
            # obbPoints = getObbEdgePoints(obb)
            # A, B = candidate_position[select_ind], candidate_position[select_ind] + candidate_direction[select_ind]
            # x = np.tile(np.array(A - B).reshape(1, -1), (500, 1))
            # sample = B.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
            # new_shape_pc = np.vstack([shape_pc, sample])
            # new_shape_pc = np.vstack([new_shape_pc, obbPoints])
            #
            # labels = np.array([np.max(shape_pc_ids) + 1] * len(sample))
            # new_shape_pc_ids = np.hstack([shape_pc_ids, labels])
            # labels = np.array([np.max(new_shape_pc_ids) + 1] * len(obbPoints))
            # new_shape_pc_ids = np.hstack([new_shape_pc_ids, labels])
            # show_points(new_shape_pc, labels=new_shape_pc_ids, window_name='obb edge')

        elif 'R' in joint_type:
            if joint_limit["noLimit"]:
                angle = 90
            else:
                angle = np.random.uniform(
                    low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
            ##################################
            # generate positive point clouds #
            ##################################
            positive_shape_pc = rotate_part(shape_pc.copy(), pointIndicator, joint_direction, joint_position, angle)
            ##################################
            # generate negative point clouds #
            ##################################
            candidate_direction = np.array([
                (A - B) / np.linalg.norm(A - B), (C - E) / np.linalg.norm(C - E),
                (H - G) / np.linalg.norm(H - G), (D - F) / np.linalg.norm(D - F), (D - F) / np.linalg.norm(D - F),

                (B - E) / np.linalg.norm(B - E), (F - G) / np.linalg.norm(F - G),
                (A - C) / np.linalg.norm(A - C), (D - H) / np.linalg.norm(D - H), (D - H) / np.linalg.norm(D - H),

                (B - F) / np.linalg.norm(B - F), (E - G) / np.linalg.norm(E - G),
                (A - D) / np.linalg.norm(A - D), (C - H) / np.linalg.norm(C - H), (C - H) / np.linalg.norm(C - H)
            ])
            candidate_position = np.array([
                A, C, H, D, (D + C) / 2,
                B, F, A, D, (D + B) / 2,
                B, E, A, C, (C + B) / 2,
            ])
            drt_cos = abs(np.sum(joint_direction * candidate_direction /
                                 (np.linalg.norm(joint_direction)), axis=-1))
            # print('drt_cos\n', drt_cos)
            inds = np.argsort(drt_cos)[-5:]
            dis = 99
            min_ind = -1
            for ind in inds:
                _dis = dist_between_3d_lines(
                    candidate_position[ind], candidate_direction[ind], joint_position, joint_direction)
                # print(ind, _dis)
                if _dis < dis:
                    dis = _dis
                    min_ind = ind
            candidate_inds = list(range(len(candidate_direction)))
            candidate_inds.remove(min_ind)
            select_ind = np.random.choice(candidate_inds, size=1)

            negative_direction = candidate_direction[select_ind].reshape(-1, )
            negative_position = candidate_position[select_ind].reshape(-1, )
            negative_shape_pc = rotate_part(shape_pc.copy(), pointIndicator,
                                            negative_direction, negative_position, angle)
            # print('select direction', candidate_direction[min_ind])
            # print('joint_direction', joint_direction)
            # print('select position', candidate_position[min_ind])
            # print('joint_position', joint_position)

            ##################################
            # argument original point clouds #
            ##################################
            if np.random.rand() > self.pair_positive:
                if joint_limit["noLimit"]:
                    angle = np.random.uniform(low=-90, high=90)
                else:
                    angle = np.random.uniform(
                        low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
                shape_pc = rotate_part(shape_pc.copy(), pointIndicator, joint_direction, joint_position, angle)

        elif 'T' in joint_type:
            amount = np.random.uniform(
                low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
            ##################################
            # generate positive point clouds #
            ##################################
            positive_shape_pc = translate_part(shape_pc.copy(), pointIndicator, joint_direction, amount)
            ##################################
            # generate negative point clouds #
            ##################################
            candidate_direction = np.array([
                (A - B) / np.linalg.norm(A - B), (C - E) / np.linalg.norm(C - E),
                (H - G) / np.linalg.norm(H - G), (D - F) / np.linalg.norm(D - F),

                (B - E) / np.linalg.norm(B - E), (F - G) / np.linalg.norm(F - G),
                (A - C) / np.linalg.norm(A - C), (D - H) / np.linalg.norm(D - H),

                (B - F) / np.linalg.norm(B - F), (E - G) / np.linalg.norm(E - G),
                (A - D) / np.linalg.norm(A - D), (C - H) / np.linalg.norm(C - H)
            ])
            drt_cos = abs(np.sum(joint_direction * candidate_direction /
                                 (np.linalg.norm(joint_direction)), axis=-1))
            # print('drt_cos\n', drt_cos)
            inds = np.argsort(drt_cos)[-4:]
            candidate_inds = list(range(len(candidate_direction)))
            for ind in inds:
                candidate_inds.remove(ind)
            select_ind = np.random.choice(candidate_inds, size=1)
            negative_direction = candidate_direction[select_ind].reshape(-1, )
            negative_shape_pc = translate_part(shape_pc.copy(), pointIndicator, negative_direction, amount)

            ##################################
            # argument original point clouds #
            ##################################
            if np.random.rand() > self.pair_positive:
                amount = np.random.uniform(
                    low=min(joint_limit['a'], joint_limit['b']), high=max(joint_limit['a'], joint_limit['b']))
                shape_pc = translate_part(shape_pc.copy(), pointIndicator, joint_direction, amount)
        return shape_pc, positive_shape_pc, negative_shape_pc, shape_pc_ids, pointIndicator.reshape(-1, 1)

    def __len__(self):
        return len(self.data_list)
