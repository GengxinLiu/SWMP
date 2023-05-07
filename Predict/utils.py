import h5py as h5
import json
import numpy as np
import torch
import math
import sys
import open3d as o3d
import matplotlib.pyplot as plt


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


def quat_conjugate(quat):
    # quat = quat.view(-1, 4)

    q0 = quat[:, :, 0]
    q1 = -1 * quat[:, :, 1]
    q2 = -1 * quat[:, :, 2]
    q3 = -1 * quat[:, :, 3]

    q_conj = torch.stack([q0, q1, q2, q3], dim=2)
    return q_conj


def hamilton_product(q1, q2):
    q_size = q1.size()
    # q1 = q1.view(-1, 4)
    # q2 = q2.view(-1, 4)
    inds = torch.LongTensor(
        [0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).view(4, 4)
    q1_q2_prods = []
    for i in range(4):
        # Hack to make 0 as positive sign. add 0.01 to all the values..
        q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
        q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

        q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
        q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

        q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
        q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

        q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
        q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
        q2_permute = torch.stack(
            [q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)

        # q1q2_v1 = torch.sum(q1 * q2_permute, dim=2)
        q1q2_v1 = torch.sum(q1 * q2_permute, dim=2,
                            keepdim=True)  
        q1_q2_prods.append(q1q2_v1)

    q_ham = torch.cat(q1_q2_prods, dim=2)
    # q_ham = q_ham.view(q_size)
    return q_ham


def quat_rot_module(points, quats):
    quatConjugate = quat_conjugate(quats)  
    # qvq^(-1)
    mult = hamilton_product(quats, points)
    mult = hamilton_product(mult, quatConjugate)
    return mult[:, :, 1:4]


# points is BxnPx3,  #Bx1x4 quat vectors
def rotate_module(points, quat):
    nP = points.size(1)
    quat_rep = quat.repeat(1, nP, 1)

    zero_points = 0 * points[:, :, 0].clone().view(-1, nP, 1)
    quat_points = torch.cat([zero_points, points], dim=2)

    rotated_points = quat_rot_module(quat_points, quat_rep)  # B x  P x 3
    return rotated_points


def transform(pc_path, motion_path, new_pc_path, new_motion_path):
    with h5.File(pc_path, 'r') as f:
        shape_pc = f['pc'][:]
        shape_pc_ids = f['seg'][:]

    motion_data = json.load(open(motion_path, 'r'))
    for part in motion_data:
        part_id = motion_data[part]['leaves']
        inds = []
        for _id in part_id:
            inds.extend(np.where(shape_pc_ids == _id)[0])
        inds = np.asarray(inds)
        if motion_data[part]['joint'] == 'slider':
            direction = np.asarray(
                motion_data[part]['jointData']['axis']['direction'])
            limit = motion_data[part]['jointData']['limit']  # a is the upper limit, b is the lower limit
            amount = np.random.uniform(
                low=min(limit['a'], limit['b']), high=max(limit['a'], limit['b']))
            # translate pts and box
            part_pts_box = np.vstack(
                [shape_pc[inds], np.array(motion_data[part]['box'])])
            part_pts_box += (direction * amount)
            shape_pc[inds] = part_pts_box[:-8]  # new shape_pc
            motion_data[part]['box'] = part_pts_box[-8:].tolist()  # new OBB

        if motion_data[part]['joint'] == 'hinge':
            direction = np.asarray(
                motion_data[part]['jointData']['axis']['direction'])
            position = np.asarray(
                motion_data[part]['jointData']['axis']['origin'])
            limit = motion_data[part]['jointData']['limit']  # a为上限，b为下限
            angle = np.random.uniform(
                low=min(limit['a'], limit['b']), high=max(limit['a'], limit['b']))
            # rotate pts and box
            part_pts_box = np.vstack(
                [shape_pc[inds], np.array(motion_data[part]['box'])])
            part_pts_box -= position
            motion_quat = np.hstack(
                [np.cos(angle / 360 * 2 * 3.14 / 2), np.sin(angle / 360 * 2 * 3.14 / 2) * direction])
            part_pts_box = \
                rotate_module(torch.from_numpy(part_pts_box).view(
                    1, -1, 3), torch.from_numpy(motion_quat).view(1, 1, 4)).numpy()[0]
            part_pts_box += position

            shape_pc[inds] = part_pts_box[:-8]  # new shape_pc
            motion_data[part]['box'] = part_pts_box[-8:].tolist()  # new OBB

    json.dump(motion_data, open(new_motion_path, 'w'))
    with h5.File(new_pc_path, 'w') as f:
        f.create_dataset('pc', data=shape_pc)
        f.create_dataset('seg', data=shape_pc_ids)


def rotate_part(pts, pointIndicator, joint_dir, joint_pos, angle):
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
    part_pts = pts[pointIndicator == 1]
    part_pts += (joint_dir * amount)
    pts[pointIndicator == 1] = part_pts
    return pts


############## visualization tools ##############
def custom_draw_geometry(obj_list, width=600, height=400, camera_parameters=None, save_path=''):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for obj in obj_list:
        vis.add_geometry(obj)
    if camera_parameters is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameters)
    for obj in obj_list:
        vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()


def create_o3d_pointclouds(coords, labels):
    """
    Draw point clods
        :param coords: [N, 3]
        :param labels: [N]
    """

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(coords)
    label_set = np.unique(labels)
    cmap = plt.cm.get_cmap('jet', len(label_set))
    label_colors = np.zeros((len(labels), 3), dtype=float)
    for i, l in enumerate(label_set):
        idx = np.where(labels == l)
        # TODO: change color
        if l == 100 or l == 100:
            label_colors[idx, :] = [0, 0, 0]
        else:
            label_colors[idx, :] = cmap(i)[:3]
            # if l == 24:
            #     label_colors[idx, :] = [255, 255, 255]
    pc.colors = o3d.utility.Vector3dVector(label_colors)
    return pc


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def get_arrow(begin=[0, 0, 0], end=[0, 0, 1]):
    begin = begin
    vec_Arr = np.array(end) - np.array(begin)

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * 1,
        # cone_radius=0.06 * 1,
        cone_radius=0.03 * 1,
        cylinder_height=0.8 * 1,
        # cylinder_radius=0.04 * 1
        cylinder_radius=0.01 * 1
    )
    mesh_arrow.paint_uniform_color([0, 1, 0])
    mesh_arrow.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow


def get_obb_pts(obb):
    obb_pts = []
    A, B, C, D, E, F, G, H = np.array(obb)
    # AB
    x = np.tile(np.array(A - B).reshape(1, -1), (500, 1))
    sample = B.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # AC
    x = np.tile(np.array(A - C).reshape(1, -1), (500, 1))
    sample = C.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # AD
    x = np.tile(np.array(A - D).reshape(1, -1), (500, 1))
    sample = D.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # DH
    x = np.tile(np.array(D - H).reshape(1, -1), (500, 1))
    sample = H.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # DF
    x = np.tile(np.array(D - F).reshape(1, -1), (500, 1))
    sample = F.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # FB
    x = np.tile(np.array(F - B).reshape(1, -1), (500, 1))
    sample = B.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # FG
    x = np.tile(np.array(F - G).reshape(1, -1), (500, 1))
    sample = G.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # BE
    x = np.tile(np.array(B - E).reshape(1, -1), (500, 1))
    sample = E.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # GE
    x = np.tile(np.array(G - E).reshape(1, -1), (500, 1))
    sample = E.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # EC
    x = np.tile(np.array(E - C).reshape(1, -1), (500, 1))
    sample = C.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # CH
    x = np.tile(np.array(C - H).reshape(1, -1), (500, 1))
    sample = H.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    # GH
    x = np.tile(np.array(G - H).reshape(1, -1), (500, 1))
    sample = H.reshape(1, -1) + np.random.rand(500).reshape(-1, 1) * x
    obb_pts.extend(sample)
    return np.asarray(obb_pts)


def show_points(coords, colors=None, labels=None, normals=None):
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
    o3d.visualization.draw_geometries([pc], point_show_normal=show_normals, width=500, height=400, window_name='base')
