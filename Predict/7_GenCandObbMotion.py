import json
import argparse
import os
import numpy as np
from configs import OBB_CANDIDATE_ROOT, MOBILITY_RESULT_GNN_ROOT, PROCESS_OBB_RESULT_ROOT, DATA_ROOT_DICT, \
    mobility2partnet, NUM_SAMPLE, TOPK
from utils import bar
from scipy.spatial.distance import cdist
import h5py as h5


def computeIR(part_id1, part_id2, pc, seg, topk=20, sample=800):
    """
    :param part_id1: id of the move part
    :param part_id2: id of the reference part
    :param pc, seg: point clouds and the segmentation
    :param topk: take the top k closest points to calculate the interaction region
    :return: Average IR position (3, )
    """

    def computeIRIndex(X, Y, topk=20):
        """ compute interaction region of two point set
        :param X: [N, 3]
        :param Y: [M, 3]
        :return: inds of size topk
        """
        size = int(topk / 2)
        distanceX = cdist(X, Y, metric='euclidean').min(axis=-1)
        distanceY = cdist(Y, X, metric='euclidean').min(axis=-1)
        return np.argsort(distanceX)[:size], np.argsort(distanceY)[:size]

    pointIndicator = np.zeros(len(pc)).astype(int)
    for pid in part_id1:
        pointIndicator[seg == int(pid)] = 1
    part1 = pc[pointIndicator == 1]
    # get part2
    for pid in part_id2:
        pointIndicator[seg == int(pid)] = 1
    part2 = pc[pointIndicator == 1]
    if sample > 0:
        inds = np.random.choice(len(part1), size=sample, replace=sample > len(part1))
        part1 = part1[inds]
        inds = np.random.choice(len(part2), size=sample, replace=sample > len(part2))
        part2 = part2[inds]
    index1, index2 = computeIRIndex(part1, part2, topk)
    region1, region2 = part1[index1], part2[index2]
    return np.mean(np.vstack([region1, region2]), axis=0)


def main(args):
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    files = os.listdir(os.path.join(MOBILITY_RESULT_GNN_ROOT, CATEGORY))
    for i, file in enumerate(files):
        file = file.split('.')[0]
        mobility_data = json.load(open(os.path.join(MOBILITY_RESULT_GNN_ROOT, CATEGORY, file + ".json"), "r"))
        for node in mobility_data:
            motion_type = node["joint"]
            if motion_type == "free":
                continue
            # point_ids = [part["id"] for part in node["parts"]]
            obb = np.loadtxt(os.path.join(PROCESS_OBB_RESULT_ROOT, "{}_{}_box".format(file, node["id"])))
            motion_splits = motion_type.split('_')
            A, B, C, D, E, F, G, H = obb
            if motion_splits[0] == 'R':
                edge_direction = np.array([
                    (A - B) / np.linalg.norm(A - B), (C - E) / np.linalg.norm(C - E),
                    (H - G) / np.linalg.norm(H - G), (D - F) / np.linalg.norm(D - F),

                    (B - E) / np.linalg.norm(B - E), (F - G) / np.linalg.norm(F - G),
                    (A - C) / np.linalg.norm(A - C), (D - H) / np.linalg.norm(D - H),

                    (B - F) / np.linalg.norm(B - F), (E - G) / np.linalg.norm(E - G),
                    (A - D) / np.linalg.norm(A - D), (C - H) / np.linalg.norm(C - H)
                ])
                edge_position = np.array([
                    (A + B) / 2, (C + E) / 2, (H + G) / 2, (D + F) / 2,
                    (B + E) / 2, (F + G) / 2, (A + C) / 2, (D + H) / 2,
                    (B + F) / 2, (E + G) / 2, (A + D) / 2, (C + H) / 2
                ])
                cos_theta = np.array([0, 0, 1]).reshape((1, 3)) * edge_direction
                cos_theta = abs(cos_theta.sum(-1))
                inds = np.argsort(cos_theta)  
                if motion_splits[1] == 'H':
                    select_inds = inds[:8]
                    select_direction = edge_direction[select_inds]
                    # joint position
                    if motion_splits[2] == 'S':
                        select_position = edge_position[select_inds]
                    if motion_splits[2] == 'C':
                        select_direction = np.vstack([select_direction[0], select_direction[4]])
                        select_edge_position = edge_position[select_inds]
                        select_position = np.vstack(
                            [np.mean(select_edge_position[:4], axis=0), np.mean(select_edge_position[4:], axis=0)])
                elif motion_splits[1] == 'V':
                    select_inds = inds[-4:]
                    select_direction = edge_direction[select_inds]
                    # joint position
                    if motion_splits[2] == 'S':
                        select_position = edge_position[select_inds]
                    if motion_splits[2] == 'C':
                        select_direction = select_direction[0].reshape(1, -1)
                        select_position = np.mean(edge_position[select_inds], axis=0).reshape(1, -1)
                else:
                    raise ValueError('Unknown motion type {}'.format(motion_type))

                path = f'{OBB_CANDIDATE_ROOT}/{CATEGORY}/{file}/{node["id"]}/'
                for n in range(len(select_direction)):
                    os.makedirs(path, exist_ok=True)
                    json.dump({'motype': motion_type,
                               'direction': select_direction[n].tolist(),
                               'origin': select_position[n].tolist()},
                              open(f'{path}/{n}.json', 'w'))
                pcsegPath = os.path.join(DATA_PATH, 'pc_seg', CATEGORY, file + '.h5')
                with h5.File(pcsegPath) as f:
                    pc = f['pc'][:]
                    seg = f['seg'][:]
                move_ids1, ref_ids2 = [], []
                for part in node["parts"]:
                    move_ids1.append(part["id"])
                for ref_node in mobility_data:
                    if ref_node["id"] == node["parent"]:
                        for part in ref_node["parts"]:
                            ref_ids2.append(part["id"])
                ir = computeIR(move_ids1, ref_ids2, pc, seg, topk=TOPK, sample=NUM_SAMPLE)

                if len(select_direction) == 8:
                    irDirection = np.vstack([select_direction[0], select_direction[4]])
                    irPosition = np.vstack([ir, ir])
                elif len(select_direction) == 2:
                    irDirection = select_direction
                    irPosition = np.vstack([ir, ir])
                elif len(select_direction) == 4:
                    irDirection = [select_direction[0]]
                    irPosition = [ir]
                elif len(select_direction) == 1:
                    irDirection = select_direction
                    irPosition = [ir]
                else:
                    raise ValueError(f"Occur {len(select_direction)} at R, ERROR case!!")
                for n in range(len(irDirection)):
                    os.makedirs(path, exist_ok=True)
                    json.dump({'motype': motion_type,
                               'direction': irDirection[n].tolist(),
                               'origin': irPosition[n].tolist()},
                              open(f'{path}/ir_{n}.json', 'w'))

            elif motion_splits[0] == 'T':
                edge_direction = np.array([
                    (A - B) / np.linalg.norm(A - B),
                    (A - C) / np.linalg.norm(A - C),
                    (A - D) / np.linalg.norm(A - D)
                ])
                cos_theta = np.array([0, 0, 1]).reshape((1, 3)) * edge_direction
                cos_theta = abs(cos_theta.sum(-1))
                inds = np.argsort(cos_theta)  

                if motion_splits[1] == 'H':
                    select_inds = inds[:2]
                    select_direction = edge_direction[select_inds]
                elif motion_splits[1] == 'V':
                    select_inds = inds[-1]
                    select_direction = edge_direction[select_inds].reshape(1, -1)
                else:
                    raise ValueError('Unknown motion type {}'.format(motion_type))
                path = f'{OBB_CANDIDATE_ROOT}/{CATEGORY}/{file}/{node["id"]}/'
                for n in range(len(select_direction)):
                    os.makedirs(path, exist_ok=True)
                    json.dump({'motype': motion_type,
                               'direction': select_direction[n].tolist(),
                               'origin': [0, 0, 0]},
                              open(f'{path}/{n}.json', 'w'))
            elif motion_splits[0] == 'TR':
                edge_direction = np.array([
                    (B - A) / np.linalg.norm(B - A),
                    (C - A) / np.linalg.norm(C - A),
                    (D - A) / np.linalg.norm(D - A)
                ])
                edge_position = np.array([
                    (A + H) / 2, (A + F) / 2, (A + E) / 2
                ])
                cos_theta = np.array([0, 0, 1]).reshape((1, 3)) * edge_direction
                cos_theta = abs(cos_theta.sum(-1))
                inds = np.argsort(cos_theta)  
                if motion_splits[1] == 'H':
                    select_inds = inds[:2]
                    select_direction = edge_direction[select_inds]
                    select_position = edge_position[select_inds]
                    
                elif motion_splits[1] == 'V':
                    select_inds = inds[-1]
                    select_direction = edge_direction[select_inds].reshape(1, -1)
                    select_position = edge_position[select_inds].reshape(1, -1)
                    # mergeDirection = np.vstack(
                    #     [select_direction, select_direction[0]])  # side + central
                    # mergePosition = np.vstack(
                    #     [edge_position[select_inds], np.mean(edge_position[select_inds], axis=0)])  # side + central
                else:
                    raise ValueError('Unknown motion type {}'.format(motion_type))

                path = f'{OBB_CANDIDATE_ROOT}/{CATEGORY}/{file}/{node["id"]}/'
                for n in range(len(select_position)):
                    os.makedirs(path, exist_ok=True)
                    json.dump({'motype': motion_type,
                               'direction': select_direction[n].tolist(),
                               'origin': select_position[n].tolist()},
                              open(f'{path}/{n}.json', 'w'))

                pcsegPath = os.path.join(DATA_PATH, 'pc_seg', CATEGORY, file + '.h5')
                with h5.File(pcsegPath) as f:
                    pc = f['pc'][:]
                    seg = f['seg'][:]
                move_ids1, ref_ids2 = [], []
                for part in node["parts"]:
                    move_ids1.append(part["id"])
                for ref_node in mobility_data:
                    if ref_node["id"] == node["parent"]:
                        for part in ref_node["parts"]:
                            ref_ids2.append(part["id"])
                ir = computeIR(move_ids1, ref_ids2, pc, seg, topk=TOPK, sample=NUM_SAMPLE)
                if len(select_direction) == 2:
                    irDirection = select_direction
                    irPosition = np.vstack([ir, ir])
                elif len(select_direction) == 1:
                    irDirection = select_direction
                    irPosition = [ir]
                else:
                    raise ValueError(f"Occur {len(select_direction)} at TR, ERROR case!!")
                for n in range(len(irDirection)):
                    os.makedirs(path, exist_ok=True)
                    json.dump({'motype': motion_type,
                               'direction': irDirection[n].tolist(),
                               'origin': irPosition[n].tolist()},
                              open(f'{path}/ir_{n}.json', 'w'))

        bar(f"7_GenCandObbMotion/{CATEGORY}", i + 1, len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='laptop')
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    MOBILITY_RESULT_GNN_ROOT = os.path.join(f"result_{args.dataset}", MOBILITY_RESULT_GNN_ROOT)
    OBB_CANDIDATE_ROOT = os.path.join(f"result_{args.dataset}", OBB_CANDIDATE_ROOT)
    PROCESS_OBB_RESULT_ROOT = os.path.join(f"result_{args.dataset}", PROCESS_OBB_RESULT_ROOT)
    DATA_PATH = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    main(args)
