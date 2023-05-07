import argparse
import os
from dataset_loader.structure_graph import PartNetDataset, collate_feats, Tree
from graph.Space_edge import Space_Category
from graph.graph_extraction import RelationshipGraph, motion
import torch
import importlib
import json
import torch.nn.functional as F
from configs import PREDICT_NETWORK_MOTION_ROOT, GT_MOTION_ROOT, \
    DATA_ROOT_DICT, DATA_MODE, GNN_NUM_POINTS, GPU, GNN_MODELS, mobility2partnet
from utils import bar


def test(model, loader, cuda, category):
    # todo set save dir
    save_network_motion_dir = f"{PREDICT_NETWORK_MOTION_ROOT}/" + category + "/"
    os.makedirs(save_network_motion_dir, exist_ok=True)
    save_gt_motion_dir = f"{GT_MOTION_ROOT}/" + category + "/"
    os.makedirs(save_gt_motion_dir, exist_ok=True)

    for k, data in enumerate(loader):
        objects = data[0]
        for obj in objects:
            if cuda:
                obj.to_cuda()
            file_name = obj.root.graph_name
            file_name = file_name.replace('\\', '/').split('/')[1]
            pred_m_type, pred_pos, pred_drt, target, e_mask, _, _ = model(obj=obj)
            if pred_m_type is None:
                print(f"{file_name} has no motion.")
                continue  
            node_name = obj.root.node_name
            node_child = obj.root.node_children
            edge_indices = obj.root.edge_indices
            node_isleaf = obj.root.node_isleaf

            m_type = target[:, 0].long()
            gt_pos = target[:, 1:4]
            gt_drt = target[:, 4:7]

            # classify motion type
            pred_probability = F.softmax(pred_m_type, dim=1)
            pred_m_type = torch.argmax(pred_m_type, dim=1)
            pred_m_type = pred_m_type.cpu().data.numpy()
            m_type = m_type.cpu().data.numpy()

            inf_dict = {}
            inf_dict[obj.root.part_id] = {}
            inf_dict[obj.root.part_id]["node_type"] = ""
            inf_dict[obj.root.part_id]["predict"] = {}
            inf_dict[obj.root.part_id]["GroundTruth"] = {}
            for name in node_name:
                inf_dict[name] = {}
                inf_dict[name]["node_type"] = ""
                inf_dict[name]["predict"] = {}
                inf_dict[name]["GroundTruth"] = {}

            inf_dict[obj.root.part_id]["node_type"] = "root"
            for name in node_isleaf.keys():
                if node_isleaf[name]:
                    inf_dict[name]["node_type"] = "leaf"

            inf_dict[obj.root.part_id]["children"] = []
            for child in obj.root.children:
                inf_dict[obj.root.part_id]["children"].append(child.part_id)
            for name in node_child.keys():
                inf_dict[name]["children"] = node_child[name]

            for i in range(len(edge_indices)):
                start_id = edge_indices[i][0]
                end_id = edge_indices[i][1]

                start_node_name = node_name[int(start_id)]
                end_node_name = node_name[int(end_id)]

                inf_dict[start_node_name]["predict"][end_node_name] = {
                    "motion_type": motion().motion_type_8[pred_m_type[i]],
                    "motion_probability": pred_probability[i].cpu().data.numpy().tolist(),
                    "motion_param": {
                        "origin": pred_pos[i].cpu().data.numpy().tolist(),
                        "direction": pred_drt[i].cpu().data.numpy().tolist()
                    }
                }
                inf_dict[start_node_name]["GroundTruth"][end_node_name] = {
                    "motion_type": motion().motion_type_8[m_type[i]],
                    "motion_param": {
                        "origin": gt_pos[i].cpu().data.numpy().tolist(),
                        "direction": gt_drt[i].cpu().data.numpy().tolist()
                    }
                }

            ##### save network motioin visualization #####
            save_network_motion_file_dir = os.path.join(save_network_motion_dir, file_name)  
            save_gt_motion_file_dir = os.path.join(save_gt_motion_dir, file_name)  
            os.makedirs(save_network_motion_file_dir, exist_ok=True)
            os.makedirs(save_gt_motion_file_dir, exist_ok=True)
            for node in inf_dict.keys():
                if len(inf_dict[node]["predict"]) == 0:
                    continue
                for ref_node in inf_dict[node]["predict"]:
                    joint_name = node + "-" + ref_node
                    # todo save predicted motion parameters
                    joint_motion = {
                        "motype": inf_dict[node]["predict"][ref_node]["motion_type"],
                        "motion_probability": inf_dict[node]["predict"][ref_node]["motion_probability"],
                        "direction": inf_dict[node]["predict"][ref_node]["motion_param"]["direction"],
                        "origin": inf_dict[node]["predict"][ref_node]["motion_param"]["origin"],
                    }
                    with open(save_network_motion_file_dir + '/' + joint_name + ".json", 'w') as f:
                        json.dump(joint_motion, f)
                    # todo save ground truth motion parameters
                    joint_motion = {
                        "motype": inf_dict[node]["GroundTruth"][ref_node]["motion_type"],
                        "direction": inf_dict[node]["GroundTruth"][ref_node]["motion_param"]["direction"],
                        "origin": inf_dict[node]["GroundTruth"][ref_node]["motion_param"]["origin"],
                    }
                    with open(save_gt_motion_file_dir + '/' + joint_name + ".json", 'w') as f:
                        json.dump(joint_motion, f)

        bar(f"1_PredictMotionType/{category}", k + 1, len(loader))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    Tree.load_semantic(DATA_PATH, args.category)

    TEST_DATASET = PartNetDataset(phase=DATA_MODE, data_root=DATA_PATH,
                                  class_name=CATEGORY, points_batch_size=GNN_NUM_POINTS)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
                                                 num_workers=4, collate_fn=collate_feats)

    '''MODEL LOADING'''
    MODEL = importlib.import_module("structure_net.model_struct")
    classifier = MODEL.RecursiveEncoder(geo_feat_size=100, attribute_size=256, node_feature_size=256, hidden_size=256,
                                        node_symmetric_type='max', edge_symmetric_type='avg', num_gnn_iterations=2,
                                        edge_types_size=256, max_child_num=1000)
    classifier.cuda()

    checkpoint = torch.load(args.model_path)
    classifier.load_state_dict(checkpoint)

    with torch.no_grad():
        test(classifier.eval(), testDataLoader, GPU != "", CATEGORY)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='bottle')
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    args.model_root = GNN_MODELS[args.category]
    args.model_path = os.path.join(args.model_root, 'model/best.pth')
    DATA_PATH = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    PREDICT_NETWORK_MOTION_ROOT = os.path.join(f"result_{args.dataset}", PREDICT_NETWORK_MOTION_ROOT)
    GT_MOTION_ROOT = os.path.join(f"result_{args.dataset}", GT_MOTION_ROOT)
    main(args)
