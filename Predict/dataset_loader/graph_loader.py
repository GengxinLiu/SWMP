from locale import normalize
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json
import random
import time
import json
# import sys
# sys.path.append(os.getcwd())
from graph.graph_extraction import RelationshipGraph
import h5py as h5


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def direction_normal(direct):
    return direct / np.linalg.norm(direct, axis=1).reshape(len(direct), 1)


######################################################
class ModelNetDataLoader(Dataset):
    def __init__(self, phase, data_root, class_name='all', points_batch_size=1024):
        super(ModelNetDataLoader, self).__init__()
        self.phase = phase
        self.data_root = data_root
        self.class_name = class_name
        self.points_batch_size = points_batch_size
        with open(data_root + "/" + phase + ".json", 'r') as load_f:
            json_data = json.load(load_f)
        self.data_list = []
        if class_name == 'all':
            for category in json_data.keys():
                self.data_list += [os.path.join(category, data_id) for data_id in json_data[category]]
        else:
            self.data_list += [os.path.join(class_name, data_id) for data_id in json_data[class_name]]
        # print("end")

    def __getitem__(self, index):
        data_id = self.data_list[index]
        graph_path = os.path.join(self.data_root, "graph", data_id + ".pkl")
        pc_path = os.path.join(self.data_root, "pc_seg", data_id + ".h5")

        graph_data = RelationshipGraph()
        graph_data.load_from_file(graph_path)
        with h5.File(pc_path, 'r') as f:
            shape_pc = f['pc'][:]
            shape_pc_ids = f['seg'][:]
        
        nodes, edges, n_mask, e_mask, pc_lists, target = graph_data.get_numpy()

        data_points = np.zeros((len(nodes), self.points_batch_size, 3))
        for i in range(len(nodes)):
            pc_list = pc_lists[i]
            points = [shape_pc[shape_pc_ids == pc_id] for pc_id in pc_list]
            points = np.concatenate(points, axis=0)
            indices = np.arange(len(points))
            if len(points) > self.points_batch_size:
                random.shuffle(indices)
                # np.random.shuffle(indices)
                indices = indices[:self.points_batch_size]
            points = points[indices]
            points[:, 0:3] = pc_normalize(points[:, 0:3])
            data_points[i] = points

        target[e_mask!=0, 4:7] = direction_normal(target[e_mask!=0, 4:7])

        batch_nodes = torch.tensor(nodes).long()
        batch_edges = torch.tensor(edges).long()
        batch_n_mask = torch.tensor(n_mask).long()
        batch_e_mask = torch.tensor(e_mask).long()
        batch_points = torch.tensor(data_points, dtype=torch.float32)  # (points_batch_size, 3)
        batch_target = torch.tensor(target, dtype=torch.float32)

        return {"nodes":batch_nodes,
                "edges":batch_edges,
                # classify mask
                "n_mask":batch_n_mask,
                # motion mask
                "e_mask":batch_e_mask,
                "points": batch_points,
                "target":batch_target}

    def __len__(self):
        return len(self.data_list)

def graph_collate_fn(batch):
    all_nodes = []
    all_edges = []
    all_n_mask = []
    all_e_mask = []
    all_data_points = []
    all_target = []

    node_offset = 0
    all_nodes_to_graph = []
    all_edges_to_graph = []
    for i, data in enumerate(batch):
        node = data["nodes"]
        edge = data["edges"]
        n_mask = data["n_mask"]
        e_mask = data["e_mask"]
        points = data["points"]
        target = data["target"]
        if node.dim() == 0 or edge.dim() == 0:
            continue
        N, E = node.size(0), edge.size(0)
        all_nodes.append(node)

        edge[:, 0] += node_offset
        edge[:, -1] += node_offset
        all_edges.append(edge)
        all_nodes_to_graph.append(torch.LongTensor(N).fill_(i))
        all_edges_to_graph.append(torch.LongTensor(E).fill_(i))
        node_offset += N

        all_n_mask.append(n_mask)
        all_e_mask.append(e_mask)
        all_data_points.append(points)
        all_target.append(target)

    all_nodes = torch.cat(all_nodes)
    all_edges = torch.cat(all_edges)
    all_n_mask = torch.cat(all_n_mask)
    all_e_mask = torch.cat(all_e_mask)
    all_data_points = torch.cat(all_data_points)
    all_target = torch.cat(all_target)

    all_nodes_to_graph = torch.cat(all_nodes_to_graph)
    all_edges_to_graph = torch.cat(all_edges_to_graph)

    out = {"nodes":all_nodes,
           "edges":all_edges,
           "n_mask":all_n_mask,
           "e_mask":all_e_mask,
           "points":all_data_points,
           "target":all_target, 
           "nodes_to_graph": all_nodes_to_graph,
           "edges_to_graph": all_edges_to_graph}
    return out

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = ModelNetDataLoader("train", '/mnt/disk2/sunqian/GNN_motion/dataset')
    trainDataLoader = DataLoader(dataset, batch_size=3, shuffle=False,
                            num_workers=0, collate_fn=graph_collate_fn)
    print(len(trainDataLoader))
    # for batch_id, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
    #     print(data)

    for data in trainDataLoader:
        print("--------------------------")
        data_nodes = data["nodes"]
        print("data_nodes: ",data_nodes.shape)
        data_edges = data["edges"]
        print("data_edges: ",data_edges.shape)
        data_pc = data["points"]
        print("data_pc: ",data_pc.shape)
        target = data["target"]
        print("target: ", target.shape)

        n_mask = data["n_mask"]
        print("mask: ", n_mask.shape)
        print(target[n_mask == 0].shape)
        print(target[n_mask == 1].shape)

    pass
