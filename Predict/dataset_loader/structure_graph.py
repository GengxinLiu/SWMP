from cProfile import label
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
from graph.graph_extraction import RelationshipGraph, motion
import h5py as h5
from collections import namedtuple
from dataset_loader.utils import one_hot
from graph.Space_edge import Space_Category


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def n_direction_normal(direct):
    return direct / np.linalg.norm(direct, axis=1).reshape(len(direct), 1)


def direction_normal(direct):
    if np.linalg.norm(direct) == 0.0:
        return direct
    return direct / np.linalg.norm(direct)


class Tree(object):
    # global object category information
    part_name2id = dict()
    part_id2name = dict()
    part_name2cids = dict()
    part_non_leaf_sem_names = []
    num_sem = 0
    semantic_list = []
    is_cuda = False

    # store a part node in the tree
    class Node(object):

        def __init__(self, part_id=0, is_leaf=False, box=None, label=None, children=None, edges=None, full_label=None,
                     geo=None, feat=None, geo_feat=None, graph_name=None):
            self.is_leaf = is_leaf  # store True if the part is a leaf node
            self.part_id = part_id  # part_id in result_after_merging.json of PartNet
            self.box = box  # box parameter for all nodes
            self.geo = geo  # 1 x 1000 x 3 point cloud
            self.geo_feat = geo_feat  # 1 x 100 geometry feature
            self.node_feat = feat
            self.label = label  # node semantic label at the current level
            self.full_label = full_label  # node semantic label from root (separated by slash)
            self.children = [] if children is None else children
            # all of its children nodes; each entry is a Node instance
            self.edges = [] if edges is None else edges
            # all of its children relationships;
            # each entry is a tuple <part_a, part_b, type, params, dist>
            self.graph_name = graph_name

            self.is_cuda = False
            """
                Here defines the edges format:
                    part_a, part_b:
                        Values are the order in self.children (e.g. 0, 1, 2, 3, ...).
                        This is an directional edge for A->B.
                        If an edge is commutative, you may need to manually specify a B->A edge.
                        For example, an ADJ edge is only shown A->B, 
                        there is no edge B->A in the json file.
                    type:
                        Four types considered in StructureNet: ADJ, ROT_SYM, TRANS_SYM, REF_SYM.
                    params:
                        There is no params field for ADJ edge;
                        For ROT_SYM edge, 0-2 pivot point, 3-5 axis unit direction, 6 radian rotation angle;
                        For TRANS_SYM edge, 0-2 translation vector;
                        For REF_SYM edge, 0-2 the middle point of the segment that connects the two box centers, 
                            3-5 unit normal direction of the reflection plane.
                    dist:
                        For ADJ edge, it's the closest distance between two parts;
                        For SYM edge, it's the chamfer distance after matching part B to part A.
            """

        def get_semantic_class(self):
            out = np.zeros((1), dtype=np.float32)
            out[0] = Tree.semantic_list.index(self.label)
            semantic = torch.tensor(out, dtype=torch.long)
            if self.is_cuda:
                semantic = semantic.cuda()
            return semantic

        def get_semantic_one_hot(self):
            out = np.zeros((1, Tree.num_sem), dtype=np.float32)
            # out[0, Tree.semantic_list.index("/" + self.full_label)] = 1
            out[0, Tree.semantic_list.index(self.label)] = 1
            semantic = torch.tensor(out, dtype=torch.float32)
            if self.is_cuda:
                semantic = semantic.cuda()
            return semantic

        def depth_first_traversal(self):
            nodes = []

            stack = [self]
            while len(stack) > 0:
                node = stack.pop()
                nodes.append(node)

                stack.extend(reversed(node.children))

            return nodes

        def edge_tensors(self, type_onehot=True):
            num_edges = len(self.edges)

            # get directed edge indices in both directions as tensor
            edge_indices = torch.tensor([[e['part_a'], e['part_b']] for e in self.edges], dtype=torch.long).view(1,
                                                                                                                 num_edges,
                                                                                                                 2)
            if self.is_cuda:
                edge_indices = edge_indices.cuda()

            # get edge type as tensor
            edge_type = torch.tensor(np.array([edge['type'] for edge in self.edges]), dtype=torch.long)
            if self.is_cuda:
                edge_type = edge_type.cuda()
            if type_onehot:
                if num_edges > 0:
                    edge_type1 = one_hot(inp=edge_type[:, 0], label_count=Space_Category.st_num).transpose(0, 1).view(1,
                                                                                                                      num_edges,
                                                                                                                      Space_Category.st_num).to(
                        dtype=torch.float32)
                    edge_type2 = one_hot(inp=edge_type[:, 1], label_count=Space_Category.dr_num).transpose(0, 1).view(1,
                                                                                                                      num_edges,
                                                                                                                      Space_Category.dr_num).to(
                        dtype=torch.float32)
                    edge_type = torch.cat([edge_type1, edge_type2], dim=2)
                else:
                    edge_type = one_hot(inp=edge_type, label_count=Space_Category.st_num).transpose(0, 1).view(1,
                                                                                                               num_edges,
                                                                                                               Space_Category.st_num).to(
                        dtype=torch.float32)
            else:
                if num_edges > 0:
                    edge_type = edge_type[:, 0:2]
                edge_type = edge_type.view(1, num_edges, 2)

            return edge_type, edge_indices

        def get_graph(self):
            # part_boxes = []
            # part_ids = []
            # part_sems = []

            # tensor 
            nodes_feat = []
            semantics = []

            edges = []
            targets = []
            e_masks = []

            # for consistancy loss
            edge_key2index = {}
            edge_forward = []
            edge_backward = []

            nodes = list(reversed(self.depth_first_traversal()))
            # for test
            self.node_name = []
            self.node_children = {}
            self.edge_indices = []
            # without root
            self.node_isleaf = {}

            box_index_offset = 0
            for node in nodes:
                child_count = 0
                box_idx = {}

                self.node_children[node.part_id] = []

                for i, child in enumerate(node.children):
                    self.node_name.append(child.part_id)
                    self.node_isleaf[child.part_id] = child.is_leaf
                    self.node_children[node.part_id].append(child.part_id)
                    # part_boxes.append(child.box)
                    # part_sems.append(child.full_label)
                    # part_ids.append(child.part_id)
                    nodes_feat.append(child.node_feat)
                    # semantics.append(child.get_semantic_one_hot())
                    semantics.append(child.get_semantic_class())

                    box_idx[i] = child_count + box_index_offset
                    child_count += 1

                for edge in node.edges:

                    # edges.append(edge.copy())
                    # edges[-1]['part_a'] = box_idx[edges[-1]['part_a']]
                    # edges[-1]['part_b'] = box_idx[edges[-1]['part_b']]
                    edge_inf = np.zeros((1, 5))
                    # start id
                    edge_inf[0, 0] = box_idx[edge['part_a']]
                    # space index
                    edge_inf[0, 1:4] = edge['type']
                    # end id
                    edge_inf[0, 4] = box_idx[edge['part_b']]

                    self.edge_indices.append([edge_inf[0, 0], edge_inf[0, 4]])

                    # key to find forward edge and backward edge
                    edge_key2index[str(edge_inf[0, 0]) + " " + str(edge_inf[0, 4])] = len(edges)
                    if str(edge_inf[0, 4]) + " " + str(edge_inf[0, 0]) in edge_key2index.keys():
                        edge_forward.append(edge_key2index[str(edge_inf[0, 0]) + " " + str(edge_inf[0, 4])])
                        edge_backward.append(edge_key2index[str(edge_inf[0, 4]) + " " + str(edge_inf[0, 0])])

                    motion = np.zeros((1, 7))
                    motion[0, 0] = edge['m_type']  # 运动类型
                    motion[0, 1:4] = edge['m_origin']  # 运动位置
                    motion[0, 4:7] = edge['m_direct']  # 运动方向

                    edges.append(torch.tensor(edge_inf).long())
                    targets.append(torch.tensor(motion, dtype=torch.float32))
                    e_masks.append(torch.tensor(edge['e_mask']).long())

                box_index_offset += child_count

            nodes_feat = torch.cat(nodes_feat)
            edges = torch.cat(edges)
            targets = torch.cat(targets)
            e_masks = torch.tensor(e_masks).long()

            semantics = torch.cat(semantics)

            if self.is_cuda:
                edges = edges.cuda()
                targets = targets.cuda()
                e_masks = e_masks.cuda()

            return nodes_feat, edges, targets, e_masks, edge_forward, edge_backward, semantics

        def to_cuda(self):
            self.is_cuda = True

            if self.box is not None:
                # self.box = self.box.to(device)
                self.box = self.box.cuda()

            if self.geo is not None:
                # self.geo = self.geo.to(device)
                self.geo = self.geo.cuda()

            for child_node in self.children:
                child_node.to_cuda()

            return self

    # functions for class Tree
    def __init__(self, root):
        self.root = root

    @staticmethod
    def load_semantic(data_path, cat):
        with open(data_path + "/semantic_merge.json", 'r') as f:
            semantic_dict = json.load(f)
        if cat in semantic_dict:
            print("using semantic_merge.json")
            Tree.semantic_list = semantic_dict[cat]
            Tree.num_sem = len(semantic_dict[cat])
        else:
            with open(data_path + "/semantic.json", 'r') as f:
                semantic_dict = json.load(f)
            print("using semantic.json")
            Tree.semantic_list = semantic_dict[cat]
            Tree.num_sem = len(semantic_dict[cat])

    def to_cuda(self):
        self.is_cuda = True
        self.root = self.root.to_cuda()
        return self


# extend torch.data.Dataset class for PartNet
class PartNetDataset(Dataset):

    def __init__(self, phase, data_root, class_name='all', points_batch_size=1000, normalize=True):
        super(PartNetDataset, self).__init__()
        self.phase = phase
        self.data_root = data_root
        self.class_name = class_name
        self.points_batch_size = points_batch_size
        with open(data_root + "/" + phase + ".struct.json", 'r') as load_f:
            json_data = json.load(load_f)
        self.data_list = []
        self.normalize = normalize
        if class_name == 'all':
            for category in json_data.keys():
                self.data_list += [os.path.join(category, data_id) for data_id in json_data[category]]
        else:
            self.data_list += [os.path.join(class_name, data_id) for data_id in json_data[class_name]]

    def __getitem__(self, index):
        obj = self.load_object(self.data_list[index])

        data_feats = ()
        data_feats = data_feats + (obj,)

        return data_feats

    def __len__(self):
        return len(self.data_list)

    # @staticmethod
    def load_object(self, fn):
        pc_path = os.path.join(self.data_root, "pc_seg", fn + ".h5")
        with h5.File(pc_path, 'r') as f:
            shape_pc = f['pc'][:]
            shape_pc_ids = f['seg'][:]

        graph_path = os.path.join(self.data_root, "graph", fn + ".pkl")
        graph_data = RelationshipGraph()
        graph_data.load_from_file(graph_path)
        root_node = graph_data.get_struct_root()[0]

        # create a virtual parent node of the root node and add it to the stack
        StackElement = namedtuple('StackElement', ['node_json', 'parent', 'parent_child_idx'])
        stack = [StackElement(node_json=root_node, parent=None, parent_child_idx=None)]

        root = None
        child_id2index = {}
        # traverse the tree, converting each node json to a Node instance
        while len(stack) > 0:
            stack_elm = stack.pop()

            parent = stack_elm.parent
            parent_child_idx = stack_elm.parent_child_idx
            node_json = stack_elm.node_json

            is_leaf = len(node_json.pc_id) == 1 and str(node_json.pc_id[0]) == node_json.id

            node = Tree.Node(
                part_id=node_json.id,
                is_leaf=(is_leaf),
                label=node_json.name)

            points = [shape_pc[shape_pc_ids == pc_id] for pc_id in node_json.pc_id]
            points = np.concatenate(points, axis=0)
            indices = np.arange(len(points))
            if len(points) > self.points_batch_size:
                random.shuffle(indices)
                # np.random.shuffle(indices)
                indices = indices[:self.points_batch_size]
            points = points[indices]
            if self.normalize:
                points[:, 0:3] = pc_normalize(points[:, 0:3])
            # pc_indf = self.deal_with_pc.get_inf(points)
            # print(pc_indf)
            node.geo = torch.tensor(points, dtype=torch.float32).view(1, -1, 3)

            box = np.concatenate((node_json.box.center, node_json.box.size, node_json.box.direction[0],
                                  node_json.box.direction[1], node_json.box.direction[2]))
            node.box = torch.from_numpy(box).to(dtype=torch.float32).view(1, 15)

            if is_leaf == False:
                for ci, child in enumerate(node_json.child):
                    stack.append(StackElement(node_json=child, parent=node, parent_child_idx=ci))
                    if child.id in child_id2index.keys():
                        print("dataset has error in child id")
                        exit()
                    child_id2index[child.id] = ci

            if is_leaf == False:
                for child in node_json.child:
                    for e in child.out_brother_edge:
                        edge = {}
                        edge['part_a'] = child_id2index[e.start_node.id]
                        edge['part_b'] = child_id2index[e.end_node.id]
                        edge['type'] = e.get_space_index()

                        edge['m_type'] = motion().motion_type_8.index(e.m_type)
                        edge['m_origin'] = e.m_origin
                        edge['m_direct'] = direction_normal(e.m_direct)
                        edge['e_mask'] = e.e_mask

                        node.edges.append(edge)

            if parent is None:
                root = node
                root.full_label = root.label
                root.graph_name = fn
            else:
                if len(parent.children) <= parent_child_idx:
                    parent.children.extend([None] * (parent_child_idx + 1 - len(parent.children)))
                parent.children[parent_child_idx] = node
                node.full_label = parent.full_label + '/' + node.label

        obj = Tree(root=root)

        return obj


def collate_feats(b):
    # list1 = []
    # for i, data in enumerate(b):
    #     list1.append(*data)
    # return list1
    return list(zip(*b))


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # from tqdm import tqdm
    DATA_PATH = '/mnt/disk2/sunqian/GNN_motion/dataset'
    train_dataset = PartNetDataset(phase='train', data_root=DATA_PATH, class_name='chair')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, \
                                                   shuffle=False, collate_fn=collate_feats)

    train_batches = enumerate(train_dataloader, 0)
    device = 0
    for train_batch_ind, batch in train_batches:
        objects = batch[0]
        # obj is Tree
        for obj in objects:
            obj.to_cuda()
            root = obj.root

            print("------------------------")
            print(root.graph_name)
            stack = [root]
            while len(stack) > 0:
                node = stack.pop()

                # print("------------------------")
                if node.is_leaf:
                    print("leaf")
                    print(node.geo.shape)
                else:
                    for child in node.children:
                        stack.append(child)
                    print(node.label)
                    print("not leaf")
                    edge_type_onehot, edge_indices = node.edge_tensors(type_onehot=True)

    pass
