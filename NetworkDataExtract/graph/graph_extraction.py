import pickle
import os
import copy
import json
# import sys
# sys.path.append(os.getcwd())
from graph.Space_edge import Space_Relationship
from graph.read_node import read_node
import numpy as np
import shutil
from graph.utils import normalize_vector


class RelationshipGraph:
    class Node:
        def __init__(self, node_id, name, box, pc_id, \
                     out_edge_indices=[], in_edge_indices=[], \
                     modelID=None, graph=None):
            self.id = node_id
            self.name = name
            self.box = box
            self.pc_id = pc_id
            self.__out_edge_indices = out_edge_indices
            self.__in_edge_indices = in_edge_indices
            self.__modelID = modelID
            self.__graph = graph

        def __repr__(self):
            rep = f'{self.name} ({self.id}) --- bbox({self.box}) --- pc_id({self.pc_id})'
            return rep

        @property
        def out_edges(self):
            return [self.__graph.edges[i] for i in self.__out_edge_indices]

        @property
        def in_edges(self):
            return [self.__graph.edges[i] for i in self.__in_edge_indices]

        @property
        def all_edges(self):
            return self.in_edges + self.out_edges

        @property
        def out_neighbors(self):
            return [e.end_node for e in self.out_edges]

        @property
        def in_neighbors(self):
            return [e.start_node for e in self.in_edges]

        @property
        def all_neighbors(self):
            return list(set(self.in_neighbors + self.out_neighbors))

        @property
        def child(self):
            return [e.end_node for e in self.out_edges if e.m_type == 'children']

        @property
        def brother(self):
            return [e.end_node for e in self.out_edges if e.m_type != 'children']

        @property
        def out_brother_edge(self):
            return [e for e in self.out_edges if e.m_type != 'children']

        @property
        def modelID(self):
            if not hasattr(self, '_Node__modelID'):
                setattr(self, '_Node__modelID', None)
            return self.__modelID

        def set_modelID(self, mid):
            self.__modelID = mid

        def clear_edges(self):
            self.__out_edge_indices = []
            self.__in_edge_indices = []

        def add_out_edge(self, edge_idx):
            self.__out_edge_indices.append(edge_idx)

        def add_in_edge(self, edge_idx):
            self.__in_edge_indices.append(edge_idx)

        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Node(self.id, self.name, self.box, self.pc_id, \
                                          self.__out_edge_indices, self.__in_edge_indices, \
                                          self.modelID, graph)

        def without_graph(self):
            return RelationshipGraph.Node(self.id, self.name, self.box, self.pc_id, \
                                          self.__out_edge_indices, self.__in_edge_indices, \
                                          self.modelID)
        # ---------------------------------------------------------------------------

    class Edge:
        def __init__(self, start_id, end_id,
                     state, direct, v_direct,
                     m_Type, m_origin=None, m_direct=None, n_mask=0, e_mask=0,
                     graph=None):
            self.__start_id = start_id
            self.__end_id = end_id

            self.state = state
            self.direct = direct
            self.v_direct = v_direct

            self.m_type = m_Type
            self.m_origin = m_origin
            self.m_direct = m_direct

            self.n_mask = n_mask
            self.e_mask = e_mask

            self.__graph = graph

        def get_space_index(self):
            space = Space_Relationship()
            return np.array([space.state.index(self.state), space.Direction.index(self.direct),
                             space.Vertical_Direction.index(self.v_direct)])

        def __repr__(self):
            edge_name = self.state + " " + self.direct + " " + self.v_direct
            edge_name = f'{edge_name}'
            return f'{self.start_node.name} ({self.start_node.id}) ---- {edge_name} ---> {self.end_node.name} ({self.end_node.id})'

        @property
        def start_node(self):
            assert (self.__graph is not None)
            return self.__graph.get_node_by_id(self.__start_id)

        @property
        def end_node(self):
            assert (self.__graph is not None)
            return self.__graph.get_node_by_id(self.__end_id)

        @property
        def neighbors(self):
            return self.start_node, self.end_node

        @property
        def start_id(self):
            return self.__start_id

        @property
        def end_id(self):
            return self.__end_id


        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Edge(self.__start_id, self.__end_id,
                                          self.state, self.direct, self.v_direct,
                                          self.m_type, self.m_origin, self.m_direct, self.n_mask, self.e_mask,
                                          graph)

        def without_graph(self):
            return RelationshipGraph.Edge(self.__start_id, self.__end_id,
                                          self.state, self.direct, self.v_direct,
                                          self.m_type, self.m_origin, self.m_direct, self.n_mask, self.e_mask)

        # ---------------------------------------------------------------------------

    # ******************************************************************************************************************
    # (Main body for RelationshipGraph)
    def __init__(self, nodes=[], edges=[]):
        self.__nodes = {n.id: n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()

    @property
    def nodes(self):
        return list(self.__nodes.values())

    def __record_node_edges(self):
        for node in self.nodes:
            node.clear_edges()
        for edge_idx in range(len(self.edges)):
            edge = self.edges[edge_idx]
            edge.start_node.add_out_edge(edge_idx)
            edge.end_node.add_in_edge(edge_idx)

    def get_node_by_id(self, id_):
        if not id_ in self.__nodes:
            print(f'Could not find node with id {id_}')
        return self.__nodes[id_]

    def add_node(self, node):
        self.__nodes[node.id] = node.with_graph(self)

    def extract_from_data(self, data_root, category, object_id):
        read = read_node(data_root, category, object_id)

        self.__nodes.clear()
        self.edges.clear()

        for _, key in enumerate(read.nodes):
            node = read.nodes[key]
            self.add_node(RelationshipGraph.Node(
                node.id, node.name, node.box, node.pc_id, modelID=node.modelID, graph=self
            ))

        for _, edge in enumerate(read.edges):
            self.edges.append(RelationshipGraph.Edge(
                edge.start_id, edge.end_id,
                edge.state, edge.direct, edge.v_direct,
                edge.m_type, edge.m_origin, edge.m_direct, edge.n_mask, edge.e_mask, graph=self
            ))

        self.__record_node_edges()

    def show(self):
        print(self.__nodes)
        print(self.edges)
        # mlab.show()

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            nodes = [n.without_graph() for n in self.nodes]
            edges = [e.without_graph() for e in self.edges]
            pickle.dump((nodes, edges), f, pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, filename):
        # fname = os.path.split(filename)[1]
        # self.id = int(os.path.splitext(fname)[0])
        with open(filename, 'rb')as f:
            nodes, edges = pickle.load(f)
        self.__nodes = {n.id: n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()
        return self

    def get_numpy(self):
        node_len = len(self.nodes)
        node_numpy = np.zeros((node_len, 15), dtype=np.float32)
        node_pc_ids = []
        nodes = {}
        for i in range(node_len):
            # node feature
            node_numpy[i, :3] = self.nodes[i].box.center
            node_numpy[i, 3:6] = self.nodes[i].box.size
            node_numpy[i, 6:9] = self.nodes[i].box.direction[0]
            node_numpy[i, 9:12] = self.nodes[i].box.direction[1]
            node_numpy[i, 12:15] = self.nodes[i].box.direction[2]
            nodes[self.nodes[i].id] = i

            # pc id
            node_pc_ids.append(self.nodes[i].pc_id)

        edge_len = len(self.edges)
        edge_numpy = np.zeros((edge_len, 5), dtype=np.float32)
        target = np.zeros((edge_len, 7), dtype=np.float32)
        n_mask_numpy = np.zeros(edge_len, dtype=np.float32)
        e_mask_numpy = np.zeros(edge_len, dtype=np.float32)
        for i in range(edge_len):
            # start id
            edge_numpy[i, 0] = nodes[self.edges[i].start_node.id]
            # space index
            edge_numpy[i, 1:4] = self.edges[i].get_space_index()
            # end id
            edge_numpy[i, 4] = nodes[self.edges[i].end_node.id]

            # motion
            target[i, 0] = motion().motion_type_8.index(self.edges[i].m_type)
            target[i, 1:4] = self.edges[i].m_origin
            target[i, 4:7] = self.edges[i].m_direct

            # mask
            n_mask_numpy[i] = self.edges[i].n_mask
            e_mask_numpy[i] = self.edges[i].e_mask

        return node_numpy, edge_numpy, n_mask_numpy, e_mask_numpy, node_pc_ids, target

    def get_struct_root(self):
        root = []
        for node in self.nodes:
            if len(node.in_edges) == 0:
                root.append(node)
        return root


class motion():
    def __init__(self):
        self.motion_type_8 = ['T_H', 'T_V',
                              'R_H_C', 'R_H_S', 'R_V_C', 'R_V_S',
                              'TR_H', 'TR_V',
                              'fixed', 'none', 'children']
        self.len = len(self.motion_type_8)


class motion2():
    def __init__(self):
        self.motion_type_8 = ['T_H', 'T_V',
                              'R_H_C', 'R_H_S', 'R_V_C', 'R_V_S',
                              'TR_H', 'TR_V',
                              '-T_H', '-T_V',
                              '-R_H_C', '-R_H_S', '-R_V_C', '-R_V_S',
                              '-TR_H', '-TR_V',
                              'fixed', 'none', 'children']
        self.len = len(self.motion_type_8)
