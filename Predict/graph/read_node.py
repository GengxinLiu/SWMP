import numpy as np
import json
import sys
import os
sys.path.append(os.getcwd())
from graph.OBB import OBB_3D
from graph.Space_edge import Space_Relationship
import h5py as h5


class Node():
    def __init__(self, id, name, obb_points, pc_id, modelID):
        self.id = id
        self.name = name
        self.box = OBB_3D(obb_points)
        self.pc_id = pc_id
        self.modelID = modelID


class Edge():
    def __init__(self, start_id, end_id, state, direct, v_direct, m_Type=None, m_origin=None, m_direct=None, n_mask=0, e_mask=0):
        '''
        state : position descriptor
        direct : direction descriptor
        v_direct : the vertical direction descriptor
        m_type : motion type
        m_origin : the motion position
        m_direct : the motion direction
        n_mask : 0 parent-child edge, 1 for motion edge
        e_mask : 0 for `fixed` or `None`, 1 for `T` or `TR`, 2 for `R`
        '''
        self.start_id = start_id
        self.end_id = end_id
        # space relation
        self.state = state
        self.direct = direct
        self.v_direct = v_direct

        # motion type + param
        self.m_type = m_Type
        self.m_origin = m_origin
        self.m_direct = m_direct

        self.n_mask = n_mask
        self.e_mask = e_mask


class read_node():
    def __init__(self, data_root, category, object_id):
        self.object_id = object_id
        self.graph_file = data_root + "/graphics" + '/' + category + '/' + object_id + ".json"
        self.pcs_file = data_root + "/pc_seg" + '/' + category + '/' + object_id + ".h5"
        self.modelID = category + "_" + object_id
        self.read_motion()
        # self.read_pc()
    
    def read_pc(self):
        with h5.File(self.pcs_file, 'r') as f:
            # print(f.keys())
            self.shape_pc = f['pc'][:]
            self.shape_pc_ids = f['seg'][:]
            # print(self.shape_pc.shape)

        node_id2pc = {}
        for node in self.nodes.values():
            pc_list = node.pc_id
            points = [self.shape_pc[self.shape_pc_ids == pc_id] for pc_id in pc_list]
            points_numpy = np.concatenate(points, axis=0)
            node_id2pc[node.id] = points_numpy
            print("p_id: ", pc_list, " ", points_numpy.shape)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        f_points = node_id2pc['4']
        # ax.scatter(self.parent.box.vertixs[:, 0], self.parent.box.vertixs[:, 1], self.parent.box.vertixs[:, 2], s=5, c='r', marker='o')
        ax.scatter(f_points[:, 0], f_points[:, 1], f_points[:, 2], s=5, c='b')
        # ax.scatter(self.child.box.vertixs[:, 0], self.child.box.vertixs[:, 1], self.child.box.vertixs[:, 2], s=5, c='r', marker='o')
        for id in ['1', '2', '3', '5']:
            c_points = node_id2pc[id]
            ax.scatter(c_points[:, 0], c_points[:, 1], c_points[:, 2], s=5, c=(0.5, 0.5, 0.5, 1.0))
        plt.axis('off') 
        plt.show()
        

    def read_motion(self):
        with open(self.graph_file, 'r') as load_f:
            graph_json = json.load(load_f)
        
        self.nodes = {}
        for id in graph_json.keys():
            node = graph_json[id]
            self.nodes[id] = Node(id, node['name'], node['box'], node["leaves"], self.modelID)

        self.edges = []
        for id in graph_json.keys():
            node = graph_json[id]
            node1 = self.nodes[id]
            box_point1 = node1.box.sample_points_random(1000)
            for end_id in node['edges']['children'].keys():
                node2 = self.nodes[end_id]
                box_point2 = node2.box.sample_points_random(1000)
                state, direct, v_direct =  Space_Relationship().get_relation(box_point1, node1.box, box_point2, node2.box)
                # print(node1.name, " ", state, " --- ", direct, " --- ", v_direct, " ", node2.name)
                self.edges.append(Edge(id, end_id, 
                                        state, direct, v_direct, 
                                        "children", np.array([0, 0, 0]), np.array([0, 0, 0]), 0, 0
                                        ))
            for end_id in node['edges']['space'].keys():
                node2 = self.nodes[end_id]
                box_point2 = node2.box.sample_points_random(1000)
                state, direct, v_direct =  Space_Relationship().get_relation(box_point1, node1.box, box_point2, node2.box)
                # print(node1.name, " ", state, " --- ", direct, " --- ", v_direct, " ", node2.name)
    
                if node['edges']['space'][end_id] == "motion":
                    # print(node['motype'])
                    if 'R' in node["motype"]: 
                        self.edges.append(Edge(
                            id, end_id, 
                            state, direct, v_direct,
                            node['motype'], node['jointData']['axis']['origin'], node['jointData']['axis']['direction'], 1, 2, 
                        ))
                    else:
                        self.edges.append(Edge(
                            id, end_id, 
                            state, direct, v_direct,
                            node['motype'], node['jointData']['axis']['origin'], node['jointData']['axis']['direction'], 1, 1, 
                        ))
                else:
                     self.edges.append(Edge(
                        id, end_id, 
                        state, direct, v_direct, 
                        node['edges']['space'][end_id], np.array([0, 0, 0]), np.array([0, 0, 0]), 1, 0
                    ))
            

if __name__ == '__main__':
    data_root = "./node_data"
    datalist_file = data_root + "/model.json"
    with open(datalist_file, 'r') as load_f:
        json_data = json.load(load_f)
    data_id = json_data['refrigerator']
    # print(data_id)

    result = read_node(data_root, 'chair', '2230')
    result.read_pc()

    pass