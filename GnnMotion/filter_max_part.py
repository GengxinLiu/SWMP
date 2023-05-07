from cmath import phase
import json
import os
from unicodedata import category 
from graph.graph_extraction import RelationshipGraph, motion


def filter_part(data_root, phase, class_name='all', max_child_num=11):
    with open(data_root + "/" + phase + ".struct.json", 'r') as load_f:
        json_data = json.load(load_f)
    
    data_list = []
    if class_name == 'all':
        for category in json_data.keys():
            data_list += [os.path.join(category, data_id) for data_id in json_data[category]]
    else:
        data_list += [os.path.join(class_name, data_id) for data_id in json_data[class_name]]
    # print(data_list)
    for fn in data_list:
        graph_path = os.path.join(data_root, "graph", fn + ".pkl")
        graph_data = RelationshipGraph()
        graph_data.load_from_file(graph_path)
        root_node = graph_data.get_struct_root()[0]

        stack = [root_node]
        while len(stack) > 0:
            node_json = stack.pop()

            is_leaf = len(node_json.pc_id) == 1 and str(node_json.pc_id[0]) == node_json.id
            if is_leaf == False:
                num_child = len(node_json.child)
                if num_child > max_child_num:
                    print(fn)
                    print(num_child)
                    # break
                for ci, child in enumerate(node_json.child):
                    stack.append(child)
    


if __name__ == "__main__":
    DATA_PATH = ''
    # category = ['bottle', 'Box', 'Bucket', 'clock', 'dishwasher', 'Dispenser', 'display', 'door_set', 'Fan', 'FoldingChair',
    #             'Globe', 'Kettle', 'KitchenPot', 'Knife', 'laptop', 'Lighter', 'Luggage', 'Mouse']

    category = ['Oven', 'Pen', 'Pliers', 'pot', 'Safe', 'scissors', 'Stapler', 'Suitcase', 'Toaster']
    for cate in category:
        filter_part(DATA_PATH, 'train', cate)
        filter_part(DATA_PATH, 'test', cate)
