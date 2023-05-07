import json



data_root = '/mnt/disk2/sunqian/GNN_motion/node_data/'
with open(data_root + "model.json", 'r') as load_f:
    json_data = json.load(load_f)

semantic_dict = {}
save_file = '/mnt/disk2/sunqian/GNN_motion/dataset/semantic.json'
for category in json_data.keys():
    data_list = json_data[category]
    name_list = []
    for data_id in data_list:
        fn = data_root + "graphics/" + category + "/" + data_id + ".json"
        with open(fn, 'r') as f:
            js_data = json.load(f)
        for key in js_data:
            if js_data[key]["name"] not in name_list:
                name_list.append(js_data[key]["name"])
    semantic_dict[category] = name_list

with open(save_file, 'w') as f:
    json.dump(semantic_dict, f)



# full label
# data_root = '/mnt/disk2/sunqian/GNN_motion/node_data/'
# with open(data_root + "model.json", 'r') as load_f:
#     json_data = json.load(load_f)

# semantic_dict = {}
# save_file = '/mnt/disk2/sunqian/GNN_motion/dataset/semantic.json'
# for category in json_data.keys():
#     data_list = json_data[category]
#     name_list = []
#     for data_id in data_list:
#         fn = data_root + "graphics/" + category + "/" + data_id + ".json"
#         with open(fn, 'r') as f:
#             js_data = json.load(f)
#         for key in js_data:
#             if js_data[key]["parent"] == -1:
#                 root = key
#                 break
#         stack = [[root, ""]]
#         while(stack):
#             stack_elem = stack.pop()

#             part_id = stack_elem[0]
#             name = js_data[part_id]["name"]
#             parent_label = stack_elem[1]

#             full_label = parent_label + "/" + name
#             if full_label not in name_list:
#                 name_list.append(full_label)

#             for child_id in js_data[part_id]["children_id"]:
#                 stack.append([str(child_id), full_label])            

#     semantic_dict[category] = name_list
#     print(name_list)

# with open(save_file, 'w') as f:
#     json.dump(semantic_dict, f)