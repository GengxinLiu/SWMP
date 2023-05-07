import os
from configs import train_ratio, DATA_ROOT_DICT
import random
import json
import argparse

random.seed(28)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    print("split train test list...")
    save_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    graphics_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graphics")
    categories = os.listdir(graphics_root)
    train_datalist = {}
    test_datalist = {}
    for category in categories:
        data_list = [file.split('.')[0] for file in os.listdir(os.path.join(graphics_root, category))]
        if args.dataset == "partnet_mobility":
            data_mid = int(len(data_list) * train_ratio)
            train_datalist[category] = data_list[:data_mid]
            test_datalist[category] = data_list[data_mid:]
        elif args.dataset == "partnet":
            test_datalist[category] = data_list

    with open(os.path.join(save_root, "train.struct.json"), 'w') as f:
        json.dump(train_datalist, f)
    with open(os.path.join(save_root, "test.struct.json"), 'w') as f:
        json.dump(test_datalist, f)
