import os
import json
import numpy as np
from tools.utils import bar
from configs import DATA_ROOT_DICT
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    graphics_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graphics")
    categories = os.listdir(graphics_root)
    for category in categories:
        files = os.listdir(os.path.join(graphics_root, category))
        for i, file in enumerate(files):
            jsonPath = os.path.join(graphics_root, category, file)
            data = json.load(open(jsonPath, 'r'))
            for node in data.keys():
                obb = np.asarray(data[node]["box"])
                inds = np.array([0, 3, 4, 1, 7, 2, 6, 5])
                data[node]["box"] = obb[inds].tolist()
            json.dump(data, open(jsonPath, 'w'))
            bar(f"4_AdjustObbOrder/{category}", i + 1, len(files))
