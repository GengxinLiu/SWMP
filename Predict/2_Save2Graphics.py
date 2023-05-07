from configs import DATA_ROOT_DICT, GRAPHICS_RESULT_ROOT, PREDICT_NETWORK_MOTION_ROOT, mobility2partnet
import argparse
import os
from utils import bar
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='laptop')
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    CATEGORY = args.category if args.dataset == "partnet_mobility" else mobility2partnet[args.category]
    DATA_PATH = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    PREDICT_NETWORK_MOTION_ROOT = os.path.join(f"result_{args.dataset}", PREDICT_NETWORK_MOTION_ROOT)
    GRAPHICS_RESULT_ROOT = os.path.join(f"result_{args.dataset}", GRAPHICS_RESULT_ROOT)
    graphics_root = os.path.join(DATA_PATH, "graphics", CATEGORY)
    predict_motion_root = os.path.join(PREDICT_NETWORK_MOTION_ROOT, CATEGORY)
    graphics_result_root = os.path.join(GRAPHICS_RESULT_ROOT, CATEGORY)
    os.makedirs(graphics_result_root, exist_ok=True)
    files = os.listdir(predict_motion_root)
    for i, file in enumerate(files):
        file = file.split('.')[0]
        graphics = json.load(open(os.path.join(graphics_root, file + ".json"), "r"))
        for node_id in graphics.keys():
            graphics[node_id]["predict"] = []

        joint_files = os.listdir(os.path.join(predict_motion_root, file))
        for joint_file in joint_files:
            data = json.load(open(os.path.join(predict_motion_root, file, joint_file), "r"))
            move_id, ref_id = joint_file.split('.')[0].split('-')
            graphics[move_id]["predict"].append({
                ref_id: data
            })
        json.dump(graphics, open(os.path.join(graphics_result_root, file + ".json"), "w"))
        bar(f"2_Save2Graphics/{CATEGORY}", i + 1, len(files))
