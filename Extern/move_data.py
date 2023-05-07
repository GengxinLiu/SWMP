import shutil
import os
import math
import sys
import argparse
from configs import SAVE_ROOT_DICT


def bar(message, now: int, total: int):
    """
    :param message: string to print.
    :param now: the i-th iteration.
    :param total: total iteration num.
    :return:
    """
    rate = now / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t\t%d/%d' % (message, "=" * rate_num,
                                      " " * (40 - rate_num), rate_nums, now, total)
    if now == total:
        r += "\n"
    sys.stdout.write(r)
    sys.stdout.flush()


parse = argparse.ArgumentParser()
parse.add_argument("--c", nargs="+", type=str, help="move categories")
args = parse.parse_args()
partnet_categories = args.c

dir_list = ["hdf5", "objects", "render", "urdf"]
partnet2mobility = {
    'Refrigerator': 'refrigerator',
    'Display': 'display',
    'Chair': 'chair',
    'Mug': 'mug',
    'Laptop': 'laptop',
    'Trashcan': 'TrashCan',
    'Knife': 'Knife',
    'Keyboard': 'keyboard',
    'Clock': 'clock',
    'Microwave': 'microwave',
    'Bottle': 'bottle',
    'Scissors': 'scissors',
    'Table': 'table',
    'Faucet': 'faucet',
    'Lamp': 'lamp',
    'Dishwasher': 'dishwasher',
    'StorageFurniture': 'storage_furniture',
    'Door': 'door_set'
}
src_root = SAVE_ROOT_DICT["partnet"]
dst_root = SAVE_ROOT_DICT["partnet_mobility"]
prefix = "partnet_axis_select"
for src_c in partnet_categories:
    dst_c = partnet2mobility[src_c]
    for dir in dir_list:
        src_root_dir = os.path.join(src_root, dir, src_c)
        dst_root_dir = os.path.join(dst_root, dir, dst_c)
        files = os.listdir(src_root_dir)
        for i, file in enumerate(files):
            src_file = os.path.join(src_root_dir, file)
            dst_file = os.path.join(dst_root_dir, f"{prefix}_{file}")
            shutil.move(src_file, dst_file)
            bar(f"moving {src_c}/{dir}", i + 1, len(files))
