from configs import MERGE_OBB_RESULT_ROOT, PROCESS_OBB_RESULT_ROOT
import tools.mobility_tool as mt
import numpy as np
import os
from utils import bar
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet_mobility", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()

    csv_files = os.listdir(MERGE_OBB_RESULT_ROOT)
    PROCESS_OBB_RESULT_ROOT = os.path.join(f"result_{args.dataset}", PROCESS_OBB_RESULT_ROOT)
    os.makedirs(PROCESS_OBB_RESULT_ROOT, exist_ok=True)
    for i, csv_file in enumerate(csv_files):
        filename = csv_file.split('.')[0]
        save_path = os.path.join(PROCESS_OBB_RESULT_ROOT, filename)
        if os.path.exists(save_path):
            continue
        csv_path = os.path.join(MERGE_OBB_RESULT_ROOT, csv_file)
        box, dirs = mt.csv2box(csv_path)
        obb = np.asarray(box)
        inds = np.array([0, 3, 4, 1, 7, 2, 6, 5])
        obb = obb[inds]
        # save obb
        np.savetxt(save_path, obb)
        bar("6_ProcessOBB", i + 1, len(csv_files))
