import json
import glob

configs = json.load(open("../config.json", "r"))
DATA_ROOT_DICT = {
    "partnet": configs["partnet"],
    "partnet_mobility": configs["partnet_mobility"],
}

DATA_MODE = "test"
GPU = "0"

# set Interaction Region parameters
TOPK = 20
NUM_SAMPLE = 800

# categories list
categories = ['refrigerator', 'display', 'chair', 'laptop', 'Trashcan', 'Knife', 'keyboard', 'clock',
              'microwave', 'bottle', 'scissors', 'table', 'faucet', 'lamp', 'dishwasher', 'storage_furniture',
              'door_set']

# search model in `MODELS` dir
GNN_MODELS = {}
AXIS_SELECTION_MODELS = {}
for key in categories:
    GNN_MODELS[key] = ""
    AXIS_SELECTION_MODELS[key] = ""
    if len(glob.glob(f"../models/gnn/{key}/*")) > 0:
        GNN_MODELS[key] = glob.glob(f"../models/gnn/{key}/*")[0]
    if len(glob.glob(f"../models/axis_selection/{key}/*")) > 0:
        AXIS_SELECTION_MODELS[key] = glob.glob(f"../models/axis_selection/{key}/*")[0]
# GNN model parameters
GNN_NUM_POINTS = 1000

# result root
PREDICT_NETWORK_MOTION_ROOT = 'RESULT_NETWORK_MOTION'
GT_MOTION_ROOT = 'RESULT_GT_MOTION'
OBB_CANDIDATE_ROOT = 'RESULT_OBB_CANDIDATE'
SELECT_MOTION_ROOT = 'RESULT_SELECT_MOTION'
GRAPHICS_RESULT_ROOT = 'RESULT_GRAPHICS'
MOBILITY_RESULT_GNN_ROOT = 'RESULT_MOBILITY_GNN'
MERGE_OBJ_RESULT_ROOT = 'OBBcalculation/Input'
MERGE_OBB_RESULT_ROOT = 'OBBcalculation/Output'
PROCESS_OBB_RESULT_ROOT = 'RESUL_PROCESS_OBB'
MOBILITY_RESULT_AXIS_SELECT_ROOT = 'RESULT_MOBILITY_AXIS_SELECT'
VISUALIZATION_OBB_CANDITATE_ROOT = 'RESULT_VISUALIZATION_CANDIDATE'
VISUALIZATION_OBB_SELECTION_ROOT = 'RESULT_VISUALIZATION_SELECT'
VISUALIZATION_NETWORK_ROOT = 'RESULT_VISUALIZATION_NETWORK'
VISUALIZATION_GT_ROOT = 'RESULT_VISUALIZATION_GT'
VISUALIZATION_CANDITATE_SUBPLOT_ROOT = 'RESULT_VISUALIZATION_CANDIDATE_SUBPLOT'

mobility2partnet = {
    'refrigerator': 'Refrigerator',
    'display': 'Display',
    'chair': 'Chair',
    'mug': 'Mug',
    'laptop': 'Laptop',
    'Trashcan': 'TrashCan',
    'Knife': 'Knife',
    'keyboard': 'Keyboard',
    'clock': 'Clock',
    'microwave': 'Microwave',
    'bottle': 'Bottle',
    'scissors': 'Scissors',
    'table': 'Table',
    'faucet': 'Faucet',
    'lamp': 'Lamp',
    'dishwasher': 'Dishwasher',
    'storage_furniture': 'StorageFurniture',
    'door_set': 'Door'
}

# visualization parameters
WIDTH, HEIGHT = 500, 400
