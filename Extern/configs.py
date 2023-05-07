import json

configs = json.load(open("../config.json", "r"))
DATA_ROOT_DICT = {
    "partnet": configs["partnet"],
    "partnet_mobility": configs["partnet_mobility"],
}

PARTNET_MOBILITY_ROOT = "../Predict/result_partnet/RESULT_MOBILITY_AXIS_SELECT"

SAVE_ROOT_DICT = {
    "partnet": "./partnet",
    "partnet_mobility": "./partnet2shapemotion"
}
