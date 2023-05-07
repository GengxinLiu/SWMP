import json

configs = json.load(open("../config.json", "r"))
DATA_ROOT_DICT = {
    "partnet": configs["partnet"],
    "partnet_mobility": configs["partnet_mobility"],
}

npoints = 1024
train_ratio = 0.9
