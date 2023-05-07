import json

configs = json.load(open("../config.json", "r"))
DATA_ROOT_DICT = {
    "partnet": configs["partnet"],
    "partnet_mobility": configs["partnet_mobility"],
}
