import os
from configs import DATA_ROOT_DICT
from tools.utils import bar
from graph.graph_extraction import RelationshipGraph
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="partnet", choices=["partnet_mobility", "partnet"])
    args = parser.parse_args()
    save_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data")
    graphics_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graphics")
    graph_root = os.path.join(DATA_ROOT_DICT[args.dataset], "network_data", "graph")
    categories = os.listdir(graphics_root)
    for category in categories:
        data_list = [file.split('.')[0] for file in os.listdir(os.path.join(graphics_root, category))]
        save_dir = os.path.join(graph_root, category)
        os.makedirs(save_dir, exist_ok=True)
        for i, object_id in enumerate(data_list):
            graph = RelationshipGraph()
            graph.extract_from_data(save_root, category, object_id)
            graph.save_to_file(os.path.join(save_dir, f"{object_id}.pkl"))
            bar(f"6_GraphDataExtract/{category}", i + 1, len(data_list))
