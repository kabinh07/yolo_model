import argparse
import yaml
import json
from addict import Dict
from modules.model import YOLOModel
import numpy as np

def remove_null_params(config):
    dic = Dict()
    for key, value in config.items():
        if isinstance(value, dict):
            value = remove_null_params(value)
        if not value is None:
            dic[key] = value
    return dic

def main():
    with open('configs/config.yaml') as f:
        config = Dict(yaml.safe_load(f))
    config = remove_null_params(config)
    print(f"Configuration:\n{json.dumps(config, indent=2)}")
    model = YOLOModel(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="type if want to train, test, predict etc.")
    args = parser.parse_args()
    results = None
    if args.method == 'train':
        model.train()
    elif args.method == 'predict':
        model.predict()
    elif args.method == 'track':
        model.track()
    elif args.method == 'count': 
        model.count()
    else:
        print("invalid arguments")
    # boxes_ids = []
    # for result in results:
    #     row = {}
    #     row['ids'] = result.boxes.id.numpy()
    #     row['boxes'] = result.boxes.xyxyn.numpy()
    #     boxes_ids.append(row)
    # with open('results.json', 'w') as f:
    #     json.dump(boxes_ids, f)
if __name__ == "__main__":
    main()
