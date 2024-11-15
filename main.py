import argparse
import yaml
from addict import Dict
from modules.model import YOLOModel

def main():
    with open('configs/config.yaml') as f:
        config = Dict(yaml.safe_load(f))
    print(config)
    model = YOLOModel(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="type if want to train, test, predict etc.")
    args = parser.parse_args()

    if args.method == 'train':
        model.train()
    elif args.method == 'predict':
        model.predict()
    else:
        print("invalid arguments")

if __name__ == "__main__":
    main()
