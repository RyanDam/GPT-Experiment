import argparse
from types import SimpleNamespace

from tinygpt.cfg import get_cfg
from tinygpt.engine import train_sequence

def parse_bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def entrypoint():
    default_cfg = get_cfg()

    parser = argparse.ArgumentParser(description='TinyGPT')
    for k, v in default_cfg:
        t = type(v)
        t = parse_bool if t == bool else (t if v is not None else str)
        parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()

    print(args)

    args = SimpleNamespace(**vars(args))

    if args.task == "train":
        train_sequence(args)