import argparse
import sys

from .config import cfg


def get_cfg():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to additional config YAML file. Overwrite order: default cfg, config file, command-line.",
    )
    parser.add_argument(
        "--opts",
        nargs="+",
        help="Overwrite config options using command line. Overwrite order: default cfg, config file, command-line.",
        default=[],
    )
    args = parser.parse_args()

    # Merge config from the YAML file if provided
    if args.config_file:
        if os.path.isfile(args.config_file):
            print(f"Loading config file from {args.config_file}")
            cfg.merge_from_file(args.config_file)
        else:
            raise FileNotFoundError(f"Config file not found: {args.config_file}")
    # Merge additional command-line inputs
    if args.opts:
        opts = []
        for i in range(0, len(args.opts), 2):
            key = args.opts[i]
            val = args.opts[i + 1]
            if "," in val:  # list inputs
                val = val.split(",")
            opts.append(key)
            opts.append(val)
        cfg.merge_from_list(opts)

    cfg.freeze()
    print("Working in " + cfg.model_dir)
    return cfg
