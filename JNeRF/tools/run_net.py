from ast import parse
import jittor as jt
from tqdm import tqdm
import argparse
import numpy as np
import os
from jnerf.runner import NeuSRunner, NeuralTORunner, NeuralTOMaterialRunner
from jnerf.utils.config import init_cfg, get_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1


def main():
    assert jt.flags.cuda_archs[0] >= 61, "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--type",
        default="novel_view",
        type=str,
    )
    parser.add_argument(
        "--mcube_threshold",
        default=0.0,
        type=float,
    )
    
    args = parser.parse_args()

    assert args.type in ["neuralTO_recon","neus"],f"{args.type} not support, please choose [novel_view, mesh]"
    assert args.task in ["train","render", "validate_mesh"],f"{args.task} not support, please choose [train, test, render, validate_mesh]"
    is_continue = False
    if args.task == 'validate_mesh':
        is_continue = True

    if args.config_file:
        init_cfg(args.config_file)

    if args.type == 'neus':
        runner = NeuSRunner(is_continue=is_continue)
    elif args.type == 'neuralTO_recon':
        runner = NeuralTORunner(is_continue=is_continue)
    elif args.type == 'neuralTO_render':
        runner = NeuralTOMaterialRunner(is_continue=is_continue)
    else:
        raise ValueError('Not support yet!')

    if args.task == "train":
        runner.train()
    elif args.task == "render":
        runner.validate_image(args.save_dir)
    elif args.task == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    

if __name__ == "__main__":
    main()
