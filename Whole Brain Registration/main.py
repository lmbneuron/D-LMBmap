import os
import argparse
from shutil import rmtree
from time import strftime, localtime
import yaml


def main():
    args = get_args()

    with open(args.config, "r") as f:
        config = yaml.load(f)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["TrainConfig"]["gpu"]
    if args.train:
        # enter the train mode
        from train import Trainer
        basedir = get_basedir(args.output, config["TrainConfig"]["start_new_model"])
        trainer = Trainer(config, basedir, args.config)
        trainer.train()
    elif args.eval:
        # enter the eval mode
        from eval import Inference
        basedir = get_basedir(args.output, start_new_model=True)
        infer = Inference(config, basedir, args.checkpoint, args.config)
        infer.inference()
    elif args.ave:
        from ms_regnet.average_template.allen_average import Averager
        basedir = get_basedir(args.output, start_new_model=True)
        infer = Averager(config, basedir, args.checkpoint, args.config)
        infer.average()
    elif args.test:
        from test import Tester
        basedir = get_basedir(args.output, start_new_model=True)
        tester = Tester(config, basedir, args.checkpoint, args.config, args.test_config, args.upsample)
        tester.test()
    else:
        raise AttributeError


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", "-t", action="store_true",
                        help="train mode, you must give the --output and --config")
    parser.add_argument("--eval", "-e", action="store_true",
                        help="eval mode, you must give the --output and --config and the --checkpoint")
    parser.add_argument("--ave", "-a", action="store_true",
                        help="average mode, you must give the --output and --config and the --checkpoint")
    parser.add_argument("--test", action="store_true",
                        help="test the accuracy use the manual mask")

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='if the mode is train: the dir to store the file;'
                             'if the mode is eval or ave: the path of output')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='used in all the modes, the path of the config yaml')

    parser.add_argument('--checkpoint', type=str,
                        help='used in the eval, ave and test mode, the path of the checkpoint')
    parser.add_argument('--test_config', type=str, default='configs/soma_nuclei_rev_test.yaml',
                        help='the test config yaml file, used in the test mode')
    parser.add_argument('--upsample', type=int, default=1,
                        help='upsample time, used in the test mode')
    args = parser.parse_args()

    return args


def get_basedir(base_dir, start_new_model=False):
    # init the output folder structure
    if base_dir is None:
        base_dir = os.path.join("./", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    if start_new_model and os.path.exists(base_dir):
        rmtree(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(os.path.join(base_dir, 'logs')):
        os.mkdir(os.path.join(base_dir, 'logs'))  ##tensorboard
    if not os.path.exists(os.path.join(base_dir, 'checkpoint')):
        os.mkdir(os.path.join(base_dir, 'checkpoint'))  ##checkpoint
    return base_dir


if __name__ == "__main__":
    main()
