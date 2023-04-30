import argparse
import os
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', type=str, default='', help='path to volumes (should have subfolders [volumes, labels, labels_sk, nonlabel(if using semi-supervising)]')
        parser.add_argument('-n', '--name', type=str, default='Sunmap', dest='name', help='name of model')
        parser.add_argument('-s', '--is_save', type=bool, default=True, dest='is_save', help='save checkpoints')
        parser.add_argument('--gpu_ids', type=str, default=None, help='cuda number')
        parser.add_argument('--model', type=str, default='segnet', help='chooses which model to use. [segnet | semiseg | evalnet]')
        parser.add_argument('--input_dim', type=int, default=128, help='input dimension of model')
        parser.add_argument('--input_nc', type=int, default=1, help='input channel of model')
        parser.add_argument('--output_nc', type=int, default=1, help='output channel of model')
        parser.add_argument('--dataset_mode', type=str, default='axons', help='axons|curaxons|augaxons')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)

        # set gpu ids
        if opt.gpu_ids:
            str_ids = opt.gpu_ids.split(',')
            opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    opt.gpu_ids.append(id)
            if len(opt.gpu_ids) > 0:
                torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
