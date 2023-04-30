from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate for adam')
        parser.add_argument('--n_epochs', type=int, default=100, help='epochs of train, test, evaluating')
        parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model')
        parser.add_argument('--batch_size', type=int, default=30, help='batch size')
        parser.add_argument('--n_samples', type=int, default=4, help='crop n volumes from one cube')
        parser.add_argument('--semi', type=bool, default=False, help='train with semi-supervised methods')
        parser.add_argument('--nonlabel', type=bool, default=False, help='train with curriculum semi-supervised methods')
        parser.add_argument('--noartifacts', action='store_true', help='train with no artifacts volumes')
        parser.add_argument('--noshuffle_train', action='store_true', help='not to shuffle the data')
        parser.add_argument('--noshuffle_val', action='store_true', help='not to shuffle the val data')
        parser.add_argument('--exp', type=str, default='Sunmap', help='name of the experiment')
        parser.add_argument('--checkpoint_name', type=str, help='name of the experiment')
        parser.add_argument('--net', type=str, help='unet | axialunet')

        self.isTrain = True
        return parser
