import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
                    
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
                    
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--save',default='output/CH/model1/', help='save path of trained model')

parser.add_argument('--resume',default='/output/result/12/epoch100.pth',help='save path of trained model')

parser.add_argument('--save_nii',default='output/CH/result', help='save path of result')

parser.add_argument('--name',default='CH', help='brain region')

parser.add_argument('--test_txt',default='brain1.txt',help='test txt name')

parser.add_argument('--dataset_path_1',default='data/train_dataset_view1', help='path of train data(view1)')
parser.add_argument('--dataset_path_2',default='data/train_dataset_view2', help='path of train data(view2)')

parser.add_argument('--label_txt_1',default='allen.txt', help='labeled data txt name(view1)')
parser.add_argument('--label_txt_2',default='allen.txt', help='labeled data txt name(view2)')
parser.add_argument('--unlabel_txt_1',default='brain1.txt', help='unlabeled data txt name(view1)')
parser.add_argument('--unlabel_txt_2',default='brain1.txt', help='unlabeled data txt name(view1)')

parser.add_argument('--img_size', type=int, default=224,help='patch size of train samples after resize')

parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')

parser.add_argument('--channel', type=int, default=1, metavar='N',help='number of channels (default: 1)')

parser.add_argument('--lr_gen', type=float, default=1e-4, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.3, metavar='M',help='SGD momentum (default: 0.5)')

parser.add_argument('--early-stop', default=None, type=int, help='early stopping (default: 20)')

parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')

parser.add_argument('--ema_decay', type=float, default=0.5, help='ema_decay')

parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')

parser.add_argument('--consistency', type=float, default=0.1, help='consistency')

parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

parser.add_argument('--batch_size', type=list, default=16, help='batch size of trainset')

parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')


parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


