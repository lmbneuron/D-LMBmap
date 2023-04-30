import os
import torch
import shutil
import tifffile
import multiprocessing
import configparser
from utils import *
from PIL import Image
from tqdm import tqdm
from . import networks
from models.losses import *
from data import create_dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader


def equal(v, mean, std):
    # return (v - v.min()) / (v.max() - v.min())
    # return (v - v.mean()) / (v.std() + 1e-8)
    return (v - mean) / (std + 1e-8)


class EvalNet:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.input_dim = self.opt.input_dim
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')


    def eval_two_volumes_maxpool(self):
        pre = read_tiff_stack(self.opt.dataroot)
        label = read_tiff_stack(self.opt.data_target)
        k = self.opt.pool_kernel
        kernel = (k, k, k)
        pre[pre < self.opt.threshold] = 0
        pre[pre >= self.opt.threshold] = 1
        label[label > 0] = 1
        pre = torch.Tensor(pre).view((1, 1, *pre.shape)).to(self.device)
        # pre = torch_dilation(pre, 5)
        label = torch.Tensor(label).view((1, 1, *label.shape)).to(self.device)
        # label = torch_dilation(label, 5)

        pre = torch.nn.functional.max_pool3d(pre, kernel, kernel, 0)
        label = torch.nn.functional.max_pool3d(label, kernel, kernel, 0)

        dice_score = dice_error(pre, label)

        total_loss_iou = iou(pre, label).cpu()
        total_loss_tiou = t_iou(pre, label).cpu()
        recall, acc = soft_cldice_f1(pre, label)
        cldice = (2. * recall * acc) / (recall + acc)

        print('\n Validation IOU: {}\n T-IOU: {}'
              '\n ClDice: {} \n ClAcc: {} \n ClRecall: {} \n Dice-score: {}'
              .format(total_loss_iou, total_loss_tiou, cldice, acc, recall, dice_score, '.8f'))

    def get_model(self):
        self.model = networks.define_net(self.opt.input_nc, self.opt.output_nc, self.opt.net, gpu_ids=self.gpu_ids)
        if self.opt.pre_trained:
            pretrain_encoder = torch.load(self.opt.pre_trained, map_location=self.device)
            self.model.load_state_dict(networks.load_my_state_dict(self.model, pretrain_encoder))
            print(f'loaded: {self.opt.pre_trained}')
        self.model.eval()

    def eval_volumes_batch(self):
        self.get_model()
        testLoader = create_dataset(self.opt)
        n_val = len(testLoader)
        loss_dir = self.eval_net(self.model, testLoader, self.device, n_val)
        iou, t_iou = loss_dir['iou'], loss_dir['tiou']
        cldice, clacc, clrecall = loss_dir['cldice'], loss_dir['cl_acc'], loss_dir['cl_recall']
        junk_rat = loss_dir['junk_ratio']
        print('\n Validation IOU: {}\n T-IOU: {}'
              '\n ClDice: {} \n ClAcc: {} \n ClRecall: {}'
              '\n Junk-ratio: {}'
              .format(iou, t_iou, cldice, clacc, clrecall, junk_rat, '.8f'))

    def test_3D_volume(self):
        self.get_model()
        multiprocessing.set_start_method('spawn')
        config = configparser.ConfigParser()

        config.read("./Axonal-semantic-segmenter/segpoints.ini", encoding="utf-8")
        volume_section = self.opt.section
        dataroot = config.get(volume_section, "dataroot")
        self.target = os.path.join('./Axonal-semantic-segmenter/segmentations', self.opt.exp)
        if os.path.exists(self.target):
            shutil.rmtree(self.target)
        os.mkdir(self.target)
        self.overlap = self.opt.overlap
        self.cube = self.input_dim - self.opt.overlap * 2

        self.files = [os.path.join(dataroot, pth) for pth in sorted(os.listdir(dataroot))]

        shape_y, shape_x = np.array(Image.open(self.files[0])).shape
        self.s_x, self.s_y, self.s_z = [int(i) for i in config.get(volume_section, "start_point").split(',')]
        self.e_x, self.e_y, self.e_z = [int(i) for i in config.get(volume_section, "end_point").split(',')]

        assert 0 <= self.s_x and self.e_x <= shape_x and 0 <= self.s_y and self.e_y <= shape_y
        assert self.s_x < self.e_x and self.s_y < self.e_y and self.s_z < self.e_z

        self.begin_x, self.begin_y = max(0, self.s_x - self.overlap), max(0, self.s_y - self.overlap)
        self.end_x, self.end_y = min(shape_x, self.e_x + self.overlap), min(shape_y, self.e_y + self.overlap)
        self.pad_s_x = self.begin_x - self.s_x + self.overlap
        self.pad_s_y = self.begin_y - self.s_y + self.overlap
        self.pad_e_x = self.e_x + self.overlap - self.end_x
        self.pad_e_y = self.e_y + self.overlap - self.end_y

        assert self.s_x - self.e_x < self.cube and self.s_y - self.e_y < self.cube and self.s_z - self.e_z < self.cube

        return [z for z in range(self.s_z, self.e_z - self.cube, self.cube)] + [int(self.e_z) - self.cube]

    def segment_brain_batch(self, z):
        volume = []
        for i in range(z - self.overlap, z + self.cube + self.overlap):
            if 0 <= i < len(self.files):
                im = np.array(Image.open(self.files[i])).astype(np.float32)
                img = im[self.begin_y: self.end_y, self.begin_x: self.end_x]
                img = np.pad(img, ((self.pad_s_y, self.pad_e_y), (self.pad_s_x, self.pad_e_x)), 'edge')
                volume.append(img)
            else:
                blank = np.zeros((self.end_y - self.begin_y + self.pad_s_y + self.pad_e_y,
                                  self.end_x - self.begin_x + self.pad_s_x + self.pad_e_x))
                volume.append(blank)
        volume = np.array(volume).astype(np.float32)
        seg_res = np.zeros_like(volume)
        shape_y, shape_x = volume.shape[1:]
        seg = []
        overlap = self.overlap
        cube = self.cube
        mean = im.mean()
        std = im.std()
        print(mean, std)
        for y in range(overlap, shape_y - cube - overlap + 1, cube):
            for x in range(overlap, shape_x - cube - overlap + 1, cube):
                v = volume[:, y - overlap: y - overlap + self.input_dim, x - overlap: x - overlap + self.input_dim]
                seg.append(equal(v[np.newaxis, ...], mean, std))
                # seg.append(equal(v)[np.newaxis, ...])
                if x + 2 * cube + overlap >= shape_x and y + 2 * cube + overlap >= shape_y:
                    v = volume[:, shape_y - self.input_dim: shape_y, shape_x - self.input_dim: shape_x][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
                if x + 2 * cube + overlap >= shape_x:
                    v = volume[:, y - overlap: y + overlap + cube, shape_x - self.input_dim: shape_x][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
                if y + 2 * cube + overlap >= shape_y:
                    v = volume[:, shape_y - self.input_dim: shape_y, x - overlap: x + overlap + cube][np.newaxis, ...]
                    seg.append(equal(v, mean, std))
                    # seg.append(equal(v))
        print('crop finished.')
        seg_sets = DataLoader(seg, batch_size=self.opt.batch_size, shuffle=False)
        segments = []
        for datas in seg_sets:
            pred = self.model(datas.to(self.device))
            if self.opt.output_nc == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=1)[:, 1, ...]

            pred = pred.reshape(-1, self.input_dim, self.input_dim, self.input_dim).detach().cpu().numpy()
            pred = pred[:, overlap: overlap + cube, overlap: overlap + cube, overlap: overlap + cube] * 255
            if len(segments) == 0:
                segments = pred
            else:
                segments = np.concatenate((segments, pred), axis=0)
        i = 0
        for y in range(overlap, shape_y - cube - overlap + 1, cube):
            for x in range(overlap, shape_x - cube - overlap + 1, cube):
                seg_res[overlap: self.input_dim - overlap, y: y + cube, x: x + cube] = segments[i]
                i += 1
                if x + 2 * cube + overlap >= shape_x and y + 2 * cube + overlap >= shape_y:
                    seg_res[overlap: self.input_dim - overlap, shape_y - overlap - cube: shape_y - overlap,
                            shape_x - cube - overlap: shape_x - overlap] = segments[i]
                    i += 1
                if x + 2 * cube + overlap >= shape_x:
                    seg_res[overlap: self.input_dim - overlap, y: y + cube, shape_x - cube - overlap: shape_x - overlap] = segments[i]
                    i += 1
                if y + 2 * cube + overlap >= shape_y:
                    seg_res[overlap: self.input_dim - overlap, shape_y - overlap - cube: shape_y - overlap, x: x + cube] = segments[i]
                    i += 1
        i = z
        for img in seg_res[overlap: self.input_dim - overlap, overlap: shape_y - overlap, overlap: shape_x - overlap]:
            tifffile.imsave(os.path.join(self.target, str(i).zfill(4) + '.tiff'), img.astype(np.uint8))
            i += 1

        print(z)

    @staticmethod
    def eval_net(model, testloader, device, n_val):
        total_loss_iou = 0
        total_loss_tiou = 0
        junk_rat = 0
        cl_recall, cl_acc = 0, 0
        global_steps = 0
        model.eval()
        with tqdm(total=n_val, desc='Validation round', unit='batch') as pbar:
            for batch, (data, label) in enumerate(testloader):
                data = Variable(data.to(device))
                label = Variable(label.clone().to(device))
                with torch.no_grad():
                    pre = model(data)
                    if len(label.shape) == 4:
                        pre = torch.argmax(torch.softmax(pre, dim=1), dim=1).unsqueeze(1).float()
                        label = label[:, np.newaxis, ...].float()
                    else:
                        pre = torch.sigmoid(pre)
                pre[pre > 0.5] = 1
                pre[pre <= 0.5] = 0
                # label[label == 255] = 0
                # label[label > 0.5] = 1
                # label[label <= 0.5] = 0
                tifffile.imsave('./predict/' + str(batch) + '.tiff', pre.cpu().numpy()[0][0])
                tifffile.imsave('./predict/' + str(batch) + '_label.tiff', label.cpu().numpy()[0][0])
                tifffile.imsave('./predict/' + str(batch) + '_data.tiff', data.cpu().numpy()[0][0])

                total_loss_iou += iou(pre, label).cpu()
                total_loss_tiou += t_iou(pre, label).cpu()
                junk_rat += junk_ratio(pre, label).cpu()
                recall, acc = soft_cldice_f1(pre, label)
                cl_recall += recall.cpu()
                cl_acc += acc.cpu()
                global_steps += 1
                pbar.update(data.shape[0])
        model.train()
        cl_recall_mean = cl_recall / global_steps
        cl_acc_mean = cl_acc / global_steps
        return {'iou': total_loss_iou / global_steps,
                'cldice': (2. * cl_recall_mean * cl_acc_mean) / (cl_recall_mean + cl_acc_mean),
                'cl_acc': cl_acc_mean,
                'cl_recall': cl_recall_mean,
                'tiou': total_loss_tiou / global_steps,
                'junk_ratio': junk_rat / global_steps}

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(semi='False')
        val_type = parser.parse_known_args()[0].val_type
        if val_type == 'cubes':
            parser.add_argument('--batch_size', type=int, default=10, help='batch size')
            parser.add_argument('--n_samples', type=int, default=4, help='crop n volumes from one cube')
            parser.add_argument('--artifacts', type=bool, default=True, help='train with artifacts volumes')
            parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model')
            parser.add_argument('--shuffle_val', type=bool, default=False, help='whether to shuffle the val data')
        elif val_type == 'volumes2':
            parser.add_argument('--data_target', type=str, help='target volume for evaluating')
            parser.add_argument('--pool_kernel', type=int, default=5, help='maxpooling kernel size')
        elif val_type == 'segment':
            parser.add_argument('--net', type=str, help='unet | axialunet')
            parser.add_argument('--overlap', type=int, default=24, help='overlap size during the prediction')
            parser.add_argument('--section', type=str, help='section for model prediction')
            parser.add_argument('--batch_size', type=int, default=10, help='batch size')
            parser.add_argument('--pre_trained', type=str, default=None, help='pre-trained model')
            parser.add_argument('--exp', type=str, required=True, help='experiment name')
            parser.add_argument('--process', type=int, default=2, help='processes number')

        return parser