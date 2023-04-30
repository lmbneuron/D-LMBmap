import numpy as np
import torch
import os
from utils.common import *
import SimpleITK as sitk
import config
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm
from load_dataset_3D import *
from torch.utils.data import DataLoader
from ce_net import cenet
from collections import OrderedDict



def load_model(model, model_path):
    print(model_path)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['net']
    # create new OrderedDict that does not contain `module.`
        
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

if __name__ == "__main__":
    args = config.args
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = cenet.CE_Net(num_classes=args.num_classes, num_channels=args.channel).cuda()
    net = load_model(net, '{}/epoch{}_1.pth'.format(args.save, args.epochs))  # View1
    net.eval()

    test_set = BasicDataset_test(args.dataset_path_1, args.test_txt)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
    for d in tqdm(test_loader,  total=len(test_loader)):
        
        img = d["img"]
        save_res_path = (d["mask_name"][0].split('/')[-1])

        img = img.float()
        img = img.cuda()

        pred = net(img)
        pred = torch.sigmoid(pred)
        pred = np.array(pred.data.cpu()[0])     
        pred = np.array(pred)

        threshold = 0.7
        pred[pred >= threshold] = 1
        pred[pred < threshold] = 0

        pred_new = np.zeros((pred.shape[1], pred.shape[2]))
        pred_new[pred[0] == 1] = 1
        pred_new[pred[1] == 1] = 2

        # Save the output
        count =  args.save.split('/')[-1].replace('model', '')
        file_name = d["mask_name"][0].split('/')[-2].replace('data', count)
        result_save_path = args.save_nii + file_name + '_' + args.name
        if not os.path.exists(result_save_path): os.mkdir(result_save_path)
        mask = sitk.GetImageFromArray(pred_new.astype(np.uint8))
        sitk.WriteImage(mask, os.path.join(result_save_path, save_res_path))

