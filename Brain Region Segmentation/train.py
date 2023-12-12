from torch.utils.data import DataLoader
import torch, os

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import torch.optim as optim
import config
from utils import common
from tqdm import tqdm
from collections import OrderedDict
from load_dataset_3D import *
import torch.nn as nn
from torch.autograd import Variable
import random
import tifffile
import cv2
import numpy as np
from cea_net import ceanet

np.seterr(divide='ignore', invalid='ignore')

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
args = config.args

# semi-supervised mean-teacher training
def train_data(model, optimizer, ema_model, iter_num, data, target, a):
    data, target = Variable(data), Variable(target)
    data = data.float()  # student model input
    data, target = data.cuda(), target.cuda()
    unlabeled_volume_batch = data[args.labeled_bs:]
    unlabeled_volume_batch_new = unlabeled_volume_batch.cpu().numpy()
    ema_inputs = []

    for i in range(unlabeled_volume_batch_new.shape[0]):  # save the before and after flip data
        img = unlabeled_volume_batch_new[i][0]
        cv2.imwrite('temp_m/data_before.tiff', img)
        temp = np.flip(img, axis=a)

        cv2.imwrite('temp_m/data_after.tiff', temp)
        temp = np.array(temp)
        temp = temp.reshape(1, img.shape[0], img.shape[1])
        ema_inputs.append(temp)

    ema_inputs = np.array(ema_inputs)
    ema_inputs = torch.from_numpy(ema_inputs).cuda()  # teacher model input

    output = model(data)   # student model output
    outputs_soft = torch.sigmoid(output)

    with torch.no_grad():
        ema_output = ema_model(ema_inputs)   # teacher model output
        ema_output_soft_old = torch.sigmoid(ema_output)

    bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50])).cuda()
    supervised_loss = bce1(output[:args.labeled_bs], target[:args.labeled_bs])   # training data loss function (BCE loss)

    for i in range(ema_output_soft_old.shape[0]):  # save the after flip prediction
        ema_output_soft_old[i] = torch.flip(ema_output_soft_old[i], dims=(a + 1,))
        temp = np.array(ema_output_soft_old[i].detach().cpu())
        cv2.imwrite('temp_m/pred1_after1.tiff', temp[0])

    consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - ema_output_soft_old) ** 2)   # unlabeled data loss function

    loss = supervised_loss + consistency_loss    # mean teacher loss function
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    update_ema_variables(model, ema_model, args.ema_decay, iter_num)
    iter_num = iter_num + 1
    return (loss, consistency_loss, supervised_loss, iter_num, outputs_soft)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def load_model(model, model_path):
    print(model_path)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['net']
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

# multi-view training
def train(iter_num_1, iter_num_2):
    common.adjust_learning_rate(optimizer_1, epoch, args)
    common.adjust_learning_rate(optimizer_2, epoch, args)
    print("=======Epoch:{}=======".format(epoch))
    model_1.train()
    model_2.train()
    loss_1_2D, loss_2_2D, loss_labeled_3d, loss_unlabeled_3d = 0, 0, 0, 0
    temp_p_l = []
    temp_p_u = []
    print('------------view1 begin----------------')
    # view1 mean teacher training
    for idx, d in tqdm(enumerate(train_loader_1), total=len(train_loader_1)):

        data_1 = d["img"]
        target_1 = d["cate"]
        name_1 = d["mask_name"]
        loss_1, consistency_loss_1_2D, supervised_loss_1_2D, iter_num_1, pred_1 = train_data(model_1,
                                                                                             optimizer_1,
                                                                                             ema_model_1,
                                                                                             iter_num_1, data_1,
                                                                                             target_1, 1,
                                                                                             args.num_classes,
                                                                                             args.epochs)

        loss_1_2D += loss_1

        data_1 = data_1.cpu().numpy()

        for i in range(args.labeled_bs):
            p = {'name': name_1[i], 'pred': pred_1[i], 'data': data_1[i][0]}
            temp_p_l.append(p)
        for i in range(args.batch_size - args.labeled_bs):
            u = {'name': name_1[i + args.labeled_bs], 'pred': pred_1[i + args.labeled_bs],
                 'data': data_1[i + args.labeled_bs][0]}
            temp_p_u.append(u)
    # sort the 2d data
    loss_1_2D /= len(train_loader_1)
    p = sorted(temp_p_l, key=lambda i: i['name'])
    u = sorted(temp_p_u, key=lambda i: i['name'])
#    p.reverse()
#    u.reverse()

    # combine the 2d data to 3d
    u_pred_1 = []
    l_pred_1 = []
    l_data_1 = []
    u_data_1 = []
    for i in range(len(p)):
        l_pred_1.append(p[i]['pred'])
        l_data_1.append(p[i]['data'])
    for i in range(len(u)):
        u_pred_1.append(u[i]['pred'])
        u_data_1.append(u[i]['data'])
    print(len(u))

    # view2 mean teacher training
    print('------------view2 begin----------------')
    temp_p_l = []
    temp_p_u = []
    for idx, d in tqdm(enumerate(train_loader_2), total=len(train_loader_2)):

        data_2 = d["img"]
        target_2 = d["cate"]

        name_2 = d["mask_name"]

        loss_2, consistency_loss_2_2D, supervised_loss_2_2D, iter_num_2, pred_2 = train_data(model_2,
                                                                                             optimizer_2,
                                                                                             ema_model_2,
                                                                                             iter_num_2, data_2,
                                                                                             target_2, 0,
                                                                                             args.num_classes,
                                                                                             args.epochs)

        loss_2_2D += loss_2

        data_2 = data_2.cpu().numpy()

        for i in range(args.labeled_bs):
            p = {'name': name_2[i], 'pred': pred_2[i], 'data': data_2[i][0]}
            temp_p_l.append(p)
        for i in range(args.batch_size - args.labeled_bs):
            u = {'name': name_2[i + args.labeled_bs], 'pred': pred_2[i + args.labeled_bs],
                 'data': data_2[i + args.labeled_bs][0]}
            temp_p_u.append(u)

    loss_2_2D /= len(train_loader_2)

    p = sorted(temp_p_l, key=lambda i: i['name'])
    u = sorted(temp_p_u, key=lambda i: i['name'])
#    p.reverse()
#    u.reverse()
    u_pred_2 = []
    l_pred_2 = []
    l_data_2 = []
    u_data_2 = []
    for i in range(len(p)):
        l_pred_2.append(p[i]['pred'])
        l_data_2.append(p[i]['data'])
    for i in range(len(u)):
        #            print(i, u[i]['name'])
        u_pred_2.append(u[i]['pred'])
        u_data_2.append(u[i]['data'])

    print('------------ready to 2D to 3D---------------')

    l_pred_1 = torch.cat([torch.unsqueeze(i, 0) for i in l_pred_1], 0)
    l_pred_2 = torch.cat([torch.unsqueeze(i, 0) for i in l_pred_2], 0)
    u_pred_1 = torch.cat([torch.unsqueeze(i, 0) for i in u_pred_1], 0)
    u_pred_2 = torch.cat([torch.unsqueeze(i, 0) for i in u_pred_2], 0)
    print('before: ', l_pred_1.shape, l_pred_2.shape, u_pred_1.shape, u_pred_2.shape)
    a, c = 320, 512  # the number of one brain

    # Handle one 3d output of two view to make their directions consistent.
    l_pred_11 = torch.transpose(l_pred_1[:c], 0, 1)
    u_pred_11 = torch.transpose(u_pred_1[:c], 0, 1)
    l_pred_21 = l_pred_2[:a]
    u_pred_21 = u_pred_2[:a]
    l_pred_21 = l_pred_21.permute(1, 3, 0, 2)
    u_pred_21 = u_pred_21.permute(1, 3, 0, 2)
    print('after: ', l_pred_11.shape, l_pred_21.shape, u_pred_11.shape, u_pred_21.shape)

    # Save the 3d output of two view
    u_pred_1_n = np.array(u_pred_11.detach().cpu())
    u_pred_2_n = np.array(u_pred_21.detach().cpu())
    l_pred_1_n = np.array(l_pred_11.detach().cpu())
    l_pred_2_n = np.array(l_pred_21.detach().cpu())
    tifffile.imwrite('temp_m/a_pred_after_u1_0.tiff', u_pred_2_n[0])
    tifffile.imwrite('temp_m/c_pred_after_u1_0.tiff', u_pred_1_n[0])
    tifffile.imwrite('temp_m/a_pred_after_l1_0.tiff', l_pred_2_n[0])
    tifffile.imwrite('temp_m/c_pred_after_l1_0.tiff', l_pred_1_n[0])

    # 3d loss function
    loss_labeled_3d = torch.mean((l_pred_11 - l_pred_21) ** 2)
    loss_unlabeled_3d = torch.mean((u_pred_11 - u_pred_21) ** 2)

    count_l = len(l_pred_1) // c
    count_u = len(u_pred_1) // c

    # Batch processing of multiple labeled brains
    for i in range(1, count_l):
        start_1 = i * c
        end_1 = (i+1) * c
        temp_pred_1 = l_pred_1[start_1: end_1]

        start_2 = i * a
        end_2 = (i+1) * a
        temp_pred_2 = l_pred_2[start_2: end_2]

        print('split: ', temp_pred_1.shape, temp_pred_2.shape)

        temp_pred_1 = torch.transpose(temp_pred_1, 0, 1)
        temp_pred_2 = temp_pred_2.permute(1, 3, 0, 2)
        print('after: ', temp_pred_1.shape, temp_pred_2.shape)

        l = torch.mean((temp_pred_1 - temp_pred_2) ** 2)
        loss_labeled_3d += l
    # Batch processing of multiple unlabeled brains
    for i in range(1, count_u):
        start_1 = i * c
        end_1 = (i+1) * c
        temp_pred_1 = u_pred_1[start_1: end_1]

        start_2 = i * a
        end_2 = (i+1) * a
        temp_pred_2 = u_pred_2[start_2: end_2]

        print('split: ', temp_pred_1.shape, temp_pred_2.shape)

        temp_pred_1 = torch.transpose(temp_pred_1, 0, 1)
        temp_pred_2 = temp_pred_2.permute(1, 3, 0, 2)
        print('after: ', temp_pred_1.shape, temp_pred_2.shape)

        l = torch.mean((temp_pred_1 - temp_pred_2) ** 2)
        loss_unlabeled_3d += l

        l_pred_2_n = np.array(temp_pred_1.detach().cpu())
        u_pred_2_n = np.array(temp_pred_2.detach().cpu())

        tifffile.imwrite('temp_m/a_pred_after_u1_' + str(i) + '.tiff', u_pred_2_n[0])
        tifffile.imwrite('temp_m/c_pred_after_u1_' + str(i) + '.tiff', u_pred_1_n[0])
        tifffile.imwrite('temp_m/a_pred_after_u2_' + str(i) + '.tiff', u_pred_2_n[1])
        tifffile.imwrite('temp_m/c_pred_after_u2_' + str(i) + '.tiff', u_pred_1_n[1])
    # Total Loss function
    train_loss = loss_1_2D + loss_2_2D + loss_unlabeled_3d + loss_labeled_3d

    train_loss_3d = loss_unlabeled_3d + loss_labeled_3d
    train_loss_3d = Variable(train_loss_3d, requires_grad=True)

    optimizer_1.zero_grad()
    optimizer_2.zero_grad()
    train_loss_3d.backward()
    optimizer_1.step()
    optimizer_2.step()

    print('-----------loss------------------')
    print('train_loss, loss_1, loss_2, loss_labeled_3d, loss_unlabeled_3d, train_loss_3d', train_loss.item(),
          loss_1_2D.item(), loss_2_2D.item(), loss_labeled_3d.item(), loss_unlabeled_3d.item(),
          train_loss_3d.item())
    #        print('loss_1', loss_1_2D.item())

    state_1 = {'net': model_1.state_dict(), 'optimizer': optimizer_1.state_dict(), 'epoch': epoch}
    state_2 = {'net': model_2.state_dict(), 'optimizer': optimizer_2.state_dict(), 'epoch': epoch}
    if (best[1] > train_loss.item()):
        print('Saving best model')
        best[0] = epoch
        best[1] = train_loss.item()
        torch.save(state_1, os.path.join(args.save, f'best_model_1.pth'))
        torch.save(state_2, os.path.join(args.save, f'best_model_2.pth'))
    print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
    if epoch == args.epochs:
        torch.save(state_1, os.path.join(args.save, f'epoch{epoch}_1.pth'))
        torch.save(state_2, os.path.join(args.save, f'epoch{epoch}_2.pth'))
    return iter_num_1, iter_num_2

if __name__ == '__main__':

    # view1
    train_set_1 = BasicDataset(args.dataset_path_1, args.label_txt_1, args.unlabel_txt_1, 'c')
    print('dataset_1:', len(train_set_1))

    labeled_num = len(open(os.path.join(args.dataset_path_1, args.label_txt_1), 'r').readlines())
    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, len(train_set_1)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    train_loader_1 = DataLoader(train_set_1, batch_sampler=batch_sampler, num_workers=16, pin_memory=False,
                                worker_init_fn=worker_init_fn)

    # view2
    train_set_2 = BasicDataset(args.dataset_path_2, args.label_txt_2, args.unlabel_txt_2, 'a')
    print('dataset_2:', len(train_set_2))
    labeled_num = len(open(os.path.join(args.dataset_path_2, args.label_txt_2), 'r').readlines())
    labeled_idxs = list(range(0, labeled_num))
    unlabeled_idxs = list(range(labeled_num, len(train_set_2)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    train_loader_2 = DataLoader(train_set_2, batch_sampler=batch_sampler, num_workers=16, pin_memory=False,
                                worker_init_fn=worker_init_fn)

    model_1 = ceanet.CEA_Net(num_classes=args.num_classes, num_channels=args.channel)
    ema_model_1 = ceanet.CEA_Net(num_classes=args.num_classes, num_channels=args.channel)
    model_1 = nn.DataParallel(model_1)
    model_1 = model_1.cuda()
    ema_model_1 = nn.DataParallel(ema_model_1)
    ema_model_1 = ema_model_1.cuda()
    print(model_1)
    optimizer_1 = optim.RMSprop(model_1.parameters(), lr=args.lr_gen, weight_decay=1e-6, momentum=0.9)
    scheduler_1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, 'min', patience=2)

    model_2 = ceanet.CEA_Net(num_classes=args.num_classes, num_channels=args.channel)
    ema_model_2 = ceanet.CEA_Net(num_classes=args.num_classes, num_channels=args.channel)
    model_2 = nn.DataParallel(model_2)
    model_2 = model_2.cuda()
    ema_model_2 = nn.DataParallel(ema_model_2)
    ema_model_2 = ema_model_2.cuda()
    optimizer_2 = optim.RMSprop(model_2.parameters(), lr=args.lr_gen, weight_decay=1e-6, momentum=0.9)
    scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, 'min', patience=2)

    best = [0, np.inf]  # Initialize epoch and performance of the Optimal Model
    trigger = 0  # early stop
    iter_num_1 = 0
    iter_num_2 = 0
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for epoch in range(1, args.epochs + 1):
        iter_num_1, iter_num_2 = train(iter_num_1, iter_num_2)
        torch.cuda.empty_cache()

