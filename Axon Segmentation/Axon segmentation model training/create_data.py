#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from PIL import Image
import random
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
from nnunet.paths import preprocessing_output_dir
from skimage.io import imread
import pdb
from tqdm import tqdm
import myutils
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms
import tifffile

join = os.path.join
artifact = True
dim = 64
sample = 10
isTrain = True


def load_tiff_convert_to_nifti(img_file, lab_file, img_out_base, anno_out):
    img = imread(img_file)
    img_itk = sitk.GetImageFromArray(img.astype(np.float32))
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if lab_file is not None:
        l = imread(lab_file)
        l[l > 0] = 1
        l_itk = sitk.GetImageFromArray(l.astype(np.float32))
        sitk.WriteImage(l_itk, anno_out)


def np_convert_to_nifti(data, label, img_out_base, anno_out):
    img_itk = sitk.GetImageFromArray(data.astype(np.float32))
    sitk.WriteImage(img_itk, join(img_out_base + "_0000.nii.gz"))

    if label is not None:
        l_itk = sitk.GetImageFromArray(label.astype(np.float32))
        sitk.WriteImage(l_itk, anno_out)


def histogram_match_data(base, source, task_id, task_name, spacing, n_samples, input_dim,
                         data_mix, flag, match_flag, cutmix, join_flag):
    p = Pool(16)
    augment = myutils.dataAugmentation()

    foldername = "Task%03.0d_%s" % (task_id, task_name)
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    origin = join(out_base, "origin")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(origin)

    train_patient_names = []
    test_patient_names = []
    res = []

    casenames = []

    datas = []
    labels = []
    datas_ori = []
    labels_ori = []
    data_path = join(base, 'train') if flag else join(base, 'val')

    volumes_folder_path = join(data_path, "volumes")
    # skeletonized annotations are used for training, while original annotations are used for evaluation
    labels_folder_path = join(data_path, "Fine-label") if flag else join(data_path, 'Rough-label')
    volumes_path = myutils.get_dir(volumes_folder_path)
    labels_path = myutils.get_dir(labels_folder_path)
    assert len(labels_path) == len(volumes_path)
    if n_samples == None:
        n_samples = len(labels_path)
    artifacts_folder_path = data_path + '/artifacts/'
    artifacts_path = myutils.get_dir(artifacts_folder_path)
    if match_flag:
        source_path = join(source, 'train')
        volumes_folder_path_s = join(source_path, "volumes")
        volumes_path_s = myutils.get_dir(volumes_folder_path_s)
        labels_folder_path_s = join(source_path, "Fine-label")
        labels_path_s = myutils.get_dir(labels_folder_path_s)
        artifacts_folder_path_s = source_path + '/artifacts/'
        artifacts_path_s = myutils.get_dir(artifacts_folder_path_s)

    total_ori_volumes = 0
    with tqdm(total=(len(volumes_path) + len(artifacts_path)) * 2, desc=f'original volume numbers') as pbar:
        for vpath, lpath in zip(volumes_path, labels_path):
            case = str(vpath).split(".")[0].split("-")[1]
            casename = task_name + case
            casenames.append(casename)

            volume = myutils.read_tiff_stack(vpath)
            label = myutils.read_tiff_stack(lpath)
            if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                    or volume.shape[2] < input_dim:
                continue
            datas_ori.append(volume)
            labels_ori.append(label)
            total_ori_volumes += 1
            pbar.update()
        if match_flag and join_flag:
            for vpath_s, lpath_s in zip(volumes_path_s, labels_path_s):
                case = str(vpath_s).split(".")[0].split("-")[1]
                casename = task_name + case
                casenames.append(casename)

                volume = myutils.read_tiff_stack(vpath_s)
                label = myutils.read_tiff_stack(lpath_s)
                if volume.shape[0] < input_dim or volume.shape[1] < input_dim \
                        or volume.shape[2] < input_dim:
                    continue
                datas_ori.append(volume)
                labels_ori.append(label)
                total_ori_volumes += 1
                pbar.update()
        axon_num = total_ori_volumes
        print("axon_num: ", axon_num)
        if match_flag:
            for i in range(axon_num):
                casename = casenames[i] + "m"
                casenames.append(casename)
                volume = datas_ori[i].copy()
                label = labels_ori[i].copy()
                match_ref_path = random.choice(volumes_path_s)
                match_ref = myutils.read_tiff_stack(match_ref_path)
                volume_match = match_histograms(volume, match_ref)
                datas_ori.append(volume_match)
                labels_ori.append(label)
                total_ori_volumes += 1
                pbar.update()
        aug_axon_num = total_ori_volumes
        print("aug_axon_num: ", aug_axon_num)
        if data_mix:
            for apath in artifacts_path:
                case_a = str(apath).split(".")[0].split("-")[1]
                casename_a = task_name + "a" + case_a
                casenames.append(casename_a)

                ak_seed = random.randint(0, 3)
                artifact = myutils.read_tiff_stack(apath)
                if artifact.shape[0] < input_dim or artifact.shape[1] < input_dim \
                        or artifact.shape[2] < input_dim:
                    print(artifact.shape)
                    continue
                datas_ori.append(artifact)
                labels_ori.append(np.zeros_like(label))
                total_ori_volumes += 1
                pbar.update()
            if match_flag and join_flag:
                for apath_s in artifacts_path_s:
                    case_a = str(apath_s).split(".")[0].split("-")[1]
                    casename_a = task_name + "a" + case_a
                    casenames.append(casename_a)
                    artifact = myutils.read_tiff_stack(apath_s)
                    if artifact.shape[0] < input_dim or artifact.shape[1] < input_dim \
                            or artifact.shape[2] < input_dim:
                        print(artifact.shape)
                        continue
                    datas_ori.append(artifact)
                    labels_ori.append(np.zeros_like(label))
                    total_ori_volumes += 1
                    pbar.update()
            junk_num = total_ori_volumes - aug_axon_num
            print("junk_num", junk_num)
            if match_flag:  # histogram matching
                for i in range(junk_num):
                    casename = casenames[i + aug_axon_num] + "m"
                    casenames.append(casename)
                    volume = datas_ori[i + aug_axon_num].copy()
                    label = labels_ori[i + aug_axon_num].copy()
                    match_ref_path = random.choice(artifacts_path_s)
                    match_ref = myutils.read_tiff_stack(match_ref_path)
                    volume_match = match_histograms(volume, match_ref)
                    datas_ori.append(volume_match)
                    labels_ori.append(label)
                    total_ori_volumes += 1
                    pbar.update()
            aug_junk_num = total_ori_volumes - aug_axon_num
            if cutmix:
                for j in range(axon_num):
                    casename = casenames[j] + "c"
                    casenames.append(casename)
                    m_seed = random.randint(0, 1)
                    volume = datas_ori[j + axon_num].copy() if match_flag and m_seed else datas_ori[j].copy()
                    label = labels_ori[j + axon_num].copy() if match_flag and m_seed else labels_ori[j].copy()
                    a_index = random.randint(0, junk_num) + aug_axon_num
                    artifact_cut = datas_ori[a_index + junk_num] if match_flag and m_seed else datas_ori[a_index]
                    z = random.randint(0, label.shape[0])
                    x = random.randint(0, label.shape[1])
                    y = random.randint(0, label.shape[2])
                    artifact_chunk = artifact_cut[:z, :x, :y].copy()
                    volume[:z, :x, :y] = artifact_chunk
                    label[:z, :x, :y] = np.zeros_like(artifact_chunk)
                    tifffile.imwrite(join(origin, casename) + "_vol.tiff", np.array(volume).astype(np.uint16))
                    tifffile.imwrite(join(origin, casename) + "_lab.tiff", np.array(label).astype(np.uint16))
                    datas_ori.append(volume)
                    labels_ori.append(label)
                    total_ori_volumes += 1
    print("{} data of original size finish.".format(total_ori_volumes))
    # data_len should be adjusted due to dataset diversity
    data_len = (total_ori_volumes - aug_junk_num) * n_samples + aug_junk_num * (
        max(1, int(len(volumes_path) / len(artifacts_path) * n_samples / 2)))
    print(data_len)

    total_volumes = 0
    with tqdm(total=data_len, desc='volumes numbers') as pbar:
        for index in range(total_ori_volumes):
            img_out_base = join(imagestr, casenames[index])
            anno_out_base = join(labelstr, casenames[index])
            if index < aug_axon_num or index >= aug_axon_num + aug_junk_num:
                volume = datas_ori[index].copy()
                label = labels_ori[index].copy()
                for j in range(int(n_samples)):
                    img_out = join(img_out_base + str(j))
                    anno_out = join(anno_out_base + str(j) + ".nii.gz")
                    casename_s = casenames[index] + str(j)
                    z = random.randint(0, label.shape[0] - input_dim)
                    x = random.randint(0, label.shape[1] - input_dim)
                    y = random.randint(0, label.shape[2] - input_dim)
                    volume_chunk = volume[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    label_chunk = label[z:z + input_dim, x:x + input_dim, y:y + input_dim].copy()
                    # ---------------- rotation ---------------- #
                    callfunc = {
                        0: lambda: [np.fliplr(volume_rot), np.fliplr(label_rot)],
                        1: lambda: [np.flipud(volume_rot), np.flipud(label_rot)],
                    }
                    k_seed = random.randint(0, 3)
                    flip_seed = random.randint(0, 1)
                    volume_rot = np.rot90(np.swapaxes(volume_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    label_rot = np.rot90(np.swapaxes(label_chunk, 0, 2), k=k_seed).swapaxes(2, 0)
                    volume_chunk, label_chunk = callfunc[flip_seed]()
                    # data augmentation
                    data, annotation = augment.data_augmentation(volume_chunk, label_chunk)
                    data = data / 6553
                    data_ori = ((data - data.min()) / (data.max() - data.min()))
                    annotation[annotation > 0] = 1

                    datas.append(data_ori[np.newaxis, :, :, :].astype(np.float32))
                    labels.append(annotation[np.newaxis, ...].astype(np.float32))
                    res.append(
                        p.starmap_async(np_convert_to_nifti, ((data_ori, annotation, img_out, anno_out),)))
                    train_patient_names.append(casename_s)

                    total_volumes += 1
                    pbar.update()
                total_volumes_axon = total_volumes
            else:
                artifact = datas_ori[index].copy()
                # changing artifact number base on dataset length and n_samples
                for k in range(max(1, int(len(volumes_path) / len(artifacts_path) * n_samples / 2))):
                    a_out = join(img_out_base + str(k))
                    l_out = join(anno_out_base + str(k) + ".nii.gz")
                    casename_as = casenames[index] + str(k)
                    z = random.randint(0, artifact.shape[0] - input_dim)
                    x = random.randint(0, artifact.shape[1] - input_dim)
                    y = random.randint(0, artifact.shape[2] - input_dim)
                    artifact = artifact[z:z + input_dim, x:x + input_dim, y:y + input_dim]
                    artifact = np.rot90(np.swapaxes(artifact, 0, 2), k=ak_seed).swapaxes(2, 0)
                    artifact = (artifact - artifact.min()) / (artifact.max() - artifact.min())
                    data_ori = artifact.copy()
                    annotation[annotation > 0] = 0
                    datas.append(data_ori[np.newaxis, :, :, :].astype(np.float32))
                    labels.append(annotation[np.newaxis, ...].astype(np.float32))
                    res.append(
                        p.starmap_async(np_convert_to_nifti, ((data_ori, annotation, a_out, l_out),)))
                    train_patient_names.append(casename_as)
                    total_volumes += 1
                    pbar.update()
        print("train {} cases finish".format(total_volumes))

        # generate test set(not necessary for training process)
        if os.path.exists(join(source, "val", "volumes")) and match_flag:
            val_volume_path = join(source, "val", "volumes")
        else:
            val_volume_path = join(base, "val", "volumes")
        vpaths = os.listdir(val_volume_path)
        for i, vpath in enumerate(vpaths):
            case = str(vpath).split(".")[0].split("-")[1]
            volume = join(val_volume_path, "volume-" + case + ".tiff")
            label = None
            casename = task_name + case
            img_out_base = join(imagests, casename)
            anno_out = None
            res.append(
                p.starmap_async(load_tiff_convert_to_nifti, ((volume, label, img_out_base, anno_out),)))
            test_patient_names.append(casename)

        # write basic information of dataset to dataset.json, needed for nnUNet preprocessing
        _ = [i.get() for i in res]
        json_dict = {'name': task_name, 'description': "", 'tensorImageSize': "4D", 'reference': "", 'licence': "",
                     'release': "0.0", 'modality': {
                      "0": "MI",  # microscope image
                      }, 'labels': {
                      "0": "background",
                      "1": "axon",
                    }, 'numTraining': len(train_patient_names), 'numTest': len(test_patient_names),
                     'training': [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                                  train_patient_names],
                     'test': ["./imagesTs/%s.nii.gz" % i for i in test_patient_names]}

        save_json(json_dict, os.path.join(out_base, "dataset.json"))
        p.close()
        p.join()


if __name__ == "__main__":
    base = ""  # train data path
    source = ""  # (if needed)data used for histogram match
    task_id = 700  # task id should be unique(better >200 to avoid conflict)
    task_name = ''
    spacing = (1, 0.126, 0.126)  # probably no need to change
    histogram_match_data(base, source, task_id, task_name, spacing,
                         2,  # number of samples(usually 2-6, depend on number of original training cubes)
                         128,  # input dimension(128*128*128)
                         data_mix=True, flag=True,  # artifact mix, train
                         # histogram match, cutmix, join(matched and origin)
                         match_flag=True, cutmix=True, join_flag=False)
