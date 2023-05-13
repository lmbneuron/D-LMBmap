import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np 
import nrrd
from math import log2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
sys.path.append("../..")
from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack
from allen.allenmap import get_allchildren_id_by_struct_name, get_struct_name_by_struct_id, get_all_children_id_by_struct_id


def main():
    # path_list = [
        
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_181208_15_21_38",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_181207_10_39_06",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_181207_18_26_44"
    # ]
    # path_list = [
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_180921_O11_488_LEFT_16-07-16",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_180725_20180724C2_LEFT_488_100ET_20-02-09",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_180724_20180723O2_LEFT_488-2_100ET_16-13-41"
    # ]
    # path_list = [
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_190312_488_LP70_ET50_Z08_HF0_17-26-21",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_190313_488_LP70_ET50_Z08_HF0_01-04-17",
    #     "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/test/reg/allen_190524_488_ET50_0HF_LP70_18-43-49"
    # ]
    path_list = [
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/soma_nuclei_serotonin/180921_O11_488_LEFT_16-07-16.tif",
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/soma_nuclei_serotonin/180725_20180724C2_LEFT_488_100ET_20-02-09.tif",
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/soma_nuclei_serotonin/180724_20180723O2_LEFT_488-2_100ET_16-13-41.tif"
    ]
    regions = ["cp", "hpf", "ctx", "cbx", "cb", "bs"]
    dst_path = "amap_iv_map.tiff"
    data_list = []
    for dir in tqdm(path_list):
        name = os.path.split(dir)[1]
        # data = read_tiff_stack(os.path.join(dir, name+".tiff"))
        data = read_tiff_stack(dir)
        data = data[:, :, 20:-20]
        # data = data[::-1, ...]
        data_list.append(data)
    datas = np.array(data_list, dtype=np.float64)
    var = np.var(datas, 0)
    var = var.astype(np.uint16)
    write_tiff_stack(var, dst_path)

    for region in regions:
        region_mask = read_tiff_stack(f"/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/soma_nuclei/data/moving/allen/allen_{region}.tiff")
        region_mask[region_mask>0] = 1
        print(f"{region}: {np.sum(var[region_mask>0]) / np.sum(region_mask)}")


def main2():
    path_list1 = [
        "/media/root/e8449930-91ce-40a4-a5c5-87d4d2cd1568/zht/result/test/reg/allen_180921_O11_488_LEFT_16-07-16",
        "/media/root/e8449930-91ce-40a4-a5c5-87d4d2cd1568/zht/result/test/reg/allen_180725_20180724C2_LEFT_488_100ET_20-02-09",
        "/media/root/e8449930-91ce-40a4-a5c5-87d4d2cd1568/zht/result/test/reg/allen_180724_20180723O2_LEFT_488-2_100ET_16-13-41"
    ]
    path_list2 = [
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/lsfm/180921_O11_488_LEFT_16-07-16.tif",
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/lsfm/180725_20180724C2_LEFT_488_100ET_20-02-09.tif",
        "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/contrast_exp/mb/lsfm/180724_20180723O2_LEFT_488-2_100ET_16-13-41.tif"
    ]
    # regions = ["Cerebral cortex", "Hippocampal formation", "Cerebellar cortex", "Cerebellum", "Brain stem"]
    regions = ["Cerebral cortex", "Hippocampal formation", "Cerebellum", "Cerebral nuclei", "Interbrain", "Midbrain", "Hindbrain"]
    iv_tool1 = IntensityVarianceTool()
    iv_tool2 = IntensityVarianceTool()
    data_list1 = []
    data_list2 = []

    for dir in tqdm(path_list1):
        name = os.path.split(dir)[1]
        data = read_tiff_stack(os.path.join(dir, name+".tiff"))
        data_list1.append(data)
    
    for dir in tqdm(path_list2):
        name = os.path.split(dir)[1]
        data = read_tiff_stack(dir)
        # data = data[:, :, 20:-20]
        # data = data[::-1, ...]
        data_list2.append(data)
    
    iv_tool1.cal_iv(data_list1)
    iv_tool2.cal_iv(data_list2)
    number_plots = len(regions)
    result = {}
    for region in regions:
        result[region] = {}
        for child in tqdm(get_allchildren_id_by_struct_name(region)):
            iv1 = iv_tool1.cal_iv_by_allen_region(child)
            iv2 = iv_tool2.cal_iv_by_allen_region(child)
            result[region][get_struct_name_by_struct_id(child)] = [log2(iv1), log2(iv2)]
        # plt.subplot(120+i+1)
        # draw_and_save_plot(result, region)
    import pickle
    with open("iv_result/nuclei_mb.pkl", "wb") as f:
         pickle.dump(result, f)
    # plt.savefig(f"{name}.png")

            
def draw_and_save_plot(data, name):
    values = [i for i in data.values() if not np.isnan(i[0]) and not np.isnan(i[1])]
    x = np.array([i[0] for i in values])
    y = np.array([i[1] for i in values])
    color = np.sign(x-y) * np.abs(x-y) / np.max(np.abs(x-y))
    color = (color + 1) / 2
    plt.scatter(x, y, c=color, cmap=cm.RdBu, norm=colors.NoNorm())
    plt.plot([0, 10], [0, 10], color='grey', linestyle='dashed')
    plt.text(0, 9.5, f"{np.round(np.sum(x<y) / np.prod(x.shape) * 100, 2)}%", fontsize=18, color="#A6172C")
    plt.text(8.5, 0, f"{np.round(np.sum(x>y) / np.prod(x.shape) * 100, 2)}%", fontsize=18, color="#3783BB")
    plt.title(name)



class IntensityVarianceTool:
    def __init__(self) -> None:
        self.iv_map = None
        self.allen_annotation, _ = nrrd.read("/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/zht/Recursive_network_pytorch/allen/annotation_25.nrrd")
        self.allen_annotation = np.transpose(self.allen_annotation, (1, 2, 0))
        self.allen_annotation = self.allen_annotation[::-1, :, ::-1]
        

    def cal_iv_by_region(self, region):
        assert self.iv_map is not None
        region_mask = read_tiff_stack(f"/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/soma_nuclei/data/moving/allen/allen_{region}.tiff")
        region_mask[region_mask>0] = 1
        return np.sum(self.iv_map[region_mask>0]) / np.sum(region_mask)
    
    def cal_iv(self, data):
        self.iv_map = np.var(data, 0)
        self.iv_map = self.iv_map.astype(np.uint16)
    
    def cal_iv_by_allen_region(self, region):
        assert self.iv_map is not None
        region_mask = np.zeros(shape=self.allen_annotation.shape, dtype=np.uint8)
        cid_list = get_all_children_id_by_struct_id(region)
        for cid in cid_list:
            region_mask[self.allen_annotation == cid] = 1
        if self.iv_map.shape != region_mask.shape:
            region_mask = region_mask[::2, ::2, ::2]
        return np.sum(self.iv_map[region_mask>0]) / np.sum(region_mask)
    

if __name__ == "__main__":
    main()
