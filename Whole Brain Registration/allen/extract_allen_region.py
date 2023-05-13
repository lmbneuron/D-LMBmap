'''
gain the mask img from allen ccf using region name
'''
import nrrd
import numpy as np
import sys
sys.path.append("../")
from allen.allenmap import get_allchildren_id_by_struct_name
from ms_regnet.tools.io import write_tiff_stack


resolution = 25  ## can be set to 10 or 25 depending on the resolution
shape = {
        10: (800, 1140, 1320),
        25: (320, 456, 528)
}[resolution]


def main():
    name_label_list = [
                        # ("Hypothalamus", 1),
                       # ("Cerebral cortex", 255),
                       ("Basolateral amygdalar nucleus", 1),
                       # ("Brain stem", 255)
    ]

    # name_label_list = [
    #                    ("Cerebellum", 1),
    #                    ]
    erase_label_list = [
                        ]


    # name_list = ["Cerebral cortex", "Cerebral nuclei", "Cerebrum",
    #              "Interbrain", "Midbrain", "Hindbrain",
    #              "Cerebellar cortex", "Cerebellar nuclei",
    #              "fiber tracts",
    #              "ventricular systems"]
    # name_color_list = [(i, get_color_by_struct_name(i)) for i in name_list]
    # name_label_list = [(i, j) for j, i in enumerate(name_list)]

    # mask = np.zeros(shape=(shape[0], shape[1], shape[2], 3), dtype=np.uint8)
    # for name, label in name_color_list:
    #     mask2 = extract_allen_region([name])
    #     mask[mask2 > 0] = label
    # write_tiff_stack(mask, r"C:\Users\haimiao\Desktop\color.tiff")
    mask = np.zeros(shape=shape, dtype=np.uint8)
    for name, label in name_label_list:
        mask2 = extract_allen_region([name])
        mask[mask2 > 0] = label
    for name, label in erase_label_list:
        mask2 = extract_allen_region([name])
        mask[mask2 > 0] = 0
    write_tiff_stack(mask, f"allen_{resolution}_bla.tiff")


def extract_allen_region(name_list):
    allen, _ = nrrd.read(f"./annotation_{resolution}.nrrd")
    mask = np.zeros(shape=allen.shape, dtype=np.uint8)
    for name in name_list:
        cid_list = get_allchildren_id_by_struct_name(name)
        for cid in cid_list:
            mask[allen == cid] = 1
    mask = rigid_deform(mask)
    return mask


def rigid_deform(vol):
    vol = np.transpose(vol, (1, 2, 0))
    vol = vol[::-1, :, ::-1]
    return vol


if __name__ == "__main__":
    main()