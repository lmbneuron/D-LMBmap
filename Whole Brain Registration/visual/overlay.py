from PIL import Image
import numpy as np
import cv2

from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack, read_binary_mhd


def main():
    vol2 = read_tiff_stack(r"H:\whole_brain\tarahe210319_Adult_Brain_3_19-37-20.tiff.tiff")
    vol2 += 40
    # vol1 = read_tiff_stack(r"D:\zht\code\align\evaluate\data\corrode3.tiff")
    vol1 = read_binary_mhd(r"H:\whole_brain\ahetarallen.tiff.tiffoutput\2\result.0.mhd")
    overlay_vol = overlay(vol1, vol2, output_need_alpha=False)
    print(overlay_vol.ndim)
    write_tiff_stack(overlay_vol, "test.tiff")


def overlay(f, m, output_need_alpha=True):
    f = f.astype(np.uint8)
    m = m.astype(np.uint8)
    image_list = []
    for slicef, slicem in zip(f, m):
        if slicef.shape != slicem.shape:
            slicef = cv2.resize(slicef, (slicem.shape[1], slicem.shape[0]))
        newf = np.zeros(shape=(slicef.shape[0], slicef.shape[1], 3), dtype=np.uint8)
        newm = np.zeros(shape=(slicef.shape[0], slicef.shape[1], 3), dtype=np.uint8)


        newf[slicef > 30] = (255, 0, 255)
        newm[slicem > 30] = (0, 255, 0)

        slicef = slicef.astype(np.float)
        slicef = (slicef - np.min(slicef)) / (np.max(slicef) - np.min(slicef)+1e-10) * 255
        slicef = slicef.astype(np.uint8)

        slicem = slicem.astype(np.float)
        slicem = (slicem - np.min(slicem)) / (np.max(slicem) - np.min(slicem)+1e-10) * 255
        slicem = slicem.astype(np.uint8)

        newf = Image.fromarray(newf)
        newf = newf.convert('RGBA')
        newf = np.array(newf)
        newf[:, :, 3] = slicef[:, :]

        newm = Image.fromarray(newm)
        newm = newm.convert('RGBA')
        newm = np.array(newm)
        newm[:, :, 3] = slicem[:, :]

        newf = Image.fromarray(newf)
        newm = Image.fromarray(newm)
        # image = Image.blend(newf, newm, 0.5)
        image = Image.alpha_composite(newf, newm)

        image = np.array(image)
        if not output_need_alpha:
            image = rgba2rgb(image)
        image = image[:, :, :]
        image_list.append(image)

    image_list = np.reshape((image_list),
                            newshape=(-1, image_list[0].shape[0], image_list[1].shape[1], image_list[1].shape[2]))
    image_list = image_list.astype(np.uint8)
    return image_list


def rgba2rgb(img):
    img = img.astype(np.float)
    newimg = 1 - img[:, :, 3][:, :, np.newaxis]/255 + img[:, :, :3]*img[:, :, 3][:, :, np.newaxis]/255
    newimg = newimg.astype(np.uint8)
    return newimg

if __name__ == "__main__":
    main()

