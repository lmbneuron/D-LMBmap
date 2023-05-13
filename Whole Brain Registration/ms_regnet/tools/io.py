from PIL import Image
import numpy as np


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)

    return np.array(images)


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)


def read_mhd(mhd):
    import SimpleITK
    image = SimpleITK.ReadImage(mhd)
    image = SimpleITK.GetArrayFromImage(image)
    return image


def read_nii(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    print(img.dtype)
    # img = img.astype(np.uint8)
    return img