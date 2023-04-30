import os
import numpy as np
from PIL import Image
from os import listdir
from os.path import join


def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)

def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)

