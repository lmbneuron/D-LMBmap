import os
import cv2
import csv
import time
import glob
import shutil
import tifffile
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool

'''
    root：Root directory of whole brain slices
    target：The target directory where the cropped cube is stored
    point_file: The path of the csv file saved after imagej has selected the center point
    indx_start：Filename suffix, modify the value to prevent overwriting in the same folder
'''
root = "./data/brain-example/"
target = "./data/cropped-cubes/"
point_file = "./data/example.csv"
indx_start = 1000

parser = argparse.ArgumentParser(description='Tiffcroper',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--source', type=str, default=root, dest='source',
                        help='source directory of the brain slices')
parser.add_argument('-t', '--target', type=str, default=target, dest='target',
                        help='target directory of the cropped volumes')
parser.add_argument('-c', '--csv', type=str, default=point_file, dest='csv',
                        help='.csv file of the coordinates')
parser.add_argument('-i', '--index', type=int, default=indx_start, dest='index',
                        help='starting index of the output files')
parser.add_argument('--labels', action='store_true', help='generate labels')

args = parser.parse_args()
root = args.source
target = args.target
point_file = args.csv
indx_start = args.index
GENERATE_LABELS = args.labels
f = pd.read_csv(open(point_file))

''' 
    width/thickness：the side length of the cut cube
'''

width = 150
thickness = 150

x = list(f['X'])
y = list(f['Y'])
z = list(f['Slice'])
points = [list(p) for p in zip(x, y, z)]
files = sorted(os.listdir(root))
offset_w = width // 2
offset_t = thickness // 2
print(len(files))

def gauss_cal(img):
    im = img.astype(np.float64)
    #blurred = cv2.GaussianBlur(im, (3,3), 10)
    blurred = cv2.blur(im, (3,3))
    im = im - blurred
    im[im < 0] = 0
    return im.astype(np.uint16)

def get_image_paths(points):
    imgs = []
    for indx,i in enumerate(points):
        for j in range(int(i[2]) - offset_t, int(i[2]) + offset_t):
            imgs.append([int(float(i[1])),int(float(i[0])), j, j -(int(i[2]) - offset_t), indx])
    return imgs
 
def create_read_img(img):
    x, y, z, num, indx = img
    print("cropped -->",num,indx)
    try:
        img = np.array(Image.open(os.path.join(root, files[z]))).astype(np.int32)
        if not os.path.exists(os.path.join(target, 'volume-' + str(indx_start + indx))):
            os.mkdir(os.path.join(target, 'volume-' + str(indx_start + indx)))

        tifffile.imsave(os.path.join(target, 'volume-' + str(indx_start + indx) + '/' + str('%05d' % num) + '.tiff'), img[x - offset_w: x + offset_w, y -offset_w: y + offset_w])
    except:
        print(z)


if __name__ == '__main__':
    start = time.time()
    imgs = get_image_paths(points)  

    pool = Pool(processes=8)
    pool.map(create_read_img,imgs)
    pool.close()
    pool.join()
    
    end = time.time()
    print(end - start)
    
    for pth in os.listdir(target):
        images = []
        if os.path.isdir(os.path.join(target, pth)):
            for imgs in sorted(os.listdir(os.path.join(target, pth))):
                img = Image.open(os.path.join(os.path.join(target, pth), imgs))
                slice = np.array(img)
                images.append(slice)
            tifffile.imwrite(os.path.join(target, pth + '.tiff'), np.array(images).astype(np.uint16))
            if GENERATE_LABELS:
                images = gauss_cal(np.array(images))
                tifffile.imwrite((os.path.join(target, pth + '.tiff')).replace('volume', 'label'), np.array(images).astype(np.uint16))

    for pth in os.listdir(target):
        if os.path.isdir(os.path.join(target, pth)) and 'volume' in pth:
            try:
                print(pth)
                shutil.rmtree(os.path.join(target, pth))
            except:
                print('Final Done.')
