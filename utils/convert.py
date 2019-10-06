# Convert the pictur format

import os
import re
import PIL.Image as Image
from mxnet import image
from matplotlib import pyplot as plt

img_format = 'jpg'
root_dir = os.path.join('..', 'data')


def convert_images(root=root_dir, data_dir='train_mask', img_format='jpg'):
    data_path = os.path.join(root, data_dir)
    save_path = os.path.join(root, (data_dir + '_%s') % img_format)
    print(data_path)
    print(save_path)
    names = os.listdir(data_path)
    print(names)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name in names:
        img_path = os.path.join(data_path, name)
        print(img_path)
        srcImg = Image.open(img_path)
        dstImg = Image.new('RGB', srcImg.size)
        dstImg.paste(srcImg)
        name = name[:name.rfind('.')]
        dstImg.save(save_path + os.sep + name + '.' + img_format)
    print('Everything is done!')


convert_images()
