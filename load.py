# Load datasets

from mxnet.gluon import data as gdata
from mxnet import image, nd
import os
from settings import COLORMAP

colormap2label = nd.zeros(256 ** 3)
for i, colormap in enumerate(COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def read_images(root=None, is_train=True):
    if is_train:
        train_dir = 'train'
        label_dir = 'train_mask_jpg'
    else:
        train_dir = 'test'
        label_dir = 'test_mask_jpg'
    train_path = os.path.join(root, train_dir)
    label_path = os.path.join(root, label_dir)
    txt_fname = os.path.join(root, 'train.txt' if is_train else 'test.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread(os.path.join(train_path, '%s.jpg' % fname))
        labels[i] = image.imread(os.path.join(label_path, '%s_mask.jpg' % fname))
    return features, labels


def label_indices(colormap, colormap2label):
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def rand_crop(feature, label, height, width):
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label


class SegDataset(gdata.Dataset):
    def __init__(self, is_train=True, is_crop=False, crop_size=None, root=None, colormap2label=None):
        self.rgb_mean = nd.array([0.448, 0.456, 0.406])
        self.rgb_std = nd.array([0.229, 0.224, 0.225])
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.colormap2label = colormap2label
        features, labels = read_images(root=root, is_train=is_train)
        if is_crop:
            self.features = [self.normalize_image(feature) for feature in self.filter(features)]
            self.labels = self.filter(labels)
        else:
            self.features = [self.normalize_image(feature) for feature in features]
            self.labels = labels

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[0] >= self.crop_size[0] and
                img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, item):
        if self.is_crop:
            feature, label = rand_crop(self.features[item], self.labels[item], *self.crop_size)
        else:
            feature, label = self.features[item], self.labels[item]
        return feature.transpose((2, 0, 1)), label_indices(label, self.colormap2label)

    def __len__(self):
        return len(self.features)
