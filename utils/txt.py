# Generate the text file of training data and their labels

import os

root_dir = os.path.join('..', 'data')
train_path = os.path.join(root_dir, 'train')

train_imgs = os.listdir(train_path)
train_fname = os.path.join(root_dir, 'train.txt')

train_names = []
with open(train_fname, 'w') as train_txt:
    for train_img in train_imgs:
        train_name = train_img[:train_img.rfind('.')]
        train_names.append(train_name + '\n')

    train_txt.writelines(train_names)
    print('Everything is done!')



