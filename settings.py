# the settings file of the Demo to set the arguments

import argparse

# the RGB label of images and the names of lables
COLORMAP = [[0, 0, 0], [255, 255, 255]]
CLASSES = ['background', 'car']


def parse_opts():
    parser = argparse.ArgumentParser(prog='settings.py',
                                     usage='python %(prog)s [options]',
                                     description='Set the arguments of the U-Net demo.',
                                     epilog='To see the settings.py for more details.',
                                     allow_abbrev=False)

    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s 1.0')
    parser.add_argument('--gpu_id',
                        help='a list to enable GPUs. (defalult: %(default)s)',
                        nargs='*',
                        type=int,
                        default=None)
    parser.add_argument('--is_crop',
                        help='a flag to enable cropping. (default: %(default)s)',
                        type=bool,
                        default=True)
    parser.add_argument('--crop_height',
                        help='the height to crop the images. (default: %(default)s)',
                        type=int,
                        default=572)
    parser.add_argument('--crop_width',
                        help='the width to crop the images. (default: %(default)s)',
                        type=int,
                        default=572)
    parser.add_argument('--learning_rate',
                        help='the learning rate of optimizer. (default: %(default)s)',
                        type=float,
                        default=0.1)
    parser.add_argument('--momentum',
                        help='the momentum of optimizer. (default: %(default)s))',
                        type=float,
                        default=0.99)
    parser.add_argument('--batch_size',
                        help='the batch size of model. (default: %(default)s)',
                        type=int,
                        default=3)
    parser.add_argument('--num_epochs',
                        help='the number of epochs to train model. (defalult: %(default)s)',
                        type=int,
                        default=5)
    parser.add_argument('--num_classes',
                        help='the classes of output. (default: %(default)s)',
                        type=int,
                        default=2)
    parser.add_argument('--optimizer',
                        help='the optimizer to optimize the weights of model. (default: %(default)s)',
                        type=str,
                        default='sgd')
    parser.add_argument('--data_dir',
                        help='the directory of datasets. (default: %(default)s)',
                        type=str,
                        default='data')
    parser.add_argument('--log_dir',
                        help="the directory of 'UNet_log.txt'. (default: %(default)s)",
                        type=str,
                        default='./')
    parser.add_argument('--checkpoints_dir',
                        help='the directory of checkpoints. (default: %(default)s)',
                        type=str,
                        default='./checkpoints')
    parser.add_argument('--is_even_split',
                        help='whether or not to even split the data to all GPUs. (default: %(default)s)',
                        type=bool,
                        default=True)
    return parser.parse_args()

