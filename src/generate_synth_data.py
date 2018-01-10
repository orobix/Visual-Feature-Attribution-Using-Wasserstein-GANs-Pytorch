#encoding: utf8

import os
import scipy
import argparse
import numpy as np
from skimage import filters

OUT_DIRECTORY = os.path.join('..', 'dataset')
OUT_DATA_DIR = 'data'
OUT_MASK_DIR = 'masks'

NUM_SAMPLES = 10000
IMAGE_SIZE = 112
SQUARE_SIZE = 20
STD_N = 50
MU_N = 0

GAUSSIAN_SIGMA = 2.5
RAND_OFFSET = 5

C0 = 0.6
C1 = 1
C2 = 1.4
C3 = 2
C4 = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        help='Root directory for the generated files',
                        default=OUT_DIRECTORY)
    parser.add_argument('--image_size',
                        type=int,
                        help='Width (and heigth) for the generated images',
                        default=IMAGE_SIZE)
    parser.add_argument('--square_size',
                        type=int,
                        help='Width (and heigth) for the inner squares in the masks (anomaly size)',
                        default=SQUARE_SIZE)
    parser.add_argument('--rand_offset',
                        type=int,
                        help='Maximum absolute offset in pixels for the off-centre inner squares in the masks (off centre anomalies max offset)',
                        default=RAND_OFFSET)
    parser.add_argument('--num_samples',
                        type=int,
                        help='number of samples to generate for each label (total samples = num_samples * 2)',
                        default=NUM_SAMPLES)
    parser.add_argument('--noise_std',
                        type=int,
                        help='Standard deviation for the Gaussian distribuited noise',
                        default=STD_N)
    parser.add_argument('--noise_mu',
                        type=int,
                        help='Mean for the Gaussian distribuited noise',
                        default=MU_N)
    parser.add_argument('--gaussian_sigma',
                        type=float,
                        help='sigma used for the gaussian filtering for the random noise (sigma for smoothing the noise)',
                        default=GAUSSIAN_SIGMA)
    parser.add_argument('--constant',
                        type=float,
                        help='Constant to add to the random noise images (multiplied for noise_std), in order to generate an image of label 1',
                        default=C2)

    opt = parser.parse_args()
    return opt


def create_dirs(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def rand_image(opt):
    return np.random.normal(loc=opt.noise_mu, scale=opt.noise_std, size=opt.image_shape)


def filter(img, sigma):
    return filters.gaussian(img, sigma, mode='reflect', preserve_range=True)


def center_square_mask(opt):
    mask = np.zeros((opt.image_shape))
    s1, s2 = center_slices(mask, opt)
    mask[s1, s2] = 1.0
    return mask


def center_slices(img, opt):
    rand_offset = np.abs(opt.rand_offset)
    width, height = opt.square_shape
    y, x = img.shape
    cx = int((x/2)-(width/2))
    cy = int((y/2)-(height/2))
    offx, offy = 0, 0
    if rand_offset:
        offx, offy = np.random.randint(
            low=-rand_offset, high=rand_offset+1, size=2)
        cx += offx
        cy += offy
    return slice(cx, cx + width), slice(cy, cy + height)


def off_centre_square_mask(opt):
    mask = np.zeros((opt.image_shape))
    width, height = opt.square_shape
    x, y = mask.shape
    cx = int(x/2)
    cy = int(y/2)
    left = np.random.randint(2)

    if left:
        s1, s2 = slice(None, cx), slice(None, cy)
    else:
        s1, s2 = slice(cx, None), slice(cy, None)
    lr_img = mask[s1, s2]

    s1, s2 = center_slices(lr_img, opt)

    lr_img[s1, s2] = 1

    return mask, not left


def generate_img_mask(opt):
    rand_img_filtered = filter(rand_image(opt), sigma=opt.gaussian_sigma)
    center_mask = center_square_mask(opt)
    lr_mask, subtype = off_centre_square_mask(opt)
    mask = center_mask + lr_mask
    mask[mask != 0] = 1
    sigma = opt.constant
    sigma *= opt.noise_std

    rand_img_filtered = (rand_img_filtered - rand_img_filtered.min()) / \
        (rand_img_filtered.max()-rand_img_filtered.min())
    rand_img_filtered *= 255

    new_mask = mask * sigma
    out = rand_img_filtered + new_mask
    out = np.clip(out, 0, 255)
    mask *= 255
    return out, mask, subtype


def generate_img_mask_nolabel(opt):
    rand_img_filtered = filter(rand_image(opt), sigma=opt.gaussian_sigma)
    rand_img_filtered = (rand_img_filtered - rand_img_filtered.min()) / \
        (rand_img_filtered.max()-rand_img_filtered.min())
    rand_img_filtered *= 255

    mask = np.zeros((opt.image_shape))

    return rand_img_filtered, mask, 0


def main():
    opt = parse_args()
    print(opt)

    opt.image_shape = (opt.image_size,)*2
    opt.square_shape = (opt.square_size,)*2

    out_data_dir = os.path.join(opt.out_dir, OUT_DATA_DIR)
    out_mask_dir = os.path.join(opt.out_dir, OUT_MASK_DIR)
    create_dirs([out_data_dir, out_mask_dir])

    labels_0s = np.zeros(int(opt.num_samples / 2))
    label_1s = np.ones(int(opt.num_samples / 2))
    labels = np.hstack([labels_0s, label_1s])
    np.random.shuffle(labels)

    for i in range(opt.num_samples):
        label = labels[i]
        if label:
            func = generate_img_mask
        else:
            func = generate_img_mask_nolabel

        img, mask, subtype = func(opt)

        scipy.misc.imsave(os.path.join(
            out_data_dir, '%d_%d_%d.png' % (i, label, subtype)), img)
        scipy.misc.imsave(os.path.join(
            out_mask_dir, '%d_%d_%d.png' % (i, label, subtype)), mask)


if __name__ == '__main__':
    main()
