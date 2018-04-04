
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        help='path to dataset',
                        default='../dataset')

    parser.add_argument('-exp', '--experiment',
                        type=str,
                        help='directory in where samples and models will be saved',
                        default='../samples')

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        help='input batch size',
                        default=32)

    parser.add_argument('-isize', '--image_size',
                        type=int,
                        help='the height / width of the input image to network',
                        default=112)

    parser.add_argument('-nc', '--channels_number',
                        type=int,
                        help='input image channels',
                        default=1)

    parser.add_argument('-ngf', '--num_filters_g',
                        type=int,
                        help='number of filters for the first layer of the generator',
                        default=16)

    parser.add_argument('-ndf', '--num_filters_d',
                        type=int,
                        help='number of filters for the first layer of the discriminator',
                        default=16)

    parser.add_argument('-nep', '--nepochs',
                        type=int,
                        help='number of epochs to train for',
                        default=1000)

    parser.add_argument('-dit', '--d_iters',
                        type=int,
                        help='number of discriminator iterations per each generator iter, default=5',
                        default=5)

    parser.add_argument('-lrG', '--learning_rate_g',
                        type=float,
                        help='learning rate for generator, default=1e-5',
                        default=1e-3)

    parser.add_argument('-lrD', '--learning_rate_d',
                        type=float,
                        help='learning rate for discriminator, default=1e-5',
                        default=1e-3)

    parser.add_argument('-b1', '--beta1',
                        type=float,
                        help='beta1 for adam. default=0.0',
                        default=0.0)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    return parser
