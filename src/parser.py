
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',
                        help='path to dataset',
                        default='../dataset')

    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='input batch size')

    parser.add_argument('--image_size',
                        type=int,
                        default=128,
                        help='the height / width of the input image to network')

    parser.add_argument('--nc',
                        type=int,
                        default=1,
                        help='input image channels')

    parser.add_argument('--nz',
                        type=int,
                        default=1,
                        help='size of the latent z vector')

    parser.add_argument('--ngf',
                        type=int,
                        default=16)

    parser.add_argument('--ndf',
                        type=int,
                        default=16)

    parser.add_argument('--niter',
                        type=int,
                        default=10000,
                        help='number of epochs to train for')

    parser.add_argument('-lrD', '--learning_rate_d',
                        type=float,
                        default=1e-5,
                        help='learning rate for Critic, default=1e-5')

    parser.add_argument('-lrG', '--learning_rate_g',
                        type=float,
                        default=1e-5,
                        help='learning rate for Generator, default=1e-5')

    parser.add_argument('--beta1',
                        type=float,
                        default=0.0,
                        help='beta1 for adam. default=0.5')

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')

    parser.add_argument('--net_g',
                        default='',
                        help="path to net_g (to continue training)")

    parser.add_argument('--net_d',
                        default='',
                        help="path to net_d (to continue training)")

    parser.add_argument('--Diters',
                        type=int,
                        default=5,
                        help='number of D iters per each G iter, default=5')

    parser.add_argument('--experiment',
                        default='../samples',
                        help='Where to store samples and models')

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        default=7,
                        help='input for the manual seeds initializations')

    return parser
