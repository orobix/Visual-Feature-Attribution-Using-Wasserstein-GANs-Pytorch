from __future__ import print_function
import os
import shutil

import torch
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable

from parser import get_parser

from synth_dataset import SynthDataset

from models.generator import UNet
from models.discriminator import DCGAN_D


LAMBDA_NORM = 1.5e-4


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)
    cudnn.benchmark = True


def init_experiment(opt):
    if opt.experiment is None:
        opt.experiment = '../samples'
    try:
        shutil.rmtree(opt.experiment)
    except:
        pass
    os.makedirs(opt.experiment)


def weights_init(m):
    '''
    Initialize cnn weithgs.
    for more info: http://cs231n.github.io/neural-networks-2/#init
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_model(opt):
    '''
    Initialize generator and disciminator
    '''
    net_g = UNet(nf=opt.ngf)
    net_d = DCGAN_D(opt.image_size, opt.nc,
                    opt.ndf)
    return net_g, net_d


def init_optimizer(opt, net_g, net_d):
    '''
    Initialize optimizers
    TODO use options for beta2 and wd
    '''
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
        opt.beta1, 0.9), weight_decay=1e-5)
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.learning_rate_d, betas=(
        opt.beta1, 0.9), weight_decay=1e-5)
    return optimizer_g, optimizer_d


def init_dataset(opt):
    '''
    Initialize both datasets and dataloaders
    '''
    dataset = SynthDataset(anomaly=False,
                           root_dir=opt.dataset_root,
                           image_size=opt.image_size,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

    anomaly_dataset = SynthDataset(anomaly=True,
                                   root_dir=opt.dataroot,
                                   image_size=opt.image_size,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

    healthy_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                     shuffle=True, drop_last=True)

    anomaly_dataloader = torch.utils.data.DataLoader(anomaly_dataset, batch_size=opt.batch_size,
                                                     shuffle=True, drop_last=True)

    return healthy_dataloader, anomaly_dataloader


def train(opt, healthy_dataloader, anomaly_dataloader, net_g, net_d, optim_g, optim_d):
    '''
    Run the trainig algorithm.
    '''
    model_input = torch.FloatTensor(
        opt.batch_size, 1, opt.image_size, opt.image_size)
    fixed_model_input = iter(anomaly_dataloader).next()[0]

    if opt.cuda:
        fixed_model_input = fixed_model_input.cuda()

    vutils.save_image(fixed_model_input.mul(0.5).add(
        0.5), '{0}/real_samples.png'.format(opt.experiment))

    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        fixed_model_input = fixed_model_input.cuda()
        one, mone = one.cuda(), mone.cuda()
        model_input = model_input.cuda()

    gen_iterations = 0
    for epoch in range(opt.niter):
        data_iter = iter(healthy_dataloader)
        anomaly_data_iter = iter(anomaly_dataloader)
        i = 0
        while i < len(anomaly_dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in net_d.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in net_g update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            labels = [mone, one]
            # occasionally switch labels
            if np.random.randint(20) == 0:
                labels = labels[::-1]
            while j < Diters and i < len(anomaly_dataloader):
                j += 1

                data = data_iter.next()
                i += 1

                # train with healthy
                real_cpu = data[0]
                net_d.zero_grad()

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                model_input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(model_input)
                err_d_real = net_d(inputv)
                err_d_real.backward(labels[1])

                # train with diseasy
                data = anomaly_data_iter.next()

                anomaly_cpu = data[0]
                net_d.zero_grad()

                if opt.cuda:
                    anomaly_cpu = anomaly_cpu.cuda()
                model_input.resize_as_(anomaly_cpu).copy_(anomaly_cpu)
                inputv = Variable(model_input)

                anomaly_map = Variable(net_g(inputv).data)

                outputv = anomaly_map
                img_sum = inputv - outputv

                err_d_anomaly_map = net_d(img_sum)
                err_d_anomaly_map.backward(labels[0])

                err_d = err_d_real - err_d_anomaly_map
                optim_d.step()

            ############################
            # (2) Update G network
            ###########################
            for p in net_d.parameters():
                p.requires_grad = False  # to avoid computation
            net_g.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            anomaly_map = net_g(inputv)
            # we want to minimize the l1 norm for the anomaly map
            (anomaly_map.norm(1) * LAMBDA_NORM).backward(retain_graph=True)
            err_g = net_d(inputv - anomaly_map)
            err_g.backward(labels[1])
            optim_g.step()
            gen_iterations += 1

            # print and save
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, opt.niter, i, len(healthy_dataloader), gen_iterations,
                     err_d.data[0], err_g.data[0], err_d_real.data[0], err_d_anomaly_map.data[0]))
            if gen_iterations % 50 == 0:
                anomaly_map = net_g(Variable(fixed_model_input, volatile=True))
                anomaly_map.data + anomaly_map.data.mul(0.5).add(0.5)
                vutils.save_image(
                    anomaly_map.data, '{:}/fake_samples_{:05d}.png'.format(opt.experiment, gen_iterations))

        # do checkpointing
        torch.save(net_g.state_dict(),
                   '{0}/net_g_chckp.pth'.format(opt.experiment, epoch))
        torch.save(net_d.state_dict(),
                   '{0}/net_d_chckp.pth'.format(opt.experiment, epoch))

    # save models at last iteration
    torch.save(net_g.state_dict(),
               '{0}/net_g_chckp_last.pth'.format(opt.experiment, epoch))
    torch.save(net_d.state_dict(),
               '{0}/net_d_chckp_last.pth'.format(opt.experiment, epoch))


def main():
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_experiment(options)
    init_seed(options)

    healthy_dataloader, anomaly_dataloader = init_dataset(options)

    net_g, net_d = init_model(options)

    net_g.apply(weights_init)
    net_d.apply(weights_init)

    optim_g, optim_d = init_optimizer(options, net_g=net_g, net_d=net_d)

    if options.cuda:
        net_g = net_g.cuda()
        net_d = net_d.cuda()

    train(options,
          healthy_dataloader, anomaly_dataloader,
          net_g=net_g, net_d=net_d,
          optim_g=optim_g, optim_d=optim_d)


if __name__ == '__main__':
    main()
