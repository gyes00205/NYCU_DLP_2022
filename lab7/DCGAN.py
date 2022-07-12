from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngf, self.nc, self.nz = args.ngf, args.nc, args.latent_dim
        self.n_classes = args.n_classes
        
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.nc),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz + self.nc, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (rgb channel = 3) x 64 x 64
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).view(-1, self.nc, 1, 1)
        # print(label_emb.shape)
        # print(noise.shape)
        gen_input = torch.cat((label_emb, noise), 1)
        out = self.main(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ndf, self.nc = args.ndf, args.nc
        self.n_classes = args.n_classes
        self.img_size = args.img_size
        self.label_emb = nn.Sequential(
            nn.Linear(self.n_classes, self.img_size * self.img_size),
            nn.LeakyReLU(0.2, True)
        )
        self.main = nn.Sequential(
            # input is (rgb chnannel + condition channel = 3 + 1) x 64 x 64, 1 is condition
            nn.Conv2d(3 + 1, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        label_emb = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_emb), dim=1)
        out = self.main(d_in)
        return out.view(-1, 1).squeeze(1)
