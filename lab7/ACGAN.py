import torch
import torch.nn as nn


## Implement with DCGAN
class Generator(nn.Module):
    def __init__(self,args):
        super(Generator, self).__init__()
        self.ngf, self.nc, self.nz = args.ngf, args.nc, args.latent_dim
        self.n_classes = args.n_classes

        # condition embedding
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
        gen_input = torch.cat((label_emb, noise), 1)
        out = self.main(gen_input)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ndf, self.nc = args.ndf, args.nc
        self.n_classes = args.n_classes
        self.img_size = args.img_size
        self.main = nn.Sequential(
            # input is (rgb chnannel = 3) x 64 x 64
            nn.Conv2d(3, self.ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*2) x 30 x 30
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*8) x 14 x 14
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (ndf*16) x 8 x 8
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),

        )
        
        # discriminator fc
        self.fc_dis = nn.Sequential(
            nn.Linear(5*5*self.ndf*32, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc
        self.fc_aux = nn.Sequential(
            nn.Linear(5*5*self.ndf*32, self.n_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
        conv = self.main(input)
        flat = conv.view(-1, 5*5*self.ndf*32)
        fc_dis = self.fc_dis(flat).view(-1, 1).squeeze(1)
        fc_aux = self.fc_aux(flat)
        return fc_dis, fc_aux