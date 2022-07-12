import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch
import DCGAN
from dataset import iclevrDataset
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model


# custom weights initialization called on Generator and Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--ngf', type=int, default=300, help="feature channels of generator")
    parser.add_argument('--ndf', type=int, default=100, help="feature channels of discriminator")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--nc", type=int, default=100, help="number of condition embedding dim")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument('--outf', default='checkpoints', help='folder to output images and model checkpoints')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--log', default='logs/', help='path to tensorboard log')
    parser.add_argument('--resume', default='', help='path to resume model weight')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.outf, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)
    # Loss functions
    criterion = nn.BCELoss()

    # Initialize generator and discriminator
    generator = DCGAN.Generator(args).to(device)
    discriminator = DCGAN.Discriminator(args).to(device)
    if args.resume == '':
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    else:
        generator.load_state_dict(torch.load(os.path.join(args.resume, 'netG.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(args.resume, 'netD.pth')))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Dataloader
    train_dataloader = DataLoader(
        iclevrDataset(mode='train', root='iclevr'),
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='iclevr'),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initilaize evaluator
    evaluator = evaluation_model()

    writer = SummaryWriter(args.log)
    if args.dry_run:
        args.n_epochs = 1
    
    iters = 0
    best_acc = 0
    for epoch in range(args.n_epochs):
        total_loss_d = 0
        total_loss_g = 0
        discriminator.train()
        generator.train()
        for i, (image, cond) in enumerate(train_dataloader):
            ############################
            # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            # train with real
            real_image = image.to(device)
            cond = cond.to(device)
            batch_size = image.size(0)
            # Use soft and noisy labels [0.7, 1.0]. Salimans et. al. 2016
            real_label = ((1.0 - 0.7) * torch.rand(batch_size) + 0.7).to(device)

            # train with fake
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_image = generator(noise, cond)
            # Use soft and noisy labels [0.0, 0.3]. Salimans et. al. 2016
            fake_label = ((0.3 - 0.0) * torch.rand(batch_size) + 0.0).to(device)

            # occasionally flip the labels when training the discriminator
            if random.random() < 0.1:
                real_label, fake_label = fake_label, real_label

            output = discriminator(real_image, cond)
            D_x = output.mean().item()
            errD_real = criterion(output, real_label)

            output = discriminator(fake_image.detach(), cond)
            errD_fake = criterion(output, fake_label)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            errD.backward()
            optimizer_D.step()

            ############################
            # (2) Update Generator: maximize log(D(G(z)))
            ########################### 
            generator_label = torch.ones(batch_size).to(device)  # fake labels are real for generator cost
            output = discriminator(fake_image, cond)
            errG = criterion(output, generator_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            writer.add_scalar('Train Step/Loss D', errD.item(), iters)
            writer.add_scalar('Train Step/Loss G', errG.item(), iters)
            writer.add_scalar('Train Step/D(x)', D_x, iters)
            writer.add_scalar('Train Step/D(G(z))', D_G_z1, iters)
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch+1, args.n_epochs, i, len(train_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            total_loss_d += errD.item()
            total_loss_g += errG.item()
            if iters % 100 == 0:
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                save_image(real_image,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
                save_image(fake_image.detach(),
                    '%s/fake_samples.png' % args.outf,
                    normalize=True)
            iters += 1
            
        
        writer.add_scalar('Train Epoch/Loss D', total_loss_d/len(train_dataloader), epoch)
        writer.add_scalar('Train Epoch/Loss G', total_loss_g/len(train_dataloader), epoch)
        
        discriminator.eval()
        generator.eval()
        with torch.no_grad():
            for i, cond in enumerate(test_dataloader):
                cond = cond.to(device)
                batch_size = cond.size(0)
                noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
                fake_image = generator(noise, cond)
                save_image(fake_image.detach(),
                    '%s/fake_test_samples_epoch_%03d.png' % (args.outf, epoch),
                    normalize=True)
                acc = evaluator.eval(fake_image, cond)
                writer.add_scalar('Test/Accuracy', acc, epoch)
                # do checkpointing
                if acc > best_acc:
                    best_acc = acc
                    torch.save(generator.state_dict(), '%s/netG.pth' % args.outf)
                    torch.save(discriminator.state_dict(), '%s/netD.pth' % args.outf)

        if args.dry_run:
            break
        # torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
        # torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
