# Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils import data
from torch import nn, optim
import torch.nn.functional as F
import argparse
from munch import Munch


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_size = args.img_size // 4
        self.from_latent = nn.Sequential(*[
            nn.Linear(args.latent_dim, 128 * self.init_size ** 2),  # 128 * 8 * 8
        ])
        self.conv_layers = nn.Sequential(*[
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 128 * 16 * 16
            nn.Conv2d(128, 128, 3, 1, 1),  # 128 * 16 * 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),  # 128 * 32 * 32
            nn.Conv2d(128, 64, 3, 1, 1),  # 64 * 32 * 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.img_channel, 3, 1, 1),
            nn.Tanh()
        ])

    def forward(self, z):
        out = self.from_latent(z)
        out = out.view(z.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_layers(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        layers = []
        in_dim = args.img_channel
        out_dim = 16
        num_blocks = 4
        for i in range(4):
            layers.append(nn.Conv2d(in_dim, out_dim, 3, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout2d(0.25))
            if i != 0:
                layers.append(nn.BatchNorm2d(out_dim))
            in_dim = out_dim
            out_dim *= 2
        out_dim /= 2
        out_dim = int(out_dim)
        out_size = args.img_size // 2 ** num_blocks
        self.conv_layers = nn.Sequential(*layers)
        self.adv_head = nn.Sequential(*[
            nn.Linear(out_dim * out_size * out_size, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(x.shape[0], -1)  # N, 512
        out = self.adv_head(out)
        return out


def train(args):
    # setup models
    nets = Munch()
    nets.G = Generator(args)
    nets.D = Discriminator(args)
    for key, value in nets.items():
        nets[key] = nets[key].cuda()

    # setup optimizers
    optims = Munch()
    optims.G = optim.Adam(nets.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optims.D = optim.Adam(nets.D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # prepare data
    os.makedirs(args.dataset_path, exist_ok=True)
    data_transforms = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * args.img_channel, [0.5] * args.img_channel)
    ])
    train_dataset = datasets.MNIST(root=args.dataset_path, download=True, train=True, transform=data_transforms)
    dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 drop_last=True)

    # sampling related
    sample_z = torch.randn(args.batch_size, args.latent_dim)
    sample_z = sample_z.cuda()
    sample_path = os.path.join("archive", "samples")
    os.makedirs(sample_path, exist_ok=True)

    # the training loop
    for epoch in range(args.num_epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.cuda()
            # train generator
            optims.G.zero_grad()
            z = torch.randn(args.batch_size, args.latent_dim)
            z = z.cuda()
            fake_x = nets.G(z)
            d_out = nets.D(fake_x)
            g_loss = F.binary_cross_entropy(d_out, torch.full_like(d_out, fill_value=1))
            g_loss.backward()
            optims.G.step()

            # train discriminator
            optims.D.zero_grad()
            d_real_out = nets.D(x)
            d_fake_out = nets.D(fake_x.detach())
            d_real_loss = F.binary_cross_entropy(d_real_out, torch.full_like(d_real_out, fill_value=1))
            d_fake_loss = F.binary_cross_entropy(d_fake_out, torch.full_like(d_fake_out, fill_value=0))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optims.D.step()

            # print log
            if i % args.log_every == 0:
                print(f"[Epoch {epoch}/{args.num_epochs}] [Iter {i}/{len(dataloader)}]: G/loss: {g_loss.item():.4f} "
                      f"D/real_loss: {d_real_loss.item():.4f} D/fake_loss: {d_fake_loss.item():.4f} D/loss: {d_loss.item():.4f}")

            # sampling
            if i % args.sample_every == 0:
                with torch.no_grad():
                    sample_x = nets.G(sample_z)
                    save_image(sample_x, os.path.join(sample_path, f"epoch{epoch:04}_iter{i:06}.png"))
        out_dict = {}
        for name, module in nets.items():
            out_dict[name] = module.state_dict()
        torch.save(out_dict, f'./archive/DCGAN.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument('--dataset_path', type=str, default='./archive/dataset')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--img_channel', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    # training related arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num_epochs', type=int, default=100)

    # others
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=50)
    opt = parser.parse_args()
    train(opt)
