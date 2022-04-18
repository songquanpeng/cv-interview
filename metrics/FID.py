import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import models
from tqdm import tqdm

from utils.data import get_dataloader
from utils.file import load_cache, save_cache, exist_cache, safe_filename
from utils.misc import str2bool
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    # Please notice, the function sqrtm() here is not element wise!
    cc, _ = linalg.sqrtm(cov @ cov2, disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)  # Cause cc may have imaginary part


@torch.no_grad()
def get_fid_mu_cov(inception, path, img_size, batch_size, device, use_cache=False):
    cache_name = safe_filename(path, 'fid')
    cache_available = use_cache and exist_cache(cache_name)
    if cache_available:
        print('Loading cache...', end=' ')
        cache = load_cache(cache_name)
        mu = cache['mu']
        cov = cache['cov']
        print('Done.')
    else:
        loader = get_dataloader(path, img_size, batch_size)
        activations = []
        for x in tqdm(loader, total=len(loader)):
            activation = inception(x.to(device))
            activations.append(activation)
        activations = torch.cat(activations, dim=0).cpu().detach().numpy()  # (N, 2048)
        mu = np.mean(activations, axis=0)  # (2048,)
        cov = np.cov(activations, rowvar=False)  # (2048, 2048), each column represents a variable

        if use_cache and not exist_cache(cache_name):
            print('Saving cache...', end=' ')
            cache = {'mu': mu, 'cov': cov}
            save_cache(cache, cache_name)
            print('Done.')
    return mu, cov


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size, batch_size, use_cache=True):
    print('Calculating FID for given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    mu1, cov1 = get_fid_mu_cov(inception, paths[0], img_size, batch_size, device, use_cache=use_cache)
    mu2, cov2 = get_fid_mu_cov(inception, paths[1], img_size, batch_size, device, use_cache=False)
    res = frechet_distance(mu1, cov1, mu2, cov2)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, required=True)
    parser.add_argument('--path2', type=str)
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--use_cache', type=str2bool, default=True)
    args = parser.parse_args()
    print(args.__dict__)
    print(f"Path 1 is: {args.path1}")
    if args.path2 is None:
        args.path2 = input("Please input path 2: ").strip("'").strip('"')
    else:
        print(f"Path 2 is: {args.path2}")
    fid_value = calculate_fid_given_paths([args.path1, args.path2], args.img_size, args.batch_size,
                                          use_cache=args.use_cache)
    print(f'FID: {fid_value:.4f}')
