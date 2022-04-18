# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
import argparse

import numpy as np
from torchvision.models.inception import inception_v3
from utils.data import NoLabelDataset
from torchvision import transforms
from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F


def inception_score(args):
    transform = transforms.Compose([
        transforms.CenterCrop(args.img_size),
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = NoLabelDataset(args.img_path, transform)
    print(f"Found {len(dataset)} images.")
    dataloader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    preds = []
    print("Initializing model...")
    model = inception_v3(pretrained=True)
    model.eval()
    print("Processing images...")
    for images in tqdm(dataloader):
        images.cuda()
        local_preds = model(images)
        local_preds = F.softmax(local_preds, dim=1).data.cpu().numpy()
        local_preds = [pred for pred in local_preds]
        preds.extend(local_preds)
    print("Calculating inception score...")
    preds = np.array(preds)
    num_per_split = preds.shape[0] // args.split_num
    # Method #1: https://gitee.com/songquanpeng/tf-fid-is/blob/master/inception_score.py#L47
    scores = []
    for i in range(args.split_num):
        split = preds[i * num_per_split:(i + 1) * num_per_split, :]
        # KL(p(y|x)||p(y)) = Sigma_i_to_n[p(y|xi)log(p(y|xi) / p(y))]
        KL_divergence = split * (np.log(split) - np.log(np.expand_dims(np.mean(split, 0), 0)))
        KL_divergence = np.mean(np.sum(KL_divergence, 1))
        # InceptionScore = exp(E(KL))
        scores.append(np.exp(KL_divergence))
    IS_mean, IS_std = np.mean(scores), np.std(scores)
    print(f"Inception Score: {IS_mean:.4f}±{IS_std:.4f}")

    # Method #2: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    # from scipy.stats import entropy
    # scores = []
    # for k in range(args.split_num):
    #     split = preds[k * num_per_split:(k + 1) * num_per_split, :]
    #     py = np.mean(split, axis=0)
    #     local_scores = []
    #     for i in range(split.shape[0]):
    #         pyx = split[i, :]
    #         # actually it's the KL divergence
    #         KL_divergence = entropy(pyx, py)
    #         local_scores.append(KL_divergence)
    #     scores.append(np.exp(np.mean(local_scores)))
    # IS_mean, IS_std = np.mean(scores), np.std(scores)
    # print(f"Inception Score: {IS_mean:.4f}±{IS_std:.4f}")
    return IS_mean, IS_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--img_size', type=int, default=229,
                        help='The default input image size of Inception-v3 is 299×299.')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--split_num', type=int, default=10)
    cfg = parser.parse_args()
    if cfg.img_path is None:
        cfg.img_path = input("Input the target image path: ").strip('"').strip("'")
    inception_score(cfg)
