import torch
import numpy as np
import torchvision

# https://github.com/harskish/ganspace/blob/65b0c4c7a4bbdcb5fedebb7c033dab59e27d61c0/models/wrappers.py#L73
def sample_np(self, z=None, n_samples=1, seed=None):
    """
    Convert generated image to numpy image.
    """
    if z is None:
        z = self.sample_latent(n_samples, seed=seed)
    elif isinstance(z, list):
        z = [torch.tensor(l).to(self.device) if not torch.is_tensor(l) else l for l in z]
    elif not torch.is_tensor(z):
        z = torch.tensor(z).to(self.device)
    img = self.forward(z)
    img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return np.clip(img_np, 0.0, 1.0).squeeze()


# https://github.com/clovaai/stargan-v2/blob/aae8bd560a8fac7be22ab42228ab704e77e1cb00/core/utils.py#L57
def tensor2img(x):
    x = (x + 1) / 2
    x = x.clamp_(0, 1)
    torchvision.utils.save_image(x.cpu())