from torch.nn import functional as F


# https://github.com/yunjey/stargan/blob/HEAD/solver.py
def classification_loss(self, logit, target, dataset='CelebA'):
    """Compute binary or softmax cross entropy loss."""
    if dataset == 'CelebA':
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
    elif dataset == 'RaFD':
        return F.cross_entropy(logit, target)
