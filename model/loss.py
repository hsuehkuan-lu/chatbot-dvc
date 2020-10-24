import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mask_nll_loss(x, y, mask):
    """
    x = [batch_size, vocab_size]
    y = [batch_size]
    mask = [batch_size]
    """
    total = mask.sum()
    cross_entropy = -torch.log(torch.gather(x, dim=1, index=y.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    return loss, total.item()
