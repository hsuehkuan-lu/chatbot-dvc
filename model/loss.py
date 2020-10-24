import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger('train')


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
    if torch.isnan(loss).any():
        logger.debug(x)
        logger.debug(y)
        logger.debug(mask)
        logger.debug(loss)
        logger.debug("---")
    return loss, total.item()
