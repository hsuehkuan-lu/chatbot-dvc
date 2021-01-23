import yaml
import torch
import numpy as np
from data_loader import preprocess
from utils import util

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(params):
    logger = util.get_logger('prepare')
    # preprocess
    preprocess.ChatbotDataPreprocess(
        data_dir=params['data_dir'],
        sent_len=params['sent_len']
    )


if __name__ == '__main__':
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    main(params)
