import argparse
import torch
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('inference')

    # setup data_loader instances
    data_loader = config.init_obj(
        'inference_data_loader',
        module_data,
        text_field_path=config.resume.parent / 'TEXT.Field',
        vocab_path=config.resume.parent / 'TEXT.Vocab'
    )
    logger.info('Load data loader')

    # build model architecture
    encoder = config.init_obj(
        'encoder_arch', module_arch,
        vocab_size=data_loader.vocab_size,
        padding_idx=data_loader.padding_idx,
        hidden_size=config['hidden_size'],
        embed_size=config['embed_size']
    )
    encoder.eval()
    logger.info(encoder)
    decoder = config.init_obj(
        'decoder_arch', module_arch,
        embedding=encoder.embedding,
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=data_loader.vocab_size
    )
    decoder.eval()
    logger.info(decoder)
    model_idx = dict([('encoder', 0), ('decoder', 1)])
    models = [encoder, decoder]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    for idx in range(len(models)):
        models[idx].load_state_dict(
            checkpoint['{}_state_dict'.format(type(models[idx]).__name__)]
        )

    greedy_decoder = config.init_obj(
        'inference_arch', module_arch,
        encoder=encoder,
        decoder=decoder,
        init_idx=data_loader.init_idx
    )
    logger.info(greedy_decoder)

    # prepare model for testing
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # greedy_decoder = greedy_decoder.to(device)
    # greedy_decoder.eval()

    with torch.no_grad():
        while True:
            text = input("Input text: ")
            x, x_len = data_loader.preprocess(text)
            # x, x_len = x.to(device), x_len.to(device)
            all_tokens, all_scores = greedy_decoder(x, x_len, data_loader.sent_len)
            converted_text = data_loader.convert_ids_to_text(all_tokens)
            print(all_scores.T[0])
            print(converted_text)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
