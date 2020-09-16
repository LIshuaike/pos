import os
import time
import argparse
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from module import CRF, CHAR_LSTM
import utils.config as config
from utils import Corpus, Vocab, Embedding, TextDataset, dataloader
from tagger import CHAR_LSTM_CRF
from model import Model

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(
        description="Create several net for POS Tagging.")
    parser.add_argument('--dropratio',
                        default=0.5,
                        type=float,
                        help='set the ratio of dropout')
    parser.add_argument('--batch_size',
                        default=50,
                        type=int,
                        help='set the size of batch')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='set the max num of epochs')
    parser.add_argument('--interval',
                        default=10,
                        type=int,
                        help='set the max interval to stop')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='set the learning rate of training')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--preprocess',
                        '-p',
                        action='store_true',
                        help='whether to preprocess corpus')
    parser.add_argument('--resume',
                        default=False,
                        help='whether to load the saved net')

    parser.add_argument('--file',
                        default='saved/',
                        help='set where to store the net')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    config = config.config['char_lstm_crf']

    # corpus
    train = Corpus.load(config.train)
    dev = Corpus.load(config.dev)
    test = Corpus.load(config.test)

    # vocab
    vocab_path = os.path.join(args.file, 'vocab')
    if not os.path.exists(vocab_path) or args.preprocess:
        print("preprocessing the data")
        vocab = Vocab.from_corpus(corpus=train, min_freq=2)
        vocab.load_embedding(Embedding.load(config.embed))
        torch.save(vocab, vocab_path)
    else:
        print('loading vocab')
        vocab = torch.load(vocab_path)

    print(vocab)

    # dataset
    ds_train = TextDataset(vocab.numericalize(train))
    ds_dev = TextDataset(vocab.numericalize(dev))
    ds_test = TextDataset(vocab.numericalize(test))

    print(f"{'':2}size of trainset: {len(ds_train)}\n"
          f"{'':2}size of devset: {len(ds_dev)}\n"
          f"{'':2}size of testset: {len(ds_test)}\n")

    # dataloader
    dl_train = dataloader(dataset=ds_train,
                          batch_size=args.batch_size,
                          shuffle=True)
    dl_dev = dataloader(dataset=ds_dev,
                        batch_size=args.batch_size,
                        shuffle=True)

    dl_test = dataloader(dataset=ds_test, batch_size=args.batch_size)

    # creating the nueral net
    print("creating neural net")
    print(f"{'':2}num_chars:{vocab.n_chars}\n"
          f"{'':2}char_embedding_dim:{config.char_embedding_dim}"
          f"{'':2}char_output_size:{config.char_output_size}"
          f"{'':2}num_words: {vocab.n_words}\n"
          f"{'':2}context_size: {config.context_size}\n"
          f"{'':2}embedding_dim: {config.embedding_dim}\n"
          f"{'':2}hidden_size: {config.hidden_size}\n"
          f"{'':2}output_size: {vocab.n_tags}\n")

    net = CHAR_LSTM_CRF(num_chars=vocab.n_chars,
                        char_embedding_dim=config.char_embedding_dim,
                        char_output_size=config.char_output_size,
                        num_words=vocab.n_words,
                        embedding_dim=config.embedding_dim,
                        hidden_size=config.hidden_size,
                        output_size=vocab.n_tags,
                        dropratio=args.dropratio)

    net.load_pretrained(vocab.embedding)
    print(f"{net}\n")

    net.to(args.device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # load checkpoint if needed
    start_epoch = 0
    max_e, max_acc = 0, 0.
    if args.resume:
        ckpt_path = os.path.join(args.file, 'checkpoint.pth')
        assert os.path.isfile(ckpt_path)
        ckpt = torch.load(ckpt_path)
        max_e = ckpt['max_e'],
        max_acc = ckpt['max_acc']
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer'])
        print("last checkpoint restored\n")

    # train
    model = Model(net, vocab, optimizer)
    print("useing Adam optimizer to train the net")
    print(f"{'':2}epochs: {args.epochs}\n"
          f"{'':2}batch_size: {args.batch_size}\n"
          f"{'':2}interval: {args.interval}\n"
          f"{'':2}lr: {args.lr}\n")
    model.fit(dl_train, dl_dev, dl_test, start_epoch, args.epochs, max_e,
              max_acc, args.interval, args.file)

    # # evaluate
    # model = Model(net, vocab, optimizer)
    # model.evaluate(dl_dev)

    # predict
    model = Model(net, vocab, optimizer)
    words, tags = model.predict(dl_test)
    with open('./test.conll', 'w', encoding='utf-8') as f:
        for word, tag in zip(words, tags):
            for w, t in zip(word, tag):
                f.write(w + '\t' + t + '\n')
            f.write('\n')
