import logging
from pathlib import Path

import numpy
import torch

from torchain import io
from torchain.functions import chain_loss, ChainResults


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--exp_dir', required=True,
                        help='kaldi s5/exp dir that must be finished before')
    parser.add_argument('--model_dir', required=True,
                        help='model dir to store pytorch params and pickle')
    # optimization
    parser.add_argument("--lr", default=1e-6, type=float,
                         help="learning rate")
    parser.add_argument("--l2_regularize", default=1e-2, type=float,
                         help="L2 regularization for LF-MMI")
    parser.add_argument("--train_minibatch_size", default="128,64,32",
                        help="number of minibatches")
    parser.add_argument("--valid_minibatch_size", default="1:64",
                        help="number of minibatches")
    parser.add_argument("--n_epoch", default=20, type=int,
                         help="number of training epochs")
    # misc
    parser.add_argument('--seed', default=777, type=int,
                        help='int random seed')
    return parser


def get_model(n_pdf, n_feat, n_ivec):
    # TODO: provide n_time_width, n_time_stride
    model = torch.nn.Sequential(
        torch.nn.Conv1d(n_feat, 512, 29, 3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(512, n_pdf, 1, 1)
    )
    return model


def train_cmd(i, egs_list):
    l = len(egs_list)
    if i == l:
        numpy.random.shuffle(egs_list)
    return  "nnet3-chain-copy-egs --frame-shift=1  ark:" + str(egs_list[i % l]) + " ark:- " \
        + "| nnet3-chain-shuffle-egs --buffer-size=5000 --srand=0 ark:- ark:- " \
        + "| nnet3-chain-merge-egs --minibatch-size=" + args.train_minibatch_size + " ark:- ark:-"


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(args)
    # init libraries
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # data preparation
    exp_dir = Path(args.exp_dir)
    chain_dir = exp_dir / "chain_train_worn_u100k_cleaned"
    egs_dir = chain_dir / "tdnn1a_sp/egs"
    egs_list = list(egs_dir.glob("cegs.*.ark"))
    valid_cmd = "nnet3-chain-copy-egs ark:" + str(egs_dir) + "/valid_diagnostic.cegs ark:- " \
                + " | nnet3-chain-merge-egs --minibatch-size=" + args.valid_minibatch_size + " ark:- ark:- "
    with io.open_example(valid_cmd) as example:
        for (mfcc, ivec), supervision in example:
            n_pdf = supervision.n_pdf
            n_feat = mfcc.shape[1]
            n_ivec = ivec.shape[1]
            break
    logging.info("shape: (n_feat: %d, n_ivec: %d, n_pdf: %d)".format(n_feat, n_ivec, n_pdf))
    den_fst_rs = chain_dir / "tdnn1a_sp/den.fst"
    den_graph = io.DenominatorGraph(str(den_fst_rs), n_pdf)

    # model preparation
    model = get_model(n_pdf, n_feat, n_ivec)
    model.cuda()
    logging.info(model)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    # main loop
    for epoch in range(args.n_epoch):
        logging.info("epoch: {}".format(epoch))
        # training
        train_result = ChainResults()
        with io.open_example(train_cmd(epoch, egs_list)) as example:
            for (mfcc, ivec), supervision in example:
                x = mfcc.cuda()
                pred = model(x)
                loss, results = chain_loss(pred, den_graph, supervision,
                                           l2_regularize=args.l2_regularize)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_result.data += results.data
                logging.info("train loss: {}, average: {}".format(results, train_result.loss))

        # validation
        valid_result = ChainResults()
        with io.open_example(valid_cmd) as example, torch.no_grad():
            for (mfcc, ivec), supervision in example:
                x = Variable(mfcc).cuda()
                pred = model(x)
                loss, results = chain_loss(pred, den_graph, supervision,
                                           l2_regularize=args.l2_regularize)
                valid_result.data += results.data
                logging.info("valid loss: {}, average: {}".format(results, valid_result.loss))

        # adaptive operations
        if valid_result.loss < best_loss:
            logging.info("update the best loss and save model")
            best_loss = valid_result.loss
            model.cpu()
            model_dir = Path(args.model_dir)
            torch.save(model, str(model_dir / "model.pickle"))
            torch.save(model.state_dict(), str(model_dir + "model.dict"))
            model.cuda()
        else:
            logging.info("reload model and half lr")
            torch.load(model, str(model_dir / "model.pickle"))
            model.cuda()
            for param_group in opt.param_groups:
                param_group['lr'] /= 2


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
