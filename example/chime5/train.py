import math
import logging
from pathlib import Path
import json
import os

import numpy
import torch
import kaldiio

from torchain import io
from torchain.functions import chain_loss, ChainResults


import models


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--exp_dir', required=True,
                        help='kaldi s5/exp dir that must be finished before')
    parser.add_argument('--model_dir', required=True,
                        help='model dir to store pytorch params and pickle')
    parser.add_argument("--lda_mat", help="lda mat file path (optional)")
    # optimization
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate")
    parser.add_argument("--optim", default="SGD", help="optimizer name")
    parser.add_argument("--weight_decay", default=5e-2, type=float,
                         help="weight decay")
    parser.add_argument("--l2_regularize", default=5e-5, type=float,
                         help="L2 regularization for LF-MMI")
    parser.add_argument("--leaky_hmm_coefficient", default=0.1, type=float,
                         help="leaky HMM coefficient for LF-MMI")
    parser.add_argument("--xent_regularize", default=0.1, type=float,
                         help="Cross entropy regularization for LF-MMI")
    parser.add_argument("--train_minibatch_size", default="256",
                        help="number of minibatches")
    parser.add_argument("--valid_minibatch_size", default="128",
                        help="number of minibatches")
    parser.add_argument("--n_epoch", default=10, type=int,
                         help="number of training epochs")
    parser.add_argument("--accum_grad", default=1, type=int,
                         help="number of gradient accumulation before update")
    # parser.add_argument("--max_loss", default=10.0, type=float,
    #                      help="max upperbound loss value to update")

    # misc
    parser.add_argument('--seed', default=777, type=int,
                        help='int random seed')
    return parser


def train_cmd(egs_path):
    return  "nnet3-chain-copy-egs --frame-shift=1  ark:" + str(egs_path) + " ark:- " \
        + "| nnet3-chain-shuffle-egs --buffer-size=5000 --srand=0 ark:- ark:- " \
        + "| nnet3-chain-merge-egs --minibatch-size=" + args.train_minibatch_size + " ark:- ark:-"


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logging.info("HOST=" + os.environ.get("HOST", ""))
    logging.info("SLURM_JOB_ID=" + os.environ.get("SLURM_JOB_ID", ""))
    logging.info(args)
    # init libraries
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # data preparation
    model_dir = Path(args.model_dir)
    exp_dir = Path(args.exp_dir)
    chain_dir = exp_dir / "chain_train_worn_u100k_cleaned"
    egs_dir = chain_dir / "tdnn1a_sp/egs"
    egs_list = list(egs_dir.glob("cegs.*.ark"))
    valid_cmd = "nnet3-chain-copy-egs ark:" + str(egs_dir) + "/valid_diagnostic.cegs ark:- " \
                + " | nnet3-chain-merge-egs --minibatch-size=" + args.valid_minibatch_size + " ark:- ark:- "
    # serialize args
    with open(model_dir / "args.json", "w") as json_file:
        json.dump(vars(args), json_file, ensure_ascii=False, indent=4)

    # get shape
    with io.open_example(valid_cmd) as example:
        for (mfcc, ivec), supervision in example:
            n_pdf = supervision.n_pdf
            n_feat = mfcc.shape[1]
            n_ivec = ivec.shape[1]
            break
    logging.info("shape: (n_feat: {}, n_ivec: {}, n_pdf: {}".format(n_feat, n_ivec, n_pdf))

    # load fst
    den_fst_rs = chain_dir / "tdnn1a_sp/den.fst"
    den_graph = io.DenominatorGraph(str(den_fst_rs), n_pdf)

    # load LDA matrix
    if args.lda_mat is not None:
        lda_mat = kaldiio.load_mat(args.lda_mat)
        lda_mat = torch.from_numpy(lda_mat)
    else:
        lda_mat = None

    # model preparation
    model = models.SimpleTDNN(n_pdf, n_feat, n_ivec, lda_mat=lda_mat)
    model = torch.nn.DataParallel(model)
    model.cuda()
    logging.info(model)
    # params = iter(p for p in model.parameters() if p.requires_grad)
    params = model.module.kaldi_like_parameters()
    opt_class = getattr(torch.optim, args.optim)
    opt = opt_class(params, lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")

    def forward(data, idx):
        (mfcc, ivec), supervision = data
        n_batch, n_freq, n_time_in = mfcc.shape
        # FIXME do not hardcode here
        n_time_out = math.floor((n_time_in - (29 - 1) -1) / 3 + 1)
        lf_mmi_pred, xe_pred = model(mfcc.cuda(), ivec.cuda())
        ref_shape = (n_batch, n_pdf, n_time_out)
        assert lf_mmi_pred.shape == ref_shape, "{} != {}".format(lf_mmi_pred.shape, ref_shape)
        return chain_loss(lf_mmi_pred, den_graph, supervision,
                          l2_regularize=args.l2_regularize,
                          leaky_hmm_coefficient=args.leaky_hmm_coefficient,
                          xent_regularize=args.xent_regularize,
                          xent_input=lf_mmi_pred, kaldi_way=True)

    # main loop
    for epoch in range(args.n_epoch):
        logging.info("epoch: {}".format(epoch))
        # training
        model.train()
        train_result = ChainResults()
        numpy.random.shuffle(egs_list)
        for egs in egs_list:
            with io.open_example(train_cmd(egs)) as example:
                for i, data in enumerate(example):
                    idx = example.indexes
                    loss, results = forward(data, idx)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_result.data += results.data
                    logging.info("train loss: {}, average: {}".format(results, train_result.loss))
        logging.info("summary train loss average: {}".format(train_result.loss))

        # validation
        model.eval()
        valid_result = ChainResults()
        with io.open_example(valid_cmd) as example, torch.no_grad():
            for i, data in enumerate(example):
                idx = example.indexes
                loss, results = forward(data, idx)
                valid_result.data += results.data
                logging.info("valid loss: {}, average: {}".format(results, valid_result.loss))
        logging.info("summary valid loss average: {}".format(valid_result.loss))

        # adaptive operations
        if valid_result.loss < best_loss:
            logging.info("update the best loss and save model")
            best_loss = valid_result.loss
            torch.save(model.module, str(model_dir / "model.pickle"))
            torch.save(model.module.state_dict(), str(model_dir / "model.dict"))
        else:
            logging.info("reload model and half lr")
            model.module = torch.load(str(model_dir / "model.pickle"))
            model.cuda()
            for param_group in opt.param_groups:
                param_group['lr'] /= 2


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
