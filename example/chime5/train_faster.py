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
    parser.add_argument('--model_dir', required=True,
                        help='dir to store model files')
    parser.add_argument('--den_fst', required=True,
                        help='denominator FST file')
    parser.add_argument('--train_scp', required=True)
    parser.add_argument('--valid_scp', required=True)
    parser.add_argument("--lda_mat", help="lda mat file path (optional)")
    # optimization
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate")
    parser.add_argument("--momentum", default=0.0, type=float,
                        help="momentum")
    parser.add_argument("--optim", default="SGD", help="optimizer name")
    parser.add_argument("--weight_decay", default=5e-2, type=float,
                         help="weight decay")
    parser.add_argument("--l2_regularize", default=5e-5, type=float,
                         help="L2 regularization for LF-MMI")
    parser.add_argument("--leaky_hmm_coefficient", default=0.1, type=float,
                         help="leaky HMM coefficient for LF-MMI")
    parser.add_argument("--xent_regularize", default=0.1, type=float,
                         help="Cross entropy regularization for LF-MMI")
    parser.add_argument("--train_minibatch_size", default=256, type=int,
                        help="number of minibatches")
    parser.add_argument("--valid_minibatch_size", default=256, type=int,
                        help="number of minibatches")
    parser.add_argument("--n_epoch", default=10, type=int,
                         help="number of training epochs")
    parser.add_argument("--accum_grad", default=1, type=int,
                         help="number of gradient accumulation before update")
    parser.add_argument("--no_ivector", action="store_true")
    # parser.add_argument("--max_loss", default=10.0, type=float,
    #                      help="max upperbound loss value to update")

    # misc
    parser.add_argument('--seed', default=777, type=int,
                        help='int random seed')
    return parser


def main():
    model_dir = Path(args.model_dir)
    # serialize args
    with open(model_dir / "args.json", "w") as json_file:
        json.dump(vars(args), json_file, ensure_ascii=False, indent=4)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logging.info("HOST=" + os.environ.get("HOST", ""))
    logging.info("SLURM_JOB_ID=" + os.environ.get("SLURM_JOB_ID", ""))
    logging.info(args)
    # init libraries
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # data preparation
    # TODO logging dataset-size
    io.set_kaldi_device()
    train_egs = io.RandExample(args.train_scp, args.seed, args.train_minibatch_size)
    valid_egs = io.RandExample(args.valid_scp, args.seed, args.valid_minibatch_size) # TODO no shuffle
    logging.info("train set batch: {}, sample: {}".format(train_egs.n_batch, train_egs.n_data))
    logging.info("valid set batch: {}, sample: {}".format(valid_egs.n_batch, valid_egs.n_data))
    # get shape
    (mfcc, ivec), supervision = valid_egs.value()
    n_pdf = supervision.n_pdf
    n_feat = mfcc.shape[1]
    n_ivec = ivec.shape[1]
    logging.info("shape-info: (n_feat: {}, n_ivec: {}, n_pdf: {})".format(n_feat, n_ivec, n_pdf))

    # load fst
    den_graph = io.DenominatorGraph(args.den_fst, n_pdf)

    # load LDA matrix
    if args.lda_mat is not None:
        lda_mat = kaldiio.load_mat(args.lda_mat)
        lda_mat = torch.from_numpy(lda_mat)
    else:
        lda_mat = None

    # model preparation
    model = models.SelfAttentionTDNN(n_pdf, n_feat, n_ivec, lda_mat=lda_mat, args=args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    logging.info(model)
    # params = iter(p for p in model.parameters() if p.requires_grad)
    params = model.module.kaldi_like_parameters()
    if args.optim == "Nesterov":
        opt = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=args.momentum)
    else:
        opt_class = getattr(torch.optim, args.optim)
        opt = opt_class(params, lr=args.lr, weight_decay=args.weight_decay)
        for p in opt.param_groups:
            if "momentum" in p:
                p["momentum"] = args.momentum
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
                          xent_input=xe_pred, kaldi_way=True)

    # main loop
    for epoch in range(args.n_epoch):
        logging.info("epoch: {}".format(epoch))
        # training
        model.train()
        train_result = ChainResults()
        train_egs.reset() # re-shuffle
        for i, data in enumerate(train_egs, 1):
            idx = None # example.indexes
            loss, results = forward(data, idx)
            loss.backward()
            if i % args.accum_grad == 0:
                opt.step()
                opt.zero_grad()
            train_result.data += results.data
            logging.info("train loss: {}, average: {}, iter {} / {}".format(
                results, train_result.loss, i, train_egs.n_batch))
        logging.info("summary train loss average: {}".format(train_result.loss))

        # validation
        model.eval()
        valid_result = ChainResults()
        valid_egs.reset()
        with torch.no_grad():
            for i, data in enumerate(valid_egs, 1):
                idx = None # example.indexes
                loss, results = forward(data, idx)
                valid_result.data += results.data
                logging.info("train loss: {}, average: {}, iter {} / {}".format(
                    results, valid_result.loss, i, valid_egs.n_batch))
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
