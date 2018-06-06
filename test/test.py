import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchain._ext import my_lib
from torchain.functions import chain_loss
from torchain import io


def test_chain():
    out = torch.cuda.FloatTensor([0])
    grads = torch.cuda.FloatTensor([0])
    my_lib.my_lib_test_chain(out, grads)
    my_lib.my_lib_set_kaldi_device(torch.cuda.FloatTensor(1))
    print(out.shape, grads.shape)


class Model(nn.Module):
    def __init__(self, n_pdf):
        super().__init__()
        self.common = torch.nn.Sequential(
            torch.nn.Conv1d(40, 512, 29, 3),
            torch.nn.ReLU(),
            torch.nn.Conv1d(512, 512, 1, 1)
        )
        self.lf_mmi = torch.nn.Conv1d(512, n_pdf, 1, 1)
        self.xent = torch.nn.Conv1d(512, n_pdf, 1, 1)

    def forward(self, x):
        h = self.common(x)
        return self.lf_mmi(h), self.xent(h)


def test_io():
    exp_root = "/data/work70/skarita/exp/chime5/kaldi-22fbdd/egs/chime5/s5/"
    den_fst_rs = exp_root + "exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/den.fst"
    cmd = "nnet3-chain-copy-egs --frame-shift=1  ark:/data/work49/skarita/repos/torchain/cegs.1.ark ark:- | nnet3-chain-shuffle-egs --buffer-size=5000 --srand=0 ark:- ark:- | nnet3-chain-merge-egs --minibatch-size=128,64,32 ark:- ark:-"
    with io.open_example(cmd) as example:
        idx = example.indexes
        print(idx.shape)
        print(idx[0])
        (mfcc, ivec), sup = example.value()
        print(mfcc.shape)
        sup = example.supervision
        assert sup.n_frame == idx.shape[1]

    for use_xent in [True]:
        for use_kaldi_way in [True]:
            print("xent: ", use_xent, "kaldi: ", use_kaldi_way)
            with io.open_example(cmd) as example:
                n_pdf = example.supervision.n_pdf
                print(n_pdf)
                # n_pdf = 2928
                den_graph = io.DenominatorGraph(den_fst_rs, n_pdf)
                model = Model(n_pdf)
                model.cuda()
                print(model)
                opt = torch.optim.SGD(model.parameters(), lr=1e-6)
                count = 0
                start = time.time()
                for (mfcc, ivec), supervision in example:
                    x = Variable(mfcc).cuda()
                    print("input:", x.shape)
                    pred, xent = model(x)
                    if not use_xent:
                        xent = None
                    loss, results = chain_loss(pred, den_graph, supervision, l2_regularize=0.01,
                                               xent_regularize=0.01, xent_input=xent, kaldi_way=use_kaldi_way)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    print(count, results)
                    count += 1
                    if count > 10:
                        break
                elapsed = time.time() - start
                print("took: %f" % (elapsed))


def test_rand_io():
    scp_path = "/data/work49/skarita/repos/torch-backup/100.scp" # example/chime5/exp/scp/egs.scp"
    seed = 1
    exp_root = "/data/work70/skarita/exp/chime5/kaldi-22fbdd/egs/chime5/s5/"
    den_fst_rs = exp_root + "exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/den.fst"

    for use_xent in [True]:
        for use_kaldi_way in [True]:
            print("xent: ", use_xent, "kaldi: ", use_kaldi_way)
            print("shuffling")
            io.set_kaldi_device()
            example = io.RandExample(scp_path, seed, 128)
            n_pdf = example.supervision.n_pdf
            print(n_pdf)
            # n_pdf = 2928
            den_graph = io.DenominatorGraph(den_fst_rs, n_pdf)
            model = Model(n_pdf)
            model.cuda()
            print(model)
            opt = torch.optim.SGD(model.parameters(), lr=1e-6)
            count = 0
            start = time.time()
            for (mfcc, ivec), supervision in example:
                x = Variable(mfcc).cuda()
                print("input:", x.shape)
                pred, xent = model(x)
                if not use_xent:
                    xent = None
                loss, results = chain_loss(pred, den_graph, supervision, l2_regularize=0.01,
                                           xent_regularize=0.01, xent_input=xent, kaldi_way=use_kaldi_way)
                opt.zero_grad()
                loss.backward()
                opt.step()
                print(count, results)
                count += 1
                if count > 10:
                    break
            elapsed = time.time() - start
            print("took: %f" % (elapsed))





if __name__ == "__main__":
    # test_chain()
    test_rand_io()
    test_io()

