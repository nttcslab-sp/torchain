import torch
import torch.nn as nn
from torch.autograd import Variable
from my_package._ext import my_lib


def test_aten():
    t = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
    my_lib.my_lib_aten_cpu(t)
    if torch.cuda.is_available():
        t = torch.cuda.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
        my_lib.my_lib_aten(t)
        print(t)
        t.mm(t)
        # t = torch.cuda.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
        # my_lib.my_lib_aten(t)
        out = torch.cuda.FloatTensor([0])
        grads = torch.cuda.FloatTensor([0])
        my_lib.my_lib_test_chain(out, grads)
        print(out.shape, grads.shape)

def test_example():
    ffi = my_lib._my_lib.ffi
    def cstr(s):
        return ffi.new("char[]", s.encode())

    exp_root = "/data/work70/skarita/exp/chime5/kaldi-22fbdd/egs/chime5/s5/"
    den_fst_rs = exp_root + "exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/den.fst"
    example_rs = "ark,bg:nnet3-chain-copy-egs --frame-shift=1  ark:" + exp_root + "exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/egs/cegs.1.ark ark:- | nnet3-chain-shuffle-egs --buffer-size=5000 --srand=0 ark:- ark:- | nnet3-chain-merge-egs --minibatch-size=128,64,32 ark:- ark:- |"

    example = my_lib.my_lib_example_reader_new(cstr(example_rs))
    # supervision = my_lib.my_lib_supervision_new(example)
    # den_fst = my_lib.my_lib_denominator_graph_new(cstr(den_fst_rs), supervision)
    # my_lib.my_lib_supervision_free(supervision)

    # # iter
    # mfcc = torch.FloatTensor([0])
    # ivec = torch.FloatTensor([0])
    # n = my_lib.my_lib_example_feats(example, mfcc, ivec)
    # assert n == 2, "number of inputs"
    # supervision = my_lib.my_lib_supervision_new(example)
    # # TODO do forward
    # my_lib.my_lib_supervision_free(supervision)
    # # if my_lib.my_lib_example_reader_next(example) == 0:
    # #     break

    # my_lib.my_lib_denominator_graph_free(den_fst)
    my_lib.my_lib_example_reader_free(example)

if __name__ == "__main__":
    # test_aten()
    test_example()
