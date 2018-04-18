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

# test_net()
test_aten()
