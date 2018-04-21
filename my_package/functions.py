# functions/add.py
import torch
from torch.autograd import Function
from ._ext import my_lib
ffi = my_lib._ffi


class ChainResults:
    def __init__(self):
        self.data = torch.zeros(3)

    def __repr__(self):
        return "ChainResults(loss=%f, objf=%f, l2_term=%f, weight=%lf)" % (
            self.loss, self.data[0], self.data[1], self.data[2])

    @property
    def loss(self):
        return -(self.data[0] / self.data[2] + self.data[1])


class ChainLoss(Function):
    @staticmethod
    def forward(ctx, input, results, den_graph, supervision,
                l2_regularize, leaky_hmm_coefficient):
        assert input.is_cuda, "Only CUDA implementation is available"
        mmi_grad = input.new(*input.shape)
        xent_grad = ffi.NULL # input.new()
        my_lib.my_lib_ComputeChainObjfAndDeriv(
            den_graph, supervision, input,
            results.data,
            mmi_grad, xent_grad,
            l2_regularize, leaky_hmm_coefficient, 0.0)
        ctx.mmi_grad = mmi_grad
        return input.new([results.loss])

    @staticmethod
    def backward(ctx, grad_output):
        return torch.autograd.Variable(-ctx.mmi_grad), None, None, None, None, None


def chain_loss(input, den_graph, supervision,
               l2_regularize=0.0, leaky_hmm_coefficient=1e-5):
    results = ChainResults()
    loss = ChainLoss.apply(input, results, den_graph, supervision,
                           l2_regularize, leaky_hmm_coefficient)
    return loss, results
