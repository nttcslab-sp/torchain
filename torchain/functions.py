# functions/add.py
import torch
from torchain import io
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


class _ChainLoss(Function):
    """
   Lattice-free MMI loss function

   Args:
   @param [in] input      The output of the neural net; dimension must equal
                          ((supervision.num_sequences * supervision.frames_per_sequence) by
                            den_graph.NumPdfs()).  The rows are ordered as: all sequences
                            for frame 0; all sequences for frame 1; etc.
   @param [in] den_graph   The denominator graph, derived from denominator fst.
   @param [in] supervision  The supervision object, containing the supervision
                            paths and constraints on the alignment as an FST

   Results:
   @param [out] objf       The [num - den] objective function computed for this
                           example; you'll want to divide it by 'tot_weight' before
                           displaying it.
   @param [out] l2_term    The l2 regularization term in the objective function, if
                           the --l2-regularize option is used.  To be added to 'o
   @param [out] weight     The weight to normalize the objective function by;
                           equals supervision.weight * supervision.num_sequences *
                           supervision.frames_per_sequence.

   Options:
   l2_regularize           l2 regularization constant on the 'chain' output;
                           the actual term added to the objf will be -0.5 times
                           this constant times the squared l2 norm.
                           (squared so it's additive across the dimensions).
                           e.g. try 0.0005.

   leaky_hmm_coefficient   Coefficient for 'leaky hmm'.  This means we have an epsilon-transition from
                           each state to a special state with probability one, and then another
                           epsilon-transition from that special state to each state, with probability
                           leaky_hmm_coefficient times [initial-prob of destination state].  Imagine
                           we make two copies of each state prior to doing this, version A and version
                           B, with transition from A to B, so we don't have to consider epsilon loops-
                           or just imagine the coefficient is small enough that we can ignore the
                           epsilon loops.

    """
    @staticmethod
    def forward(ctx, input, results, den_graph, supervision,
                l2_regularize, leaky_hmm_coefficient):
        assert input.is_cuda, "Only CUDA implementation is available"
        if isinstance(supervision, io.Supervision):
            supervision = supervision.ptr
        if isinstance(den_graph, io.DenominatorGraph):
            den_graph = den_graph.ptr
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
    if input.dim() == 3:
        n_pdf = input.shape[1]
        input = input.transpose(1, 2).contiguous().view(-1, n_pdf)
    results = ChainResults()
    loss = _ChainLoss.apply(input, results, den_graph, supervision,
                            l2_regularize, leaky_hmm_coefficient)
    return loss, results
