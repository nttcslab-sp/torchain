from contextlib import contextmanager

import torch
from ._ext import my_lib


ffi = my_lib._my_lib.ffi


def cstr(s):
    return ffi.new("char[]", s.encode())


def set_kaldi_device(device_id=0):
    # FIXME: do not allocate memory
    my_lib.my_lib_set_kaldi_device(torch.zeros(1).cuda(device_id))


class Supervision:
    def __init__(self, egs):
        assert isinstance(egs, Example)
        self.ptr = my_lib.my_lib_supervision_new(egs.ptr)
        self.n_pdf = my_lib.my_lib_supervision_num_pdf(self.ptr)
        self.n_batch = my_lib.my_lib_supervision_num_sequence(self.ptr)
        self.n_frame = my_lib.my_lib_supervision_num_frame(self.ptr)
        self.shape = (self.n_batch, self.n_frame, self.n_pdf)

    def __del__(self):
        my_lib.my_lib_supervision_free(self.ptr)


class DenominatorGraph:
    def __init__(self, rspec, n_pdf):
        self.rspec = rspec
        self.ptr = my_lib.my_lib_denominator_graph_new(cstr(rspec), n_pdf)

    def __del__(self):
        my_lib.my_lib_denominator_graph_free(self.ptr)


class Example:
    def __init__(self, rspec):
        self.rspec = rspec
        self.ptr = my_lib.my_lib_example_reader_new(cstr(rspec))

    def __del__(self):
        my_lib.my_lib_example_reader_free(self.ptr)

    def next(self):
        return my_lib.my_lib_example_reader_next(self.ptr) == 1

    @property
    def supervision(self):
        return Supervision(self)

    @property
    def inputs(self):
        inp = torch.FloatTensor()
        aux = torch.FloatTensor()
        n = my_lib.my_lib_example_feats(self.ptr, inp, aux)
        if n == 1:
            return inp, None
        elif n == 2:
            return inp, aux
        else:
            raise ValueError("unsupported number of inputs (up to 2): %d" % n)

    def __iter__(self):
        while self.next():
            supervision = self.supervision
            n_batch, n_out_frame, n_pdf = supervision.shape
            inp, aux = self.inputs
            if inp is not None:
                inp = inp.view(n_batch, -1, inp.shape[1]).transpose(1, 2)
            yield (inp, aux), supervision
