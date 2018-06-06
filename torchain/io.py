import os
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
    def __init__(self, ptr):
        self.ptr = ptr
        if self.ptr == ffi.NULL:
            raise ValueError("null supervision ptr")
        self.n_pdf = my_lib.my_lib_supervision_num_pdf(self.ptr)
        self.n_batch = my_lib.my_lib_supervision_num_sequence(self.ptr)
        self.n_frame = my_lib.my_lib_supervision_num_frame(self.ptr)
        self.shape = (self.n_batch, self.n_frame, self.n_pdf)

    def __del__(self):
        my_lib.my_lib_supervision_free(self.ptr)


def feats(egs):
    inp = torch.FloatTensor()
    aux = torch.FloatTensor()
    if isinstance(egs, Example):
        n = my_lib.my_lib_example_feats(self.ptr, inp, aux)
        # TODO add RandExample here
    else:
        raise ValueError("unknown reader type")

    if n == 1:
        return inp, None
    elif n == 2:
        return inp, aux
    else:
        raise ValueError("unsupported number of inputs (up to 2): %d" % n)


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

    def load_feats(self, inp, aux):
        return my_lib.my_lib_example_feats(self.ptr, inp, aux)

    @property
    def supervision(self):
        return Supervision(my_lib.my_lib_supervision_new(self.ptr))

    @property
    def indexes(self):
        idx = torch.LongTensor()
        err = my_lib.my_lib_example_reader_indexes(self.ptr, idx);
        assert err != 0
        return idx

    @property
    def inputs(self):
        inp = torch.FloatTensor()
        aux = torch.FloatTensor()
        n = self.load_feats(inp, aux)
        if n == 1:
            return inp, None
        elif n == 2:
            return inp, aux
        else:
            raise ValueError("unsupported number of inputs (up to 2): %d" % n)

    def value(self):
        supervision = self.supervision
        n_batch, n_out_frame, n_pdf = supervision.shape
        inp, aux = self.inputs
        if inp is not None:
            inp = inp.view(n_batch, -1, inp.shape[1]).transpose(1, 2)
        return (inp, aux), supervision

    def __iter__(self):
        while self.next():
            try:
                yield self.value()
            except ValueError:
                continue


@contextmanager
def open_example(cmd):
    import subprocess
    import os
    import tempfile
    tmpdir = tempfile.mkdtemp()
    FIFO = os.path.join(tmpdir, 'myfifo.ark')
    os.mkfifo(FIFO)
    # FIXME: maybe Popen is better? (in terms of safety)
    subprocess.run(cmd + " > " + FIFO + " &", shell=True, check=True)
    example_rs = "ark,bg:" + FIFO
    set_kaldi_device()
    example = Example(example_rs)
    yield example
    os.remove(FIFO)
    os.rmdir(tmpdir)
    del example


def print_key_length(scp_path, len_file="/dev/stdout"):
    my_lib.print_key_length(cstr(scp_path), cstr(len_file))


class RandExample(Example):
    """
    native C++ example reader without subprocess
    """
    def __init__(self, scp_path, seed, batchsize, len_file=""):
        assert os.path.exists(scp_path)
        self.scp_path = scp_path
        self.ptr = my_lib.my_lib_example_rand_reader_new(cstr(scp_path), seed, batchsize, cstr(len_file))

    def __del__(self):
        my_lib.my_lib_example_rand_reader_free(self.ptr)

    def reset(self):
        my_lib.my_lib_example_rand_reader_reset(self.ptr)

    @property
    def n_batch(self):
        return my_lib.my_lib_example_rand_reader_num_batch(self.ptr)

    @property
    def n_data(self):
        return my_lib.my_lib_example_rand_reader_num_data(self.ptr)

    # override functions
    def next(self):
        return my_lib.my_lib_example_rand_reader_next(self.ptr) == 1

    def load_feats(self, inp, aux):
        return my_lib.my_lib_example_rand_feats(self.ptr, inp, aux)

    @property
    def supervision(self):
        return Supervision(my_lib.my_lib_supervision_rand_new(self.ptr))

    @property
    def indexes(self):
        idx = torch.LongTensor()
        err = my_lib.my_lib_example_reader_indexes(self.ptr, idx);
        assert err != 0
        return idx
