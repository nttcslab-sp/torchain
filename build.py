import os
import torch
from torch.utils.ffi import create_extension


headers = ['src/my_lib.h', 'src/my_lib_cuda.h']
extra_objects = ["libmy_lib.a", "libmy_lib_cuda.a"]
extra_link_args = [
    "-lgcov",
    "-lstdc++",
    "-L" + os.environ["KALDI_ROOT"] + "/src/lib",
    "-lkaldi-matrix",
    "-lkaldi-cudamatrix",
    "-lkaldi-chain",
    "-lkaldi-nnet3"
]
pwd = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(pwd, fname) for fname in extra_objects]
ffi = create_extension(
    'torchain._ext.my_lib',
    headers=headers,
    package=True,
    sources=[],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    extra_compile_args=["-std=c99", "-fopenmp"],
    extra_link_args=extra_link_args,
    extra_objects=extra_objects,
    with_cuda=True,
    verbose=True
)

if __name__ == '__main__':
    ffi.build()
