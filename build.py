import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)
sources = []
headers = ['src/my_lib.h']
defines = []
extra_objects = ["libmy_lib.a"]
extra_link_args = ["-lgcov", "-lstdc++"]
extra_link_args += ["-L" + os.environ["KALDI_ROOT"] + "/src/matrix",  "-lkaldi-matrix"]
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    extra_objects += ["libmy_lib_cuda.a"]
    extra_link_args += ["-L" + os.environ["KALDI_ROOT"] + "/src/cudamatrix",  "-lkaldi-cudamatrix"]
    extra_link_args += ["-L" + os.environ["KALDI_ROOT"] + "/src/chain",  "-lkaldi-chain"]
    with_cuda = True

# need for linking
this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# extra_objects += [os.environ["KALDI_ROOT"] + "/src/matrix/kaldi-matrix.a"]
# extra_objects += [os.environ["KALDI_ROOT"] + "/src/cudamatrix/kaldi-cudamatrix.a"]

ffi = create_extension(
    'my_package._ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    extra_compile_args=["-fopenmp"],
    extra_link_args=extra_link_args,
    extra_objects=extra_objects,
    with_cuda=with_cuda,
    verbose=True
)

if __name__ == '__main__':
    ffi.build()
