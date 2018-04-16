# PyTorch FFI package with Kaldi and ATen

``` console
$ make test KALDI_ROOT=/data/work44/public/kaldi
```


This example shows how to structure the code to create an ffi package for
PyTorch. It can be later distributed via pip.

### Required files:

* `setup.py` - setuptools file, that defines package metadata and some extension
    options
* `my_package/build.py` - cffi build file. Defines the extensions and builds
    them if executed.
