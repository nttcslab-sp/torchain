# PyTorch FFI package with Kaldi and ATen

``` console
$ source /data/work70/skarita/exp/chime5/venv/bin/activate
$ make test KALDI_ROOT=/data/work70/skarita/exp/chime5/kaldi-22fbdd
```

note that 
- I configured my kaldi with `./configure --mkl-root=$CONDA_PREFIX --mkl-libdir=$CONDA_PREFIX/lib`
- I patched `src/cudamatrix/cu-device.h` with

``` diff
index 99105355a..3bcbcfc5a 100644
--- a/src/cudamatrix/cu-device.h
+++ b/src/cudamatrix/cu-device.h
@@ -158,7 +158,7 @@ class CuDevice {
   CuDevice(CuDevice&); // Disallow.
   CuDevice &operator=(CuDevice&);  // Disallow.
 
-
+public:
   static CuDevice global_device_;
   cublasHandle_t handle_;
   cusparseHandle_t cusparse_handle_;
```

This example shows how to structure the code to create an ffi package for
PyTorch. It can be later distributed via pip.

### Required files:

* `setup.py` - setuptools file, that defines package metadata and some extension
    options
* `build.py` - cffi build file. Defines the extensions and builds
    them if executed.

