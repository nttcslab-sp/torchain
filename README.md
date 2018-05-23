# PyTorch FFI package with Kaldi and ATen

``` console
# with pytorch 0.4.0
$ source /data/work49/skarita/tool/miniconda3/bin/activate torch
# or with pytorch 0.3.1
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

this diff will be not required in the future

