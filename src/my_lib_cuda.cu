#define HAVE_CUDA 1

#include <iostream>
#include <memory>
#include <matrix/kaldi-matrix.h>
#include <cudamatrix/cu-matrix.h>

#include <THC/THC.h>
#include <ATen/ATen.h>

#include "common.hpp"


extern "C"
{
    // this symbol will be resolved automatically from PyTorch libs
    extern THCState *state;

    int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
                                THCudaTensor *output)
    {
        if (!THCudaTensor_isSameSizeAs(state, input1, input2))
            return 0;
        THCudaTensor_resizeAs(state, output, input1);
        THCudaTensor_cadd(state, output, input1, 1.0, input2);
        return 1;
    }

    int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
    {
        THCudaTensor_resizeAs(state, grad_input, grad_output);
        THCudaTensor_fill(state, grad_input, 1);
        return 1;
    }

    // re-initialize kaldi with gpu-id of tensor t
    void set_kaldi_device(THCudaTensor* t) {
        // NOTE: need to patch kaldi for making all the private fields
        //       e.g., active_gpu_id_ public
        auto gpu_id = THCudaTensor_getDevice(state, t);
        auto& device = kaldi::CuDevice::Instantiate();
        if (device.ActiveGpuId() == gpu_id) return; // maybe no need to reset
        kaldi::CuDevice::Instantiate().AllowMultithreading();
        device.active_gpu_id_ = gpu_id;
        device.handle_ = THCState_getCurrentBlasHandle(state);
        device.cusparse_handle_ = THCState_getCurrentSparseHandle(state);
        device.properties_ = *THCState_getCurrentDeviceProperties(state);
        assert(kaldi::CuDevice::Instantiate().Enabled());
    }

    int my_lib_aten(THCudaTensor* t)
    {
        // NOTE: do not forget to set
        set_kaldi_device(t);
        at::Tensor a = at::CUDA(at::kFloat).unsafeTensorFromTH(t, true);

        // test cublas_copy (cublas handler works)
        {
            kaldi::CuMatrix<float> m(3, 3);
            kaldi::CuVector<float> v(3);
            m.CopyColFromVec(v, 0);
        }

        // test sharing kaldi -> torch
        {
            auto m = std::make_shared<kaldi::CuMatrix<float>>(3, 4);
            auto a = common::make_tensor(m);
            a[0][0] = 23;
            m->Add(100);
            std::cout << a << std::endl;
            assert((a[0][0] == 123).all());
        }

        // test sharing torch -> kaldi
        {
            auto m = common::make_matrix<kaldi::CuSubMatrix<float>>(a);
            a[0][0] = 23;
            m.Add(100);
            std::cout << a << std::endl;
            assert((a[0][0] == 123).all());
        }

        return 1;
    }
}