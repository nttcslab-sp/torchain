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

    int my_lib_aten(THCudaTensor* t)
    {
        at::Tensor a = at::CUDA(at::kFloat).unsafeTensorFromTH(t, true);
        std::cout << "aten device: " << a.get_device() << std::endl;
        //kaldi::CuDevice::Instantiate().AllowMultithreading();
        //kaldi::CuDevice::Instantiate().SelectGpuId("yes");
        //assert(kaldi::CuDevice::Instantiate().ActiveGpuId() == a.get_device());
        // test sharing kaldi -> torch
        {
            auto m = std::make_shared<kaldi::CuMatrix<float>>(3, 4);
            std::cout << *m << std::endl;
            auto a = common::make_tensor(m);
            a[0][0] = 23;
            std::cout << *m << std::endl;
            m->Add(100);
            std::cout << *m << std::endl;
            // FIXME: cannot cudaMemcpy in torch
            // std::cout << a << std::endl;
            // assert(*(a.toBackend(at::kCPU).template data<float>()) == 123);
        }        
        // return 1;

        // test sharing torch -> kaldi
        {
            // FIXME: cannot do this here
            // at::Tensor a = at::CUDA(at::kFloat).unsafeTensorFromTH(t, true);
            // std::cout << a << std::endl;
            auto m = common::make_matrix<kaldi::CuSubMatrix<float>>(a);
            // FIXME: cannot do this here in torch
            // auto a = common::make_tensor(m);
            // a[0][0] = 23;
            // std::cout << m << std::endl;
            m.Add(100);
            // std::cout << m << std::endl;
            // std::cout << a << std::endl;
            // assert(*(a.toBackend(at::kCPU).template data<float>()) == 123);
        }

        return 1;
    }
}