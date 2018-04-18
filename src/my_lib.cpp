#include <assert.h>
#include <iostream>
#include <memory>

#include <matrix/kaldi-matrix.h>
#include <TH/TH.h>
#include <ATen/ATen.h>

#include "common.hpp"


extern "C" {

    int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
                           THFloatTensor *output)
    {
        if (!THFloatTensor_isSameSizeAs(input1, input2))
            return 0;
        THFloatTensor_resizeAs(output, input1);
        THFloatTensor_cadd(output, input1, 1.0, input2);
        return 1;
    }

    int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
    {
        THFloatTensor_resizeAs(grad_input, grad_output);
        THFloatTensor_fill(grad_input, 1);
        return 1;
    }


    int my_lib_aten_cpu(THFloatTensor* t) {
        // test sharing kaldi -> torch
        {
            auto m = std::make_shared<kaldi::Matrix<float>>(3, 4);
            auto a = common::make_tensor(m);
            a[0][0] = 23;
            std::cout << *m << std::endl;
            m->Add(100);
            std::cout << a << std::endl;
            assert(*(a[0][0].template data<float>()) == 123);
        }

        // test sharing torch -> kaldi
        {
            at::Tensor a = at::CPU(at::kFloat).unsafeTensorFromTH(t, true);
            std::cout << a << std::endl;
            // auto m = common::make_matrix<kaldi::SubMatrix<float>>(a);
            auto m = common::make_matrix(t);
            a[0][0] = 23;
            std::cout << m << std::endl;
            m.Add(100);
            std::cout << a << std::endl;
            assert(*(a[0][0].template data<float>()) == 123);
        }
        return 1;
    }
}

