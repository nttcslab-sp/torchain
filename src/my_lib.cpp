#include <iostream>
#include <memory>

#include <matrix/kaldi-matrix.h>
#include <TH/TH.h>
#include <ATen/ATen.h>

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
        auto m = std::make_shared<kaldi::Matrix<kaldi::BaseFloat>>(3, 4);
        at::Tensor a = at::CPU(at::kFloat).unsafeTensorFromTH(t, true);
        std::cout << a << std::endl;
        return 1;
    }
}

