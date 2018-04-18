int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output);
int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input);
int my_lib_aten(THCudaTensor* t);
int my_lib_ComputeChainObjfAndDeriv(
        // inputs
        void* den_graph_ptr, void* supervision_ptr, THCudaTensor* nnet_output_ptr,
        // outputs
        float* objf, float* l2_term, float* weight,
        // grads
        THCudaTensor* nnet_output_deriv_ptr, THCudaTensor* xent_output_deriv_ptr,
        // hyper params
        float l2_regularize, float leaky_hmm_coefficient, float xent_regularize);

int my_lib_test_chain(THCudaTensor* out, THCudaTensor* grad);

