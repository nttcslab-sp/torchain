// chain example
void* my_lib_example_reader_new(const char* examples_rspecifier);
int my_lib_example_reader_next(void* reader_ptr);
void my_lib_example_reader_free(void* reader_ptr);
int my_lib_example_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux);
void* my_lib_supervision_new(void* reader_ptr);
void my_lib_supervision_free(void* supervision_ptr);
int my_lib_supervision_num_pdf(void* supervision_ptr);
int my_lib_supervision_num_sequence(void* supervision_ptr);
int my_lib_supervision_num_frame(void* supervision_ptr);
void* my_lib_denominator_graph_new(const char* rxfilename, int num_pdf);
void my_lib_denominator_graph_free(void* den_graph_ptr);


int my_lib_ComputeChainObjfAndDeriv(
    // inputs
    void* den_graph_ptr, void* supervision_ptr, THCudaTensor* nnet_output_ptr,
    // outputs
    // float* objf, float* l2_term, float* weight,
    THFloatTensor* results,
    // grads
    THCudaTensor* nnet_output_deriv_ptr, THCudaTensor* xent_output_deriv_ptr,
    // hyper params
    float l2_regularize, float leaky_hmm_coefficient, float xent_regularize);

int my_lib_test_chain(THCudaTensor* out, THCudaTensor* grad);
void my_lib_set_kaldi_device(THCudaTensor* t);
