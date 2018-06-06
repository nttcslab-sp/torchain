/// chain example is a minibatch of utterance chunks and targets
void* my_lib_example_reader_new(const char* examples_rspecifier);
int my_lib_example_reader_next(void* reader_ptr);
void my_lib_example_reader_free(void* reader_ptr);
int my_lib_example_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux);

/// custom example reader with shuffle and minibatch
void* my_lib_example_rand_reader_new(const char* examples_rspecifier, int seed, int batchsize,
                                     const char* len_file);
void my_lib_example_rand_reader_reset(void* reader_ptr);
int my_lib_example_rand_reader_num_batch(void* reader_ptr);
int my_lib_example_rand_reader_num_data(void* reader_ptr);

int my_lib_example_rand_reader_next(void* reader_ptr);
void my_lib_example_rand_reader_free(void* reader_ptr);
int my_lib_example_rand_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux);
void* my_lib_supervision_rand_new(void* reader_ptr);
void print_key_length(const char* rspec, const char* len_file);

/// chain supervision is target data in example
void* my_lib_supervision_new(void* reader_ptr);
void my_lib_supervision_free(void* supervision_ptr);
int my_lib_supervision_num_pdf(void* supervision_ptr);
int my_lib_supervision_num_sequence(void* supervision_ptr);
int my_lib_supervision_num_frame(void* supervision_ptr);
int my_lib_example_reader_indexes(void* reader_ptr, THLongTensor* index_tensor);

/// chain denominator holds lattices per utterances in a dataset
void* my_lib_denominator_graph_new(const char* rxfilename, int num_pdf);
void my_lib_denominator_graph_free(void* den_graph_ptr);

/// chain loss function
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

/// execute C++ native test function in kaldi
int my_lib_test_chain(THCudaTensor* out, THCudaTensor* grad);

/// set kaldi CUDA device/handlers equal to torch cuda tensor
void my_lib_set_kaldi_device(THCudaTensor* t);
