int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
		       THFloatTensor *output);
int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);
int my_lib_aten_cpu(THFloatTensor* t);

// chain example
void* my_lib_example_reader_new(const char* examples_rspecifier);
int my_lib_example_reader_next(void* reader_ptr);
void my_lib_example_reader_free(void* reader_ptr);
int my_lib_example_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux);
void* my_lib_supervision_new(void* reader_ptr);
void my_lib_supervision_free(void* supervision_ptr);
void* my_lib_denominator_graph_new(const char* rxfilename, void* supervision_ptr);
void my_lib_denominator_graph_free(void* den_graph_ptr);
