#include "common.hpp"
#include <THC/THC.h>

#include <nnet3/nnet-example.h>
#include <nnet3/nnet-chain-example.h>
#include <nnet3/nnet-chain-training.h>

#include <chain/chain-supervision.h>
#include <base/kaldi-common.h>
#include <util/common-utils.h>

// official example usage in kaldi
// see https://github.com/kaldi-asr/kaldi/blob/19dc26ff833cbaedb5a2ffd2609d7cd8d0a8e6a5/src/chainbin/nnet3-chain-train.cc

// how to create char[] in python: ffi.new("char[]", "hello")
// see http://cffi.readthedocs.io/en/latest/using.html

// how to create nullptr in python: ffi.NULL
// http://cffi.readthedocs.io/en/latest/ref.html#ffi-null


using ExampleReader = kaldi::nnet3::SequentialNnetChainExampleReader;
using Example = kaldi::nnet3::NnetChainExample;
using kaldi::chain::Supervision;

void copy_to_mat(kaldi::GeneralMatrix& src, THFloatTensor* dst) {
    THFloatTensor_resize2d(dst, src.NumRows(), src.NumCols());
    auto mat = common::make_matrix(dst);
    src.CopyToMat(&mat);
}

extern "C" {
    void* my_lib_example_reader_new(const char* examples_rspecifier) {
        auto example_reader = new ExampleReader(examples_rspecifier);
        return static_cast<void*>(example_reader);
    }

    int my_lib_example_reader_next(void* reader_ptr) {
        auto reader = static_cast<ExampleReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        reader->Next();
        return 1; // success
    }

    void my_lib_example_reader_free(void* reader_ptr) {
        delete static_cast<ExampleReader*>(reader_ptr);
    }

    // NOTE: this function returns size of inputs instead of success/fail
    int my_lib_example_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux) {
        auto reader = static_cast<ExampleReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        auto&& egs = reader->Value();

        // read input feats. e.g., mfcc
        if (input != nullptr) {
            copy_to_mat(egs.inputs[0].features, input);
        }

        // read aux feats. e.g., i-vector
        if (aux != nullptr && egs.inputs.size() > 1) {
            copy_to_mat(egs.inputs[1].features, aux);
        }
        return egs.inputs.size(); // success
    }

    void* my_lib_supervision_new(void* reader_ptr) {
        auto reader = static_cast<ExampleReader*>(reader_ptr);
        if (reader->Done()) return nullptr; // fail
        auto&& egs = reader->Value();
        return static_cast<void*>(new Supervision(egs.outputs[0].supervision));
    }

    void my_lib_supervision_free(void* supervision_ptr) {
        delete static_cast<Supervision*>(supervision_ptr);
    }

    int my_lib_supervision_num_pdf(void* supervision_ptr) {
        auto supervision = static_cast<Supervision*>(supervision_ptr);
        return supervision->label_dim;
    }

    int my_lib_supervision_num_sequence(void* supervision_ptr) {
        auto supervision = static_cast<Supervision*>(supervision_ptr);
        return supervision->num_sequences;
    }

    int my_lib_supervision_num_frame(void* supervision_ptr) {
        auto supervision = static_cast<Supervision*>(supervision_ptr);
        return supervision->frames_per_sequence;
    }

    void* my_lib_denominator_graph_new(const char* rxfilename, int num_pdfs) {
        fst::StdVectorFst den_fst;
        fst::ReadFstKaldi(rxfilename, &den_fst);
        auto den_graph = new kaldi::chain::DenominatorGraph(den_fst, num_pdfs);
        return static_cast<void*>(den_graph);
    }

    void my_lib_denominator_graph_free(void* den_graph_ptr) {
        delete static_cast<kaldi::chain::DenominatorGraph*>(den_graph_ptr);
    }
}
