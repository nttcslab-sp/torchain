#include <stdio.h>

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


using SeqReader = kaldi::nnet3::SequentialNnetChainExampleReader;
using RandReader = kaldi::nnet3::RandomAccessNnetChainExampleReader;

using Example = kaldi::nnet3::NnetChainExample;
using kaldi::chain::Supervision;


extern "C" {
    void* my_lib_example_reader_new(const char* examples_rspecifier) {
        auto example_reader = new SeqReader(examples_rspecifier);
        return static_cast<void*>(example_reader);
    }

    int my_lib_example_reader_next(void* reader_ptr) {
        auto reader = static_cast<SeqReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        reader->Next();
        return 1; // success
    }

    void my_lib_example_reader_free(void* reader_ptr) {
        auto reader = static_cast<SeqReader*>(reader_ptr);
        reader->Close();
        delete reader;
    }

    // NOTE: this function returns size of inputs instead of success/fail
    int my_lib_example_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux) {
        auto reader = static_cast<SeqReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        auto&& egs = reader->Value();

        // read input feats. e.g., mfcc
        if (input != nullptr) {
            common::copy_to_mat(egs.inputs[0].features, input);
        }

        // read aux feats. e.g., i-vector
        if (aux != nullptr && egs.inputs.size() > 1) {
            common::copy_to_mat(egs.inputs[1].features, aux);
        }
        return egs.inputs.size(); // success
    }

    void* my_lib_supervision_new(void* reader_ptr) {
        auto reader = static_cast<SeqReader*>(reader_ptr);
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

    int my_lib_example_reader_indexes(void* reader_ptr, THLongTensor* index_tensor) {
        /// The indexes that the output corresponds to.  The size of this vector will
        /// be equal to supervision.num_sequences * supervision.frames_per_sequence.
        /// Be careful about the order of these indexes-- it is a little confusing.
        /// The indexes in the 'index' vector are ordered as: (frame 0 of each sequence);
        /// (frame 1 of each sequence); and so on.  But in the 'supervision' object,
        /// the FST contains (sequence 0; sequence 1; ...).  So reordering is needed
        /// when doing the numerator computation.
        /// We order 'indexes' in this way for efficiency in the denominator
        /// computation (it helps memory locality), as well as to avoid the need for
        /// the nnet to reorder things internally to match the requested output
        /// (for layers inside the neural net, the ordering is (frame 0; frame 1 ...)
        /// as this corresponds to the order you get when you sort a vector of Index).
        auto reader = static_cast<SeqReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        auto&& egs = reader->Value();

        // TODO support multiple outputs
        const auto& chain_supervision = egs.outputs[0];
        const auto& indexes = chain_supervision.indexes;
        const auto& supervision = chain_supervision.supervision;
        THLongTensor_resize2d(index_tensor, supervision.frames_per_sequence, supervision.num_sequences);
        auto index_data = THLongTensor_data(index_tensor);
        for (const auto& idx : indexes) {
            // printf("{n:%d, t:%d, x:%d}\n", idx.n, idx.t, idx.x);
            *index_data = idx.t;
            ++index_data;
        }
        THLongTensor_transpose(index_tensor, nullptr, 0, 1); // (B, T)
        return egs.inputs.size(); // success
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
