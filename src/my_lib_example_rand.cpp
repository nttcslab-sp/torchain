#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <assert.h>

#include <unordered_map>
#include <unordered_set>

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

using Example = kaldi::nnet3::NnetChainExample;
using kaldi::chain::Supervision;

struct RandReader {
    using BaseReader = kaldi::nnet3::RandomAccessNnetChainExampleReader;
    using T = BaseReader::T;

    BaseReader reader;
    std::mt19937 engine;
    std::vector<std::vector<std::string>> key_batch;
    std::unordered_map<size_t, std::vector<std::string>> length_to_keys;
    std::unordered_set<size_t> lengths;
    int batchsize = 1;
    bool verbose = false;
    bool compress = false;
    Example minibatch;
    int current_pos = 0;
    int prev_pos = -1;

    RandReader(const std::string& rspec, int seed, int batchsize,
               bool verbose=true, bool compress=false) :
        reader(BaseReader("scp:" + rspec)),
        engine(seed),
        batchsize(batchsize),
        verbose(verbose),
        compress(compress)
    {
        if (verbose) std::cerr << "read keys" << std::endl;
        this->read_keys(rspec);
        if (verbose) std::cerr << "shuffle keys" << std::endl;
        this->shuffle_keys();
    }

    void read_keys(const std::string& rspec) {
        std::ios::sync_with_stdio(false);
        std::ifstream ifs(rspec);
        if (!ifs.is_open()) {
            throw std::runtime_error(rspec + " does not exist");
        }
        bool key = true;
        std::vector<std::string> keys;
        if (verbose) std::cerr << "accumurate keys" << std::endl;
        while (ifs) {
            std::string line;
            ifs >> line;
            if (key && !line.empty()) {
                keys.push_back(line);
            }
            key = !key;
        }

        if (verbose) std::cerr << "sort length" << std::endl;
        // create length_to_keys
        for (auto&& k : keys) {
            auto l = this->reader.Value(k).outputs[0].supervision.frames_per_sequence;
            if (this->length_to_keys.count(l) == 0) {
                this->length_to_keys[l] = {};
            }
            this->length_to_keys[l].push_back(k);
        }
    }

    void shuffle_keys() {
        this->key_batch.clear();
        std::vector<std::string> batch;
        batch.reserve(this->batchsize);
        // shuffle batch per length
        for (auto kv : this->length_to_keys) {
            auto& keys = kv.second;
            std::shuffle(keys.begin(), keys.end(), this->engine);
            for (const auto& k : keys) {
                batch.push_back(k);
                if (batch.size() == this->batchsize) {
                    this->key_batch.push_back(batch);
                    batch.clear();
                }
            }
            if (batch.size() > 0) {
                this->key_batch.push_back(batch);
            }
            batch.clear();
        }
        std::shuffle(this->key_batch.begin(), this->key_batch.end(), this->engine);
    }

    void reset() {
        this->current_pos = 0;
        this->prev_pos = -1;
        this->shuffle_keys();
    }

    /// use sequential reader interface
    const T& Value() {
        std::vector<Example> list;
        list.reserve(this->batchsize);
        if (this->current_pos != this->prev_pos) {
            this->prev_pos = this->current_pos;
            for (const auto& k : this->key_batch[this->current_pos]) {
                const auto& v = this->reader.Value(k);
                list.push_back(v);
                // std::cout << "=== debug ===" << std::endl;
                // std::cout << "key: " << k << std::endl;
                // const auto& sup = v.outputs[0].supervision;
                // const auto& inp = v.inputs[0].features;
                // std::cout << "supervision.label_dim: " << sup.label_dim << std::endl;
                // std::cout << "supervision.num_sequences: " << sup.num_sequences << std::endl;
                // std::cout << "supervision.frames_per_sequence: " << sup.frames_per_sequence << std::endl;
                // std::cout << "inp.NumRows(): "  << inp.NumRows() << std::endl;
                // std::cout << "inp.NumCols(): "  << inp.NumCols() << std::endl;
            }
            kaldi::nnet3::MergeChainExamples(this->compress, &list, &this->minibatch);
        }
        return this->minibatch;
    }


    bool Done() const {
        return this->current_pos >= this->key_batch.size();
    }

    void Next() {
        ++this->current_pos;
    }

    void Close() {
        this->reader.Close();
    }
};

extern "C" {
    // TODO refactor this with template<class Reader>

    void* my_lib_example_rand_reader_new(const char* examples_rspecifier, int seed, int batchsize) {
        auto example_reader = new RandReader(examples_rspecifier, seed, batchsize);
        return static_cast<void*>(example_reader);
    }

    int my_lib_example_rand_reader_next(void* reader_ptr) {
        auto reader = static_cast<RandReader*>(reader_ptr);
        if (reader->Done()) return 0; // fail
        reader->Next();
        return 1; // success
    }

    void my_lib_example_rand_reader_free(void* reader_ptr) {
        auto reader = static_cast<RandReader*>(reader_ptr);
        reader->Close();
        delete reader;
    }

    // NOTE: this function returns size of inputs instead of success/fail
    int my_lib_example_rand_feats(void* reader_ptr, THFloatTensor* input, THFloatTensor* aux) {
        auto reader = static_cast<RandReader*>(reader_ptr);
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

    void* my_lib_supervision_rand_new(void* reader_ptr) {
        auto reader = static_cast<RandReader*>(reader_ptr);
        if (reader->Done()) return nullptr; // fail
        auto&& egs = reader->Value();
        return static_cast<void*>(new Supervision(egs.outputs[0].supervision));
    }

    int my_lib_example_rand_reader_indexes(void* reader_ptr, THLongTensor* index_tensor) {
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
        auto reader = static_cast<RandReader*>(reader_ptr);
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
}
