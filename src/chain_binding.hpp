#pragma once
#define HAVE_CUDA 1

#include <iostream>
#include <memory>

// torch
#include <THC/THC.h>
#include <ATen/ATen.h>

// kaldi
#include <matrix/kaldi-matrix.h>
#include <cudamatrix/cu-matrix.h>
#include <chain/chain-training.h>

#include "common.hpp"


extern "C" {
    /**
       This function computes the loss and grads of LF-MMI objective

       Inputs

       @param [in] nnet_output
           The output of the neural net; dimension must equal
           ((supervision.num_sequences * supervision.frames_per_sequence) by
           den_graph.NumPdfs()).  The rows are ordered as: all sequences
           for frame 0; all sequences for frame 1; etc.
       @param [in] den_graph
           The denominator graph, derived from denominator fst.
       @param [in] supervision
           The supervision object, containing the supervision
           paths and constraints on the alignment as an FST


       Outputs:

       @param [out] objf
           The [num - den] objective function computed for this
           example; you'll want to divide it by 'tot_weight' before
           displaying it.
       @param [out] l2_term
           The l2 regularization term in the objective function, if
           the --l2-regularize option is used.  To be added to 'o
       @param [out] weight
           The weight to normalize the objective function by;
           equals supervision.weight * supervision.num_sequences *
           supervision.frames_per_sequence.


       Gradients:

       @param [out] nnet_output_deriv
           The derivative of the objective function w.r.t.
           the neural-net output.  Only written to if non-NULL.
           You don't have to zero this before passing to this function,
           we zero it internally.
       @param [out] xent_output_deriv [NOT SUPPPORTED]
           If non-NULL, then the numerator part of the derivative
           (which equals a posterior from the numerator
           forward-backward, scaled by the supervision weight)
           is written to here (this function will set it to the
           correct size first; doing it this way reduces the
           peak memory use).  xent_output_deriv will be used in
           the cross-entropy regularization code; it is also
           used in computing the cross-entropy objective value.


       Hyper parameters from ChainTrainingOptions:

       @param [in] l2_regularize
           l2 regularization constant for 'chain' training,
           applied to the output of the neural net.
       @param [in] leaky_hmm_coefficient,
           Coefficient that allows transitions from each HMM state to each other 
           HMM state, to ensure gradual forgetting of context (can improve generalization).
           For numerical reasons, may not be exactly zero.
       @param [in] xent_regularize,
           Cross-entropy regularization constant for 'chain' training.
           If nonzero, the network is expected to have an output 
           named 'output-xent', which should have a softmax as its final nonlinearity.

     */
    int my_lib_ComputeChainObjfAndDeriv(
        // inputs
        void* den_graph_ptr, void* supervision_ptr, THCudaTensor* nnet_output_ptr,
        // outputs
        float* objf, float* l2_term, float* weight,
        // grads
        THCudaTensor* nnet_output_deriv_ptr, THCudaTensor* xent_output_deriv_ptr,
        // hyper params
        float l2_regularize, float leaky_hmm_coefficient, float xent_regularize)
    {
        common::set_kaldi_device(nnet_output_ptr);
        auto nnet_output = common::make_matrix(nnet_output_ptr);
        auto nnet_output_deriv = common::make_matrix(nnet_output_deriv_ptr);

        const auto& den_graph = *static_cast<kaldi::chain::DenominatorGraph*>(den_graph_ptr);
        const auto& supervision = *static_cast<kaldi::chain::Supervision*>(supervision_ptr);

        kaldi::chain::ChainTrainingOptions opts;
        opts.l2_regularize = l2_regularize;
        opts.leaky_hmm_coefficient = leaky_hmm_coefficient;
        opts.xent_regularize = 0.0f; // xent_regularize;
        kaldi::chain::ComputeChainObjfAndDeriv(opts, den_graph, supervision, nnet_output,
                                               objf, l2_term, weight, &nnet_output_deriv, nullptr);
        return 1;
    }
}
