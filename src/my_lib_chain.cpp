#define HAVE_CUDA 1

#include <iostream>
#include <memory>

#include "cu-device.h"

// torch
#include <THC/THC.h>
#include <ATen/ATen.h>

// kaldi
#include <matrix/kaldi-matrix.h>
#include <cudamatrix/cu-matrix.h>
#include <chain/chain-training.h>
#include <nnet3/nnet-chain-example.h>

#include "common.hpp"
#include "./chain-supervision-test.hpp"


void copy_to_mat(kaldi::CuMatrix<float>& src, THCudaTensor* dst) {
    THCudaTensor_resize2d(state, dst, src.NumRows(), src.NumCols());
    auto aten = common::make_tensor(src);
    auto src_tensor = reinterpret_cast<THCudaTensor*>(aten.unsafeGetTH(true));
    THCudaTensor_copy(state, dst, src_tensor);
    // auto mat = common::make_matrix(dst);
    // src.CopyToMat(&mat);
}


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
        // float* objf, float* l2_term, float* weight,
        THFloatTensor* results,
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
        opts.xent_regularize = xent_regularize;
        float* data = THFloatTensor_data(results);
        kaldi::CuMatrix<BaseFloat> xent_deriv;
        auto xent_deriv_ptr = xent_regularize != 0.0 ? &xent_deriv : nullptr;
        kaldi::chain::ComputeChainObjfAndDeriv(opts, den_graph, supervision, nnet_output,
                                               data, data+1, data+2,
                                               &nnet_output_deriv, xent_deriv_ptr);
        if (xent_regularize != 0.0) {
            copy_to_mat(xent_deriv, xent_output_deriv_ptr);
        }
        return 1;
    }

    int my_lib_test_chain(THCudaTensor* out, THCudaTensor* grad) {
        using namespace kaldi;
        using namespace kaldi::chain;
        common::set_kaldi_device(out);

        ContextDependency *ctx_dep;
        TransitionModel *trans_model = GenRandTransitionModel(&ctx_dep);
        const std::vector<int32> &phones = trans_model->GetPhones();

        int32 subsample_factor = RandInt(1, 3);

        int32 phone_sequence_length = RandInt(1, 20);
        std::vector<std::pair<int32, int32> > phones_durations(phone_sequence_length);

        CompactLattice clat;
        int32 cur_state = clat.AddState();
        clat.SetStart(cur_state);

        for (int32 i = 0; i < phone_sequence_length; i++) {
            int32 phone = phones[RandInt(0, phones.size() - 1)];
            int32 min_length = trans_model->GetTopo().MinLength(phone),
                headroom = 5,
                duration = RandInt(subsample_factor * min_length,
                                   subsample_factor * min_length + headroom);
            phones_durations[i].first = phone;
            phones_durations[i].second = duration;
            int32 next_state = clat.AddState();
            std::vector<int32> ones(duration, 1);
            clat.AddArc(cur_state,
                        CompactLatticeArc(phone, phone,
                                          CompactLatticeWeight(LatticeWeight::One(),
                                                               ones), next_state));
            cur_state = next_state;
        }
        clat.SetFinal(cur_state, CompactLatticeWeight::One());
        ProtoSupervision proto_sup1, proto_sup2;
        SupervisionOptions opts;
        opts.frame_subsampling_factor = subsample_factor;
        bool ans1 = AlignmentToProtoSupervision(opts, phones_durations, &proto_sup1),
            ans2 = PhoneLatticeToProtoSupervision(opts, clat, &proto_sup2);
        KALDI_ASSERT(ans1 && ans2);
        KALDI_ASSERT(proto_sup1 == proto_sup2);

        Supervision supervision;
        if (!ProtoSupervisionToSupervision(*ctx_dep, *trans_model,
                                           proto_sup1, &supervision)) {
            // we shouldn't fail because we multiplied by
            // 'subsample_factor' when creating the duration.
            KALDI_ERR << "Failed creating supervision.";
        }
        supervision.Check(*trans_model);
        TestSupervisionIo(supervision);
        TestSupervisionSplitting(*ctx_dep, *trans_model, supervision);
        TestSupervisionAppend(*trans_model, supervision);

        {
            fst::StdVectorFst den_fst;
            ComputeExampleDenFst(*ctx_dep, *trans_model, &den_fst);
            DenominatorGraph den_graph(den_fst, trans_model->NumPdfs());
            ChainDenominatorTest(den_graph);
            if (RandInt(0, 1) == 0)
                supervision.weight = 0.5;
            fst::StdVectorFst normalization_fst;
            den_graph.GetNormalizationFst(den_fst, &normalization_fst);
            // add the weight to the numerator FST so we can assert objf <= 0.
            bool ans = AddWeightToSupervisionFst(normalization_fst, &supervision);
            KALDI_ASSERT(ans);
            // TODO: still have to test for appended sequences.
            _ChainTrainingTest(out, grad, den_graph, supervision);
        }

        delete ctx_dep;
        delete trans_model;

        return 1;
    }
}
