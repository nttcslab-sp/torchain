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
#include "./chain-supervision-test.hpp"


void _ChainTrainingTest(
    THCudaTensor* out, THCudaTensor* grad,
    const kaldi::chain::DenominatorGraph &den_graph,
    const kaldi::chain::Supervision &supervision)
{
    using namespace kaldi;
    using namespace kaldi::chain;
    int32 num_sequences = supervision.num_sequences,
        frames_per_sequence = supervision.frames_per_sequence;
    if (frames_per_sequence == 1)  // this will break some code.
        return;

    THCudaTensor_resize2d(state, out, num_sequences * frames_per_sequence, den_graph.NumPdfs());
    auto nnet_output = common::make_matrix(out);
    //CuMatrix<BaseFloat> nnet_output(num_sequences * frames_per_sequence,
    //                                den_graph.NumPdfs());

    bool zero_output = (RandInt(0, 3) == 0);
    if (!zero_output)
        nnet_output.SetRandn();

    ChainTrainingOptions opts;
    if (RandInt(0, 1) == 1)
        opts.leaky_hmm_coefficient = 0.2;

    // CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
    //                                      nnet_output.NumCols(),
    //                                      kUndefined);
    THCudaTensor_resizeAs(state, grad, out);
    auto nnet_output_deriv = common::make_matrix(grad);


    BaseFloat objf, l2_term, weight;

    ComputeChainObjfAndDeriv(opts, den_graph, supervision,
                             nnet_output, &objf, &l2_term, &weight,
                             &nnet_output_deriv);

    {
        // make sure each row of nnet_output_deriv sums to one (shift invariance of
        // the nnet output).
        CuVector<BaseFloat> nnet_output_deriv_row_sums(nnet_output_deriv.NumRows());
        nnet_output_deriv_row_sums.AddColSumMat(1.0, nnet_output_deriv, 0.0);
        KALDI_ASSERT(nnet_output_deriv_row_sums.Norm(2.0) < 0.1);
    }

    KALDI_LOG << "Chain objf per frame is " << (objf / weight)
              << " over " << weight << " frames (weighted)";

    { // a check
        BaseFloat output_deriv_sum = nnet_output_deriv.Sum();
        KALDI_LOG << "Sum of nnet-output-deriv is " << output_deriv_sum
                  << " vs. expected 0.";
        KALDI_ASSERT(output_deriv_sum < 0.2);
    }

    KALDI_ASSERT(objf <= 0.0);

    int32 num_tries = 5;
    BaseFloat epsilon = 1.0e-04;
    Vector<BaseFloat> predicted_objf_changes(num_tries),
        observed_objf_changes(num_tries);
    for (int32 p = 0; p < num_tries; p++) {
        CuMatrix<BaseFloat> nnet_delta_output(nnet_output.NumRows(),
                                              nnet_output.NumCols());
        nnet_delta_output.SetRandn();
        nnet_delta_output.Scale(epsilon);
        predicted_objf_changes(p) = TraceMatMat(nnet_output_deriv,
                                                nnet_delta_output, kTrans);
        CuMatrix<BaseFloat> nnet_output_perturbed(nnet_delta_output);
        nnet_output_perturbed.AddMat(1.0, nnet_output);

        BaseFloat objf_modified, l2_term_modified, weight_modified;

        ComputeChainObjfAndDeriv(opts, den_graph, supervision,
                                 nnet_output_perturbed,
                                 &objf_modified, &l2_term_modified,
                                 &weight_modified,
                                 NULL);

        observed_objf_changes(p) = objf_modified - objf;
    }
    KALDI_LOG << "Predicted objf changes are " << predicted_objf_changes;
    KALDI_LOG << "Observed objf changes are " << observed_objf_changes;
    {
        Vector<BaseFloat> error(predicted_objf_changes);
        error.AddVec(-1.0, observed_objf_changes);
        KALDI_LOG << "num-sequences = " << num_sequences << ", frames-per-sequence = "
                  << frames_per_sequence << ", relative accuracy is "
                  << (error.Norm(2.0) / predicted_objf_changes.Norm(2.0));
    }

    {
        // we get inaccuracy for long segments, I think because there is a bias when we
        // add random noise for it to increase the likelihood (for winner-take-all reasons)
        // and for long utterances this bias adds up over the frames and tends to
        // outweigh the random component that the gradient predicts (which will tend to
        // cancel).  Try to correct for this...
        BaseFloat correction = (predicted_objf_changes.Sum() - observed_objf_changes.Sum()) /
            predicted_objf_changes.Dim();
        observed_objf_changes.Add(correction);
        KALDI_LOG << "Correcting observed objf changes for statistical effects, to "
                  << observed_objf_changes;
        if (frames_per_sequence > 2 &&
            predicted_objf_changes.Norm(2.0) > 0.1 * epsilon) {
            // if we only have the initial and final frames, due to the scaling-down
            // of pdfs not in the numerator sequence the derivative might be zero,
            // which would cause problems doing the comparison.
            // note, epsilon = 1.0e-04.
            KALDI_ASSERT(predicted_objf_changes.ApproxEqual(observed_objf_changes, 0.25));
        }
    }
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
