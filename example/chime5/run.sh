#!/usr/bin/env bash

set -e
set -u
set -o pipefail
set -x

KALDI_ROOT=/data/work70/skarita/exp/chime5/kaldi-22fbdd/
exp_dir=$KALDI_ROOT/egs/chime5/s5/exp
# TODO make this rspec work
example_rs="ark,bg:nnet3-chain-copy-egs --frame-shift=1  ark:${exp_dir}/chain_train_worn_u100k_cleaned/tdnn1a_sp/egs/cegs.1.ark ark:- | nnet3-chain-shuffle-egs --buffer-size=5000 --srand=0 ark:- ark:- | nnet3-chain-merge-egs --minibatch-size=128,64,32 ark:- ark:- |"
example_rs="ark:/home/skarita/work/repos/extension-ffi-kaldi/package/mb.ark"
denominator_fst_rs="${exp_dir}/chain_train_worn_u100k_cleaned/tdnn1a_sp/den.fst"
recog_set="dev_worn"
model_dir="./model"
stage=0

. ./parse_options.sh || exit 1;
. ./path.sh
. ./cmd.sh


if [ -d $exp_dir/tree_sp ]; then
    echo "$exp_dir is not finished"
    exit 1;
fi

mkdir -p $model_dir

if [ $stage -le 1 ]; then
    echo "=== stage 1: acoustic model training ==="
    python train.py \
           --exp_dir $exp_dir \
           --model_dir $model_dir
fi

if [ $stage -le 2 ]; then
    echo "=== stage 2: calc acoustic log-likelihood ==="
    for label in $recog_set; do
        forward_dir=$model_dir/forward/$label
        mkdir -p $forward_dir
        ${train_cmd} python forward.py \
               --example_rs $example_rs \
               --model_dir $model_dir \
               --forward_dir $forward_dir
    done
fi

if [ $stage -le 3 ]; then
    echo "=== stage 3: decoding ==="
    # see also
    # - nnet1 https://github.com/kaldi-asr/kaldi/blob/72d89cedd064f879d08aef2d048cde8cf1dc687f/egs/wsj/s5/steps/nnet/decode.sh#L156
    # - chime5 https://github.com/kaldi-asr/kaldi/blob/72d89cedd064f879d08aef2d048cde8cf1dc687f/egs/chime5/s5/local/chain/tuning/run_tdnn_1a.sh#L236
    for label in $recog_set; do
        forward_dir=$model_dir/forward/$label
        # ${decode_cmd} latgen-faster-mapped
    done
fi
