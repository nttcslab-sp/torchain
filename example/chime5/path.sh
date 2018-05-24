export KALDI_ROOT=/data/work70/skarita/exp/chime5/kaldi-22fbdd
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$KALDI_ROOT/egs/wsj/s5/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$LD_LIBRARY_PATH
# export LC_ALL=C

