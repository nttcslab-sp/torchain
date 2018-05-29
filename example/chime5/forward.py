import os
import logging
from pathlib import Path

import torch
import numpy

import kaldiio
import kaldi_io

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # IO configuration
    parser.add_argument('--input_rs', required=True,
                        help='input feat (e.g., MFCC, FBANK) rspecifier in decoding')
    parser.add_argument('--aux_scp', required=False,
                        help='aux feat (e.g., i-vector) scp in decoding')
    parser.add_argument('--model_dir', required=True,
                        help='dir storing pytorch model.pickle')
    parser.add_argument('--forward_ark', required=True,
                        help='dir to store logprob ark/scp')
    # FIXME do not provide n_time_width=29 manually
    parser.add_argument('--min_time_width', type=int, default=33,
                        help='minumum time width of input. input will be padded if time width is smaller')
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logging.info("HOST=" + os.environ.get("HOST", ""))
    logging.info("SLURM_JOB_ID=" + os.environ.get("SLURM_JOB_ID", ""))

    model_dir = Path(args.model_dir)
    forward_dir = model_dir / "forward"
    aux_scp = kaldiio.load_scp(args.aux_scp)
    model = torch.load(model_dir / "model.pickle", map_location="cpu")
    model.eval()
    with torch.no_grad(), open(args.forward_ark, "wb") as f:
        for key, feat in kaldi_io.read_mat_ark(args.input_rs):
            aux = torch.from_numpy(aux_scp[key])
            logging.info("input: key={} feat={} aux={}".format(key, feat.shape, aux.shape))
            # feat is (time, freq) shape
            x = torch.from_numpy(feat.T).unsqueeze(0)
            if x.shape[2] < args.min_time_width:
                remain = args.min_time_width - x.shape[2] + 1
                lpad = torch.zeros(1, x.shape[1], remain / 2)
                rpad = torch.zeros(1, x.shape[1], remain / 2)
                x = torch.cat((lpad, x, rpad), dim=2)

            n_aux = aux.shape[0]
            # take center ivector frame
            aux = aux[n_aux//2].unsqueeze(0)
            # forward
            y, _ = model(x, aux)
            y = torch.nn.functional.log_softmax(y, dim=1).squeeze(0)
            logging.info("output: {}".format(y.shape))
            kaldi_io.write_mat(f, y.numpy().T, key)

args = get_parser().parse_args()
main()
