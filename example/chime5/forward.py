from pathlib import Path

import torch
import numpy

import kaldi_io


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # IO configuration
    parser.add_argument('--input_rs', required=True,
                        help='input feat (e.g., MFCC, FBANK) rspecifier in decoding')
    parser.add_argument('--aux_rs', required=False,
                        help='aux feat (e.g., i-vector) rspecifier in decoding')
    parser.add_argument('--model_dir', required=True,
                        help='dir storing pytorch model.pickle')
    parser.add_argument('--forward_ark', required=True,
                        help='dir to store logprob ark/scp')
    return parser


def main():
    assert args.aux_rs is not None, "not supported"
    model_dir = Path(args.model_dir)
    forward_dir = model_dir / "forward"
    model = torch.load(model_dir / "model.pickle", map_location="cpu")
    with torch.no_grad(), open(args.forward_ark, "wb") as f:
        for key, feat in kaldi_io.read_mat_ark(args.input_rs):
            print(key, feat.shape)
            # feat is (time, freq) shape
            x = torch.from_numpy(feat.T).unsqueeze(0)
            y = model(x)
            kaldi_io.write_mat(f, y.numpy().T, key)

args = get_parser().parse_args()
main()
