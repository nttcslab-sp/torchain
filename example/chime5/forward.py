from pathlib import Path

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--example_rs', required=True,
                        help='kaldi s5/exp dir that must be finished before')
    parser.add_argument('--model_dir', required=True,
                        help='dir to store pytorch params and pickle')
    parser.add_argument('--forward_dir', required=True,
                        help='dir to store logprob ark/scp')


def main(args):
    io.set_kaldi_device()
    example = io.Example(args.example_rs)
    model_dir = Path(args.model_dir)
    forward_dir = model_dir / "forward"
    model = torch.load(model_dir / "model.pickle")
    count = 0
    for (mfcc, ivec), _ in io.Example(example_rs):
        x = Variable(mfcc)
        pred = model(x)
        count += 1
        if count > 20:
            break

parser = get_parser()
args = parser.parse_args()
main(args)
