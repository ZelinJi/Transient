import argparse

def get_args():
    parser = argparse.ArgumentParser(description='LSTM2')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='batch size (default: 400)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 1e-3)')
    args = parser.parse_args()

    # args.cuda = not args.no_cuda and tf.test.is_gpu_available()
    return args