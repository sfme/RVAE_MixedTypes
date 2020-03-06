import argparse

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="AE Model",)

    parser.add_argument('--batch-size', type=int, default=150, metavar="N",
                        help="batch size for training")

    parser.add_argument("--number-epochs", type=int, default=5, metavar="N",
                        help="number of epochs to run for training")

    parser.add_argument("--lr", type=float, default=0.001, metavar="lr",
                        help="initial learning rate for optimizer")

    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument("--cuda-on", action="store_true", default=False,
                        help="Use CUDA (GPU) to run experiment")

    parser.add_argument("--dataset-folder", default="adult_standard", type=str,
                        dest="data_folder", help="Input dataset folder to use in experiments save")

    parser.add_argument("--save-on", action="store_true", default=False,
                        help="True / False on saving experiment data")

    parser.add_argument("--output-folder", type=str, default="./dummy/",
                        help="output folder path where experiment data is saved")

    parser.add_argument('--latent-dim', type=int, default=20, metavar='N',
                        help='dimension of the latent space')

    parser.add_argument('--layer-size', type=int, default=400, metavar='N',
                        help='capacity of the internal layers of the models')

    # -- low lambda means S has a lot of outliers; -- higher lambda means more sparse S
    parser.add_argument('--lambda-param', default=0.15, type=float) # (range of value is sensitive to type of dataset, and noise scenario)

    parser.add_argument('--number-ADMM-iters', default=20, type=int)

    parser.add_argument('--turn-on-validation', action='store_true', default=False)

    parser.add_argument('--eps-bar-X', default=1e-7, type=float)

    parser.add_argument('--eps-bar-diff-iter', default=1e-7, type=float)

    parser.add_argument('--l1-method', action='store_true', default=False)

    parser.add_argument('--l21-method', action='store_true', default=False)

    parser.add_argument('--cat-fout', type=str, default='softmax') # either softmax or sigmoid

    parser.add_argument('--l2-reg', default=0., type=float, # if user wants gradient-based optimizer to apply weight-decay
                        help="recommended values lie between 0.1 and 100. if high corruption, default turned off")

    parser.add_argument('--activation', default='relu', type=str,
                        help="either choose ''relu'' or ''hardtanh'' (computationally cheaper than tanh):")

    return parser.parse_args(argv)

