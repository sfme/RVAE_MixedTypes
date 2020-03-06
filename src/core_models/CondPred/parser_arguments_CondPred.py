
import argparse

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="NN Conditional Prediction Model",)

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

    parser.add_argument('--l2-reg', default=0., type=float,
                        help="recommended values lie between 0.1 and 100. if high corruption, default turned off")

    parser.add_argument('--activation', default='relu', type=str,
                        help="either choose ''relu'' or ''hardtanh'' (computationally cheaper than tanh):")

    parser.add_argument('--base-type', default='deep', type=str,
                        help='choose if using linear or deep as base type for conditional prediction')

    parser.add_argument('--embedding-size', type=int, default=50, metavar='N',
                        help='size of the embeddings for the categorical attributes')

    parser.add_argument("--nest-mom", action="store_true", default=False,
                        help="for linear base predictor, use SGD wit momentum?")

    parser.add_argument("--mom-val", type=float, default=0.9,
                        help="momentum value to use in SGD (linear predictor option)")

    return parser.parse_args(argv)

