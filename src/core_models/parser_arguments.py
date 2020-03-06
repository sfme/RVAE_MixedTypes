#!/usr/bin/env python3

import argparse

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="VAE Models",)

    parser.add_argument('--batch-size', type=int, default=150, metavar="N",
                        help="batch size for training")

    parser.add_argument("--number-epochs", type=int, default=5, metavar="N",
                        help="number of epochs to run for training")

    parser.add_argument("--lr", type=float, default=1e-3, metavar="lr",
                        help="initial learning rate for optimizer")

    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument("--cuda-on", action="store_true", default=False,
                        help="Use CUDA (GPU) to run experiment")

    parser.add_argument("--dataset-folder", default="adult_standard", type=str,
                        dest="data_folder", help="Input dataset folder to use in experiments")

    parser.add_argument("--save-on", action="store_true", default=False,
                        help="True / False on saving experiment data")

    parser.add_argument("--is-one-hot", action="store_true", default=False,
                        help="True when using one hot encoding") 

    parser.add_argument("--outlier-model", type=str, default="VAE",
                        help="name of the file with the outlier model")

    parser.add_argument("--output-folder", type=str, default="./dummy/",
                        help="output folder path where experiment data is saved")

    parser.add_argument("--alpha-prior", type=float, default=0.95,
                        help="prior value that defines assumption on cleanliness of cells")

    parser.add_argument('--embedding-size', type=int, default=50, metavar='N',
                        help='size of the embeddings for the categorical attributes')

    parser.add_argument('--latent-dim', type=int, default=20, metavar='N',
                        help='dimension of the latent space')

    parser.add_argument('--layer-size', type=int, default=400, metavar='N',
                        help='capacity of the internal layers of the models')

    parser.add_argument("--verbose-metrics-epoch", action="store_true", default=False,
                        help="show / no show the metrics for each epoch")

    parser.add_argument("--verbose-metrics-feature-epoch", action="store_true", default=False,
                        help="show / no show the metrics for each epoch -- feature stuff")

    parser.add_argument("--inference-type", type=str, default="vae",
                        help="choose either: 'vae' or 'seq_vae' (pseudo-Gibbs Sampling: MCMC / refeeding option)")

    parser.add_argument('--AVI', action="store_true", default=False,
                        help='whether an encoder for pi is used or not')

    parser.add_argument('--std-gauss-nll', default=2.0, type=float) # 2.0

    parser.add_argument('--l2-reg', default=0., type=float,
                        help="e.g. values lie between 0.1 and 100. if high corruption, default turned off")

    parser.add_argument('--activation', default='relu', type=str,
                        help="either choose ''relu'' or ''hardtanh'' (computationally cheaper than tanh):")

    ## load model flags
    parser.add_argument('--load-model', action='store_true', default=False)
    parser.add_argument('--load-model-path', type=str)

    ## seq_vae specific flags (when option --inference-type seqvae is chosen, the MCMC option)
    #  these options are used in pseudo-Gibbs Sampling, MCMC / refeeding option
    parser.add_argument('--seqvae-bprop', action='store_true', default=False) # bprop the sequence during training
    parser.add_argument('--seqvae-two-stage', action='store_true', default=False) # whether estimate mask first, and then repair (like Rezende et al. 2014)
    parser.add_argument('--seqvae-steps', default=4, type=int) # number of steps at first stage
    parser.add_argument('--steps-2stage', default=4, type=int) # number of steps of Rezende et. al at second stage

    return parser.parse_args(argv)


