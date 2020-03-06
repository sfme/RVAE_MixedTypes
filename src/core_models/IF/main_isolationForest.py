#!/usr/bin/env python3

import sys
# sys.path.append("..")
# sys.path.append("../..")

import os, errno
import core_models.parser_arguments as parser_arguments

import pandas as pd
import core_models.utils as utils
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score as auc_compute
from sklearn.metrics import average_precision_score as avpr_compute


def main(args):

    # Load datasets
    _, X_train, target_errors_train, _, _ = utils.load_data(args.data_folder, args.batch_size, is_train=True)
    #_, X_test, target_errors_test, _, _ = utils.load_data(args.data_folder, args.batch_size, is_train=False) # NOTE: used in hyper-parameter selection

    # Best parameters from CV
    clf = IsolationForest(n_estimators=100, max_samples=0.5, contamination=0.2, behaviour='new')
    clf.fit(X_train)

    target_row = (target_errors_train.sum(dim=1)>0).numpy()
    outlier_score_row = -clf.score_samples(X_train)

    auc_row = auc_compute(target_row, outlier_score_row)
    avpr_row = avpr_compute(target_row, outlier_score_row)

    print('Isolation Forest Train - AUC: ' + str(auc_row) + ', AVPR: ' + str(avpr_row))

    #Save results into csv
    if args.save_on:

        # create folder for saving experiment data (if necessary)
        folder_output = args.output_folder + "/" + args.outlier_model

        try:
            os.makedirs(folder_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        columns = ['AUC','AVPR']
        results = {'AUC': [auc_row], 'AVPR': [avpr_row]}

        #Dataframe
        df_out = pd.DataFrame(data=results, columns=columns)
        df_out.index.name = "Epochs"
        df_out.to_csv(folder_output + "/train_epochs_data.csv")

if __name__ == '__main__':

    args = parser_arguments.getArgs(sys.argv[1:])

    main(args)