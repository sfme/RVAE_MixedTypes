#!/usr/bin/env python3

import sys
# sys.path.append("..")
# sys.path.append("../..")

import os, errno
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import roc_auc_score as auc_compute
from sklearn.metrics import average_precision_score as avpr_compute
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from core_models.Marginals.main_Marginals import get_prob_matrix
import core_models.parser_arguments as parser_arguments
from core_models.utils import get_auc_metrics, get_avpr_metrics
import core_models.utils as utils

import warnings

def main(args):

    # Load datasets
    train_loader, X_train, target_errors_train, dataset_obj_train, attributes = utils.load_data(args.data_folder, args.batch_size,
                                                                                                is_train=True, is_one_hot=args.is_one_hot)
    test_loader, X_test, target_errors_test, _, _ = utils.load_data(args.data_folder, args.batch_size, is_train=False)

    df_data_train = dataset_obj_train.df_dataset_instance

    # Run Marginals to obtain cell log probs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_mat_train, _, _, _ = get_prob_matrix(df_data_train, dataset_obj_train.cat_cols, n_comp_max=40)
    nll_marginal_cell = -np.log(p_mat_train + 1e-8)


    target_errors_row_train = (target_errors_train.sum(dim=1)>0)
    target_row_train = target_errors_row_train.numpy()


    target_errors_row_test = (target_errors_test.sum(dim=1)>0)
    target_row_test = target_errors_row_test.numpy()

    # Run OCSVM row outlier detection
    clf = svm.OneClassSVM(nu=0.2, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    outlier_score_row_train = -clf.score_samples(X_train)
    outlier_score_row_test = -clf.score_samples(X_test)

    # Platt Scaling (uses Logistic Regression) of OCSVM scores
    lr_calib = LogisticRegression(solver='lbfgs')
    lr_calib.fit(outlier_score_row_test.reshape(-1,1), target_row_test)
    p_inlier_train = lr_calib.predict_proba(outlier_score_row_train.reshape(-1,1))[:,0]
    nll_inlier_row_train = -np.log(p_inlier_train + 1e-8) # -log (p_inlier)


    # Row metrics
    auc_row_train = auc_compute(target_row_train, outlier_score_row_train)
    avpr_row_train = avpr_compute(target_row_train, outlier_score_row_train)
    ll_row_train = log_loss(target_row_train, outlier_score_row_train)

    auc_row_train_calibed = auc_compute(target_row_train, nll_inlier_row_train)
    avpr_row_train_calibed = avpr_compute(target_row_train, nll_inlier_row_train)
    ll_row_train_calibed = log_loss(target_row_train, 1.-p_inlier_train)


    print("AUC Prev. Calib.: {}".format(auc_row_train))
    print("AVPR Prev. Calib.: {}".format(avpr_row_train))
    print("Cross-Entropy Prev. Calib. {}".format(ll_row_train))

    # Re-check score is still good after calibration (AVPR and AUC should be same);
    # then Cross-Entropy should drop !!
    print("AUC Post. Calib.: {}".format(auc_row_train_calibed))
    print("AVPR Post. Calib.: {}".format(avpr_row_train_calibed))
    print("Cross-Entropy Post. Calib. {}".format(ll_row_train_calibed))

    # combine calibrated OCSVM and Marginals for cell outlier detection
    nll_cells_final_train = nll_inlier_row_train.reshape(-1,1) + nll_marginal_cell

    # Cell metrics
    auc_cell_train, auc_feats = get_auc_metrics(target_errors_train, nll_cells_final_train)
    avpr_cell_train, avpr_feats = get_avpr_metrics(target_errors_train, nll_cells_final_train)

    print('Combined: OCSVM + Marginals Train -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {}'.format(
                        auc_cell_train, avpr_cell_train, auc_row_train, avpr_row_train))


    #Save results into csv
    if args.save_on:

        # create folder for saving experiment data (if necessary)
        folder_output = args.output_folder + "/" + args.outlier_model

        try:
            os.makedirs(folder_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        columns = ['AUC row','AVPR row','AUC cell','AVPR cell']
        results = {'AUC row': [auc_row_train], 'AVPR row': [avpr_row_train],
                   'AUC cell': [auc_cell_train], 'AVPR cell': [avpr_cell_train]}

        #Dataframe
        df_out = pd.DataFrame(data=results, columns=columns)
        df_out.index.name = "Epochs"
        df_out.to_csv(folder_output + "/train_epochs_data.csv")

        # store AVPR for features (cell only)
        df_avpr_feat_cell = pd.DataFrame([], index=['AVPR'], columns=attributes)
        df_avpr_feat_cell.loc['AVPR'] = avpr_feats
        df_avpr_feat_cell.to_csv(folder_output + "/train_avpr_features.csv")

        # store AUC for features (cell only)
        df_auc_feat_cell = pd.DataFrame([], index=['AUC'], columns=attributes)
        df_auc_feat_cell.loc['AUC'] = auc_feats
        df_auc_feat_cell.to_csv(folder_output + "/train_auc_features.csv")

if __name__ == '__main__':

    args = parser_arguments.getArgs(sys.argv[1:])

    main(args)