
import sys
# sys.path.append("..")
# sys.path.append("../..")

import os, errno
import core_models.parser_arguments as parser_arguments
import warnings

import numpy as np
import pandas as pd
import core_models.utils as utils
import operator

from sklearn.metrics import roc_auc_score as auc_compute
from sklearn.metrics import average_precision_score as avpr_compute
from core_models.utils import get_auc_metrics, get_avpr_metrics
from sklearn import mixture

def is_number(val):
    # some outliers will be cast to NaN
    # when reading from file (eg. csv)
    # since they do not conform with data types

    # checks if numerical
    try:
        float(val)

        # check if nan
        if np.isnan(float(val)):
            return False

        return True

    except ValueError:
        return False

def get_prob_matrix(df_dataset, cat_columns, z_mtx=None, dict_densities=None, n_comp_max=6):

    '''
    Computes:
            -> Marginal Histograms (Categorical feature)
                and
            -> GMM (Gaussian Mixture Model) Density with BIC selection (Continous feature)
    '''
    # z_mtx is matrix that indicates whether some cell in dataset is clean (=1) or dirty (=0)
    if z_mtx is None:
        z_mtx = np.ones(df_dataset.shape, dtype=bool)

    # obtain vectorized version of is_nan
    is_number_vec = np.vectorize(is_number, otypes=[np.bool])

    # get indexes for columns
    col_idx_map = dict((name, index) for index, name in enumerate(df_dataset.columns))

    # get continuous features’ names
    cont_columns = [col for col in df_dataset.columns if col not in cat_columns]

    # GMM model selected dictionary (which GMM has been selected in terms of components, for each column)
    gmm_selected = dict.fromkeys(cont_columns)

    # dict of dicts with density/probability values for the domain of each feature
    if dict_densities is None:
        # initialize dictionary for each number of components, for each GMM
        dict_densities = dict()
        for col_name in cont_columns:
            dict_densities[col_name] = dict()
            for n_components in range(1, n_comp_max+1,2):
                dict_densities[col_name][n_components] = \
                mixture.GaussianMixture(n_components=n_components, warm_start=True)

    # density/probability matrix (to be returned for the dataset)
    prob_mat = np.empty(df_dataset.shape)
    repair_mat = np.empty(df_dataset.shape)

    # calculate histogram values for categorical features
    for col_name in cat_columns:

        # build density for discrete variable
        dict_densities[col_name] = df_dataset[col_name][z_mtx[:,col_idx_map[col_name]]].value_counts(normalize=True).to_dict()

        # insert normalized histogram for feature
        lf = lambda cell: dict_densities[col_name][cell] if cell in dict_densities[col_name] else 0.0
        prob_mat[:, col_idx_map[col_name]] = np.array([*map(lf, df_dataset[col_name].values)])


    # calculate GMM values for continuous features
    for col_name in cont_columns:

        # the feature data
        col_data = df_dataset[col_name].values.reshape(-1,1)

        col_data = (col_data - np.mean(col_data))/np.std(col_data)

        # select indexes of number cells (not nan)
        idx_bool = is_number_vec(col_data).flatten()

        # select clean data as defined by the z variables, and that is number (not nan)
        aux_idxs =  np.logical_and(z_mtx[:,col_idx_map[col_name]], idx_bool)
        col_data_clean = col_data[aux_idxs]

        # select best number of components for GMM
        best_bic = np.inf
        best_GMM = None
        for n_components in range(1, n_comp_max+1,2):
            gmm_mdl = dict_densities[col_name][n_components]
            gmm_mdl.fit(col_data_clean)
            bic_val = gmm_mdl.bic(col_data_clean)
            if best_bic > bic_val:
                best_bic = bic_val
                best_GMM = gmm_mdl

        # for output
        gmm_selected[col_name] = best_GMM

        # obtain density values for feature’s cells, using best current GMM model
        prob_mat[:, col_idx_map[col_name]][idx_bool] = np.exp(best_GMM.score_samples(col_data[idx_bool]))
        prob_mat[:, col_idx_map[col_name]][np.logical_not(idx_bool)] = 0.0

        #Select closest mean for each value
        index = np.argmin((col_data_clean-best_GMM.means_.T)**2,1)
        repair_mat[:, col_idx_map[col_name]][idx_bool] = best_GMM.means_[index].squeeze()
        repair_mat[:, col_idx_map[col_name]][np.logical_not(idx_bool)] = np.mean(best_GMM.means_)

    return prob_mat, dict_densities, gmm_selected, repair_mat

def error_computation(dataset_obj, X_true, X_hat, dict_densities, mask):

    cursor_feat = 0
    feature_errors_arr = []
    for feat_select, (feat_name, col_type, feat_size) in enumerate(dataset_obj.feat_info):

        select_cell_pos = np.argwhere(mask[:,cursor_feat]==1)

        if select_cell_pos.sum() == 0:
            feature_errors_arr.append(-1.)
        else:
            # Brier Score (score ranges between 0-1)
            if col_type == 'categ':
                true_feature_one_hot = np.zeros((X_true.shape[0], feat_size))
                true_index = [int(elem) for elem in X_true[:,cursor_feat]]
                true_feature_one_hot[np.arange(X_true.shape[0]), true_index] = 1.

                mean_est_probs = [dict_densities[feat_name][key] for key in dataset_obj.cat_to_idx[feat_name].keys()]

                error_brier = np.sum((mean_est_probs - true_feature_one_hot[select_cell_pos].squeeze())**2) / (2*float(len(select_cell_pos)))

                feature_errors_arr.append(error_brier)

            # Standardized Mean Square Error (SMSE)
            # SMSE (score ranges betweem 0,1)
            elif col_type == 'real':

                true_feature = X_true[:,cursor_feat]
                reconstructed_feature = X_hat[:,cursor_feat]

                smse_error = np.sum((true_feature[select_cell_pos] - reconstructed_feature[select_cell_pos])**2)
                sse_div = np.sum(true_feature[select_cell_pos]**2) # (y_ti - avg(y_i))**2, avg(y_i)=0. due to data standardization
                smse_error = smse_error / sse_div # SMSE does not need to div 1/N_mask, since it cancels out.

                feature_errors_arr.append(smse_error.item())

        cursor_feat +=1

    # Global error (adding all the errors and dividing by number of features)
    mean_error = np.array([val for val in feature_errors_arr if val >= 0.]).mean() # / torch.sum(mask.type(dtype_float))

    return mean_error, feature_errors_arr


def main(args):

    # Load datasets
    train_loader, X_train, target_errors_train, dataset_obj_train, attributes = utils.load_data(args.data_folder, args.batch_size, 
                                                                                                is_train=True)
    train_loader_clean, X_train_clean, _, dataset_obj_clean, _ = utils.load_data(args.data_folder, args.batch_size,
                                                                        is_train=True, is_clean=True, stdize_dirty=True)

    dataset_obj = dataset_obj_train
    df_data_train = dataset_obj_train.df_dataset_instance

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_mat_train, dict_densities, _, repair_mat = get_prob_matrix(df_data_train, dataset_obj.cat_cols, n_comp_max=40)


    mean_error_dirty, features_errors_dirty = error_computation(dataset_obj_clean, X_train_clean.detach().numpy(),
                                                    repair_mat, dict_densities, target_errors_train.detach().numpy())
    mean_error_clean, features_errors_clean = error_computation(dataset_obj_clean, X_train_clean.detach().numpy(),
                                                    repair_mat, dict_densities, (1-target_errors_train).detach().numpy())

    #print(features_errors)
    logp_mat_train = np.log(p_mat_train + 1e-9)

    target_row_train = (target_errors_train.sum(dim=1)>0).numpy()

    # Uses the NLL score as outlier score (just like VAE outlier score)
    outlier_score_cell_train = -logp_mat_train
    outlier_score_row_train = -logp_mat_train.sum(axis=1)


    ## Cell metrics
    auc_cell_train, auc_feats = get_auc_metrics(target_errors_train, outlier_score_cell_train)
    avpr_cell_train, avpr_feats = get_avpr_metrics(target_errors_train, outlier_score_cell_train)

    print("AVPR per feature")
    print(avpr_feats)
    print("AUC per feature")
    print(auc_feats)

    ## Row metrics
    auc_row_train = auc_compute(target_row_train, outlier_score_row_train)
    avpr_row_train = avpr_compute(target_row_train, outlier_score_row_train)


    print('Marginals Prob. Train - Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {}'.format(
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

        columns = ['AUC row','AVPR row','AUC cell','AVPR cell','Error repair on dirty pos', 'Error repair on clean pos']
        results = {'AUC row': [auc_row_train], 'AVPR row': [avpr_row_train],
                   'AUC cell': [auc_cell_train], 'AVPR cell': [avpr_cell_train],
                   'Error repair on dirty pos': [mean_error_dirty], 'Error repair on clean pos': [mean_error_clean]}


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

        df_errors_repair = pd.DataFrame([], index=['error_repair_dirtycells','error_repair_cleancells'], columns=attributes)
        df_errors_repair.loc['error_repair_dirtycells'] = features_errors_dirty
        df_errors_repair.loc['error_repair_cleancells'] = features_errors_clean
        df_errors_repair.to_csv(folder_output + "/train_error_repair_features.csv")


if __name__ == '__main__':

    args = parser_arguments.getArgs(sys.argv[1:])

    main(args)
