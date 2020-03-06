#!/usr/bin/env python3

import torch
import torch.utils.data
import numpy as np
import pandas as pd
from core_models.mixedDataset import mixedDatasetInstance
from core_models.imageDataset import imageDatasetInstance
from core_models.model_utils import nll_categ_global, nll_gauss_global

from sklearn.metrics import roc_auc_score as auc_compute
from sklearn.metrics import average_precision_score as avpr_compute
from sklearn.metrics import auc
from collections import OrderedDict
import json

from sklearn.metrics.cluster import contingency_matrix



def load_data(folder_path, batch_size, is_train, is_clean=False,
              get_data_idxs=False, is_one_hot=False,
              stdize_dirty=False):

    # Get column / feature information
    with open(folder_path + 'cols_info.json') as infile:
        data_load = json.load(infile)

    num_feat_names = data_load['num_cols_names']
    cat_feat_names = data_load['cat_cols_names']
    dataset_type = data_load['dataset_type']

    type_load = "clean" if is_clean else "noised"

    get_indexes_flag = True if get_data_idxs else False

    if is_train:
        folder_path_set = folder_path + "train"
    else:
        get_indexes_flag = False
        folder_path_set = folder_path + "validation"

    #Get data folders
    csv_file_path_all = folder_path + "full/data_{}.csv".format(type_load)
    csv_file_path_instance = folder_path_set + "/data_{}.csv".format(type_load)
    csv_file_cell_outlier_mtx = folder_path_set + "/cells_changed_mtx.csv"

    if stdize_dirty and type_load == 'clean':
        dirty_path = folder_path + "full/data_{}.csv".format('noised')
    else:
        dirty_path = None

    # Get train and test data
    if dataset_type == 'image':

        cont_flag = True if num_feat_names else False # Binarized images

        dataset = imageDatasetInstance(csv_file_path_all, csv_file_path_instance,
                                       csv_file_cell_outlier_mtx, get_indexes=get_indexes_flag,
                                       cont_flag=cont_flag, standardize_dirty=stdize_dirty,
                                       dirty_csv_file_path=dirty_path)
    else:

        dataset = mixedDatasetInstance(csv_file_path_all,
                                       csv_file_path_instance,
                                       num_feat_names, cat_feat_names,
                                       csv_file_cell_outlier_mtx, get_indexes=get_indexes_flag,
                                       use_one_hot=is_one_hot, standardize_dirty=stdize_dirty,
                                       dirty_csv_file_path=dirty_path)

    # dataloaders for back-prop
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True)


    # get outputs from the dataset columns names in order
    X = torch.Tensor(dataset.df_dataset_instance_standardized.values).type(torch.FloatTensor)
    target_errors = torch.Tensor(dataset.cell_outlier_mtx).type(torch.ByteTensor)
    attributes = dataset.df_dataset_instance.columns


    return data_loader, X, target_errors, dataset, attributes


def get_auc_metrics(target_matrix, score_matrix):

    auc_feats = np.zeros(target_matrix.shape[1])

    for ii in range(target_matrix.shape[1]):
        if target_matrix[:,ii].any():
            auc_feats[ii] = auc_compute(target_matrix[:,ii], score_matrix[:,ii])
        else:
            auc_feats[ii] = -10.

    # macro average of auc through feature set
    macro_auc = auc_feats[auc_feats>=0].mean()

    return macro_auc, auc_feats

def get_avpr_metrics(target_matrix, score_matrix):

    avpr_feats = np.zeros(target_matrix.shape[1])

    for ii in range(target_matrix.shape[1]):
        if target_matrix[:,ii].any():
            avpr_feats[ii] = avpr_compute(target_matrix[:,ii], score_matrix[:,ii])
        else:
            avpr_feats[ii] = -10.

    # macro average of avpr through feature set
    macro_avpr = avpr_feats[avpr_feats>=0].mean()

    return macro_avpr, avpr_feats

def brier_score(pred_probs, values, numb_elem=1.):
    # insert feat. individually
    # Sizes: pred_probs: NxC (probability vector) ; values NxC (one-hot)
    # NOTE: /2. is to norm. to 0-1
    return torch.sum((pred_probs - values)**2) / (2*float(numb_elem))


def generate_score_outlier_matrix(p_params, input_data, dataset):

    # Score is based on the clean component decoder, or just decoder for VAE. (isRobust=False)

    shape_out_mat = (p_params['x'].shape[0], len(dataset.feat_info))
    outlier_score_mat = torch.zeros(shape_out_mat).type(p_params['x'].type())

    if dataset.dataset_type == "image" and (not dataset.cat_cols):

        # nll (vectorized)
        outlier_score_mat = nll_gauss_global(p_params['x'],
                                      input_data,
                                      p_params['logvar_x'], isRobust=False,
                                      shape_feats=[len(dataset.num_cols)]) 

    else:

        start = 0
        cursor_num_feat = 0 #

        for feat_select, (col_name, col_type, feat_size) in enumerate(dataset.feat_info):

            if col_type == "categ":

                nll = nll_categ_global(p_params['x'][:,start:(start + feat_size)], 
                                          input_data[:,feat_select].long(), feat_size, isRobust=False)
                start += feat_size

            else: # "real"

                nll = nll_gauss_global(p_params['x'][:,start:(start + 1)], # + 2
                                input_data[:,feat_select], 
                                p_params['logvar_x'][:,cursor_num_feat], isRobust=False)
                start += 1 # 2
                cursor_num_feat += 1 #

            outlier_score_mat[:,feat_select] = nll.view(-1) # negative log-likelihood score

    return outlier_score_mat


def generate_score_outlier_matrix_complete(p_params, q_params, input_data, dataset):

    # Score is based on entire RVAE reconstruction loss (isRobust=True)

    shape_out_mat = (p_params['x'].shape[0], len(dataset.feat_info))
    outlier_score_mat = torch.zeros(shape_out_mat).type(p_params['x'].type())

    w = torch.sigmoid(q_params['w']['logit_pi']).clamp(1e-6, 1-1e-6)

    if dataset.dataset_type == "image" and (not dataset.cat_cols):

        outlier_score_mat = nll_gauss_global(p_params['x'],
                                             input_data,
                                             p_params['logvar_x'], isRobust=True,
                                             w=w, shape_feats=[len(dataset.num_cols)])

    else:
        start = 0
        cursor_num_feat = 0

        for feat_select, (col_name, col_type, feat_size) in enumerate(dataset.feat_info):

            if col_type == "categ":

                nll_recon = nll_categ_global(p_params['x'][:,start:(start + feat_size)], 
                                          input_data[:,feat_select].long(), feat_size, isRobust=True,
                                          w=w[:,feat_select])
                start += feat_size

            else: # "real"

                nll_recon = nll_gauss_global(p_params['x'][:,start:(start + 1)], # + 2
                                               input_data[:,feat_select],
                                               p_params['logvar_x'][:,cursor_num_feat],
                                               isRobust=True, w=w[:,feat_select])
                start += 1 # 2
                cursor_num_feat += 1 #

            outlier_score_mat[:,feat_select] = nll_recon.view(-1) # reconstruction loss cost

    return outlier_score_mat


def cell_metrics(target_errors, outlier_score_mat, weights=False):

    target_errors = target_errors.cpu()
    outlier_score_mat = outlier_score_mat.cpu()

    target_cells = target_errors.numpy()

    if weights: # assumes outlier_score_mat is pi's
        outlier_score_cells = -outlier_score_mat.log().numpy() # // 1.- equivalent: 1.-q_params['w']['pi']
    else: # assumes outlier_score_mat is -log p(x | ...)
        outlier_score_cells = outlier_score_mat.numpy()

    auc_cell, auc_vec = get_auc_metrics(target_cells, outlier_score_cells)
    avpr_cell, avpr_vec = get_avpr_metrics(target_cells, outlier_score_cells)

    return auc_cell, auc_vec, avpr_cell, avpr_vec

def row_metrics(target_errors, outlier_score_mat, weights=False):

    target_errors = target_errors.cpu()
    outlier_score_mat = outlier_score_mat.cpu()

    target_errors_row = (target_errors.sum(dim=1)>0)

    if weights: # assumes outlier_score_mat is pi's
        outlier_score_row = - outlier_score_mat.log().sum(dim=1)
    else: # assumes outlier_score_mat is -log p(x | ... )
        outlier_score_row = outlier_score_mat.sum(dim=1)

    target_row = target_errors_row.numpy()
    outlier_score_row = outlier_score_row.numpy()

    if target_row.any():
        auc_row = auc_compute(target_row, outlier_score_row)
        avpr_row = avpr_compute(target_row, outlier_score_row)
    else:
        auc_row = np.ones(target_row.shape)*-10.
        avpr_row = np.ones(target_row.shape)*-10.

    return auc_row, avpr_row

def error_computation(model, X_true, X_hat, mask, x_input_size=False):

    # This function computes the proper error for each type of variable
    use_device = "cuda" if model.args.cuda_on else "cpu"

    start = 0
    cursor_feat = 0
    feature_errors_arr = []
    for feat_select, (_, col_type, feat_size) in enumerate(model.dataset_obj.feat_info):

        select_cell_pos = mask[:,cursor_feat].bool()

        if select_cell_pos.sum() == 0:
            feature_errors_arr.append(-1.)
        else:
            # Brier Score (score ranges betweem 0-1)
            if col_type == 'categ':
                true_feature_one_hot = torch.zeros((X_true.shape[0], feat_size)).to(use_device)
                true_feature_one_hot[torch.arange(X_true.shape[0], device=use_device), X_true[:,cursor_feat].long()] = 1.

                if x_input_size:
                    reconstructed_feature = torch.zeros((X_true.shape[0], feat_size)).to(use_device)
                    reconstructed_feature[torch.arange(X_true.shape[0], device=use_device), X_hat[:,cursor_feat].long()] = 1.
                    feat_size = 1
                else:
                    reconstructed_feature = torch.exp(X_hat[:,start:(start + feat_size)] + 1e-6) # exp of log_probs

                error_brier = brier_score(reconstructed_feature[select_cell_pos],
                                                true_feature_one_hot[select_cell_pos,:], select_cell_pos.sum().item())

                feature_errors_arr.append(error_brier.item())
                start += feat_size

            # Standardized Mean Square Error (SMSE)
            # SMSE (score ranges betweem 0,1)
            elif col_type == 'real':
                true_feature = X_true[:,cursor_feat]
                reconstructed_feature = X_hat[:,start:(start + 1)].view(-1)

                smse_error = torch.sum((true_feature[select_cell_pos] - reconstructed_feature[select_cell_pos])**2)
                sse_div = torch.sum(true_feature[select_cell_pos]**2) # (y_ti - avg(y_i))**2, avg(y_i)=0. due to data standardizaton
                smse_error = smse_error / sse_div # SMSE does not need to div 1/N_mask, since it cancels out.

                feature_errors_arr.append(smse_error.item())
                start += 1 # 2

        cursor_feat +=1

    # Average error per feature, with selected cells only
    error_per_feature = torch.tensor(feature_errors_arr).type(X_hat.type())

    # Global error (adding all the errors and dividing by number of features)
    mean_error = np.array([val for val in feature_errors_arr if val >= 0.]).mean()

    return mean_error, error_per_feature