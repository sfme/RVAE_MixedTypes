
import sys

# sys.path.append("..")
# sys.path.append("../..")
# sys.path.append(".")

import parser_arguments_deepRPCA
import torch
import numpy as np
from torch import nn, optim
import core_models.utils as utils

from sklearn.metrics import roc_auc_score as auc_compute
from sklearn.metrics import average_precision_score as avpr_compute
from core_models.utils import get_auc_metrics, get_avpr_metrics

import json
import os, errno
import pandas as pd
from sklearn.metrics import auc

from collections import OrderedDict


# RAE baseline architecture -- DeepRPCA
class RAE(nn.Module):

    def __init__(self, dataset_obj, args_in):

        super(RAE, self).__init__()

        self.dataset_obj = dataset_obj
        self.args_in = args_in

        self.size_output = np.array([c_size for _, col_type, c_size in dataset_obj.feat_info
                                   if col_type=="categ"], dtype='int').sum()
        self.size_output += len(dataset_obj.num_cols)
        self.size_input = self.size_output

        ## Encoder Params
        self.fc1 = nn.Linear(self.size_input, self.args_in.layer_size)
        self.fc2 = nn.Linear(self.args_in.layer_size, self.args_in.latent_dim)


        ## Decoder Params
        self.fc3 = nn.Linear(self.args_in.latent_dim, self.args_in.layer_size)

        if dataset_obj.dataset_type == "image" and (not dataset_obj.cat_cols):
            self.out_cat_linears = nn.Linear(self.args_in.layer_size, self.size_output)
        else:
            self.out_cat_linears = nn.ModuleList([nn.Linear(self.args_in.layer_size, c_size)
                                                 for _, col_type, c_size in dataset_obj.feat_info])


        ## Other

        if args_in.activation == 'relu':
            self.activ = nn.ReLU()
        elif args_in.activation == 'hardtanh':
            self.activ = nn.Hardtanh()

        if args_in.cat_fout == 'sigmoid':
            self.cat_fout = nn.Sigmoid()
        elif args_in.cat_fout == 'softmax':
            self.cat_fout = nn.Softmax(dim=1)

        # define encoder / decoder easy access lists
        # encoder params
        encoder_list = [self.fc1, self.fc2]
        self.encoder_mod = nn.ModuleList(encoder_list)
        self.encoder_param_list = nn.ParameterList(self.encoder_mod.parameters())

        # decoder params
        decoder_list = [self.fc3, self.out_cat_linears]
        self.decoder_mod = nn.ModuleList(decoder_list)
        self.decoder_param_list = nn.ParameterList(self.decoder_mod.parameters())


    def encode(self, x_data):
        fc1_out = self.fc1(x_data)
        h1 = self.activ(fc1_out)

        return self.fc2(h1)

    def decode(self, z):
        h3 = self.activ(self.fc3(z))

        if self.dataset_obj.dataset_type == 'image' and (not self.dataset_obj.cat_cols):
            h_out = self.out_cat_linears(h3)

            # tensor with dims (batch_size, self.size_output)
            out_vals = h_out

        else:
            out_cat_list = []
            for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):

                h_out = out_cat_layer(h3)

                if self.dataset_obj.feat_info[feat_idx][1] == "categ": # coltype check
                    out_cat_list.append(self.cat_fout(h_out))

                elif self.dataset_obj.feat_info[feat_idx][1] == "real":
                    out_cat_list.append(h_out)

            # tensor with dims (batch_size, self.size_output)
            out_vals = torch.cat(out_cat_list, 1)

        return out_vals

    def forward(self, x_data):
        z = self.encode(x_data)
        return self.decode(z)


def loss_function_mixed_bp(recon_LD, input_LD, dataset_obj):
    loss_out = torch.zeros(1).type(recon_LD.type())

    if dataset_obj.dataset_type == 'image' and (not dataset_obj.cat_cols):
        loss_out = ((input_LD - recon_LD)**2).sum()

    else:
        start = 0
        for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):
            if col_type == 'categ':
                log_p = torch.log(recon_LD[:,start:(start + feat_size)] + 1e-8)
                loss_out += (-log_p*input_LD[:,start:(start + feat_size)]).sum()
                start += feat_size

            elif col_type == 'real':
                loss_out += ((input_LD[:,start] - recon_LD[:,start])**2).sum()
                start += 1

    return loss_out

def l1_prox_operator(S_in, lambda_val):

    # proximal operator for l1 norm: shrinkage operator
    # -- accounts for sparse random errors

    S = S_in.clone().detach()

    S[S>lambda_val] = S[S>lambda_val] - lambda_val

    S[S<-lambda_val] = S[S<-lambda_val] + lambda_val

    aux_log = (S<=lambda_val) * (S>=-lambda_val)

    S[aux_log] = 0.0

    return S

def l21_prox_operator(S_in, lambda_val, transpose=True):

    # proximal operator for l21 norm: block-wise soft-thresholding
    # -- accounts for groups of sparse random errors

    S = S_in.clone().detach()

    # in order to apply prox operator
    if transpose:
        S = S.t()

    ej_vec = (S**2).sum(dim=0).sqrt()

    S[:,ej_vec>lambda_val] = S[:,ej_vec>lambda_val] - lambda_val*S[:,ej_vec>lambda_val]/ej_vec[ej_vec>lambda_val]

    S[:,ej_vec<=lambda_val] = 0.0

    # in order to return right dimensions
    if transpose:
        S = S.t()

    return S

def l21_norm(A):

    return (A**2).sum(dim=0).sqrt().sum()

def loss_function_RAE_opt_l1(recon_LD, input_LD, S, lambda_val):

    loss = ((input_LD - recon_LD)**2).sum().sqrt()

    loss = (loss + S.norm(1)*lambda_val)/float(recon_LD.shape[0])

    return loss

def loss_function_RAE_opt_l21(recon_LD, input_LD, S, lambda_val, transpose=True):

    if transpose:
        S_calc = S.t()
    else:
        S_calc = S

    loss = ((input_LD - recon_LD)**2).sum().sqrt()

    loss = (loss + lambda_val*l21_norm(S_calc))/float(recon_LD.shape[0])

    return loss


def error_detection(dataset_obj, S, data_in, data_recon, l21=False):

    """
        Uses 1) outlier matrix S; 2) reconstruction error
    """

    ##### S based outlier detection

    S_out = S.data

    ## Cell
    if dataset_obj.dataset_type == 'image' and (not dataset_obj.cat_cols):
        error_mtx_cell_S = S_out**2

    else:
        start = 0
        error_mtx_cell_S = torch.zeros((S_out.shape[0], len(dataset_obj.feat_info))).type_as(S)

        for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):
            if col_type == "categ":
                error_mtx_cell_S[:, feat_select] = (S_out[:, start:(start + feat_size)]**2).sum(dim=1)
                start += feat_size

            elif col_type == "real":
                error_mtx_cell_S[:, feat_select] = S_out[:, start]**2
                start += 1

    ## Row
    error_mtx_row_S = S_out.norm(p=2, dim=1) ## like original authors used in their paper code as outlier score!


    ##### Recon. based outlier detection

    ## Cell
    if dataset_obj.dataset_type == 'image' and (not dataset_obj.cat_cols):
        error_mtx_cell_recon = ((data_in - data_recon)**2)

    else:
        start = 0
        error_mtx_cell_recon = torch.zeros((data_recon.shape[0], len(dataset_obj.feat_info))).type(data_recon.type())

        for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):
            if col_type == 'categ':
                log_p = torch.log(data_recon[:,start:(start + feat_size)] + 1e-8)
                error_mtx_cell_recon[:, feat_select] = (-log_p*data_in[:,start:(start + feat_size)]).sum(dim=1)
                start += feat_size

            elif col_type == 'real':
                error_mtx_cell_recon[:, feat_select] = ((data_in[:,start] - data_recon[:,start])**2)
                start += 1

    ## Row
    error_mtx_row_recon = error_mtx_cell_recon.sum(dim=1)

    return error_mtx_cell_S, error_mtx_row_S, error_mtx_cell_recon, error_mtx_row_recon



def repair_error(args, dataset_obj, X_true, X_hat, mask):

    # This function computes the proper error for each type of variable
    use_device = "cuda" if args.cuda_on else "cpu"

    start = 0
    cursor_feat = 0
    feature_errors_arr = []
    for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):

        select_cell_pos = mask[:,cursor_feat].astype(bool)

        if select_cell_pos.sum() == 0:
            feature_errors_arr.append(-1.)
        else:
            # Brier Score (score ranges betweem 0-1)
            if col_type == 'categ':

                reconstructed_feature = X_hat[:,start:(start + feat_size)]
                true_feature = X_true[:,start:(start + feat_size)]

                error_brier = utils.brier_score(reconstructed_feature[select_cell_pos],
                                    true_feature[select_cell_pos], select_cell_pos.sum().item())

                feature_errors_arr.append(error_brier.item())
                start += feat_size

            # Standardized Mean Square Error (SMSE)
            # SMSE (score ranges betweem 0,1)
            elif col_type == 'real':
                true_feature = X_true[:,start:(start + 1)]
                reconstructed_feature = X_hat[:,start:(start + 1)].view(-1)

                smse_error = torch.sum((true_feature[select_cell_pos] - reconstructed_feature[select_cell_pos].view(-1,1))**2)
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


def repair_phase(args, model, data_mtxs, data_dirty, data_clean, mask, iterations=1):

    model.eval()
    with torch.no_grad():
        # model under dirty data input (uses previous estimate)
        data_recon_xd = data_mtxs['LD']
        # model under clean data input (data is clean, no subtraction)
        data_recon_xc = model(data_clean)


    # error (MSE) lower bound, on dirty cell positions only
    error_lb_dc, error_lb_dc_per_feat = repair_error(args, args.dataset_defs, data_clean, data_recon_xc, mask) # x_truth - f_vae(x_clean)

    # error repair, on dirty cell positions only
    error_repair_dc, error_repair_dc_per_feat = repair_error(args, args.dataset_defs, data_clean, data_recon_xd, mask) # x_truth - f_vae(x_dirty)

    # error upper bound, on dirty cell positions only
    error_up_dc, error_up_dc_per_feat = repair_error(args, args.dataset_defs, data_clean, data_dirty, mask) # x_truth - x_dirty

    # error on clean cell positions only (to test impact on dirty cells on clean cells under model)
    error_repair_cc, error_repair_cc_per_feat = repair_error(args, args.dataset_defs, data_clean, data_recon_xd, 1-mask)

    # AE loss only for cc and dc positions, under dirty data
    # dc
    dirty_row_pos = mask.any(axis=1)
    n_dirty_rows = dirty_row_pos.sum()
    # nll
    ae_loss_dc = loss_function_mixed_bp(data_recon_xd[dirty_row_pos,:], data_clean[dirty_row_pos,:], args.dataset_defs)

    # cc
    clean_row_pos = 1-dirty_row_pos
    n_clean_rows = clean_row_pos.sum()
    # nll
    ae_loss_cc = loss_function_mixed_bp(data_recon_xd[clean_row_pos,:], data_clean[clean_row_pos,:], args.dataset_defs)

    # RAE (DeepRPCA) loss, under dirt data
    # dc
    rae_loss_dc = loss_function_RAE_opt_l21(data_recon_xd[dirty_row_pos,:], data_clean[dirty_row_pos,:],
                                                    data_mtxs['S'][dirty_row_pos,:], args.lambda_param)
    # cc
    rae_loss_cc = loss_function_RAE_opt_l21(data_recon_xd[clean_row_pos,:], data_clean[clean_row_pos,:],
                                                    data_mtxs['S'][clean_row_pos,:], args.lambda_param)



    eval_data_len = data_dirty.shape[0]
    losses = OrderedDict([('eval_loss_final_clean_dc', rae_loss_dc.item()/n_dirty_rows), ('eval_nll_final_clean_dc',ae_loss_dc.item()/n_dirty_rows),
              ('eval_loss_final_clean_cc', rae_loss_cc.item()/n_clean_rows), ('eval_nll_final_clean_cc',ae_loss_cc.item()/n_clean_rows),
              ('eval_loss_final_clean_all', (rae_loss_cc+rae_loss_dc).item()/eval_data_len), ('eval_nll_final_clean_all',(ae_loss_cc+ae_loss_dc).item()/eval_data_len),
              ('mse_lower_bd_dirtycells', error_lb_dc.item()), ('mse_upper_bd_dirtycells', error_up_dc.item()), ('mse_repair_dirtycells', error_repair_dc.item()),
              ('mse_repair_cleancells', error_repair_cc.item()),
              ('errors_per_feature', [error_lb_dc_per_feat, error_repair_dc_per_feat, error_up_dc_per_feat, error_repair_cc_per_feat])])

    return losses


def run_iteration_ADMM(args_in, AE_model, optim_proc, X_data, data_loader, data_mtxs, AE_train_mode=True):

    # Runs full ADMM algo. with BP for AE model, for one iteration (several BP epochs)

    ## Run BP on AE model
    if AE_train_mode:

        AE_model.train()

        for epoch in range(1, args_in.number_epochs + 1):

            train_loss = 0.

            for batch_idx, unpack in enumerate(data_loader):

                X_data_batch = unpack[0] # raw data
                true_errors_batch = unpack[1] # ground-truth labels
                row_idxs = unpack[2] # datapoint indexes

                if args_in.cuda_on:
                    X_data_batch = X_data_batch.cuda()

                with torch.no_grad():
                    LD_batch = X_data_batch - data_mtxs['S'][row_idxs,:]

                optim_proc.zero_grad()

                recon_LD_batch = AE_model(LD_batch)

                loss = loss_function_mixed_bp(recon_LD_batch, LD_batch, args_in.dataset_defs)

                loss.backward()

                train_loss += loss.item()

                optim_proc.step()

                if batch_idx % args_in.log_interval == 0:

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(X_data_batch), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader),
                        loss.item() / len(X_data_batch)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / len(data_loader.dataset)))


    ## Project back to the Constraint Manifold
    AE_model.eval()

    with torch.no_grad():

        data_mtxs['LD'] = AE_model(X_data - data_mtxs['S'])

        if args_in.l1_method: # unstructured outliers
            data_mtxs['S'] = l1_prox_operator(X_data - data_mtxs['LD'], args_in.lambda_param)
        elif args_in.l21_method: # structured outliers on the rows
            data_mtxs['S'] = l21_prox_operator(X_data - data_mtxs['LD'], args_in.lambda_param)

        X_data_norm = X_data.norm(2)

        c1 = (X_data - data_mtxs['LD'] - data_mtxs['S']).norm(2) / X_data_norm

        c2 = (data_mtxs['LS'] - data_mtxs['LD'] - data_mtxs['S']).norm(2) / X_data_norm

        data_mtxs['LS'] = data_mtxs['LD'] + data_mtxs['S']

    return c1.item(), c2.item()


def train(args_in, model, optim_proc, data_mtxs, x_data, x_data_clean, admm_iter):

    converged_flag = False

    # Run ADMM iteration
    print("\nADMM Iteration Number:{}; Total Epochs: {}\n\n".format(admm_iter, admm_iter*args_in.number_epochs))

    c1, c2 = run_iteration_ADMM(args_in, model, optim_proc, x_data, args_in.train_loader,
                                data_mtxs)

    print("\nTrain Data:")

    # Print convergence values (lower is better)
    print("\n\t\t(train) c1 value: {}".format(c1))
    print("\t\t(train) c2 value: {}\n\n".format(c2))

    # Check convergence
    if c1 < args_in.eps_bar_X or c2 < args_in.eps_bar_diff_iter:
        converged_flag = True

    # Print DeepRPCA cost (is minimized by RAE -- DeepRPCA)
    model.eval()
    with torch.no_grad():
        LD_train_recon = model(data_mtxs['LD'])
    if args_in.l1_method:
        loss_RAE = loss_function_RAE_opt_l1(LD_train_recon, data_mtxs['LD'], data_mtxs['S'], args_in.lambda_param).item()
    elif args_in.l21_method:
        loss_RAE = loss_function_RAE_opt_l21(LD_train_recon, data_mtxs['LD'], data_mtxs['S'], args_in.lambda_param).item()

    print("\n\t\t(train) Loss for RAE cost (opt via ADMM): {}; Lambda val: {} \n".format(loss_RAE, args_in.lambda_param))

    # Check outlier metrics
    outlier_score_cell_S, outlier_score_row_S, outlier_score_cell_recon, outlier_score_row_recon = \
        error_detection(args_in.dataset_defs, data_mtxs['S'], x_data, data_mtxs['LD'], args_in.l21_method)

    outlier_score_cell_S = outlier_score_cell_S.cpu().numpy()
    outlier_score_cell_recon = outlier_score_cell_recon.cpu().numpy()
    outlier_score_row_S = outlier_score_row_S.cpu().numpy()
    outlier_score_row_recon = outlier_score_row_recon.cpu().numpy()

    target_errors_train_cell = args_in.target_errors_train
    target_errors_train_cell = target_errors_train_cell.cpu().numpy().astype(np.uint8)
    target_errors_train_row = (target_errors_train_cell.sum(axis=1)>0).astype(np.uint8)

    ## Cell metrics
    auc_cell_train_S, _ = get_auc_metrics(target_errors_train_cell, outlier_score_cell_S)
    avpr_cell_train_S, avpr_feats_S = get_avpr_metrics(target_errors_train_cell, outlier_score_cell_S)

    auc_cell_train_recon, _ = get_auc_metrics(target_errors_train_cell, outlier_score_cell_recon)
    avpr_cell_train_recon, avpr_feats_recon = get_avpr_metrics(target_errors_train_cell, outlier_score_cell_recon)


    ## Row metrics
    auc_row_train_S = auc_compute(target_errors_train_row, outlier_score_row_S)
    avpr_row_train_S = avpr_compute(target_errors_train_row, outlier_score_row_S)

    auc_row_train_recon = auc_compute(target_errors_train_row, outlier_score_row_recon)
    avpr_row_train_recon = avpr_compute(target_errors_train_row, outlier_score_row_recon)

    outlier_metrics = OrderedDict([('auc_cell_S',auc_cell_train_S), ('avpr_cell_S',avpr_cell_train_S), 
                       ('auc_cell_recon',auc_cell_train_recon), ('avpr_cell_recon',avpr_cell_train_recon),
                       ('auc_row_S',auc_row_train_S), ('avpr_row_S',avpr_row_train_S), 
                       ('auc_row_recon',auc_row_train_recon), ('avpr_row_recon',avpr_row_train_recon), 
                       ('avpr_per_feature',[avpr_feats_S,avpr_feats_recon])])

    print('Train (S) -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {} \n\n'.format(
        auc_cell_train_S, avpr_cell_train_S, auc_row_train_S, avpr_row_train_S))

    print('Train (Recon.) -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {}\n'.format(
        auc_cell_train_recon, avpr_cell_train_recon, auc_row_train_recon, avpr_row_train_recon))

    # Repair analysis
    repair_metrics = repair_phase(args_in, model, data_mtxs, x_data, x_data_clean, 
                                  target_errors_train_cell)


    store_metrics_iter('train', args_in, loss_RAE, outlier_metrics, repair_metrics,
                       c1, c2, admm_iter)

    return converged_flag



def test(args_in, model, optim_proc, data_mtxs, x_data, x_data_clean, admm_iter):

    converged_flag = False

    # Run ADMM iteration
    c1, c2 = run_iteration_ADMM(args_in, model, optim_proc, x_data, args_in.test_loader,
                                data_mtxs, AE_train_mode=False)

    print("\nTest Data:")

    # Print convergence values (akin to losses, lower is better)
    print("\n\t\t(test) c1 value: {}".format(c1))
    print("\t\t(test) c2 value: {}\n\n".format(c2))

    # Check convergence
    if c1 < args_in.eps_bar_X or c2 < args_in.eps_bar_diff_iter:
        converged_flag = True

    # Print DeepRPCA cost (is minimized by RAE -- DeepRPCA)
    model.eval()
    with torch.no_grad():
        LD_test_recon = model(data_mtxs['LD'])

    if args_in.l1_method:
        loss_RAE = loss_function_RAE_opt_l1(LD_test_recon, data_mtxs['LD'], data_mtxs['S'], args_in.lambda_param).item()
    elif args_in.l21_method:
        loss_RAE = loss_function_RAE_opt_l21(LD_test_recon, data_mtxs['LD'], data_mtxs['S'], args_in.lambda_param).item()


    print("\n\t\t(test) Loss for RAE cost (opt via ADMM): {}; Lambda val: {} \n".format(loss_RAE, args_in.lambda_param))

    # Check outlier metrics
    outlier_score_cell_S, outlier_score_row_S, outlier_score_cell_recon, outlier_score_row_recon = \
        error_detection(args_in.dataset_defs, data_mtxs['S'], x_data, data_mtxs['LD'], args_in.l21_method) 

    outlier_score_cell_S = outlier_score_cell_S.cpu().numpy()
    outlier_score_cell_recon = outlier_score_cell_recon.cpu().numpy()
    outlier_score_row_S = outlier_score_row_S.cpu().numpy()
    outlier_score_row_recon = outlier_score_row_recon.cpu().numpy()

    target_errors_test_cell = args_in.target_errors_test
    target_errors_test_cell = target_errors_test_cell.cpu().numpy().astype(np.uint8)
    target_errors_test_row = (target_errors_test_cell.sum(axis=1)>0).astype(np.uint8)


    ## Cell metrics
    auc_cell_test_S, _ = get_auc_metrics(target_errors_test_cell, outlier_score_cell_S)
    avpr_cell_test_S, avpr_feats_S = get_avpr_metrics(target_errors_test_cell, outlier_score_cell_S)

    auc_cell_test_recon, _ = get_auc_metrics(target_errors_test_cell, outlier_score_cell_recon)
    avpr_cell_test_recon, avpr_feats_recon = get_avpr_metrics(target_errors_test_cell, outlier_score_cell_recon)


    ## Row metrics
    auc_row_test_S = auc_compute(target_errors_test_row, outlier_score_row_S)
    avpr_row_test_S = avpr_compute(target_errors_test_row, outlier_score_row_S)

    auc_row_test_recon = auc_compute(target_errors_test_row, outlier_score_row_recon)
    avpr_row_test_recon = avpr_compute(target_errors_test_row, outlier_score_row_recon)

    outlier_metrics = OrderedDict([('auc_cell_S',auc_cell_test_S), ('avpr_cell_S',avpr_cell_test_S), 
                       ('auc_cell_recon',auc_cell_test_recon), ('avpr_cell_recon',avpr_cell_test_recon),
                       ('auc_row_S',auc_row_test_S), ('avpr_row_S',avpr_row_test_S), 
                       ('auc_row_recon',auc_row_test_recon), ('avpr_row_recon',avpr_row_test_recon), 
                       ('avpr_per_feature',[avpr_feats_S,avpr_feats_recon])])

    print('Test (S) -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {} \n\n'.format(
        auc_cell_test_S, avpr_cell_test_S, auc_row_test_S, avpr_row_test_S))

    print('Test (Recon.) -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {}\n'.format(
        auc_cell_test_recon, avpr_cell_test_recon, auc_row_test_recon, avpr_row_test_recon))

    # Repair analysis
    repair_metrics = repair_phase(args_in, model, data_mtxs, x_data, x_data_clean, target_errors_test_cell)


    store_metrics_iter('test', args_in, loss_RAE, outlier_metrics, repair_metrics,
                       c1, c2, admm_iter)

    return converged_flag


def store_metrics_iter(mode, args_in, loss_RAE, outlier_metrics, repair_metrics,
                       c1, c2, iter_numb):

    # save to file step
    if args_in.save_on:
        # save to dataframe table
        args_in.losses_save[mode][iter_numb] = [loss_RAE, c1, c2]
        del outlier_metrics['avpr_per_feature']
        args_in.losses_save[mode][iter_numb].extend(list(outlier_metrics.values()))
        del repair_metrics['errors_per_feature']
        args_in.losses_save[mode][iter_numb].extend(list(repair_metrics.values()))


def evaluation_phase(args_in, model, data_mtxs, x_data, x_data_clean, target_errors_cell, data_loader, iterations=1):

    model.eval()

    # Check outlier metrics
    outlier_score_cell_S, outlier_score_row_S, outlier_score_cell_recon, outlier_score_row_recon = \
        error_detection(args_in.dataset_defs, data_mtxs['S'], x_data, data_mtxs['LD'], args_in.l21_method)

    outlier_score_cell_S = outlier_score_cell_S.cpu().numpy()
    outlier_score_cell_recon = outlier_score_cell_recon.cpu().numpy()
    outlier_score_row_S = outlier_score_row_S.cpu().numpy()
    outlier_score_row_recon = outlier_score_row_recon.cpu().numpy()

    target_errors_cell = target_errors_cell.cpu().numpy().astype(np.uint8)
    target_errors_row = (target_errors_cell.sum(axis=1)>0).astype(np.uint8)


    ## Cell metrics
    auc_cell_S, auc_feats_S = get_auc_metrics(target_errors_cell, outlier_score_cell_S)
    avpr_cell_S, avpr_feats_S = get_avpr_metrics(target_errors_cell, outlier_score_cell_S)

    auc_cell_recon, auc_feats_recon = get_auc_metrics(target_errors_cell, outlier_score_cell_recon)
    avpr_cell_recon, avpr_feats_recon = get_avpr_metrics(target_errors_cell, outlier_score_cell_recon)

    ## Row metrics
    auc_row_S = auc_compute(target_errors_row, outlier_score_row_S)
    avpr_row_S = avpr_compute(target_errors_row, outlier_score_row_S)

    auc_row_recon = auc_compute(target_errors_row, outlier_score_row_recon)
    avpr_row_recon = avpr_compute(target_errors_row, outlier_score_row_recon)

    outlier_metrics = OrderedDict([('score_cell_S',outlier_score_cell_S),('score_row_S',outlier_score_row_S),
                       ('score_cell_recon',outlier_score_cell_recon),('score_row_recon',outlier_score_row_recon),
                       ('auc_cell_S',auc_cell_S), ('avpr_cell_S',avpr_cell_S),
                       ('auc_cell_recon',auc_cell_recon), ('avpr_cell_recon',avpr_cell_recon),
                       ('auc_row_S',auc_row_S), ('avpr_row_S',avpr_row_S),
                       ('auc_row_recon',auc_row_recon), ('avpr_row_recon',avpr_row_recon),
                       ('avpr_per_feature',[avpr_feats_S,avpr_feats_recon]),
                       ('auc_per_feature',[auc_feats_S,auc_feats_recon])])

    # Repair analysis
    repair_metrics = repair_phase(args_in, model, data_mtxs, x_data, x_data_clean, target_errors_cell)

    return outlier_metrics, repair_metrics


def store_metrics_final(mode, dataset_obj, attributes, outlier_metrics, repair_metrics, targets, S, args_in):

    ## Outlier metrics

    # store AVPR for features (cell only)
    df_avpr_feat_cell = pd.DataFrame([], index=['AVPR_recon', 'AVPR_S'], columns=attributes)
    df_avpr_feat_cell.loc['AVPR_recon'] = outlier_metrics['avpr_per_feature'][1]
    df_avpr_feat_cell.loc['AVPR_S'] = outlier_metrics['avpr_per_feature'][0]
    df_avpr_feat_cell.to_csv(args_in.folder_output + "/" + mode + "_avpr_features.csv")

    # store AUC for features (cell only)
    df_auc_feat_cell = pd.DataFrame([], index=['AUC_recon', 'AUC_S'], columns=attributes)
    df_auc_feat_cell.loc['AUC_recon'] = outlier_metrics['auc_per_feature'][1]
    df_auc_feat_cell.loc['AUC_S'] = outlier_metrics['auc_per_feature'][0]
    df_auc_feat_cell.to_csv(args_in.folder_output + "/" + mode + "_auc_features.csv")

    ## Store data from Epochs (includes repair metrics)

    columns = ['Avg. Loss DeepRPCA', 'c1', 'c2',
               'AUC Cell S', 'AVPR Cell S', 'AUC Cell Recon', 'AVPR Cell Recon',
               'AUC Row S', 'AVPR Row S', 'AUC Row Recon', 'AVPR Row Recon',
               'Avg. Loss -- p(x_clean | x_dirty) on dirty pos', 'Avg. NLL -- p(x_clean | x_dirty) on dirty pos',
               'Avg. Loss -- p(x_clean | x_dirty) on clean pos', 'Avg. NLL -- p(x_clean | x_dirty) on clean pos',
               'Avg. Loss -- p(x_clean | x_dirty) on all', 'Avg. NLL -- p(x_clean | x_dirty) on all',
               'Error lower-bound on dirty pos', 'Error upper-bound on dirty pos',
               'Error repair on dirty pos', 'Error repair on clean pos']

    df_out = pd.DataFrame.from_dict(args_in.losses_save[mode], orient="index",
                                    columns=columns)
    df_out.index.name = "Epochs"
    df_out.to_csv(args_in.folder_output + "/" + mode + "_epochs_data.csv")

    ### Repair: store errors per feature

    df_errors_repair = pd.DataFrame([], index=['error_lowerbound_dirtycells','error_repair_dirtycells',
            'error_upperbound_dirtycells','error_repair_cleancells'], columns=attributes)

    df_errors_repair.loc['error_lowerbound_dirtycells'] = repair_metrics['errors_per_feature'][0].cpu()
    df_errors_repair.loc['error_repair_dirtycells'] = repair_metrics['errors_per_feature'][1].cpu()
    df_errors_repair.loc['error_upperbound_dirtycells'] = repair_metrics['errors_per_feature'][2].cpu()
    df_errors_repair.loc['error_repair_cleancells'] = repair_metrics['errors_per_feature'][3].cpu()

    df_errors_repair.to_csv(args_in.folder_output + "/" + mode + "_error_repair_features.csv")


    ## Save Matrix S
    if S.is_cuda:
        S_out = S.cpu()
    else:
        S_out = S
    np.save(args_in.folder_output + "/{}_S_matrix.npy".format(mode), S_out.numpy())





def main(args_in):

    #### MAIN ####

    # saving data of experiment to folder is on
    if args_in.save_on:
        # create folder for saving experiment data (if necessary)
        folder_output = args_in.output_folder
        args_in.folder_output=folder_output

        # structs for saving data
        args_in.losses_save = {"train":{},"test":{}}

        try:
            os.makedirs(folder_output + '/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    # dtype definitions for runing
    if args_in.cuda_on:
        dtype_float = torch.cuda.FloatTensor
        dtype_byte = torch.cuda.ByteTensor
    else:
        dtype_float = torch.FloatTensor
        dtype_byte = torch.ByteTensor


    # only one type of prior assumption on errors / outliers
    if (not args_in.l1_method) and (not args_in.l21_method):
        args_in.l21_method = True
    elif args_in.l1_method and args_in.l21_method:
        args_in.l21_method = False

    # Choose dataset to run on
    folder_path = args_in.data_folder


    # Load datasets
    train_loader, X_train, target_errors_train, dataset_obj_train, attributes = utils.load_data(folder_path, args_in.batch_size,
                                                                is_train=True, get_data_idxs=True, is_one_hot=True)
    args_in.dataset_defs = dataset_obj_train
    args_in.train_loader = train_loader
    args_in.target_errors_train = target_errors_train.type(dtype_byte)
    X_train = X_train.type(dtype_float)

    test_loader, X_test, target_errors_test, dataset_obj_test, _ = utils.load_data(folder_path, args_in.batch_size,
                                                                is_train=False, get_data_idxs=True, is_one_hot=True)
    args_in.test_loader = test_loader
    args_in.target_errors_test = target_errors_test.type(dtype_byte)
    X_test = X_test.type(dtype_float)

    # -- clean versions for data repair evaluation (standardized according to the dirty data statistics)
    train_loader_clean, X_train_clean, _, dataset_obj_clean, _ = utils.load_data(args_in.data_folder, args_in.batch_size,
                                                                is_train=True, is_clean=True, is_one_hot=True, stdize_dirty=True)

    args_in.train_loader_clean = train_loader_clean
    X_train_clean = X_train_clean.type(dtype_float)

    test_loader_clean, X_test_clean, _, _, _ = utils.load_data(args_in.data_folder, args_in.batch_size, is_train=False,
                                                                is_clean=True, is_one_hot=True, stdize_dirty=True)

    args_in.test_loader_clean = test_loader_clean
    X_test_clean = X_test_clean.type(dtype_float)


    # RAE model matrices
    rae_data_train = dict()
    rae_data_test = dict()

    rae_data_train['LD'] = torch.zeros_like(X_train).type(dtype_float)
    rae_data_test['LD'] = torch.zeros_like(X_test).type(dtype_float)

    rae_data_train['LS'] = X_train.clone()
    rae_data_test['LS'] = X_test.clone()

    rae_data_train['S'] = torch.zeros_like(X_train).type(dtype_float)
    rae_data_test['S'] = torch.zeros_like(X_test).type(dtype_float)


    # Run RAE model
    model = RAE(args_in.dataset_defs, args_in)

    if args_in.cuda_on:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args_in.lr, weight_decay=args_in.l2_reg)

    for admm_iter in range(1, args_in.number_ADMM_iters + 1):

        # train
        converged_train = train(args_in, model, optimizer, rae_data_train, X_train, X_train_clean, admm_iter)

        if converged_train:
            print("--> RAE for train data has converged!")

        # validation
        if args_in.turn_on_validation:
            test(args_in, model, optimizer, rae_data_test, X_test, X_test_clean, admm_iter)


    if args_in.save_on:

        ### Train Data
        outlier_metrics_train, repair_metrics_train  = evaluation_phase(args_in, model, rae_data_train,
                                                                        X_train, X_train_clean, 
                                                                        target_errors_train, train_loader)

        store_metrics_final('train', args_in.dataset_defs, attributes, outlier_metrics_train, repair_metrics_train,
                            target_errors_train, rae_data_train['S'], args_in)

        ### Test Data
        outlier_metrics_test, repair_metrics_test = evaluation_phase(args_in, model, rae_data_test,
                                                                    X_test, X_test_clean, 
                                                                    target_errors_test, test_loader)

        store_metrics_final('test', args_in.dataset_defs, attributes, outlier_metrics_test, repair_metrics_test,
                            target_errors_test, rae_data_test['S'], args_in)


        # save model parameters
        model.cpu()
        torch.save(model.state_dict(), folder_output + "/model_params.pth")

        # remove non-serializable stuff
        del args_in.dataset_defs # = []
        del args_in.train_loader # = []
        del args_in.target_errors_train # = []
        del args_in.test_loader # = []
        del args_in.target_errors_test # = []
        del args_in.train_loader_clean
        del args_in.test_loader_clean
        del args_in.folder_output
        del args_in.losses_save

        # save to .json file the args that were used for running the model
        with open(folder_output + "/args_run.json", "w") as outfile:
            json.dump(vars(args_in), outfile, indent=4, sort_keys=True)


if __name__ == '__main__':

    # Needed to run in command line
    args = parser_arguments_deepRPCA.getArgs(sys.argv[1:])
    main(args)


