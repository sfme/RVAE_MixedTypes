
import sys
# sys.path.append("..")
# sys.path.append("../..")

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
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import auc

from collections import OrderedDict

from core_models.model_utils import nll_categ_global, nll_gauss_global
import parser_arguments_CondPred


def create_predictor(layers_list, activ_func, pred_type, out_func):

    net_build = []

    if pred_type != 'linear':
        for lay_idx, (layer_in, layer_out) in enumerate(layers_list[:-1]):
            net_build.append(('fc_'+str(lay_idx), nn.Linear(layer_in,layer_out)))
            net_build.append(('activ_'+str(lay_idx), activ_func))

    net_build.append(('fc_out', nn.Linear(layers_list[-1][0],layers_list[-1][1])))

    if out_func:
        net_build.append(('activ_out', out_func))

    return nn.Sequential(OrderedDict(net_build))


class SinglePred(nn.Module):

    def __init__(self, dataset_obj, args_in, col_name):

        super(SinglePred, self).__init__()

        self.dataset_obj = dataset_obj
        self.args_in = args_in
        self.col_name_cond = col_name

        self.feat_info_dict = OrderedDict([(k,(v1,v2)) for k, v1, v2 in dataset_obj.feat_info])

        self.col_type_cond = self.feat_info_dict[col_name][0]
        self.col_size_cond = self.feat_info_dict[col_name][1]
        self.col_idx = [col_idx_curr for col_idx_curr, (col_name_curr, _, _ ) in enumerate(dataset_obj.feat_info)
                            if col_name_curr==self.col_name_cond][0]

        if args_in.activation == 'relu':
            self.activ = nn.ReLU()
        elif args_in.activation == 'hardtanh':
            self.activ = nn.Hardtanh()

        if self.col_type_cond == 'categ':
            self.out_func = nn.LogSoftmax(dim=1)
        else:
            self.out_func = None

        self.size_input_total = len(dataset_obj.cat_cols)*self.args_in.embedding_size + len(dataset_obj.num_cols)
        self.size_output_total = len(dataset_obj.cat_cols) + len(dataset_obj.num_cols) # 2*

        if self.col_type_cond == 'categ':
            self.size_input_cond = self.size_input_total - self.args_in.embedding_size
        else:
            self.size_input_cond = self.size_input_total - self.col_size_cond

        self.feat_embedd = nn.ModuleDict([(col_name_curr, nn.Embedding(c_size_curr, self.args_in.embedding_size, max_norm=1))
                                         for col_name_curr, col_type_curr, c_size_curr in dataset_obj.feat_info
                                         if (col_type_curr=="categ" and col_name_curr!=self.col_name_cond)])

        if self.args_in.base_type == 'linear':
            self.pred_nn = create_predictor([(self.size_input_cond,self.col_size_cond)], self.activ,
                                            self.args_in.base_type, self.out_func)
        else:
            self.pred_nn = create_predictor([(self.size_input_cond,200),(200,50), (50, self.col_size_cond)],
                                            self.activ, self.args_in.base_type, self.out_func)

        if self.col_type_cond == 'real':
            self.logvar_x = nn.Parameter(torch.zeros(1).float())
        else:
            self.logvar_x = None


    def get_inputs(self, x_data):
        # mixed data, or just real or just categ

        input_list = []

        for feat_idx, (col_name, col_type, col_size) in enumerate(self.dataset_obj.feat_info):

            if col_name != self.col_name_cond:

                if col_type == "categ": # categorical (uses embeddings)
                    input_list.append(self.feat_embedd[col_name](x_data[:,feat_idx].long()))

                elif col_type == "real": # numerical
                    input_list.append(x_data[:,feat_idx].view(-1,1))

        return torch.cat(input_list, 1)


    def forward(self, x_data):

        input_values = self.get_inputs(x_data)
        out_nn = self.pred_nn(input_values)

        p_params = dict()
        p_params['x'] = out_nn
        if self.col_type_cond == 'real':
            p_params['logvar_x'] = self.logvar_x.clamp(-3,3)

        return p_params

    def loss_function(self, input_data, p_params):

        feat_select = self.col_idx
        feat_size = self.col_size_cond

        if self.col_type_cond == 'categ':
            nll_val = nll_categ_global(p_params['x'],
                                       input_data[:,feat_select].long(),
                                       feat_size,
                                       isRobust=False).sum()

        elif self.col_type_cond == 'real':
            nll_val = nll_gauss_global(p_params['x'].view(-1,1), # 2
                                       input_data[:,feat_select],
                                       p_params['logvar_x'],
                                       isRobust=False).sum()

        return nll_val # / float(input_data.shape[0])


# Conditional Predictor Baseline (Analogous to Pseudo-Likelihood)
class CondPred(nn.Module):

    """ Note: Use in tabular data only, not images -- overcapacity """

    def __init__(self, dataset_obj, args_in):

        super(CondPred, self).__init__()

        self.dataset_obj = dataset_obj
        self.args_in = args_in

        self.size_input = len(dataset_obj.cat_cols)*self.args_in.embedding_size + len(dataset_obj.num_cols)
        self.size_output = len(dataset_obj.cat_cols) + len(dataset_obj.num_cols) # 2

        self.cond_models = nn.ModuleDict([(col_name, SinglePred(dataset_obj, args_in, col_name))
                                            for col_name, col_type, col_size in dataset_obj.feat_info])

    def forward(self, x_data):

        out_cond_list = []
        for col_name, _, _ in self.dataset_obj.feat_info:
            p_params_cond = self.cond_models[col_name](x_data)
            loss_cond = self.cond_models[col_name].loss_function(x_data, p_params_cond)

            out_cond_list.append((col_name, (p_params_cond,loss_cond)))

        return out_cond_list



def get_params_matrix(args_in, out_condpred_list):

    p_params_mtx = dict()
    list_params_cond_x = []
    list_params_cond_logvar_x = []

    for feat_idx, (_, col_type, _) in enumerate(args_in.dataset_defs.feat_info):
        list_params_cond_x.append(out_condpred_list[feat_idx][1][0]['x'])
        if col_type == 'real':
            list_params_cond_logvar_x.append(out_condpred_list[feat_idx][1][0]['logvar_x'].view(-1,1))

    p_params_mtx['x'] = torch.cat(list_params_cond_x,1)
    if list_params_cond_logvar_x:
        p_params_mtx['logvar_x'] = torch.cat(list_params_cond_logvar_x,1)

    return p_params_mtx


def error_computation(model, X_true, X_hat, mask, x_input_size=False):

    # This function computes the proper error for each type of variable
    use_device = "cuda" if model.args_in.cuda_on else "cpu"

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

                error_brier = utils.brier_score(reconstructed_feature[select_cell_pos],
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


def repair_phase(args, model, data_dirty, data_clean, mask):

    mask_b = mask.bool()

    model.eval()
    with torch.no_grad():
        out_cpred_list_xc = model(data_clean)
        p_params_xc = get_params_matrix(args, out_cpred_list_xc)

        out_cpred_list_xd = model(data_dirty)
        p_params_xd = get_params_matrix(args, out_cpred_list_xd)

    # error (MSE) lower bound, on dirty cell positions only
    error_lb_dc, error_lb_dc_per_feat = error_computation(model, data_clean, p_params_xc['x'], mask_b) # x_truth - f_vae(x_clean)

    # error repair, on dirty cell positions only
    error_repair_dc, error_repair_dc_per_feat = error_computation(model, data_clean, p_params_xd['x'], mask_b) # x_truth - f_vae(x_dirty)

    # error upper bound, on dirty cell positions only
    error_up_dc, error_up_dc_per_feat = error_computation(model, data_clean, data_dirty, mask_b, x_input_size=True) # x_truth - x_dirty

    # error on clean cell positions only (to test impact on dirty cells on clean cells under model)
    error_repair_cc, error_repair_cc_per_feat = error_computation(model, data_clean, p_params_xd['x'], ~mask_b)


    # Get NLLloss only for cc and dc positions, under dirty data
    dict_slice = lambda dict_op, row_pos: {key:(value[row_pos,:] \
        if value.shape[0]==data_dirty.shape[0] else value) for key, value in dict_op.items()}

    # dc
    dirty_row_pos = mask_b.any(dim=1)
    n_dirty_rows = dirty_row_pos.sum().item()
    # nll
    nll_cond_dc = 0.0
    for col_name, (p_params_cond_xd, _) in out_cpred_list_xd:
        p_params_cond_xd_sliced = dict_slice(p_params_cond_xd, dirty_row_pos)
        nll_cond_dc += model.cond_models[col_name].loss_function(data_clean[dirty_row_pos,:], p_params_cond_xd_sliced).item()

    # cc
    clean_row_pos = ~dirty_row_pos
    n_clean_rows = clean_row_pos.sum().item()
    # nll
    nll_cond_cc = 0.0
    for col_name, (p_params_cond_xd, _) in out_cpred_list_xd:
        p_params_cond_xd_sliced = dict_slice(p_params_cond_xd, clean_row_pos)
        nll_cond_cc += model.cond_models[col_name].loss_function(data_clean[clean_row_pos,:], p_params_cond_xd_sliced).item()


    eval_data_len = data_dirty.shape[0]
    losses = OrderedDict([('eval_nll_final_clean_dc',nll_cond_dc/n_dirty_rows),
                          ('eval_nll_final_clean_cc',nll_cond_cc/n_clean_rows),
                          ('eval_nll_final_clean_all',(nll_cond_cc+nll_cond_dc)/eval_data_len),
                          ('mse_lower_bd_dirtycells', error_lb_dc.item()), ('mse_upper_bd_dirtycells', error_up_dc.item()), ('mse_repair_dirtycells', error_repair_dc.item()),
                          ('mse_repair_cleancells', error_repair_cc.item()),
                          ('errors_per_feature', [error_lb_dc_per_feat.cpu().numpy(), error_repair_dc_per_feat.cpu().numpy(),
                                                  error_up_dc_per_feat.cpu().numpy(), error_repair_cc_per_feat.cpu().numpy()])])

    return losses


def training_phase(args_in, model, optimizer_dict, data_loader, epoch):

    model.train()

    train_loss = 0.0

    for batch_idx, unpack in enumerate(data_loader):

        data_input = unpack[0]

        if args.cuda_on:
            data_input = data_input.cuda()

        for col_idx, (col_name, col_optim) in enumerate(optimizer_dict.items()):
            col_optim.zero_grad()

        out_condpred_list = model(data_input)

        loss = 0.0
        for col_idx, (col_name, col_optim) in enumerate(optimizer_dict.items()):

            out_condpred_list[col_idx][1][1].backward()

            loss += out_condpred_list[col_idx][1][1].item()
            train_loss += out_condpred_list[col_idx][1][1].item()

            col_optim.step()

        if batch_idx % args_in.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_input), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader),
                    loss / len(data_input)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))

    return train_loss



def evaluation_phase(args_in, model, data_eval, data_clean, target_errors, losses_save,
                     epoch_numb, mode='train'):

    model.eval()
    with torch.no_grad():
        condpred_list = model(data_eval)
    p_params = get_params_matrix(args_in, condpred_list)

    nll_score_mtx = utils.generate_score_outlier_matrix(p_params, data_eval, args_in.dataset_defs)
    nll_score_rows = nll_score_mtx.sum(dim=1)

    # check outlier results
    ## cells
    auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = utils.cell_metrics(target_errors,
                                                                        nll_score_mtx, weights=False)

    ## rows
    auc_row_nll, avpr_row_nll = utils.row_metrics(target_errors, nll_score_mtx, weights=False)

    outlier_metrics = OrderedDict([('auc_cell_nll',auc_cell_nll), ('avpr_cell_nll',avpr_cell_nll),
                                   ('auc_row_nll',auc_row_nll), ('avpr_row_nll',avpr_row_nll),
                                   ('auc_per_feature',auc_vec_nll),
                                   ('avpr_per_feature',avpr_vec_nll)])


    repair_out = repair_phase(args_in, model, data_eval, data_clean, target_errors)

    if epoch_numb>=0:
        print('{} -- Cell AUC: {}, Cell AVPR: {}, Row AUC: {}, Row AVPR: {}\n'.format(
              mode, auc_cell_nll, avpr_cell_nll, auc_row_nll, avpr_row_nll))

    # save to file step
    if args_in.save_on and epoch_numb>=0:
        # save to dataframe table
        losses_save[mode][epoch_numb] = [nll_score_mtx.sum().item()/float(data_eval.shape[0])]
        del outlier_metrics['auc_per_feature']
        del outlier_metrics['avpr_per_feature']
        losses_save[mode][epoch_numb].extend(list(outlier_metrics.values()))
        del repair_out['errors_per_feature']
        losses_save[mode][epoch_numb].extend(list(repair_out.values()))

    return outlier_metrics, repair_out, (nll_score_mtx.cpu().numpy(), nll_score_rows.cpu().numpy())

def store_metrics_final(mode, outlier_scores, dataset_obj, attributes, outlier_metrics, repair_metrics, targets,
                        losses_save,
                        folder_output):

    ## Outlier metrics

    # store AVPR for features (cell only)
    df_avpr_feat_cell = pd.DataFrame([], index=['AVPR'], columns=attributes)
    df_avpr_feat_cell.loc['AVPR'] = outlier_metrics['avpr_per_feature']
    df_avpr_feat_cell.to_csv(folder_output + "/" + mode + "_avpr_features.csv")

    # store AUC for features (cell only)
    df_auc_feat_cell = pd.DataFrame([], index=['AUC'], columns=attributes)
    df_auc_feat_cell.loc['AUC'] = outlier_metrics['auc_per_feature']
    df_auc_feat_cell.to_csv(folder_output + "/" + mode + "_auc_features.csv")

    ## Store data from Epochs (includes repair metrics)
    columns = ['Avg. Loss',
               'AUC Cell', 'AVPR Cell',
               'AUC Row', 'AVPR Row',
               'Avg. Loss -- p(x_clean | x_dirty) on dirty pos',
               'Avg. Loss -- p(x_clean | x_dirty) on clean pos',
               'Avg. Loss -- p(x_clean | x_dirty) on all',
               'Error lower-bound on dirty pos', 'Error upper-bound on dirty pos',
               'Error repair on dirty pos', 'Error repair on clean pos']

    df_out = pd.DataFrame.from_dict(losses_save[mode], orient="index",
                                    columns=columns)
    df_out.index.name = "Epochs"
    df_out.to_csv(folder_output + "/" + mode + "_epochs_data.csv")

    ### Repair: store errors per feature
    df_errors_repair = pd.DataFrame([], index=['error_lowerbound_dirtycells','error_repair_dirtycells',
            'error_upperbound_dirtycells','error_repair_cleancells'], columns=attributes)

    df_errors_repair.loc['error_lowerbound_dirtycells'] = repair_metrics['errors_per_feature'][0]
    df_errors_repair.loc['error_repair_dirtycells'] = repair_metrics['errors_per_feature'][1]
    df_errors_repair.loc['error_upperbound_dirtycells'] = repair_metrics['errors_per_feature'][2]
    df_errors_repair.loc['error_repair_cleancells'] = repair_metrics['errors_per_feature'][3]

    df_errors_repair.to_csv(folder_output + "/" + mode + "_error_repair_features.csv")




def main(args_in):

    #### MAIN ####

    # saving data of experiment to folder is on
    if args_in.save_on:
        # create folder for saving experiment data (if necessary)
        folder_output = args_in.output_folder + "/"

        try:
            os.makedirs(folder_output)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # structs for saving data
        losses_save = {"train":{},"test":{}}


    # dtype definitions for runing
    if args_in.cuda_on:
        dtype_float = torch.cuda.FloatTensor
        dtype_byte = torch.cuda.ByteTensor
    else:
        dtype_float = torch.FloatTensor
        dtype_byte = torch.ByteTensor

    print(args_in)

    # Load datasets
    train_loader, X_train, target_errors_train, dataset_obj_train, attributes = utils.load_data(args_in.data_folder, args_in.batch_size,
                                                                is_train=True, get_data_idxs=True)
    args_in.dataset_defs = dataset_obj_train
    X_train = X_train.type(dtype_float)

    test_loader, X_test, target_errors_test, dataset_obj_test, _ = utils.load_data(args_in.data_folder, args_in.batch_size, is_train=False)
    X_test = X_test.type(dtype_float)

    # -- clean versions for data repair evaluation (standardized according to the dirty data statistics)
    train_loader_clean, X_train_clean, _, dataset_obj_clean, _ = utils.load_data(args_in.data_folder, args_in.batch_size,
                                                                is_train=True, is_clean=True, stdize_dirty=True)
    X_train_clean = X_train_clean.type(dtype_float)

    test_loader_clean, X_test_clean, _, _, _ = utils.load_data(args_in.data_folder, args_in.batch_size, is_train=False,
                                                                is_clean=True, stdize_dirty=True)
    X_test_clean = X_test_clean.type(dtype_float)


    ### Run CondPred Model ###
    model = CondPred(args_in.dataset_defs, args_in)

    if args_in.cuda_on:
        model.cuda()


    # define optimizers for each cond pred model
    optimizer_dict = OrderedDict()
    for col_name, col_type, col_size in args_in.dataset_defs.feat_info:
        if args_in.base_type == 'linear':
            optimizer_dict[col_name] = optim.SGD(model.cond_models[col_name].parameters(),
                                                lr=args_in.lr,
                                                weight_decay=args_in.l2_reg,
                                                nesterov=args_in.nest_mom, # default: False
                                                momentum=args_in.mom_val)
        else:
            optimizer_dict[col_name] = optim.Adam(model.cond_models[col_name].parameters(),
                                                lr=args_in.lr,
                                                weight_decay=args_in.l2_reg)


    # Run epochs
    for epoch in range(1, args_in.number_epochs + 1):

        training_phase(args_in, model, optimizer_dict, train_loader, epoch)

        # Train set evaluation
        evaluation_phase(args_in, model, X_train, X_train_clean, target_errors_train,
                         losses_save, epoch, mode='train')
        # Test set evaluation
        evaluation_phase(args_in, model, X_test, X_test_clean, target_errors_test,
                         losses_save, epoch, mode='test')


    if args_in.save_on:

        ### Train Data
        outlier_metrics_train, repair_metrics_train, outlier_scores_train = \
            evaluation_phase(args_in, model, X_train, X_train_clean, target_errors_train, [], -1, mode='train')
            # (outlier_score_cells_train, outlier_score_rows_train)

        store_metrics_final('train', outlier_scores_train, args_in.dataset_defs, attributes, outlier_metrics_train, repair_metrics_train,
                            target_errors_train, losses_save,
                            folder_output)

        ### Test Data
        outlier_metrics_test, repair_metrics_test, outlier_scores_test = \
            evaluation_phase(args_in, model, X_test, X_test_clean, target_errors_test, [], -1, mode='test')

        store_metrics_final('test', outlier_scores_test, args_in.dataset_defs, attributes, outlier_metrics_test, repair_metrics_test,
                            target_errors_test, losses_save,
                            folder_output)


        # save model parameters
        model.cpu()
        torch.save(model.state_dict(), folder_output + "/model_params.pth")

        # remove non-serializable stuff
        del args_in.dataset_defs

        # save to .json file the args that were used for running the model
        with open(folder_output + "/args_run.json", "w") as outfile:
            json.dump(vars(args_in), outfile, indent=4, sort_keys=True)



if __name__ == '__main__':

    args = parser_arguments_CondPred.getArgs(sys.argv[1:])
    main(args)
