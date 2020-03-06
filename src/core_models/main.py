#!/usr/bin/env python3

import sys
# sys.path.append("..")

#import argparse
import json
import os, errno
import pandas as pd
import numpy as np

import torch
from torch import optim

import utils
import parser_arguments
from train_eval_models import training_phase, evaluation_phase, repair_phase


def compute_metrics(model, X, dataset_obj, args, epoch, losses_save,
                    logit_pi_prev, X_clean, target_errors, mode):

    # get epoch metrics on outlier detection for train dataset
    if args.outlier_model == "VAE":
        # outlier analysis
        loss_ret, metric_ret = evaluation_phase(model, X, dataset_obj, args, epoch)

        # repair analysis
        clean_loss_ret = repair_phase(model, X, X_clean, dataset_obj, args, target_errors, mode, epoch)

    else:
        # outlier analysis
        loss_ret, metric_ret = evaluation_phase(model, X, dataset_obj, args, epoch,
                                                            clean_comp_show=True,
                                                            logit_pi_prev=logit_pi_prev,
                                                            w_conv=True,
                                                            mask_err=target_errors)

        # repair analysis
        clean_loss_ret = repair_phase(model, X, X_clean, dataset_obj, args, target_errors, mode, epoch)

    print('\n\n\n\n')
    print('====> ' + mode + ' set: Epoch: {} Avg. AVI loss: {:.3f}\tAvg. AVI NLL: {:.3f}\tAvg. AVI KLD_Z: {:.3f}\tAvg. AVI KLD_W: {:.3f}'.format(
          epoch, loss_ret['eval_loss_vae'], loss_ret['eval_nll_vae'], loss_ret['eval_z_kld_vae'], loss_ret['eval_w_kld_vae']))

    print('\n')
    print('====> ' + mode + ' set: -- clean component | reparability (all data): p_recon(x_clean | x_dirty) -- \n \t\t Epoch: {} Avg. loss: {:.3f}\tAvg. NLL: {:.3f}\tAvg. KLD_Z: {:.3f}\tAvg. KLD_W: {:.3f}'.format(
          epoch, clean_loss_ret['eval_loss_final_clean_all'], clean_loss_ret['eval_nll_final_clean_all'],
          clean_loss_ret['eval_z_kld_final_clean_all'], clean_loss_ret['eval_w_kld_final_clean_all']))

    print('====> ' + mode + ' set: -- clean component | reparability (dirty pos): p_recon(x_clean | x_dirty) -- \n \t\t Epoch: {} Avg. loss: {:.3f}\tAvg. NLL: {:.3f}\tAvg. KLD_Z: {:.3f}\tAvg. KLD_W: {:.3f}'.format(
          epoch, clean_loss_ret['eval_loss_final_clean_dc'], clean_loss_ret['eval_nll_final_clean_dc'],
          clean_loss_ret['eval_z_kld_final_clean_dc'], clean_loss_ret['eval_w_kld_final_clean_dc']))


    print('====> ' + mode + ' set: cell error (lower bound dirty pos): {:.3f}, cell error (upper bound dirty pos): {:.3f}, cell error (repair dirty pos): {:.3f}, cell error (repair clean pos): {:.3f}'.format(
          clean_loss_ret['mse_lower_bd_dirtycells'], clean_loss_ret['mse_upper_bd_dirtycells'], clean_loss_ret['mse_repair_dirtycells'], clean_loss_ret['mse_repair_cleancells']))


    if args.inference_type == 'seqvae':
        print('\n')
        print('\n\nAdditional Info: Avg. SEQ-VAE Total loss: {:.3f}\tAvg. SEQ-VAE loss: {:.3f}\tAvg. SEQ-VAE NLL: {:.3f}\tAvg. SEQ-VAE KLD_Z: {:.3f}\tAvg. SEQ-VAE KLD_W: {:.3f}'.format(
              loss_ret['eval_total_loss_seq'], loss_ret['eval_loss_seq'], loss_ret['eval_nll_seq'], loss_ret['eval_z_kld_seq'], loss_ret['eval_w_kld_seq']))


    if args.outlier_model == "RVAE":
        print('\n\n')
        print('====> ' + mode + ' set: -- clean component: p_recon(x_dirty | x_dirty) -- \n \t\t Epoch: {} Avg. loss: {:.3f}\tAvg. NLL: {:.3f}\tAvg. KLD_Z: {:.3f}\tAvg. KLD_W: {:.3f}'.format(
              epoch, loss_ret['eval_loss_final_clean'], loss_ret['eval_nll_final_clean'],
              loss_ret['eval_z_kld_final_clean'], loss_ret['eval_w_kld_final_clean']))


    # calc cell metrics
    auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = utils.cell_metrics(target_errors, metric_ret['nll_score'], weights=False)
    if args.outlier_model == "RVAE":
        auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = utils.cell_metrics(target_errors, metric_ret['pi_score'], weights=True)
    else:
        auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = 4*[-10]

    # calc row metrics
    auc_row_nll, avpr_row_nll = utils.row_metrics(target_errors, metric_ret['nll_score'], weights=False)
    if args.outlier_model == "RVAE":
        auc_row_pi, avpr_row_pi = utils.row_metrics(target_errors, metric_ret['pi_score'], weights=True)
    else:
        auc_row_pi, avpr_row_pi = 2*[-10]


    if args.verbose_metrics_epoch:
        print('         (Cell) Avg. ' + mode + ' AUC: {} '.format(auc_cell_nll))
        print('         (Cell) Avg. ' + mode + ' AVPR: {} '.format(avpr_cell_nll))
        print("\n\n")
        if args.verbose_metrics_feature_epoch:
            print('         AUC per feature: \n {}'.format(auc_vec_nll))
            print('         AVPR per feature: \n {}'.format(avpr_vec_nll))
            print("\n\n")
        print('         (Row) ' + mode + ' AUC: {} '.format(auc_row_nll))
        print('         (Row) ' + mode + ' AVPR: {} '.format(avpr_row_nll))

        if args.outlier_model == "RVAE":
            print('         (Cell) Avg. ' + mode + ' AUC: {} '.format(auc_cell_pi))
            print('         (Cell) Avg. ' + mode + ' AVPR: {} '.format(avpr_cell_pi))
            print("\n\n")
            if args.verbose_metrics_feature_epoch:
                print('         AUC per feature: \n {}'.format(auc_vec_pi))
                print('         AVPR per feature: \n {}'.format(avpr_vec_pi))
                print("\n\n")
            print('         (Row) ' + mode + ' AUC: {} '.format(auc_row_pi))
            print('         (Row) ' + mode + ' AVPR: {} '.format(avpr_row_pi))


    # save to file step
    if args.save_on:
        if args.inference_type == 'vae':

            loss_ret.update(dict.fromkeys(['eval_loss_seq','eval_nll_seq',
                                           'eval_z_kld_seq','eval_w_kld_seq'],-10))

        if args.outlier_model == "VAE":
            loss_ret.update(dict.fromkeys(['eval_loss_final_clean','eval_nll_final_clean',
                                           'eval_z_kld_final_clean','eval_w_kld_final_clean'],-10))

            clean_loss_ret.update(dict.fromkeys(['eval_loss_final_clean','eval_nll_final_clean',
                                           'eval_z_kld_final_clean','eval_w_kld_final_clean'],-10))

        losses_save[mode][epoch] = [loss_ret['eval_loss_vae'], loss_ret['eval_nll_vae'],
                                    loss_ret['eval_z_kld_vae'], loss_ret['eval_w_kld_vae'],
                                    loss_ret['eval_loss_seq'], loss_ret['eval_nll_seq'],
                                    loss_ret['eval_z_kld_seq'], loss_ret['eval_w_kld_seq'],
                                    loss_ret['eval_loss_final_clean'], loss_ret['eval_nll_final_clean'],
                                    loss_ret['eval_z_kld_final_clean'], loss_ret['eval_w_kld_final_clean'],
                                    clean_loss_ret['eval_loss_final_clean_dc'], clean_loss_ret['eval_nll_final_clean_dc'],
                                    clean_loss_ret['eval_z_kld_final_clean_dc'], clean_loss_ret['eval_w_kld_final_clean_dc'],
                                    clean_loss_ret['eval_loss_final_clean_cc'], clean_loss_ret['eval_nll_final_clean_cc'],
                                    clean_loss_ret['eval_z_kld_final_clean_cc'], clean_loss_ret['eval_w_kld_final_clean_cc'],
                                    clean_loss_ret['eval_loss_final_clean_all'], clean_loss_ret['eval_nll_final_clean_all'],
                                    clean_loss_ret['eval_z_kld_final_clean_all'], clean_loss_ret['eval_w_kld_final_clean_all'],
                                    metric_ret['converg_norm_w'], auc_cell_nll, avpr_cell_nll, auc_row_nll, avpr_row_nll,
                                    auc_cell_pi, avpr_cell_pi, auc_row_pi, avpr_row_pi,
                                    clean_loss_ret['mse_lower_bd_dirtycells'], clean_loss_ret['mse_upper_bd_dirtycells'],
                                    clean_loss_ret['mse_repair_dirtycells'], clean_loss_ret['mse_repair_cleancells']]



def save_to_csv(model, X_data, X_data_clean, target_errors, attributes, losses_save,
                dataset_obj, folder_output, args, epoch, mode='train'):

    """ This method performs all operations needed to save the data to csv """

    #Create saving folderes
    try:
        os.makedirs(folder_output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    ### Evaluate model
    _, metric_ret = evaluation_phase(model, X_data, dataset_obj, args, epoch)

    clean_loss_ret = repair_phase(model, X_data, X_data_clean, dataset_obj, args, target_errors, mode, epoch)


    ## calc cell metrics
    auc_cell_nll, auc_vec_nll, avpr_cell_nll, avpr_vec_nll = utils.cell_metrics(target_errors, metric_ret['nll_score'], weights=False)
    if args.outlier_model == "RVAE":
        auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = utils.cell_metrics(target_errors, metric_ret['pi_score'], weights=True)
    else:
        auc_cell_pi, auc_vec_pi, avpr_cell_pi, avpr_vec_pi = -10, np.zeros(len(attributes))*-10, -10, np.zeros(len(attributes))*-10


    # store AVPR for features (cell only)
    df_avpr_feat_cell = pd.DataFrame([], index=['AVPR_nll', 'AVPR_pi'], columns=attributes)
    df_avpr_feat_cell.loc['AVPR_nll'] = avpr_vec_nll
    df_avpr_feat_cell.loc['AVPR_pi'] = avpr_vec_pi
    df_avpr_feat_cell.to_csv(folder_output + "/" + mode + "_avpr_features.csv")

    # store AUC for features (cell only)
    df_auc_feat_cell = pd.DataFrame([], index=['AUC_nll', 'AUC_pi'], columns=attributes)
    df_auc_feat_cell.loc['AUC_nll'] = auc_vec_nll
    df_auc_feat_cell.loc['AUC_pi'] = auc_vec_pi
    df_auc_feat_cell.to_csv(folder_output + "/" + mode + "_auc_features.csv")

    ### Store data from Epochs
    columns = ['Avg. AVI Loss', 'Avg. AVI NLL', 'Avg. AVI KLD_Z', 'Avg. AVI KLD_W',
               'Avg. SEQ Loss', 'Avg. SEQ NLL', 'Avg. SEQ KLD_Z', 'Avg. SEQ KLD_W',
               'Avg. Loss -- p(x_dirty | x_dirty) on all', 'Avg. NLL -- p(x_dirty | x_dirty) on all', 'Avg. KLD_Z -- p(x_dirty | x_dirty) on all', 'Avg. KLD_W -- p(x_dirty | x_dirty) on all',
               'Avg. Loss -- p(x_clean | x_dirty) on dirty pos', 'Avg. NLL -- p(x_clean | x_dirty) on dirty pos', 'Avg. KLD_Z -- p(x_clean | x_dirty) on dirty pos', 'Avg. KLD_W -- p(x_clean | x_dirty) on dirty pos',
               'Avg. Loss -- p(x_clean | x_dirty) on clean pos', 'Avg. NLL -- p(x_clean | x_dirty) on clean pos', 'Avg. KLD_Z -- p(x_clean | x_dirty) on clean pos', 'Avg. KLD_W -- p(x_clean | x_dirty) on clean pos',
               'Avg. Loss -- p(x_clean | x_dirty) on all', 'Avg. NLL -- p(x_clean | x_dirty) on all', 'Avg. KLD_Z -- p(x_clean | x_dirty) on all', 'Avg. KLD_W -- p(x_clean | x_dirty) on all',
               'W Norm Convergence', 'AUC Cell nll score', 'AVPR Cell nll score', 'AUC Row nll score', 'AVPR Row nll score',
               'AUC Cell pi score', 'AVPR Cell pi score', 'AUC Row pi score', 'AVPR Row pi score',
               'Error lower-bound on dirty pos', 'Error upper-bound on dirty pos', 'Error repair on dirty pos', 'Error repair on clean pos']

    df_out = pd.DataFrame.from_dict(losses_save[mode], orient="index",
                                    columns=columns)
    df_out.index.name = "Epochs"
    df_out.to_csv(folder_output + "/" + mode + "_epochs_data.csv")

    ### Store errors per feature

    df_errors_repair = pd.DataFrame([], index=['error_lowerbound_dirtycells','error_repair_dirtycells',
            'error_upperbound_dirtycells','error_repair_cleancells'], columns=attributes)
    df_errors_repair.loc['error_lowerbound_dirtycells'] = clean_loss_ret['errors_per_feature'][0].cpu()
    df_errors_repair.loc['error_repair_dirtycells'] = clean_loss_ret['errors_per_feature'][1].cpu()
    df_errors_repair.loc['error_upperbound_dirtycells'] = clean_loss_ret['errors_per_feature'][2].cpu()
    df_errors_repair.loc['error_repair_cleancells'] = clean_loss_ret['errors_per_feature'][3].cpu()
    df_errors_repair.to_csv(folder_output + "/" + mode + "_error_repair_features.csv")


# Running Options:
#
#   RVAE-AVI: + is args.outlier_model='RVAE'; args.inference_type='vae';
#               args.AVI=True;
#
#   RVAE-CVI: + is args.outlier_model='RVAE'; args.inference_type='vae';
#               args.AVI=False
#
#   VAE: + is outlier_model='VAE'; args.inference_type='vae';
#
#   SEQ-RVAE-CVI (MCMC - sequence: Pseudo-Gibbs): + is outlier_model='RVAE';
#                        args.inference_type='seq_vae'; args.AVI=False
#
#

def main(args):

    # Load datasets
    train_loader, X_train, target_errors_train, dataset_obj, attributes = utils.load_data(args.data_folder, args.batch_size,
                                                                                        is_train=True,
                                                                                        get_data_idxs=False)

    test_loader, X_test, target_errors_test, _, _ = utils.load_data(args.data_folder, args.batch_size, is_train=False)
    # -- clean versions for evaluation
    _, X_train_clean, _, _, _ = utils.load_data(args.data_folder, args.batch_size,
                                                                    is_train=True, is_clean=True, stdize_dirty=True)
    _, X_test_clean, _, _, _ = utils.load_data(args.data_folder, args.batch_size, is_train=False,
                                                                is_clean=True, stdize_dirty=True)

    # if runnin on gpu, then load data there
    if args.cuda_on:
        X_test = X_test.cuda()
        target_errors_test = target_errors_test.cuda()
        X_train_clean = X_train_clean.cuda()
        X_test_clean = X_test_clean.cuda()
        target_errors_train = target_errors_train.cuda()
        X_train = X_train.cuda()


    # for checking w (pi) raw convergence
    logit_pi_prev_train = torch.tensor([])
    logit_pi_prev_test = torch.tensor([])


    # Import the model from the correct file
    outlier_model = __import__(args.outlier_model)
    model = outlier_model.VAE(dataset_obj, args)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))

    print(args)

    if args.cuda_on:
        model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.l2_reg)  # excludes frozen params / layers


    # structs for saving data
    losses_save = {"train":{},"test":{}, "train_per_feature":{}, "test_per_feature":{}}

    # Run epochs
    for epoch in range(1, args.number_epochs + 1):

        # Training Phase
        _train_loader, _dataset_obj = train_loader, dataset_obj

        training_phase(model, optimizer, _train_loader, args, epoch)

        #Compute all the losses and metrics per epoch (Train set)
        compute_metrics(model, X_train, _dataset_obj, args, epoch, losses_save,
                        logit_pi_prev_train, X_train_clean, target_errors_train, mode="train")

        #Test Phase
        compute_metrics(model, X_test, dataset_obj, args, epoch, losses_save,
                        logit_pi_prev_test, X_test_clean, target_errors_test, mode="test")


    # save to folder AVPR / AUC per feature
    if args.save_on:

        # create folder for saving experiment data (if necessary)
        folder_output = args.output_folder + "/" + args.outlier_model

        ### Train Data
        save_to_csv(model, X_train, X_train_clean, target_errors_train, attributes, losses_save,
                dataset_obj, folder_output, args, epoch, mode='train')


        ### Test Data
        save_to_csv(model, X_test, X_test_clean, target_errors_test, attributes, losses_save,
                dataset_obj, folder_output, args, epoch, mode='test')


        # save model parameters
        model.cpu()
        torch.save(model.state_dict(), folder_output + "/model_params.pth")

        # save to .json file the args that were used for running the model
        with open(folder_output + "/args_run.json", "w") as outfile:
            json.dump(vars(args), outfile, indent=4, sort_keys=True)



if __name__ == '__main__':

    args = parser_arguments.getArgs(sys.argv[1:])

    main(args)
