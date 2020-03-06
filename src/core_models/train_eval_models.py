#!/usr/bin/env python3

import torch
from torch import optim
import torch.nn.functional as F

import argparse
from sklearn.metrics import mean_squared_error
import numpy as np
import json

import utils
from model_utils import get_pi_exact_vec, rnn_vae_forward_one_stage, rnn_vae_forward_two_stage


def training_phase(model, optimizer, train_loader, args, epoch):

    model.train()

    train_loss_vae, train_nll_vae, train_z_kld_vae, train_w_kld_vae = 4*[0]
    train_loss_seq, train_nll_seq, train_z_kld_seq, train_w_kld_seq = 4*[0]

    train_total_loss_seq_vae, train_loss_seq_vae, train_nll_seq_vae, train_z_kld_seq_vae, train_w_kld_seq_vae = 5*[0]

    for batch_idx, unpack in enumerate(train_loader):

        data_input = unpack[0]

        if args.cuda_on:
            data_input = data_input.cuda()

        optimizer.zero_grad()

        ## first foward-pass
        p_params, q_params, q_samples = model(data_input, n_epoch=epoch-1)

        if not args.AVI:
            get_pi_exact_vec(model, data_input, p_params, q_params, args, logit_ret=True) # get pi, saves to q_params (with no_grad)

        vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(data_input, p_params, q_params, q_samples)

        train_loss_vae += vae_loss.item()
        train_nll_vae += vae_nll.item()
        train_z_kld_vae += vae_z_kld.item()
        train_w_kld_vae += vae_w_kld.item()

        if args.inference_type == 'vae':

            vae_loss.backward()

        elif args.inference_type == 'seqvae':

            if args.seqvae_bprop: # NOTE: rolls out iterations through time and bprops

                params_in = (p_params, q_params, q_samples)

                seq_loss_pack, _, _ = rnn_vae_forward_one_stage(params_in, data_input, model, vae_loss, args,
                                                                number_steps=args.seqvae_steps, loss_per_iter=True, epoch_id=epoch)
                seq_total_loss, seq_final_loss, seq_final_nll, seq_final_z_kld, seq_final_w_kld = seq_loss_pack

                train_total_loss_seq_vae += seq_total_loss.item()
                train_loss_seq_vae += seq_final_loss.item()
                train_nll_seq_vae += seq_final_nll.item()
                train_z_kld_seq_vae += seq_final_z_kld.item()
                train_w_kld_seq_vae += seq_final_w_kld.item()

            else:
                vae_loss.backward()

                train_total_loss_seq_vae += vae_loss.item()
                train_loss_seq_vae += vae_loss.item()
                train_nll_seq_vae += vae_nll.item()
                train_z_kld_seq_vae += vae_z_kld.item()
                train_w_kld_seq_vae += vae_w_kld.item()


                seq_total_loss = torch.tensor(0.0)
                seq_final_loss = torch.tensor(0.0)
                seq_final_nll = torch.tensor(0.0)
                seq_final_z_kld = torch.tensor(0.0)
                seq_final_w_kld = torch.tensor(0.0)


        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('\n\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tVAE Loss: {:.3f}\tVAE NLL: {:.3f}\tVAE KLD_Z: {:.3f}\tVAE KLD_W: {:.3f}'.format(
                  epoch, batch_idx * len(data_input), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  vae_loss.item()/len(data_input), vae_nll.item()/len(data_input),
                  vae_z_kld.item()/len(data_input), vae_w_kld.item()/len(data_input)))

            if args.inference_type == 'seqvae':
                print('\n')
                print('\n\nAdditional Info:\tTotal Seq Loss: {:.3f}\tFinal Seq Loss: {:.3f}\tFinal Sep NLL: {:.3f}\tFinal Sep KLD_Z: {:.3f}\tFinal Sep KLD_W: {:.3f}\n'.format(
                      seq_total_loss.item()/len(data_input), seq_final_loss.item()/len(data_input),
                      seq_final_nll.item()/len(data_input), seq_final_z_kld.item()/len(data_input),
                      seq_final_w_kld.item()/len(data_input)))


    dataset_len = float(len(train_loader.dataset))

    ret = {'train_loss_vae': train_loss_vae/dataset_len, 'train_nll_vae': train_nll_vae/dataset_len,
           'train_z_kld_vae': train_z_kld_vae/dataset_len, 'train_w_kld_vae': train_w_kld_vae/dataset_len}

    if args.inference_type == "seqvae":
        ret_seq = {'train_loss_seq': train_loss_seq_vae/dataset_len, 'train_nll_seq': train_nll_seq_vae/dataset_len,
                   'train_z_kld_seq': train_z_kld_seq_vae/dataset_len,'train_w_kld_seq': train_w_kld_seq_vae/dataset_len,
                   'train_total_loss_seq':train_total_loss_seq_vae/dataset_len}
        ret = {**ret, **ret_seq}

    return ret


def evaluation_phase(model, data_eval, dataset_obj, args, epoch,
                     clean_comp_show=False, data_eval_clean=False, logit_pi_prev=torch.tensor([]), w_conv=False, mask_err=None):

    # if args.cuda_on:
    #     model.cpu()

    if type(mask_err) != type(None):
        mask_err = mask_err.bool()

    model.eval()

    p_params, q_params, q_samples = model(data_eval)

    if not args.AVI:
        get_pi_exact_vec(model, data_eval, p_params, q_params, args, logit_ret=True) # get pi

    vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(data_eval, p_params,
                                                                  q_params, q_samples)

    eval_data_len = data_eval.shape[0]
    losses = {'eval_loss_vae': vae_loss.item()/eval_data_len, 'eval_nll_vae':vae_nll.item()/eval_data_len,
              'eval_z_kld_vae': vae_z_kld.item()/eval_data_len, 'eval_w_kld_vae':vae_w_kld.item()/eval_data_len}


    # SEQ-VAE
    if args.inference_type == 'seqvae':
        #with torch.no_grad():
        params_in = (p_params, q_params, q_samples)

        if args.seqvae_two_stage:
            seq_loss_pack, _, seq_param_pack  = rnn_vae_forward_two_stage(params_in, data_eval, model, vae_loss, args,
                                                    number_steps=args.seqvae_steps, number_steps_second_stage=args.steps_2stage,
                                                    loss_per_iter=True, mask_err=mask_err, epoch_id=epoch)
        else:
            seq_loss_pack, _, seq_param_pack  = rnn_vae_forward_one_stage(params_in, data_eval, model, vae_loss, args,
                                                    number_steps=args.seqvae_steps, loss_per_iter=True, mask_err=mask_err, epoch_id=epoch)

        seq_total_loss, seq_final_loss, seq_final_nll, seq_final_z_kld, seq_final_w_kld = seq_loss_pack
        p_params_final, q_params_final, q_samples_final = seq_param_pack

        losses_seq_vae = {'eval_loss_seq': seq_final_loss.item()/eval_data_len, 'eval_nll_seq': seq_final_nll.item()/eval_data_len,
                          'eval_z_kld_seq': seq_final_z_kld.item()/eval_data_len, 'eval_w_kld_seq': seq_final_w_kld.item()/eval_data_len,
                          'eval_total_loss_seq': seq_total_loss.item()/eval_data_len}

        losses = {**losses, **losses_seq_vae}


    if args.inference_type == 'seqvae':
        p_params_metric, q_params_metric, q_samples_metric = p_params_final, q_params_final, q_samples_final
    else:
        p_params_metric, q_params_metric, q_samples_metric = p_params, q_params, q_samples

    #Getting scores and clean component if neededin_aux_samples
    with torch.no_grad():
        if args.outlier_model == "VAE": # VAE models only (no w's or pi's)

            # generative model only p(x|z, ...)
            nll_score_mat = utils.generate_score_outlier_matrix(p_params_metric, data_eval, dataset_obj)

            pi_score_mat = -10
            converg_norm_w = -10

        else:
            if clean_comp_show:
                loss_clean, nll_clean, z_kld_clean, w_kld_clean = model.loss_function(data_eval, p_params_metric,
                                                                                      q_params_metric, q_samples_metric,
                                                                                      clean_comp_only=True,
                                                                                      data_eval_clean=data_eval_clean)

                losses_add = {'eval_loss_final_clean': loss_clean.item()/eval_data_len,
                              'eval_nll_final_clean': nll_clean.item()/eval_data_len,
                              'eval_z_kld_final_clean': z_kld_clean.item()/eval_data_len,
                              'eval_w_kld_final_clean': w_kld_clean.item()/eval_data_len
                             }

                losses = {**losses, **losses_add}

            # q(w|x, ...) param (pi), used in outlier score
            pi_score_mat = torch.sigmoid(q_params_metric['w']['logit_pi']).clamp(1e-6, 1-1e-6)

            # -log p(x|z, ...) used as outlier score
            nll_score_mat = utils.generate_score_outlier_matrix(p_params_metric, data_eval, dataset_obj)

            # check convergence of weights (pi's)
            if w_conv:
                if logit_pi_prev.nelement() == 0:
                    logit_pi_prev = torch.zeros_like(q_params_metric['w']['logit_pi'])
                converg_norm_w = (q_params_metric['w']['logit_pi'] - logit_pi_prev).norm().item()
                logit_pi_prev = q_params_metric['w']['logit_pi'].clone().detach()
            else:
                converg_norm_w = -10

            # insert here measurement of calibration of pi's using MSE or cross-entropy
            if isinstance(mask_err, torch.Tensor):
                pi_mtx = pi_score_mat
                pi_mtx_true = (~mask_err).float()
                err_pi = ((pi_mtx - pi_mtx_true)**2).mean()
                ce_pi = F.binary_cross_entropy(pi_mtx, pi_mtx_true)
                print('MSE on pi pred: {}'.format(err_pi))
                print('CE on pi pred: {}'.format(ce_pi))
                print('dirt pi median: {} std: {}'.format(torch.sigmoid(q_params_metric['w']['logit_pi'][mask_err]).median(), torch.sigmoid(q_params_metric['w']['logit_pi'][mask_err]).std()))
                print('clean pi median: {} std: {}'.format(torch.sigmoid(q_params_metric['w']['logit_pi'][~mask_err]).median(), torch.sigmoid(q_params_metric['w']['logit_pi'][~mask_err]).std()))

    metrics = {'nll_score': nll_score_mat, 'pi_score': pi_score_mat, 'converg_norm_w': converg_norm_w}

    return losses, metrics


def repair_phase(model, data_dirty, data_clean, dataset_obj, args, mask, mode, epoch):

    model.eval()

    # model params with input: dirty data
    if args.inference_type == 'seqvae':
        p_params_xd, q_params_xd, q_samples_xd = model(data_dirty)
        if not args.AVI:
            get_pi_exact_vec(model, data_dirty, p_params_xd, q_params_xd, args, logit_ret=True)
        params_xd_in = (p_params_xd, q_params_xd, q_samples_xd)

        if args.seqvae_two_stage:
             _, _, (p_params_xd, q_params_xd, q_samples_xd) = rnn_vae_forward_two_stage(params_xd_in, data_dirty, model,
                                                                              torch.tensor(0.0, device=data_dirty.device),
                                                                              args, number_steps=args.seqvae_steps,
                                                                              number_steps_second_stage=args.steps_2stage,
                                                                              loss_per_iter=True, epoch_id=epoch)
        else:
            _, _, (p_params_xd, q_params_xd, q_samples_xd) = rnn_vae_forward_one_stage(params_xd_in, data_dirty, model,
                                                                             torch.tensor(0.0, device=data_dirty.device),
                                                                             args, number_steps=args.seqvae_steps, loss_per_iter=True, epoch_id=epoch)

    else: # standard 'vae' type inference
        p_params_xd, q_params_xd, q_samples_xd = model(data_dirty)
        if not args.AVI:
            get_pi_exact_vec(model, data_dirty, p_params_xd, q_params_xd, args, logit_ret=True) # get pi

    # model params with input: underlying clean data
    if args.inference_type == 'seqvae':
        p_params_xc, q_params_xc, q_samples_xc = model(data_clean)
        if not args.AVI:
            get_pi_exact_vec(model, data_dirty, p_params_xc, q_params_xc, args, logit_ret=True)
        params_xc_in = (p_params_xc, q_params_xc, q_samples_xc)

        if args.seqvae_two_stage:
            _, _, (p_params_xc, q_params_xc, q_samples_xc) = rnn_vae_forward_two_stage(params_xc_in, data_clean, model,
                                                                 torch.tensor(0.0, device=data_clean.device),
                                                                 args, number_steps=args.seqvae_steps,
                                                                 number_steps_second_stage=args.steps_2stage,
                                                                 loss_per_iter=True, epoch_id=epoch)

        else:
            _, _, (p_params_xc, q_params_xc, q_samples_xc) = rnn_vae_forward_one_stage(params_xc_in, data_clean, model,
                                                                torch.tensor(0.0, device=data_clean.device),
                                                                args, number_steps=args.seqvae_steps, loss_per_iter=True, epoch_id=epoch)
    else: # 'vae' type inference
        p_params_xc, q_params_xc, q_samples_xc = model(data_clean)
        # no need to get pi, not used after

    # error (MSE) lower bound, on dirty cell positions only
    error_lb_dc, error_lb_dc_per_feat = utils.error_computation(model, data_clean, p_params_xc['x'], mask) # x_truth - f_vae(x_clean)

    # error repair, on dirty cell positions only
    error_repair_dc, error_repair_dc_per_feat = utils.error_computation(model, data_clean, p_params_xd['x'], mask) # x_truth - f_vae(x_dirty)

    print("\n\n {} REPAIR ERROR (DIRTY POS):{}".format(mode, error_repair_dc))

    # error upper bound, on dirty cell positions only
    error_up_dc, error_up_dc_per_feat = utils.error_computation(model, data_clean, data_dirty, mask, x_input_size=True) # x_truth - x_dirty

    # error on clean cell positions only (to test impact on dirty cells on clean cells under model)
    error_repair_cc, error_repair_cc_per_feat = utils.error_computation(model, data_clean, p_params_xd['x'], 1-mask)

    print("\n\n {} REPAIR ERROR (CLEAN POS):{}".format(mode, error_repair_cc))


    # Get NLL (predict. posterior approx) under dirty data

    dict_slice = lambda dict_op, row_pos: {key:(value[row_pos,:] \
        if value.shape[0]==data_dirty.shape[0] else value) for key, value in dict_op.items()}

    dirty_row_pos = mask.any(dim=1).bool()
    n_dirty_rows = dirty_row_pos.sum().item()

    p_params_xd_sliced = dict_slice(p_params_xd, dirty_row_pos)
    q_params_xd_sliced = dict()
    if args.outlier_model == 'RVAE':
        q_params_xd_sliced['w'] = dict_slice(q_params_xd['w'], dirty_row_pos)
    q_params_xd_sliced['z'] = dict_slice(q_params_xd['z'], dirty_row_pos)
    q_samples_xd_sliced = dict_slice(q_samples_xd, dirty_row_pos)

    vae_loss_dc, vae_nll_dc, vae_z_kld_dc, vae_w_kld_dc = model.loss_function(data_clean[dirty_row_pos,:], p_params_xd_sliced,
                                                                  q_params_xd_sliced, q_samples_xd_sliced,
                                                                  clean_comp_only=True,
                                                                  data_eval_clean=True)
    clean_row_pos = ~dirty_row_pos
    n_clean_rows = clean_row_pos.sum().item()

    p_params_xd_sliced = dict_slice(p_params_xd, clean_row_pos)
    q_params_xd_sliced = dict()
    if args.outlier_model == 'RVAE':
        q_params_xd_sliced['w'] = dict_slice(q_params_xd['w'], clean_row_pos)
    q_params_xd_sliced['z'] = dict_slice(q_params_xd['z'], clean_row_pos)
    q_samples_xd_sliced = dict_slice(q_samples_xd, clean_row_pos)

    vae_loss_cc, vae_nll_cc, vae_z_kld_cc, vae_w_kld_cc = model.loss_function(data_clean[clean_row_pos,:], p_params_xd_sliced,
                                                                  q_params_xd_sliced, q_samples_xd_sliced,
                                                                  clean_comp_only=True,
                                                                  data_eval_clean=True)

    eval_data_len = data_dirty.shape[0]
    losses = {'eval_loss_final_clean_dc': vae_loss_dc.item()/n_dirty_rows, 'eval_nll_final_clean_dc':vae_nll_dc.item()/n_dirty_rows,
              'eval_z_kld_final_clean_dc': vae_z_kld_dc.item()/n_dirty_rows, 'eval_w_kld_final_clean_dc':vae_w_kld_dc.item()/n_dirty_rows,
              'eval_loss_final_clean_cc': vae_loss_cc.item()/n_clean_rows, 'eval_nll_final_clean_cc':vae_nll_cc.item()/n_clean_rows,
              'eval_z_kld_final_clean_cc': vae_z_kld_cc.item()/n_clean_rows, 'eval_w_kld_final_clean_cc':vae_w_kld_cc.item()/n_clean_rows,
              'eval_loss_final_clean_all': (vae_loss_cc+vae_loss_dc).item()/eval_data_len, 'eval_nll_final_clean_all':(vae_nll_cc+vae_nll_dc).item()/eval_data_len,
              'eval_z_kld_final_clean_all': (vae_z_kld_cc+vae_z_kld_dc).item()/eval_data_len, 'eval_w_kld_final_clean_all':(vae_w_kld_cc+vae_w_kld_dc).item()/eval_data_len,
              'mse_lower_bd_dirtycells': error_lb_dc.item(), 'mse_upper_bd_dirtycells': error_up_dc.item() , 'mse_repair_dirtycells': error_repair_dc.item(),
              'mse_repair_cleancells': error_repair_cc.item(),
              'errors_per_feature': [error_lb_dc_per_feat, error_repair_dc_per_feat, error_up_dc_per_feat, error_repair_cc_per_feat]}

    return losses


