#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F

import torch.distributions as dists


def logit_fn(x):
    return (x + 1e-9).log() - (1. - x + 1e-9).log()

def nll_categ_global(categ_logp_feat, input_idx_feat, cat_feat_size,
                     isRobust=False, w=None, isClean=False):

    # Normal
    if not isRobust:
        return F.nll_loss(categ_logp_feat, input_idx_feat, reduction='none').view(-1,1)

    # Robust
    w_r = w.view(-1,1)
    if isClean:
        return F.nll_loss(w_r*categ_logp_feat, input_idx_feat, reduction='none').view(-1,1)
    else:
        categ_logp_robust = torch.log(torch.tensor(1.0/cat_feat_size))
        categ_logp_robust = categ_logp_robust*torch.ones(categ_logp_feat.shape)
        categ_logp_robust = categ_logp_robust.type(categ_logp_feat.type())
        two_compnts = w_r*categ_logp_feat + (1-w_r)*categ_logp_robust

        return F.nll_loss(two_compnts, input_idx_feat, reduction='none').view(-1,1)


def nll_gauss_global(gauss_params, input_val_feat, logvar, isRobust=False, w=None,
                     std_0_scale=2, isClean=False, shape_feats=[1]):

    # Normal
    mu = gauss_params.view([-1] + shape_feats)
    logvar_r = (logvar.exp() + 1e-9).log()

    data_compnt = 0.5*logvar_r + (input_val_feat.view([-1] + shape_feats) - mu)**2 / (2.* logvar_r.exp() + 1e-9)

    if not isRobust:
        return data_compnt.view([-1] + shape_feats)

    # Robust
    w_r = w.view([-1] + shape_feats)
    if isClean:
        return (w_r*data_compnt).view([-1] + shape_feats)
    else:
        #Outlier model
        mu_0 = 0.0
        var_0 = torch.tensor(std_0_scale**2).type(gauss_params.type())
        robust_compnt = 0.5*torch.log(var_0) + (input_val_feat.view([-1] + shape_feats) - mu_0)**2 / (2.* var_0  + 1e-9)

        return (w_r*data_compnt + (1-w_r)*robust_compnt).view([-1] + shape_feats)



def get_pi_exact_categ(categ_logp_feat, input_idx_feat, cat_feat_size, prior_sig):

    input_dims = categ_logp_feat.shape

    # robust component
    categ_logp_robust = torch.log(torch.tensor(1.0/cat_feat_size))
    categ_logp_robust = categ_logp_robust*torch.ones(input_dims)
    categ_logp_robust = categ_logp_robust.type(categ_logp_feat.type())

    with torch.no_grad():
        pi = torch.sigmoid(categ_logp_feat - categ_logp_robust + \
                           torch.log(prior_sig) - torch.log(1-prior_sig))

    return pi

def get_pi_exact_gauss(gauss_params, input_val_feat, logvar, prior_sig, std_0_scale=2, mu_0=0.,
                       shape_feats=[1]):

    # clean component
    mu = gauss_params.view([-1] + shape_feats)
    logvar_r = (logvar.exp() + 1e-9).log()

    # robust component
    mu_0 = 0.0
    var_0 = torch.tensor(std_0_scale**2).type(gauss_params.type())

    # log p(x|z)
    data_compnt = -(0.5*logvar_r + (input_val_feat.view([-1] + shape_feats) - mu)**2 / (2.* logvar_r.exp() + 1e-9))
    robust_compnt = -(0.5*torch.log(var_0) + (input_val_feat.view([-1] + shape_feats) - mu_0)**2 / (2.* var_0  + 1e-9))

    with torch.no_grad():
        pi = torch.sigmoid(data_compnt - robust_compnt + \
                           torch.log(prior_sig) - torch.log(1-prior_sig))

    return pi



def get_pi_exact_vec(model, data, p_params, q_params, args, logit_ret=True):

    # NOTE: p_params is obtained after forward-pass on decoder, reconstruct probability
    #       q_params is the dict with variational params

    prior_sig = torch.tensor(args.alpha_prior).type(q_params['z']['mu'].data.type())

    if model.dataset_obj.dataset_type == 'image' and (not model.dataset_obj.cat_cols):

        pi_vector = get_pi_exact_gauss(p_params['x'],
                                       data,
                                       p_params['logvar_x'],
                                       prior_sig,
                                       shape_feats=[len(model.dataset_obj.num_cols)])

        pi_vector = torch.clamp(pi_vector, 1e-6, 1-1e-6)

    else:

        pi_vector = []

        start = 0
        cursor_num_feat = 0

        for feat_select, (_, col_type, feat_size) in enumerate(model.dataset_obj.feat_info):

            if col_type == 'categ':

                pi_feat = get_pi_exact_categ(p_params['x'][:,start:(start + feat_size)],
                                             data[:,feat_select].long(),
                                             feat_size, prior_sig)

                pi_feat = torch.clamp(pi_feat, 1e-6, 1-1e-6)
                pi_feat = torch.gather(pi_feat, 1, data[:,feat_select].long().view(-1,1))

                pi_vector.append(pi_feat)

                start += feat_size

            elif col_type == 'real':

                pi_feat = get_pi_exact_gauss(p_params['x'][:,start:(start + 1)],
                                             data[:,feat_select],
                                             p_params['logvar_x'][:,cursor_num_feat],
                                             prior_sig)

                pi_feat = torch.clamp(pi_feat, 1e-6, 1-1e-6).view(-1,1)
                pi_vector.append(pi_feat)

                start += 1 # 2
                cursor_num_feat +=1

        pi_vector = torch.cat(pi_vector, 1)

    if logit_ret:
        q_params['w'] = {'logit_pi': logit_fn(pi_vector)}
        return q_params['w']['logit_pi']
    else:
        q_params['w'] = {'pi': pi_vector}
        return q_params['w']['pi']


def reparam_categ(log_probs, stochastic=False, gs_temp=0.5):

    if stochastic:
        gumbel_dist_noise = dists.gumbel.Gumbel(torch.tensor([0.0]),torch.tensor([1.0]))
        g_noise = gumbel_dist_noise.sample(sample_shape=log_probs.shape).type(log_probs.type()).squeeze(dim=-1)
    else:
        g_noise = 0.0

    inner = (log_probs + g_noise) / gs_temp
    samples = torch.softmax(inner, dim=-1)

    # ST estimation, needed for using Embeddings that bprop (one-hot)
    shape = samples.size()
    _, ind = samples.max(dim=-1)
    samples_hard = torch.zeros_like(samples).view(-1, shape[-1])
    samples_hard.scatter_(1, ind.view(-1, 1), 1)
    samples_hard = samples_hard.view(*shape)

    # note: ind not bpropable
    return (samples_hard - samples).detach() + samples, ind.float().view(-1,1)


def reparam_real(mu_x, logvar_x, stochastic=False, eps=None):

    if stochastic:
        if eps is None:
            eps = torch.randn_like(mu_x)
        std = logvar_x.mul(0.5).exp_()
        return eps.mul(std).add_(mu_x).clamp(-2.,2.) # -50., 50.

    else:
        return mu_x.clamp(-2.,2.) # -50., 50.

def reparam_gen_model(p_params, dataset_obj, sampling=False):

    if dataset_obj.dataset_type == 'image' and (not dataset_obj.cat_cols):
        aux_sampl = reparam_real(p_params['x'], p_params['logvar_x'], sampling)
        samples_out_oh = aux_sampl
        samples_out_ind = aux_sampl.detach()

    else:
        # mixed datasets, or just categorical / continuous with medium number of features
        start = 0
        cursor_num_feat = 0
        samples_out_oh, samples_out_ind = [], []

        for feat_select, (_, col_type, feat_size) in enumerate(dataset_obj.feat_info):
            if col_type == 'categ':
                aux_sampl_oh, aux_sampl_ind = reparam_categ(p_params['x'][:,start:(start + feat_size)], sampling, gs_temp=0.75) # 0.5 # 5
                samples_out_oh.append(aux_sampl_oh)
                samples_out_ind.append(aux_sampl_ind.detach())
                start += feat_size

            elif col_type == 'real':
                aux_sampl = reparam_real(p_params['x'][:,start:(start + 1)], p_params['logvar_x'][:,cursor_num_feat], sampling)
                samples_out_oh.append(aux_sampl)
                samples_out_ind.append(aux_sampl.detach())
                start += 1 # 2
                cursor_num_feat +=1

        samples_out_oh = torch.cat(samples_out_oh, 1)
        samples_out_ind = torch.cat(samples_out_ind, 1)

    return samples_out_oh, samples_out_ind


def copy_params_vae(params_tuple, batch_size, args, require_grad=False):

    p_params, q_params, q_samples = params_tuple

    cp_dict_func = lambda dict_op: {key:value.clone().detach().requires_grad_(require_grad)
                                            for key, value in dict_op.items()}

    p_params_cp = cp_dict_func(p_params)
    q_params_cp = dict()
    if args.outlier_model == 'RVAE':
        q_params_cp['w'] = cp_dict_func(q_params['w'])
    q_params_cp['z'] = cp_dict_func(q_params['z'])
    q_samples_cp = cp_dict_func(q_samples)

    return (p_params_cp, q_params_cp, q_samples_cp)


def masker(logit_pi, gs_temp=1, stochastic=True, dropout=False, out_shape=[]):

    if dropout:
        # note that here logit_pi only has one value for the entire batch / features
        p_dropout = 1.-torch.sigmoid(logit_pi)

        if stochastic:
            return (torch.empty(out_shape, device=logit_pi.device).uniform_() > p_dropout).float()
        else:
            return torch.ones(out_shape, device=logit_pi.device).float()

    else:

        if stochastic:
            gumbel_dist_noise = dists.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))
            g_noise = gumbel_dist_noise.sample(sample_shape=logit_pi.shape).type(logit_pi.type()).squeeze(dim=-1)
        else:
            g_noise = 0.0

        inner = (logit_pi + g_noise) / gs_temp
        samples = torch.sigmoid(inner)

        samples_hard = torch.round(samples)

        return (samples_hard - samples).detach() + samples


def combine_x_data(x_fixed, x_new, drop_mask, model, one_hot_categ=False):

    "only indexes yet, not one-hot"

    input_list = []
    start = 0

    for feat_idx, ( _, col_type, feat_size ) in enumerate(model.dataset_obj.feat_info):

        if one_hot_categ:
            if col_type == "categ":
                aux = (x_fixed[:,start:(start + feat_size)]*drop_mask[:,feat_idx].view(-1,1)) \
                            + ((1.-drop_mask[:,feat_idx].view(-1,1))*x_new[:,start:(start + feat_size)])
                input_list.append(aux)
                start += feat_size

            elif col_type == "real":
                aux = (x_fixed[:,start]*drop_mask[:,feat_idx]).view(-1,1) + ((1.-drop_mask[:,feat_idx])*x_new[:,start]).view(-1,1)
                input_list.append(aux)
                start += 1

        else:
            if col_type == "categ":
                aux = (x_fixed[:,feat_idx]*drop_mask[:,feat_idx]).view(-1,1) + ((1.-drop_mask[:,feat_idx])*x_new[:,feat_idx]).view(-1,1)
                input_list.append(aux)

            elif col_type == "real":
                aux = (x_fixed[:,feat_idx]*drop_mask[:,feat_idx]).view(-1,1) + ((1.-drop_mask[:,feat_idx])*x_new[:,feat_idx]).view(-1,1)
                input_list.append(aux)

    return torch.cat(input_list,1)



def rnn_vae_forward_one_stage(params_init, data_out, model, loss, args, number_steps=4, loss_per_iter=False, mask_err=None, epoch_id=None):

    """
    Regular refeeding, allows full bprop for training, if user wants.
    OneStage: asssume everything is dirty.

    Used at evaluation time for AISTATS 2020, not used at training -- see Supp. Material.
    """

    total_loss = loss.clone() if loss_per_iter else 0.0
    p_params, q_params, q_samples = params_init

    if type(mask_err) != type(None):
        print("Pi values iter: {} val: {}".format(0, torch.sigmoid(q_params['w']['logit_pi'])[mask_err].median() ) )

    for jj in range(number_steps):

        # sample fantasy x from generative model (decoder)
        in_samples_oh, in_samples_ind = reparam_gen_model(p_params, model.dataset_obj, model.training)

        p_params, q_params, q_samples = model(in_samples_oh, one_hot_categ=True) # n_epochs=...

        if not args.AVI:
            get_pi_exact_vec(model, in_samples_ind, p_params, q_params, args, logit_ret=True) # get pi, saves to q_params (with no_grad)
            if type(mask_err) != type(None):
                print("Pi values iter: {} val: {}".format(jj+1, torch.sigmoid(q_params['w']['logit_pi'])[mask_err].median() ) )

        if loss_per_iter:
            vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(in_samples_ind, p_params, q_params, q_samples)
            total_loss = total_loss + vae_loss

    if not loss_per_iter:
            vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(in_samples_ind, p_params, q_params, q_samples)
            total_loss = vae_loss

    if model.training:
        total_loss.backward() # bprop-through-time option

    losses = (total_loss, vae_loss, vae_nll, vae_z_kld, vae_w_kld)
    params_final = (p_params, q_params, q_samples)

    # get pi(x^0 | z^T)
    params_eval = copy_params_vae(params_final, data_out.shape[0], args)
    if not args.AVI:
        get_pi_exact_vec(model, data_out, params_eval[0], params_eval[1], args, logit_ret=True)

    return losses, params_final, params_eval



def rnn_vae_forward_two_stage(params_init, data_out, model, loss, args, number_steps=4,
                              number_steps_second_stage=15, loss_per_iter=False, mask_err=None, epoch_id=None):

    """ TwoStage: Double chain; bprop is not possible here, to be used at evaluation time only, does not allow bprop at training (use OneStage).

        Used at evaluation time for AISTATS 2020 -- see Supp. Material.
    """

    p_params, q_params, q_samples = params_init
    if type(mask_err) != type(None):
        mask_err = mask_err.bool()

    if type(mask_err) != type(None):
        print("Pi values iter: {} val: {}".format(0, torch.sigmoid(q_params['w']['logit_pi'])[mask_err].median()))

    total_loss = loss.clone() if loss_per_iter else 0.0

    # 1st Stage (OneStage)
    for jj in range(number_steps):

        # sample fantasy x from generative model (decoder)
        _, in_samples_ind = reparam_gen_model(p_params, model.dataset_obj, model.training)

        p_params, q_params, q_samples = model(in_samples_ind, one_hot_categ=False) # n_epochs=...

        if not args.AVI:
            get_pi_exact_vec(model, in_samples_ind, p_params, q_params, args, logit_ret=True) # get pi, saves to q_params (with no_grad)
            if type(mask_err) != type(None):
                print("Pi values iter: {} val: {}".format(jj+1, torch.sigmoid(q_params['w']['logit_pi'])[mask_err].median()))

        if loss_per_iter:
            vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(in_samples_ind, p_params, q_params, q_samples)

    losses = (total_loss, vae_loss, vae_nll, vae_z_kld, vae_w_kld)
    params_first_stage = (p_params, q_params, q_samples)

    # get pi(x^0 | z^T)
    params_eval = copy_params_vae(params_first_stage, data_out.shape[0], args)
    if not args.AVI:
        get_pi_exact_vec(model, data_out, params_eval[0], params_eval[1], args, logit_ret=True)

    # 2nd Stage
    if epoch_id > 1:
        p_params, q_params, q_samples = params_eval

        drop_mask = masker(q_params['w']['logit_pi'], gs_temp=0.75, stochastic=model.training, dropout=False, out_shape=[]) # gs_temp=0.5

        _, in_samples_ind = reparam_gen_model(p_params, model.dataset_obj, True) # NOTE: using it for sampling at eval now!! (True flag)

        combined_sample = combine_x_data(data_out, in_samples_ind, drop_mask, model)
        p_params, q_params, q_samples = model(combined_sample, one_hot_categ=False)

        if not args.AVI:
            get_pi_exact_vec(model, combined_sample, p_params, q_params, args, logit_ret=True)

        vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(combined_sample, p_params, q_params, q_samples)
        total_loss = vae_loss

    else:
        total_loss = loss.clone() if loss_per_iter else 0.0

    for jj in range(number_steps_second_stage):

        # sample fantasy x from generative model (decoder)
        _, in_samples_ind = reparam_gen_model(p_params, model.dataset_obj, True) # NOTE: using sampling at eval now!! (True flag)

        if epoch_id > 1:
            combined_sample = combine_x_data(data_out, in_samples_ind, drop_mask, model)
        else:
            combined_sample = in_samples_ind

        # forward entire model
        if args.outlier_model == 'RVAE':
            q_params = model.encode(combined_sample, False, False, [], [])
        else:
            q_params = model.encode(combined_sample, False)
        q_samples = model.reparameterize(q_params)
        p_params = model.decode(q_samples['z'])

        if not args.AVI:
            get_pi_exact_vec(model, combined_sample, p_params, q_params, args, logit_ret=True) # get pi, saves to q_params (with no_grad)
            if type(mask_err) != type(None):
                print("Pi values iter: {} val: {}".format(jj+1, torch.sigmoid(q_params['w']['logit_pi'])[mask_err.bool()].median()))

        if loss_per_iter:
            vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(combined_sample, p_params, q_params, q_samples)
            total_loss = total_loss + vae_loss

    if not loss_per_iter:
        vae_loss, vae_nll, vae_z_kld, vae_w_kld = model.loss_function(combined_sample, p_params, q_params, q_samples)
        total_loss = vae_loss

    losses = (total_loss, vae_loss, vae_nll, vae_z_kld, vae_w_kld)
    params_final = (p_params, q_params, q_samples)

    # get pi(x^0 | z^T)
    params_eval = copy_params_vae(params_final, data_out.shape[0], args)
    if not args.AVI:
        get_pi_exact_vec(model, data_out, params_eval[0], params_eval[1], args, logit_ret=True)

    # like in Rezende et. al 2014 (impute only missing cells, in our case dirty ones given samping from \pi):
    # if repair_mode:
    #     params_eval[0]['x'] = combine_x_data() # combined_sample to impute only bad cells given by mask

    return losses, params_final, params_eval


