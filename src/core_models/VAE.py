#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import functional as F
from model_utils import nll_categ_global, nll_gauss_global

from EmbeddingMul import EmbeddingMul


class VAE(nn.Module):

    def __init__(self, dataset_obj, args):

        super(VAE, self).__init__()
        # NOTE: for feat_select, (col_name, col_type, feat_size) in enumerate(dataset_obj.feat_info)

        self.dataset_obj = dataset_obj
        self.args = args

        self.size_input = len(dataset_obj.cat_cols)*self.args.embedding_size + len(dataset_obj.num_cols)
        self.size_output = len(dataset_obj.cat_cols) + len(dataset_obj.num_cols) # 2*

        ## Encoder Params

        # define a different embedding matrix for each feature
        if (dataset_obj.dataset_type == "image") and (not dataset_obj.cat_cols):
            self.feat_embedd = nn.ModuleList([])
        else:
            self.feat_embedd = nn.ModuleList([nn.Embedding(c_size, self.args.embedding_size, max_norm=1)
                                             for _, col_type, c_size in dataset_obj.feat_info 
                                             if col_type=="categ"])

        self.fc1 = nn.Linear(self.size_input, self.args.layer_size)
        self.fc21 = nn.Linear(self.args.layer_size, self.args.latent_dim)
        self.fc22 = nn.Linear(self.args.layer_size, self.args.latent_dim)

        ## Decoder Params

        self.fc3 = nn.Linear(self.args.latent_dim, self.args.layer_size)

        if dataset_obj.dataset_type == "image" and (not dataset_obj.cat_cols):
            self.out_cat_linears = nn.Linear(self.args.layer_size, self.size_output)
        else:
            self.out_cat_linears = nn.ModuleList([nn.Linear(self.args.layer_size, c_size) if col_type=="categ" 
                                                 else nn.Linear(self.args.layer_size, c_size) # 2*
                                                 for _, col_type, c_size in dataset_obj.feat_info])

        ## Log variance of the decoder for real attributes
        if dataset_obj.dataset_type == "image" and (not dataset_obj.cat_cols):
            self.logvar_x = nn.Parameter(torch.zeros(1).float())
        else:
            if dataset_obj.num_cols:
                self.logvar_x = nn.Parameter(torch.zeros(1,len(dataset_obj.num_cols)).float())
            else:
                self.logvar_x = []

        ## Other

        if args.activation == 'relu':
            self.activ = nn.ReLU()
        elif args.activation == 'hardtanh':
            self.activ = nn.Hardtanh()

        self.logSoftmax = nn.LogSoftmax(dim=1)

        # define encoder / decoder easy access parameter list
        # encoder params
        encoder_list = [self.fc1, self.fc21, self.fc22]
        self.encoder_mod = nn.ModuleList(encoder_list)
        if self.feat_embedd:
            self.encoder_mod.append(self.feat_embedd)
        self.encoder_param_list = nn.ParameterList(self.encoder_mod.parameters())

        # decoder params
        decoder_list = [self.fc3, self.out_cat_linears]

        self.decoder_mod = nn.ModuleList(decoder_list)
        self.decoder_param_list = nn.ParameterList(self.decoder_mod.parameters())
        if len(self.logvar_x):
            self.decoder_param_list.append(self.logvar_x)


    def get_inputs(self, x_data, one_hot_categ=False):

        if self.dataset_obj.dataset_type == "image" and (not self.dataset_obj.cat_cols):
            # image data, hence real
            return x_data

        else:
            # mixed data, or just real or just categ
            input_list = []
            cursor_embed = 0
            start = 0

            for feat_idx, ( _, col_type, feat_size ) in enumerate(self.dataset_obj.feat_info):

                if one_hot_categ:
                    if col_type == "categ": # categorical (uses embeddings)
                        func_embedd = EmbeddingMul(self.args.embedding_size, x_data.device)
                        func_embedd.requires_grad = True
                        categ_val = func_embedd(x_data[:,start:(start + feat_size)].view(1,x_data.shape[0],-1),
                                    self.feat_embedd[cursor_embed].weight,-1, max_norm=1, one_hot_input=True)
                        input_list.append(categ_val.view(x_data.shape[0],-1))

                        start += feat_size
                        cursor_embed += 1

                    elif col_type == "real": # numerical
                        input_list.append(x_data[:,start].view(-1,1))
                        start += 1

                else:
                    if col_type == "categ": # categorical (uses embeddings)
                        input_list.append(self.feat_embedd[cursor_embed](x_data[:,feat_idx].long()))
                        cursor_embed += 1

                    elif col_type == "real": # numerical
                        input_list.append(x_data[:,feat_idx].view(-1,1))

            return torch.cat(input_list, 1)


    def encode(self, x_data, one_hot_categ=False):

        q_params = dict()

        input_values = self.get_inputs(x_data, one_hot_categ)

        h1 = self.activ(self.fc1(input_values))

        q_params['z'] = {'mu': self.fc21(h1), 'logvar': self.fc22(h1)}

        return q_params


    def sample_normal(self, q_params_z, eps=None):

        if self.training:

            if eps is None:
                eps = torch.randn_like(q_params_z['mu'])

            std = q_params_z['logvar'].mul(0.5).exp_()

            return eps.mul(std).add_(q_params_z['mu'])

        else:
            return q_params_z['mu']


    def reparameterize(self, q_params, eps_samples=None):

        q_samples = dict()

        q_samples['z'] = self.sample_normal(q_params['z'], eps_samples)

        return q_samples


    def decode(self, z):

        p_params = dict()

        h3 = self.activ(self.fc3(z))

        if self.dataset_obj.dataset_type == 'image' and (not self.dataset_obj.cat_cols):

            # tensor with dims (batch_size, self.size_output)
            p_params['x'] = self.out_cat_linears(h3)
            p_params['logvar_x'] = self.logvar_x.clamp(-3,3)

        else:
            out_cat_list = []

            for feat_idx, out_cat_layer in enumerate(self.out_cat_linears):

                if self.dataset_obj.feat_info[feat_idx][1] == "categ": # coltype check
                    out_cat_list.append(self.logSoftmax(out_cat_layer(h3)))

                elif self.dataset_obj.feat_info[feat_idx][1] == "real":
                    out_cat_list.append(out_cat_layer(h3))

            # tensor with dims (batch_size, self.size_output)
            p_params['x'] = torch.cat(out_cat_list, 1)

            if self.dataset_obj.num_cols:
                p_params['logvar_x'] = self.logvar_x.clamp(-3,3) #p_params['logvar_x'] = self.logvar_x

        return p_params

    def forward(self, x_data, n_epoch=None, one_hot_categ=False):

        q_params = self.encode(x_data, one_hot_categ)
        q_samples = self.reparameterize(q_params)

        return self.decode(q_samples['z']), q_params, q_samples


    def loss_function(self, input_data, p_params, q_params, q_samples, clean_comp_only=False, data_eval_clean=False):

        """ ELBO: reconstruction loss for each variable + KL div losses summed over elements of a batch """

        # categorical reconstruction value
        dtype_float = torch.cuda.FloatTensor if self.args.cuda_on else torch.FloatTensor
        nll_val = torch.zeros(1).type(dtype_float)

        if self.dataset_obj.dataset_type == 'image' and (not self.dataset_obj.cat_cols):
            # image datasets, large number of features (so vectorize loss)
            # compute NLL
            nll_val = nll_gauss_global(p_params['x'],
                                input_data, 
                                p_params['logvar_x'], isRobust=False,
                                shape_feats=[len(self.dataset_obj.num_cols)]).sum()

        else:
            start = 0
            cursor_num_feat = 0

            for feat_select, (_, col_type, feat_size) in enumerate(self.dataset_obj.feat_info):
                # compute NLL
                if col_type == 'categ':
                    nll_val += nll_categ_global(p_params['x'][:,start:(start + feat_size)], 
                                          input_data[:,feat_select].long(), feat_size, isRobust=False).sum()
                    start += feat_size

                elif col_type == 'real':
                    nll_val += nll_gauss_global(p_params['x'][:,start:(start + 1)], # 2
                                         input_data[:,feat_select], 
                                         p_params['logvar_x'][:,cursor_num_feat], isRobust=False).sum()

                    start += 1 # 2
                    cursor_num_feat +=1


        # kld regularizer on the latent space
        kld = -0.5 * torch.sum(1 + q_params['z']['logvar'] - q_params['z']['mu'].pow(2) - q_params['z']['logvar'].exp())

        return nll_val + kld, nll_val, kld, torch.zeros_like(kld)


