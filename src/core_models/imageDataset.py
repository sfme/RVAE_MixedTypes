
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce

import abc


class imageDataset(Dataset):

    def __init__(self, csv_file_path, cont_flag=True, standardize_dirty=False, dirty_csv_file_path=None):

        self.df_dataset = pd.read_csv(csv_file_path)
        self.dataset_type = 'image'
        self.cont_flag = cont_flag
        self.len_data = self.df_dataset.shape[0]

        if cont_flag:
            self.num_cols = self.df_dataset.columns.tolist()
            self.cat_cols = []
        else:
            self.num_cols = []
            self.cat_cols = self.df_dataset.columns.tolist()

        if standardize_dirty:
            self.df_dataset_dirty = pd.read_csv(dirty_csv_file_path)

        ## feature type cast -- all features are real
        for col_name in self.num_cols:
            self.df_dataset[col_name] = self.df_dataset[col_name].astype(float, copy=False)
            if standardize_dirty:
                self.df_dataset_dirty[col_name] = self.df_dataset_dirty[col_name].astype(float, copy=False)

        # standardize (re-scale) continuous features defintions
        if cont_flag:
            if standardize_dirty:
                # standardize using dirty statistics -- e.g. useful running clean data on dirty models.
                self.cont_means = self.df_dataset_dirty[self.num_cols].stack().mean()
                self.cont_stds = self.df_dataset_dirty[self.num_cols].stack().std()
            else:
                self.cont_means = self.df_dataset[self.num_cols].stack().mean()
                self.cont_stds = self.df_dataset[self.num_cols].stack().std()

        ## global defs
        self.size_tensor = len(self.df_dataset.columns)

        self.feat_info = []
        if cont_flag:
            for col_name in self.df_dataset.columns:
                # numerical (real)
                self.feat_info.append((col_name, "real", 1))
        else:
            for col_name in self.df_dataset.columns:
                # categorical (categ)
                self.feat_info.append((col_name, "categ", 2)) # black and white

    def standardize_dataset(self, dataset):

        """ NOTE: this method standardize images or not depending on
                  whether they are binarized or not."""

        if self.cont_flag:
            dataset = self.from_raw_to_tensor_cont(dataset)
        else:
            dataset = self.from_raw_to_tensor_categ(dataset)

        return dataset

    def from_raw_to_tensor_cont(self, dataset):

        return (dataset-self.cont_means)/self.cont_stds

    def from_raw_to_tensor_categ(self, dataset):

        return dataset

    @abc.abstractmethod
    def __len__(self):
        """Not implemented"""

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Not implemented"""


class imageDatasetInstance(imageDataset):

    def __init__(self, csv_file_path_all, csv_file_path_instance,
                 csv_file_cell_outlier_mtx=[], get_indexes=False, cont_flag=True,
                 standardize_dirty=False, dirty_csv_file_path=None):

        super().__init__(csv_file_path_all, cont_flag=cont_flag,
                                standardize_dirty=standardize_dirty,
                                dirty_csv_file_path=dirty_csv_file_path)

        self.df_dataset_instance = pd.read_csv(csv_file_path_instance)
        self.get_indexes = get_indexes

        # get ground-truth cell error matrix, if provided
        if csv_file_cell_outlier_mtx:
            self.cell_outlier_mtx = pd.read_csv(csv_file_cell_outlier_mtx).values

        else:
            self.cell_outlier_mtx = np.array([])

        # make sure of data types in the dataframe
        for col_name in self.num_cols:
            self.df_dataset_instance[col_name] = self.df_dataset_instance[col_name].astype(float, copy=False)

        #Directly standardize the dataset here (instead of row by row)
        self.df_dataset_instance_standardized = self.standardize_dataset(self.df_dataset_instance)

    def __len__(self):

        return self.df_dataset_instance.shape[0]

    def __getitem__(self, idx):

        if self.get_indexes:
            index_ret = [idx]
        else:
            index_ret = []

        if self.cell_outlier_mtx.size:
            cell_outlier_ret = [self.cell_outlier_mtx[idx,:]]
        else:
            cell_outlier_ret = []

        ret_list = [torch.tensor(self.df_dataset_instance_standardized.iloc[idx].values, dtype=torch.float)]
#        ret_list = [torch.tensor(self.df_dataset_instance.iloc[idx].values, dtype=torch.float)]

        ret_list += cell_outlier_ret
        ret_list += index_ret

        return ret_list


