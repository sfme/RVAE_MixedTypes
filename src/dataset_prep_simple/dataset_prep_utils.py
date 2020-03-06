
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import json
import os, errno
from cleaningbenchmark.NoiseModels.RandomNoise import CategoricalNoiseModel, GaussianNoiseModel, LaplaceNoiseModel, ZipfNoiseModel
from cleaningbenchmark.NoiseModels.RandomNoise import MixedNoiseModel, ImageSaltnPepper, ImageAdditiveGaussianNoise
from cleaningbenchmark.NoiseModels.RandomNoise import LogNormalNoiseModel, Mixture2GaussiansNoiseModel
from cleaningbenchmark.Utils.Utils import pd_df_diff
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy


def create_data_folders(run_stats, path_to_folder):

    """ create folders """

    # path to folder where to save to
    path_saving = path_to_folder + run_stats["name"] + "/"

    # try to create folder if not exists yet
    try:
        os.makedirs(path_saving)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for full dirty dataset
    try:
        os.makedirs(path_saving + "/full/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    # path for train dataset
    try:
        os.makedirs(path_saving + "/train/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for validation dataset
    try:
        os.makedirs(path_saving + "/validation/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # path for test dataset
    try:
        os.makedirs(path_saving + "/test/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return path_saving


def create_data_splits(run_stats, data):

    cond_test = (run_stats["train_size"]+run_stats["valid_size"]+run_stats["test_size"])==1.0
    assert cond_test, "dataset size percentages (train; valid; test) must match!"

    splitter = ShuffleSplit(n_splits=1, test_size=(1.0-run_stats["train_size"]), random_state=1)
    train_idxs, test_idxs = [x for x in splitter.split(data)][0]

    test_size_prop = float(run_stats["test_size"]) / (run_stats["valid_size"] + run_stats["test_size"])

    splitter_cv = ShuffleSplit(n_splits=1, test_size=test_size_prop, random_state=1)
    rel_valid_idxs, rel_test_idxs  = [x for x in splitter_cv.split(test_idxs)][0]

    validation_idxs = test_idxs[rel_valid_idxs]
    test_idxs = test_idxs[rel_test_idxs]

    return train_idxs, validation_idxs, test_idxs

def save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs,
                  noised_data_df, cells_changed, tuples_changed):


    ## train dataset

    # clean data
    df_train = df_data.iloc[train_idxs,:]
    df_train = df_train.reset_index(drop=True)

    # dirty data
    df_train_noised = noised_data_df.iloc[train_idxs,:]
    df_train_noised = df_train_noised.reset_index(drop=True)

    # ground-truth of errors
    df_changes_train, _, _ = pd_df_diff(df_train, df_train_noised)

    # ground-truth of errors - error matrix (i.e. the matrix completion matrix)
    cells_changed_train = cells_changed[train_idxs,:]

    # ground-truth of errors - rows with errors
    tuples_changed_train = tuples_changed[train_idxs]

    # save
    df_train.to_csv(path_saving + "/train/" + "data_clean.csv", index=False)
    df_train_noised.to_csv(path_saving + "/train/" + "data_noised.csv", index=False)
    df_changes_train.to_csv(path_saving + "/train/" + "changes_summary.csv")

    df_cells_changed_train = pd.DataFrame(cells_changed_train, columns=df_data.columns)
    df_cells_changed_train.to_csv(path_saving + "/train/" + "cells_changed_mtx.csv", index=False)

    df_tuples_changed_train = pd.DataFrame(tuples_changed_train, columns=["rows_with_outlier"])
    df_tuples_changed_train.to_csv(path_saving + "/train/" + "tuples_changed_mtx.csv", index=False)

    df_train_idxs = pd.DataFrame(train_idxs, columns=["original_idxs"])
    df_train_idxs.to_csv(path_saving + "/train/" + "original_idxs.csv", index=False)


    ## validation dataset

    # clean data
    df_validation = df_data.iloc[validation_idxs,:]
    df_validation = df_validation.reset_index(drop=True)

    # dirty data
    df_validation_noised = noised_data_df.iloc[validation_idxs,:]
    df_validation_noised = df_validation_noised.reset_index(drop=True)

    # ground-truth of errors - dataframe value changes
    df_changes_validation, _, _ = pd_df_diff(df_validation, df_validation_noised)

    # ground-truth of errors - error matrix (i.e. the matrix completion matrix)
    cells_changed_validation = cells_changed[validation_idxs,:]

    # ground-truth of errors - rows with errors
    tuples_changed_validation = tuples_changed[validation_idxs]


    # save
    df_validation.to_csv(path_saving + "/validation/" + "data_clean.csv", index=False)
    df_validation_noised.to_csv(path_saving + "/validation/" + "data_noised.csv", index=False)
    df_changes_validation.to_csv(path_saving + "/validation/" + "changes_summary.csv")

    df_cells_changed_validation = pd.DataFrame(cells_changed_validation, columns=df_data.columns)
    df_cells_changed_validation.to_csv(path_saving + "/validation/" + "cells_changed_mtx.csv", index=False)

    df_tuples_changed_validation = pd.DataFrame(tuples_changed_validation, columns=["rows_with_outlier"])
    df_tuples_changed_validation.to_csv(path_saving + "/validation/" + "tuples_changed_mtx.csv", index=False)

    df_validation_idxs = pd.DataFrame(validation_idxs, columns=["original_idxs"])
    df_validation_idxs.to_csv(path_saving + "/validation/" + "original_idxs.csv", index=False)


    ## test dataset

    # clean data
    df_test = df_data.iloc[test_idxs,:]
    df_test = df_test.reset_index(drop=True)

    # dirty data
    df_test_noised = noised_data_df.iloc[test_idxs,:]
    df_test_noised = df_test_noised.reset_index(drop=True)

    # ground-truth of errors
    df_changes_test, _, _ = pd_df_diff(df_test, df_test_noised)

    # ground-truth of errors - error matrix (i.e. the matrix completion matrix)
    cells_changed_test = cells_changed[test_idxs,:]

    # ground-truth of errors - rows with errors
    tuples_changed_test = tuples_changed[test_idxs]


    # save
    df_test.to_csv(path_saving + "/test/" + "data_clean.csv", index=False)
    df_test_noised.to_csv(path_saving + "/test/" + "data_noised.csv", index=False)
    df_changes_test.to_csv(path_saving + "/test/" + "changes_summary.csv")

    df_cells_changed_test = pd.DataFrame(cells_changed_test, columns=df_data.columns)
    df_cells_changed_test.to_csv(path_saving + "/test/" + "cells_changed_mtx.csv", index=False)

    df_tuples_changed_test = pd.DataFrame(tuples_changed_test, columns=["rows_with_outlier"])
    df_tuples_changed_test.to_csv(path_saving + "/test/" + "tuples_changed_mtx.csv", index=False)

    df_test_idxs = pd.DataFrame(test_idxs, columns=["original_idxs"])
    df_test_idxs.to_csv(path_saving + "/test/" + "original_idxs.csv", index=False)


    ## full dataset

    # ground-truth of errors
    df_changes_full, _, _ = pd_df_diff(df_data, noised_data_df)

    # save
    df_data.to_csv(path_saving + "/full/" + "data_clean.csv", index=False)
    noised_data_df.to_csv(path_saving + "/full/" + "data_noised.csv", index=False)
    df_changes_full.to_csv(path_saving + "/full/" + "changes_summary.csv")

    df_cells_changed = pd.DataFrame(cells_changed, columns=df_data.columns)
    df_cells_changed.to_csv(path_saving + "/full/" + "cells_changed_mtx.csv", index=False)

    df_tuples_changed = pd.DataFrame(tuples_changed, columns=["rows_with_outlier"])
    df_tuples_changed.to_csv(path_saving + "/full/" + "tuples_changed_mtx.csv", index=False)



def dirtify_and_save_categorical(df_data, run_stats, path_to_folder):

    """
    df_data := must be pandas dataframe with category dtype columns

    run_stats := definitions to dirtify dataset

    path_to_folder := folder where to save datasets
    """

    ## make sure data is categorical (convert first to string)
    df_data = df_data.apply(lambda x: x.astype(str))
    df_data = df_data.apply(lambda x: x.astype('category'))

    ## create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    ## dirtify dataset

    # helper structures ; NOTE: leave or remove missing sign?
    dict_categories_per_feat = dict([(feat_name, [x for x in df_data[feat_name].cat.categories.tolist()]) #  if x != '?' 
                                    for feat_name in df_data.columns.tolist()])

    categories_per_feat_list = [dict_categories_per_feat[col] for col in df_data.columns.tolist()]

    if run_stats["use_cat_probs"]:
        cats_probs_list = [np.array(df_data[col].value_counts()[dict_categories_per_feat[col]].values / 
                                  float(df_data[col].value_counts()[dict_categories_per_feat[col]].sum()))
                            for col in df_data.columns.tolist()]
    else:
        cats_probs_list = []


    # define categorical noise model
    cat_mdl = CategoricalNoiseModel(df_data.shape, categories_per_feat_list,
                                    probability=run_stats["p_row"], cats_probs_list=cats_probs_list, 
                                    typo_prob=run_stats["typo_prob"], p_cell=run_stats["p_cell"],
                                    alpha_prob=run_stats["alpha_prob"], 
                                    one_cell_flag=run_stats["one_cell_per_row"])

    # apply noise model to data
    noised_data, _ = cat_mdl.apply(df_data)
    noised_data_df = pd.DataFrame(noised_data, columns=df_data.columns, index=df_data.index)

    # get ground-truth
    df_changes, cells_changed, tuples_changed = pd_df_diff(df_data, noised_data_df)

    cells_changed = cells_changed.astype(int)
    tuples_changed = tuples_changed.astype(int)


    ## get dataset splits (Train; Valid; Test) and entire dataset
    train_idxs, validation_idxs, test_idxs = create_data_splits(run_stats, df_data)

    ## create dataset splits and save to folders
    save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs, 
                  noised_data_df, cells_changed, tuples_changed)

    ## save num_cols and cat_cols names for future reading by models
    cols_info = {"cat_cols_names":df_data.columns.tolist(), 
                 "num_cols_names":[],
                 "dataset_type": "categorical"
                }

    with open(path_saving + 'cols_info.json', 'w') as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    with open(path_saving + 'noising_info.json', 'w') as outfile:
        json.dump(run_stats, outfile, indent=4, sort_keys=True)


def dirtify_and_save_real(df_data, run_stats, path_to_folder):

    """
    df_data := must be pandas dataframe with real (continous) dtype columns

    run_stats := definitions to dirtify dataset

    path_to_folder := folder where to save datasets
    """

    ## create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    ## dirtify dataset

    # numerical column definition
    means_df = df_data.mean().values # means of numerical features
    stds_df = df_data.std().values # standard deviations of numerical features

    # define numerical noise model
    if run_stats["type"]=="Gaussian":
        # usual context outliers: 0.5*sigma
        num_mdl = GaussianNoiseModel((df_data.shape[0], len(df_data.columns)),
                                     probability=run_stats["p_row"], 
                                     one_cell_flag=run_stats["one_cell_per_row"],
                                     mu=run_stats["mu"], sigma=run_stats["sigma"],
                                     scale=stds_df, int_cast=run_stats["int_cast_array"],
                                     p_cell=run_stats["p_cell"])

    elif run_stats["type"]=="Laplace":

        num_mdl =  LaplaceNoiseModel((df_data.shape[0], len(df_data.columns)),
                                     probability=run_stats["p_row"],
                                     one_cell_flag=run_stats["one_cell_per_row"],
                                     mu=run_stats["mu"], b=run_stats["b"],
                                     scale=stds_df, int_cast=run_stats["int_cast_array"],
                                     p_cell=run_stats["p_cell"])

    elif run_stats["type"]=="Zipf":
        # usual value for context outliers: z = 3 or 4; big marginal outliers z = 1.3
        num_mdl = ZipfNoiseModel((df_data.shape[0], len(df_data.columns)),
                                 probability=run_stats["p_row"],
                                 one_cell_flag=run_stats["one_cell_per_row"],
                                 z=run_stats["z"], scale=stds_df,
                                 int_cast=run_stats["int_cast_array"],
                                 active_neg=run_stats["active_neg"], # active_neg=True/False
                                 p_cell=run_stats["p_cell"])

    elif run_stats["num_model"]["type"]=="LogNormal":

        num_mdl = LogNormalNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                       mu=run_stats["num_model"]["mu"], sigma=run_stats["num_model"]["sigma"],
                                       scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])

    elif run_stats["num_model"]["type"]=="Mixture2Gaussians":

        num_mdl = Mixture2GaussiansNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                              pc_1=run_stats["num_model"]["pc_1"],
                                              mu_1=run_stats["num_model"]["mu_1"], sigma_1=run_stats["num_model"]["sigma_1"],
                                              mu_2=run_stats["num_model"]["mu_2"], sigma_2=run_stats["num_model"]["sigma_2"],
                                              scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])

    else:
        print("Numerical noise model does not exist!!")
        return


    # apply noise model to data
    noised_data, _ = num_mdl.apply(df_data)
    noised_data_df = pd.DataFrame(noised_data, columns=df_data.columns, index=df_data.index)

    # get ground-truth
    df_changes, cells_changed, tuples_changed = pd_df_diff(df_data, noised_data_df)

    cells_changed = cells_changed.astype(int)
    tuples_changed = tuples_changed.astype(int)


    ## get dataset splits (Train; Valid; Test) and entire dataset
    train_idxs, validation_idxs, test_idxs = create_data_splits(run_stats, df_data)

    ## create dataset splits and save to folders
    save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs,
                  noised_data_df, cells_changed, tuples_changed)



    ## save num_cols and cat_cols names for future reading by models
    cols_info = {"cat_cols_names":[],
                 "num_cols_names":df_data.columns.tolist(),
                 "dataset_type": "real"
                }

    with open(path_saving + 'cols_info.json', 'w') as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    with open(path_saving + 'noising_info.json', 'w') as outfile:
        json.dump(run_stats, outfile, indent=4, sort_keys=True)


def dirtify_and_save_mixed_simple(df_data, run_stats, path_to_folder):

    """ Simple:

        assumes same noise model for the categorical features;
        assumes same noise model for the continous features
    """

    ### create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    ### dirtify dataset

    # convert categorical features to categorical type (note must be converted to string first)
    df_data[run_stats["cat_cols_names"]] = df_data[run_stats["cat_cols_names"]].apply(lambda x: x.astype(str))
    df_data[run_stats["cat_cols_names"]] = df_data[run_stats["cat_cols_names"]].apply(lambda x: x.astype('category'))

    # dictionaries mapping the indexes between full dataframe and helper structures
    idx_map_cat = dict([(df_data.columns.get_loc(col), i) for i, col in enumerate(run_stats["cat_cols_names"])])
    idx_map_num = dict([(df_data.columns.get_loc(col), i) for i, col in enumerate(run_stats["num_cols_names"])])

    ## categorical columns definition
    dict_categories_per_feat = dict([(feat_name, [x for x in df_data[feat_name].cat.categories.tolist()]) #  if x != '?'
                                    for feat_name in run_stats["cat_cols_names"]])

    categories_per_feat_list = [dict_categories_per_feat[col] for col in run_stats["cat_cols_names"]]

    if run_stats["cat_model"]["use_cat_probs"]:
        cats_probs_list = [np.array(df_data[col].value_counts()[dict_categories_per_feat[col]].values /
                                  float(df_data[col].value_counts()[dict_categories_per_feat[col]].sum()))
                            for col in run_stats["cat_cols_names"]]
    else:
        cats_probs_list = []

    # boolean array defining which features are categories (following the same order as the dataframe)
    cat_array_bool = (df_data.dtypes.apply(lambda t: str(t)) == 'category').values

    # define categorical noise model
    cat_mdl = CategoricalNoiseModel(df_data.shape, categories_per_feat_list,
                                    cats_probs_list=cats_probs_list,
                                    typo_prob=run_stats["cat_model"]["typo_prob"],
                                    alpha_prob=run_stats["cat_model"]["alpha_prob"])

    ## numerical column definition
    means_df = df_data[run_stats["num_cols_names"]].mean().values # means of numerical features
    stds_df = df_data[run_stats["num_cols_names"]].std().values # standard deviations of numerical features

    # define numerical noise model
    if run_stats["num_model"]["type"]=="Gaussian":
        # usual context outliers: 0.5*sigma
        num_mdl = GaussianNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                     mu=run_stats["num_model"]["mu"], sigma=run_stats["num_model"]["sigma"],
                                     scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])

    elif run_stats["num_model"]["type"]=="Laplace":

        num_mdl =  LaplaceNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                     mu=run_stats["num_model"]["mu"], b=run_stats["num_model"]["b"],
                                     scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])

    elif run_stats["num_model"]["type"]=="Zipf":
        # usual value for context outliers: z = 3 or 4; big marginal outliers z = 1.3
        num_mdl = ZipfNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                 z=run_stats["num_model"]["z"], scale=stds_df,
                                 int_cast=run_stats["num_model"]["int_cast_array"],
                                 active_neg=run_stats["num_model"]["active_neg"]) # active_neg=True/False


    elif run_stats["num_model"]["type"]=="LogNormal":

        num_mdl = LogNormalNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                       mu=run_stats["num_model"]["mu"], sigma=run_stats["num_model"]["sigma"],
                                       scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])

    elif run_stats["num_model"]["type"]=="Mixture2Gaussians":

        num_mdl = Mixture2GaussiansNoiseModel((df_data.shape[0], len(run_stats["num_cols_names"])),
                                              pc_1=run_stats["num_model"]["pc_1"],
                                              mu_1=run_stats["num_model"]["mu_1"], sigma_1=run_stats["num_model"]["sigma_1"],
                                              mu_2=run_stats["num_model"]["mu_2"], sigma_2=run_stats["num_model"]["sigma_2"],
                                              scale=stds_df, int_cast=run_stats["num_model"]["int_cast_array"])
    else:
        print("Numerical noise model does not exist!!")
        return

    # define mixed data noise model
    mix_mdl = MixedNoiseModel(df_data.shape, cat_array_bool,
                              idx_map_cat, idx_map_num, [cat_mdl],
                              [num_mdl], probability=run_stats["p_row"],
                              p_row=run_stats["p_cell"],
                              one_cell_flag=run_stats["one_cell_per_row"])

    ## apply noise model to data
    noised_data, _ = mix_mdl.apply(df_data)
    noised_data_df = pd.DataFrame(noised_data, columns=df_data.columns, index=df_data.index)
    noised_data_df[run_stats["cat_cols_names"]] = \
    noised_data_df[run_stats["cat_cols_names"]].apply(lambda x: x.astype('category'))

    ## get ground-truth
    df_changes, cells_changed, tuples_changed = pd_df_diff(df_data, noised_data_df)

    cells_changed = cells_changed.astype(int)
    tuples_changed = tuples_changed.astype(int)


    ## get dataset splits (Train; Valid; Test) and entire dataset
    train_idxs, validation_idxs, test_idxs = create_data_splits(run_stats, df_data)

    ## create dataset splits and save to folders
    save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs,
                  noised_data_df, cells_changed, tuples_changed)

    ## save num_cols and cat_cols names for future reading by models
    cols_info = {"cat_cols_names":run_stats["cat_cols_names"],
                 "num_cols_names":run_stats["num_cols_names"],
                 "dataset_type": "mixed"
                }

    with open(path_saving + 'cols_info.json', 'w') as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    ## save run_stats information about noising
    # make ndarray JSON serializable
    run_stats_json = deepcopy(run_stats)
    run_stats_json["num_model"]["int_cast_array"] = \
    run_stats_json["num_model"]["int_cast_array"].tolist()

    with open(path_saving + 'noising_info.json', 'w') as outfile:
        json.dump(run_stats_json, outfile, indent=4, sort_keys=True)


def dirtify_and_save_image_data(array_data, run_stats, path_to_folder, shape_dim=3,
                                image_dim_in=[28,28]):

    """
    array_data := must be numpy array with image data (examples, x_dim, y_dim)

    run_stats := definitions to dirtify dataset

    path_to_folder := folder where to save datasets
    """

    ## create folders
    path_saving = create_data_folders(run_stats, path_to_folder)

    if shape_dim == 2:
        # lat_shape_img = int(np.sqrt(array_data.shape[1]))
        array_data = array_data.reshape(-1, image_dim_in[0], image_dim_in[1])

    # helper definitions
    num_points = array_data.shape[0]
    x_dim = array_data.shape[1]
    y_dim = array_data.shape[2]
    img_num_pixels = x_dim*y_dim
    col_names = ['pixel_{}'.format(n) for n in range(img_num_pixels)]

    if run_stats['conv_to_int']:
        array_data = array_data.astype(int)

    ## dirtify dataset

    # define numerical noise model
    if run_stats["type"]=="SaltnPepper":

        # standard Salt and Pepper Noise
        noise_mdl = ImageSaltnPepper(array_data.shape,
                                     probability=run_stats["p_img"],
                                     one_cell_flag=run_stats["one_cell_per_row"],
                                     min_val=run_stats["min_val"],
                                     max_val=run_stats["max_val"],
                                     p_min=run_stats['p_min'],
                                     p_pixel=run_stats['p_pixel'],
                                     conv_to_int=run_stats['conv_to_int'])

    elif run_stats["type"]=="AdditiveGaussian":

        noise_mdl = ImageAdditiveGaussianNoise(array_data.shape,
                                               probability=run_stats["p_img"],
                                               one_cell_flag=False,
                                               min_val=run_stats["min_val"],
                                               max_val=run_stats["max_val"],
                                               mu=run_stats["mu"],
                                               sigma=run_stats["sigma"],
                                               scale=np.array([run_stats["scale"]]),
                                               p_pixel=run_stats['p_pixel'])

    else:
        print("Numerical noise model does not exist!!")
        return

    # apply noise model to data
    noised_data, _ = noise_mdl.apply(array_data)

    # reshape image dataset to traditional format (examples, features) dims
    array_data_collapsed = array_data.reshape((num_points,-1))
    noised_data_collapsed = noised_data.reshape((num_points,-1))

    # save array data to DataFrames
    df_data = pd.DataFrame(array_data_collapsed, columns=col_names)
    df_noised_data = pd.DataFrame(noised_data_collapsed, columns=col_names, index=df_data.index)

    # get ground-truth
    df_changes, cells_changed, tuples_changed = pd_df_diff(df_data, df_noised_data)

    cells_changed = cells_changed.astype(int)
    tuples_changed = tuples_changed.astype(int)

    ## get dataset splits (Train; Valid; Test) and entire dataset
    train_idxs, validation_idxs, test_idxs = create_data_splits(run_stats, df_data)

    ## create dataset splits and save to folders NOTE: this needs to be changed for images?
    save_datasets(path_saving, df_data, train_idxs, validation_idxs, test_idxs,
                  df_noised_data, cells_changed, tuples_changed)

    ## save num_cols and cat_cols names for future reading by models
    if run_stats['conv_to_int']:
        num_cols_names = []
        cat_cols_names = df_data.columns.tolist()
    else:
        num_cols_names = df_data.columns.tolist()
        cat_cols_names = []

    cols_info = {"cat_cols_names":cat_cols_names,
                 "num_cols_names":num_cols_names,
                 "dataset_type": "image"
                }

    with open(path_saving + 'cols_info.json', 'w') as outfile:
        json.dump(cols_info, outfile, indent=4, sort_keys=True)

    with open(path_saving + 'noising_info.json', 'w') as outfile:
        json.dump(run_stats, outfile, indent=4, sort_keys=True)


