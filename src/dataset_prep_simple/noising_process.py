
import pandas as pd
import numpy as np
import os, errno
from dataset_prep_utils import dirtify_and_save_mixed_simple, dirtify_and_save_categorical

##### STEPS
# 1 - Select dataset
# 2 - Select corruption process
# 3 - Select ranges of corruptions and number of experiments
# 4 - Create corrupted datasets
############

# 1 - Select dataset
dataset = 'Wine' # 'Letter'

if dataset is 'Adult':

    #### Noise Adult Dataset as Mixed Data
    name_file = "adult.csv"
    folder_path = "../../data_simple/adult/"

    df_data = pd.read_csv(folder_path + name_file)

    # numerical features
    num_feat_names = ['age',
                      'fnlwgt',
                      'hours-per-week',
                      'capital-gain',
                      'capital-loss']

    ## if noise + data should be rounded to next integer value
    ##      Note: follows the same order as num_feat_names list!
    int_cast_array = np.array([1,1,1,1,1,1], dtype=bool)

    # categorical features
    cat_feat_names = ['bracket-salary',
                      'native-country',
                      'sex',
                      'race',
                      'relationship',
                      'occupation',
                      'marital-status',
                      'education-num',
                      'education',
                      'workclass']

elif dataset is 'Wine':

    name_file = "wine.csv"
    folder_path = "../../data_simple/Wine/"

    df_data = pd.read_csv(folder_path + name_file)

    # real features
    num_feat_names = ['fixed_acidity',
                      'volatile_acidity',
                      'citric_acid',
                      'residual_sugar',
                      'chlorides',
                      'free_sulfur_dioxide',
                      'total_sulfur_dioxide',
                      'density',
                      'pH',
                      'sulphates',
                      'alcohol',
                      'quality']

    ## if noise + data should be rounded to next integer value
    ##      Note: follows the same order as num_feat_names list!
    int_cast_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)

    # categorical features
    cat_feat_names = ['wine_type']

elif dataset is 'Letter':
    name_file = "letter.csv"
    folder_path = "../../data_simple/Letter/"

    df_data = pd.read_csv(folder_path + name_file)

    # numerical features
    num_feat_names = []

    ## if noise + data should be rounded to next integer value
    ##      Note: follows the same order as num_feat_names list!
    int_cast_array = np.array([], dtype=bool)

    # categorical features
    cat_feat_names = ['letter',
                      'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                      'F11', 'F12', 'F13', 'F14', 'F15', 'F16']# 1 - Select dataset


elif dataset is 'DefaultCredit':

    name_file = "DefaultCredit.csv"
    folder_path = "../../data_simple/DefaultCredit/"

    df_data = pd.read_csv(folder_path + name_file)

    # real features
    num_feat_names = ["LIMIT_BAL",
                      "AGE",
                      "BILL_AMT1",
                      "BILL_AMT2",
                      "BILL_AMT3",
                      "BILL_AMT4",
                      "BILL_AMT5",
                      "BILL_AMT6",
                      "PAY_AMT1",
                      "PAY_AMT2",
                      "PAY_AMT3",
                      "PAY_AMT4",
                      "PAY_AMT5",
                      "PAY_AMT6"]

    ## if noise + data should be rounded to next integer value
    ##      Note: follows the same order as num_feat_names list!
    # int_cast_array = np.array([0,0,0,0,0,0,0,0,0,0,0,1], dtype=bool)
    int_cast_array = np.ones(14, dtype=bool)

    # categorical features
    cat_feat_names = ["SEX",
                      "EDUCATION",
                      "MARRIAGE",
                      "PAY_0",
                      "PAY_2",
                      "PAY_3",
                      "PAY_4",
                      "PAY_5",
                      "PAY_6",
                      "default payment next month"]


# 2 - Select corruption process
# Categorical model
alpha_cat = 0.0 # e.g. alpha_cat=0.5, or alpha_cat=0.0 -- uniform noise.
if alpha_cat > 0.0 and alpha_cat < 1.0:
    cat_model = {"typo_prob": 0.0, "use_cat_probs": True, "alpha_prob": alpha_cat}
else:
    cat_model = {"typo_prob": 0.0, "use_cat_probs": False, "alpha_prob": 0.0}

cat_name = "categorical_alpha" + str(alpha_cat)

# Numerical model - change model corruption parameters here
model_real = 'gaussian' # 'laplace'
if model_real == 'gaussian':
    mu = 0
    sigma = 5
    num_model = {"type": "Gaussian", "mu": 0, "sigma": sigma, "int_cast_array": int_cast_array}
    num_name = model_real + "_m" + str(mu) + "s" + str(sigma)
elif model_real == 'laplace':
    mu = 0
    lap_b = 4
    num_model = {"type": "Laplace", "mu": 0.0, "b": lap_b, "int_cast_array": int_cast_array}
    num_name = model_real + "_m" + str(mu) + "b" + str(lap_b)
elif model_real == 'lognormal':
    mu = 0
    sigma = 0.75
    num_model = {"type": "LogNormal", "mu": 0.0, "sigma": sigma, "int_cast_array": int_cast_array}
    num_name = model_real + "_m" + str(mu) + "s" + str(sigma)
elif model_real == 'mix2gaussians':
    pc_1 = 0.6
    mu_1 = 0.5
    mu_2 = -0.5
    sigma_1 = 3.0
    sigma_2 = 3.0
    num_model = {"type": "Mixture2Gaussians", "pc_1": pc_1,
                  "mu_1": mu_1, "mu_2": mu_2, "sigma_1": sigma_1, "sigma_2": sigma_2,
                  "int_cast_array": int_cast_array}
    num_name = model_real + "_p" + str(pc_1) + "_m1_" + str(mu_1) + "_m2_" + str(mu_2) + "_s1_" + str(sigma_1) + "_s2_" + str(sigma_2)


# 3 - Select corruption choices
train_size = 0.8
valid_size = 0.1
test_size = 0.1
p_row_range = [0.01, 0.05, 0.10, 0.20, 0.50]; p_col_range = [0.20]
runs = 1 # 5

# 4 - Create corrupted datasets

#Different corruption process, since every feature is categorical
if dataset == 'Letter':
    folder_path_final = folder_path + cat_name + "/"
    for jj in range(runs):
        for p_row in p_row_range:
            for p_col in p_col_range:
                # define noising running stats (note that here p_cell means p_col)
                run_stats_mixed = {"name": str(int(p_row*100)) + "pc_rows_" + str(int(p_col*100)) + "pc_cols" + "_run_" + str(jj+1),
                                 "p_row": p_row, "p_cell": p_col, "one_cell_per_row":False,
                                 "train_size": train_size, "valid_size":valid_size, "test_size":test_size,
                                 "typo_prob": cat_model['typo_prob'], "use_cat_probs":cat_model['use_cat_probs'], "alpha_prob": cat_model['alpha_prob']}

                dirtify_and_save_categorical(df_data, run_stats_mixed, folder_path_final)

else:
    folder_path_final = folder_path + num_name + "_" + cat_name + "/"
    for jj in range(runs):
        for p_row in p_row_range:
            for p_col in p_col_range:
                # define noising running stats (note that here p_cell means p_col)
                run_stats_mixed = {"name": str(int(p_row*100)) + "pc_rows_" + str(int(p_col*100)) + "pc_cols" + "_run_" + str(jj+1),
                                 "p_row": p_row, "p_cell": p_col, "one_cell_per_row":False,
                                 "train_size": train_size, "valid_size":valid_size, "test_size":test_size,
                                 "cat_cols_names":cat_feat_names, "num_cols_names":num_feat_names,
                                 "cat_model":cat_model, "num_model":num_model
                                }

                dirtify_and_save_mixed_simple(df_data, run_stats_mixed, folder_path_final)