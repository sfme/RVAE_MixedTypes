#!/bin/bash

declare dataset="Wine" # "DefaultCredit"

declare rows_perc="5"
declare cols_perc="20"
declare run="1"

declare data_file="${dataset}/gaussian_m0s5_categorical_alpha0.0/${rows_perc}pc_rows_${cols_perc}pc_cols_run_${run}"
declare data_folder="../../data_simple/${data_file}/"

python ../core_models/IF/main_isolationForest.py --dataset-folder ${data_folder}
python ../core_models/OCSVM/main_OCSVM.py --dataset-folder ${data_folder}
python ../core_models/Marginals/main_Marginals.py --dataset-folder ${data_folder}
python ../core_models/OCSVMnMarginals/main_OCSVMnMarginals.py --dataset-folder ${data_folder}

