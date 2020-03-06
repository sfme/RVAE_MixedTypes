declare dataset="Wine" # "DefaultCredit"

declare rows_perc="5"
declare cols_perc="20"
declare run="1"

declare data_file="${dataset}/gaussian_m0s5_categorical_alpha0.0/${rows_perc}pc_rows_${cols_perc}pc_cols_run_${run}"
declare data_folder="../../data_simple/${data_file}/"

declare out_folder="../../outputs_experiments_i/${data_file}/DeepRPCA"

declare ACTIVATION="relu"
declare L2_REG="0."

python -u ../core_models/DeepRPCA/main_deepRPCA.py --dataset-folder ${data_folder} --output-folder ${out_folder} --l21-method --turn-on-validation \
       --latent-dim 20 --layer-size 400 --number-epochs 10 --lr 0.001 --number-ADMM-iters 50 --lambda-param 0.01 \
       --activation ${ACTIVATION} --l2-reg ${L2_REG} --save-on --cuda-on