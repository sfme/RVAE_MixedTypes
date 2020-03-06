declare dataset="Wine" # "DefaultCredit"

declare rows_perc="5"
declare cols_perc="20"
declare run="1"

declare data_file="${dataset}/gaussian_m0s5_categorical_alpha0.0/${rows_perc}pc_rows_${cols_perc}pc_cols_run_${run}"
declare data_folder="../../data_simple/${data_file}/"

declare out_folder="../../outputs_experiments_i/${data_file}/VAE"

declare ACTIVATION="relu"
declare L2_REG="1." # 50

python -u ../core_models/main.py --dataset-folder ${data_folder} --output-folder ${out_folder} --outlier-model VAE --number-epochs 100 \
       --save-on --latent-dim 20 --layer-size 400 --verbose-metrics-epoch --activation ${ACTIVATION} --l2-reg ${L2_REG} --alpha-prior 0.95 --cuda-on
