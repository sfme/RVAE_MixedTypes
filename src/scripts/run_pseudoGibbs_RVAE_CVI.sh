declare dataset="Wine" # "DefaultCredit"

declare rows_perc="5"
declare cols_perc="20"
declare run="1"

declare data_file="${dataset}/gaussian_m0s5_categorical_alpha0.0/${rows_perc}pc_rows_${cols_perc}pc_cols_run_${run}"
declare data_folder="../../data_simple/${data_file}/"

declare out_folder="../../outputs_experiments_i/${data_file}/SEQ_RVAE_CVI" # MCMC at eval time (pseudo-Gibbs)

declare ACTIVATION="relu"
declare L2_REG="0." # 50

# check paser_arguments and paper supplementary material for more info (below twoStage, with 15 samples at the second stage chain, giving final repair estimate)
#                                                                      (one stage uses 5 sample of chain only)

python -u ../core_models/main.py --dataset-folder ${data_folder} --output-folder ${out_folder} --outlier-model RVAE --number-epochs 100 \
       --save-on --latent-dim 20 --layer-size 400 --verbose-metrics-epoch --activation ${ACTIVATION} --l2-reg ${L2_REG} --alpha-prior 0.95 --cuda-on \
       --inference-type seqvae --seqvae-steps 5 --seqvae-two-stage --steps-2stage 15

