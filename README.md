# RVAE for Mixed Type Features (Tabular Data)

Code for paper: "Robust Variational Autoencoders for Outlier Detection and Repair of Mixed-Type Data" AISTATS 2020. Check it here https://arxiv.org/abs/1907.06671.
**Please consider citing us if you use our code.**

## Instalation

+ Please install ./setup.py in folder ./src in order to use core_models package.

+ Use Pytorch 1.3.1 at least

## Usage

### Data folder
+ Clean is found (or inserted) in data_simple:
  + e.g. ./data_simple/Wine/Wine.csv
  + the folder contains both clean data, and then after nosing, several noisy replicas given by ```python noising_process.py``` in separate folders.

### Output folder
+ Current scripts generate folder ./outputs_experiments_i/{dataset}/{noise_type}/{corruption_level}_run_j/{Model_Name}/
+ Therein results for outlier detection metrics (cell and row), and repair of cells are presented. The latter only if algorithm provides this.

### Simple Work Flow

#### Noising a dataset:
+ Go to ./src/dataset_prep_simple/ to run noising of datasets in data folder:
  + Open noising_process.py
  + Edit script definitions: dataset; noise type; corruption level;
  + Run ```python noising_process.py```

#### Running a model:

+ Go to ./src/scripts/ to run a specific model (choose from scripts therein):
  + Make sure you pick correct hyper-parameters (see paper), and turn ```--cuda-on``` for GPU. For instance:
    + ```sh run_RVAE_CVI.sh``` , for our main algorithm.
    + ```sh run_VAE_l2.sh``` , for VAE-L2 baseline.
    + ```sh run_CondPred.sh``` , for NN-based Conditional Predictor (pseudo-likelihood).
    + ```sh run_baselines.sh``` , for assorted baselines in paper.


## License

 MIT
