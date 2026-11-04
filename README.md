# Accompanying Code for
# "Evaluating Transferability and Robustness of Process-Guided Neural Networks in Forest Carbon Flux Modelling"

This repository combines the semi-empirical process model **PRELES** with neural networks to build process-guided neural networks (PGNNs) for forest carbon flux modelling (GPP, ET). Models are evaluated with nested cross-validation (NCV) across sites (spatial) and across years at one site (temporal), at full and sparse data resolution.

Available model types:
- `naive` (meteorological inputs only)
- `bias`
- `parallel`
- `regularized`
- `residual`
- `Mixture` (mixture of experts with PRELES as a fixed expert)
- `process` (PRELES baseline, used in result summaries)

## Versions
This repository is the code for the manuscript. It supersedes the first preprint version. The Zenodo record [10.5281/zenodo.18631792](https://doi.org/10.5281/zenodo.23942881).

## 1) Environment

Python 3.12 is the reference version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

or `conda env create -f environment.yml && conda activate bo2-pgnn`.

Training logs to [Weights & Biases](https://wandb.ai). Either run `wandb login`, or switch it off with `export WANDB_MODE=disabled` (or `offline`).

Run all scripts from the repository root; data, configs and checkpoint paths are relative to it. Optional settings go in a `.env` file (see `.env.example`).

## 2) Build PRELES

### `preles_c` (required)
The pipeline runs PRELES through this C extension (`PM_Extension/prelesR/src`). Needs a C/C++ compiler, `pybind11` and `numpy`.

```bash
cd PM_Extension/prelesR/src
python setup.py build
cd ../../..
```

The build lands in `PM_Extension/prelesR/src/build/lib.<platform>-<python>/`, which `PRELES_dataset.py` finds automatically.

Check:
```bash
python -c "import PRELES_dataset as P; print(P.preles)"
```
This should print the `preles_c` module, not `None`.

### `preles` torch rewrite (optional)
`PM_Extension/PRELES_rewrite` is the differentiable, row-by-row C++/PyTorch version used by the first preprint version. The pipeline only uses it when a config selects `Preles::backend = "preles_rewrite"` or `"torch"`. Build instructions: `PM_Extension/PRELES_rewrite/README.md`.

## 3) Data and calibrations

- `Data/preprocessed_data/station_{5,12,14,21}.csv`: daily forcing and fluxes for Collelongo (5), Hyytiälä (12), Le Bray (14) and Sorø (21).
- `Data/calibrations/preles/spatial/{multi_site,single_site}/{full,sparse7,sparse30,sparse50}/<train split>/`: calibrated PRELES parameters (`params.json`) and provenance (`metadata.json`) for every outer and inner train split of the manuscript runs listed in `batch_scripts/metaschedule_paper1.sh`. Temporal calibrations are single-site bundles in the same tree. Other CV designs need calibrating first (section 4).

Check that a CV design has all of its calibrations:
```bash
python Utilityscripts/preflight_calibrations.py \
    --cv-config configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json [--sparsity 30]
```

## 4) Running

### CV designs
`configs/CV_splits/Paper_1_preles_ncvs/`:
- **Spatial** (leave-one-site-out, train on the other three sites): `CV_station_len{3,6,9,12}_config.json`, where `lenN` is the number of training years. Outer folds `outer_station_<id>`.
- **Temporal** (one site, held-out years): `temporal_cv_train_len{1,2,3,5,8}_station_5.json`, `new_temporal_cv_train_len{1,2,3}_station_{12,14}.json`. Outer folds `out_l_<n>`.

### Frozen configs
`configs/config_spatial/<model>[_sparse<N>].json` (model = `naive`, `bias`, `parallel`, `regularized`, `residual`) and `configs/config_spatial_mixture/parallel[_sparse<N>].json` for the mixture of experts. Spatial and temporal runs both use these.

### Calibrate PRELES
Only needed for CV designs or sparsities without shipped bundles:
```bash
python process_model_calibration_manager.py \
    --config configs/config_spatial/naive.json \
    --cv-config configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json \
    --outer-fold outer_station_5 [--inner-loop inner_loop_1] \
    --calibrate auto --no-plot
```
Without `--inner-loop` it calibrates the outer train split. Use a `_sparse<N>` config for sparse calibrations.

### Train (nested CV)
```bash
export spatial_training_years_len=len3   # names the output folder; match the CV file
python Optuna_spatial_NCV.py \
    --config configs/config_spatial/residual.json \
    --CV_config_path configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json \
    --outer_fold outer_station_5 \
    --n_trials 100 --top_trials 5 --workers_per_fold 1 --worker_id 0
```
- Temporal: `Optuna_temporal_NCV.py` with a temporal CV file and an `out_l_<n>` fold (set `temporal_training_years_len` instead).
- Mixture of experts: `Optuna_spatial_NCV_noisy_expert.py` / `Optuna_temporal_NCV_noisy_expert.py` with a `config_spatial_mixture` config.
- Set `calibration_flag=require_existing` to fail instead of calibrating when a bundle is missing.
- These and the other optional settings can also go in a `.env` file in the repository root; see `.env.example`.

Outputs go to `checkpoints/preles/<full|sparseN>/<experiment>/logs_spatial_<model>/<outer fold>/`, with the final model in `models/best_model_<k>.pth` and its normalisation constants in `models/normalization.json`.

### SLURM
`batch_scripts/` holds the scripts used for the manuscript runs (written for the University of Freiburg cluster; adjust partitions, resources and `--exclude` for your system). Submit from the repository root:
- `metaschedule_paper1.sh`: all architectures × all paper CV designs/sparsities; checks calibrations first.
- `submit_paper1.sh`: one CV design and architecture (`LANE=spatial|temporal`, `CV_CONFIG_PATH`, `prediction_loss`, `SPARSITY`).
- `batch_calibrations_spatial.sh`, `batch_calibrations_temporal.sh`: PRELES calibration arrays.

### ALE
`XAI/scripts/run_ale.py` computes ALE explanations for trained models: `main`, `pairwise` or `full`, optionally through PRELES (`ALE_CHAIN_PROCESS=1`). See `XAI/scripts/README_ALE.md`.

## 5) Notebooks and figure scripts

- `Load_and_test_model.ipynb`: loads one trained model (`naive`, `bias`, `parallel`, `regularized` or `residual`) from `checkpoints/preles/full/station_<length>/`, rebuilds its test data with the shipped PRELES calibration and the model's saved normalisation, and plots predicted vs. observed GPP and ET in physical units. Set `CHECKPOINT_PATH` if the models live elsewhere.
- `XAI/run_ale.ipynb` computes and plots ALE explanations for trained models; `XAI/scripts/run_ale.ipynb` computes them only.
- `Visualisation_generalisation_error_violin.ipynb`, `Visualisation_generalisation_error_ID.py` and `Visualsation_generalisation_error_OOD.py` plot the generalisation errors from `results/result_sum/`. Download that folder from the Zenodo record of this paper first.

## 6) Repository structure

- `Optuna_{spatial,temporal}_NCV[_noisy_expert].py`: nested-CV drivers (HPO + outer training)
- `process_model_calibration_manager.py`, `calibration_paths.py`, `calibration_dataloader_handoff.py`: PRELES calibration and the process-model → dataset handoff
- `PRELES_dataset.py`, `Basic_dataset.py`: PRELES runs and data loading
- `NN_framework.py`, `MLP_Model.py`, `noisy_mixture_v2.py`: models and training
- `Config_Files/`, `configs/`: config builders, frozen configs, CV splits
- `PM_Extension/prelesR/src/`: `preles_c` source; `PM_Extension/PRELES_rewrite/`: torch rewrite
- `Utilityscripts/`: calibration preflight
- `XAI/`: ALE utilities
- `batch_scripts/`: SLURM scripts

Only PRELES is included and supported here. Some modules keep unused hooks for other process models (e.g. BASFOR).

## 7) Citation

This code accompanies the manuscript "Evaluating Transferability and Robustness of Process-Guided Neural Networks in Forest Carbon Flux Modelling". If you use it, please cite the paper and the PRELES sources.

## 8) License

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the paper (section 7).

Not covered by this license:
- The PRELES source and its rewrite in `PM_Extension/`, which keep their own terms (for the rewrite, see `PM_Extension/PRELES_rewrite/README.md`).
- The station data in `Data/preprocessed_data/`, which derive from openly available third-party datasets and remain under the terms of those sources.
