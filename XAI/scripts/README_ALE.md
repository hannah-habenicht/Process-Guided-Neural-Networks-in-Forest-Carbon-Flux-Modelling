# ALE for trained spatial models

`XAI/scripts/run_ale.py` and the notebooks `XAI/run_ale.ipynb` (with plots) and `XAI/scripts/run_ale.ipynb` compute Accumulated Local Effects (ALE) for the models of a spatial CV design and save per-model-type caches plus a bundle. Run the script from the repository root; start each notebook from its own folder.

## Variants
- `main`: 1D ALE of each feature.
- `pairwise`: 2D ALE, interaction only.
- `full`: 2D ALE, main effects + interaction. It equals `main` + `pairwise` on the same grid.

By default PRELES is held fixed, so the ALE shows the network alone; for `parallel` that is the network part without the PRELES offset. With `ALE_CHAIN_PROCESS=1` (`main` and `full`, hybrid models only) each perturbed day is re-run through PRELES and then the network, which gives the full response to the weather.

## Settings (environment variables)
| Variable | Meaning | Default |
|---|---|---|
| `ALE_VARIANT` | `main`, `pairwise` or `full` | `pairwise` in the script, `main` in the notebooks |
| `ALE_LENGTH` | `len3`, `len6`, `len9` or `len12` | `len12` |
| `ALE_GRID_SIZE` | grid resolution; finer is slower | `20` |
| `ALE_CHAIN_PROCESS` | `1` to run through PRELES | off |
| `ALE_PAIRS` | notebooks only, with `full`, e.g. `PAR_VPD,Tair_VPD` | all pairs |
| `ALE_PREDICTION_TYPES` | script only, comma-separated | `naive,bias,parallel,regularized,residual` |
| `ALE_LOGS_ROOT` | script only: folder holding `logs_spatial_<model type>/` | `checkpoints/preles/full/station_<length>` |
| `CHECKPOINT_PATH` | notebooks only: replaces `checkpoints/` | `checkpoints` |
| `ALE_SAVE_DIR_BASE` | script only: output folder | `XAI/results/ALE/paper1_spatial` |
| `ALE_IMPUTE_EMPTY_CELLS` | script only: `0` keeps empty `pairwise` bins empty | on |

## Inputs
- The CV design `configs/CV_splits/Paper_1_preles_ncvs/CV_station_<length>_config.json`.
- Trained models in `<logs root>/logs_spatial_<model type>/outer_station_<site>/` (`config_<k>_summary.json`, `models/best_model_<k>.pth`, `models/normalization.json`).
- The PRELES calibrations for the models' training splits in `Data/calibrations/`.

## Outputs
Under `<save dir>/spatial_<length>/grid<grid size>/<variant>[_chain]/`:
- `<variant>_<model type>.joblib`: per-type caches. Existing caches are reused, so a stopped run resumes.
- `<variant>_bundle.joblib`: all results, keyed by model type, block, fold, draw (`config_0` is the best HPO configuration), split (`train`, `test`), target and feature.
- `<variant>_normalization_stats.json`: the train-fitted constants per fold and draw.
- `<variant>_meta.json`: the run settings.

Model ALEs are stored in the z-scored model space, except that chained curves keep physical x values. Process-model ALEs are physical. `XAI.scripts.plot_ale.to_physical_units(bundle, stats)` converts the model curves to physical units; the notebook and `plot_ale.py` plot in physical units through it.

## Plots
`XAI/run_ale.ipynb` plots the run it computed; `python XAI/scripts/plot_ale.py` plots a saved bundle (edit its CONFIG block). Both show the `config_0` draw.
