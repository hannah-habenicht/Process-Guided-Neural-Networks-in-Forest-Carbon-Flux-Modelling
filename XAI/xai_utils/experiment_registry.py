# XAI/xai_utils/experiment_registry.py
"""
Registry of experiment configurations.

Each entry defines the paths and settings for a specific experiment variant.
Used by the ALE scripts (run_ale.py / plot_ale.py).

Keys per entry
--------------
cv_config_name  : CV split config JSON, relative to configs/CV_splits/.
                  None = derived from `length` as Paper_1_preles_ncvs/CV_station_{length}_config.json.
logs_root       : trained-model checkpoint directory, relative to the repository root.
                  None = derived from `length` as checkpoints/preles/full/station_{length}.
save_dir_base   : base directory under which ALE results are written.
                  Set to None if no ALE has been computed for this experiment.
ncv_kind        : CV kind string used in the ALE output path ("spatial", "koppen", ...).
length          : fold-length string (e.g. "6folds", "s-fold", "f-fold").
use_climate_plot: if True, plot_ale.py calls plot_ale_by_climate instead of plot_ale_results.
dashboard_key   : short key used by the OOD dashboard (e.g. "s", "w-v2").
                  Set to None for experiments not shown in the dashboard.
fold_label      : human-readable label used in dashboard plots.
                  Set to None for experiments not shown in the dashboard.
dashboard_active: if True, this fold type is included in the default FOLD_DIRS set
                  (i.e. loaded by run_dashboard.py by default).
                  False = registered for ALE/reference use but not in the default dashboard run.
"""

EXPERIMENT_CONFIGS: dict = {
    # ----- Paper-1 spatial NCV (PRELES) -----
    # cv_config_name / logs_root are derived from `length`; set ALE_LENGTH=len3|len6|len9|len12.
    "spatial": {
        "cv_config_name": None,
        "logs_root": None,
        "save_dir_base": "XAI/results/ALE/paper1_spatial",
        "ncv_kind": "spatial",
        "length": "len12",
        "use_climate_plot": False,
        "dashboard_key": None,
        "fold_label": None,
        "dashboard_active": False,
    },
}
