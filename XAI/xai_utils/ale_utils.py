# xai_utils/ale_utils.py

from collections import defaultdict
import os
import sys
import json
from PyALE import ale
import glob
import joblib
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Sequence, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from process_model_calibration_manager import ProcessModelCalibrationManager
from XAI.xai_utils.batch_utils import batch_explain
from XAI.xai_utils.model_utils import (
    TorchPredictWrapper,
    ProcessModelPredictWrapper,
    ChainedPredictWrapper,
    Wrapper,
    _prepare_fold_config,
    _load_fold_summary,
    _load_model_and_dataset,
    infer_ncv_kind,
    iter_summary_dirs,
    iter_fold_configs,
)

#PAIRWISE_BASE_FEATURES = ["PAR", "Tair", "VPD", "Precip", "fapar"]
RESIDUAL_PRED_FEATURES = ["GPP_pred", "ET_pred"]

# Meteorological seasons by calendar month
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def ale_correlated_full_effect(X, model, features, grid_size=5, impute_empty_cells=True):
    """
    Combined 2D ALE surface: main ALE(x0) + main ALE(x1) + centred pairwise interaction.

    The interaction is PyALE's 2D ALE (identical to the ``pairwise`` variant); the main
    effects are PyALE's 1D ALE of each feature on the same X and grid_size (identical to
    the ``main`` variant), interpolated onto the 2D grid edges. Each part is centred on X,
    so the surface shows the response relative to the average, not the level.

    Returns:
        pd.DataFrame: rows = grid edges of features[0], columns = grid edges of features[1].
    """
    import numpy as np
    from PyALE._src.ALE_1D import aleplot_1D_continuous
    from PyALE._src.ALE_2D import aleplot_2D_continuous

    X = X.reset_index(drop=True)
    inter = aleplot_2D_continuous(X, model, features, grid_size=grid_size,
                                  impute_empty_cells=impute_empty_cells)
    main_0 = aleplot_1D_continuous(X, model, features[0], grid_size=grid_size, include_CI=False)
    main_1 = aleplot_1D_continuous(X, model, features[1], grid_size=grid_size, include_CI=False)

    def _main_on(edges, main):
        return np.interp(np.asarray(edges, dtype=float),
                         main.index.to_numpy(dtype=float), main["eff"].to_numpy(dtype=float))

    f0 = _main_on(inter.index, main_0)
    f1 = _main_on(inter.columns, main_1)
    return inter + f0[:, None] + f1[None, :]


def split_dates(frame: pd.DataFrame, station_split: dict) -> pd.DatetimeIndex:
    """Calendar date of every row of a fold frame.

    The loader concatenates sites in split order, each site's years sorted and
    whole (Jan 1 - Dec 31), so the dates follow from the split. Asserted against
    the frame's DOY column, which fails if rows were dropped or reordered.
    """
    if isinstance(next(iter(station_split.values())), dict):
        raise ValueError("split_dates needs a {station: [years]} split, not date ranges.")
    dates = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{y}-01-01", f"{y}-12-31").to_numpy()
        for sid in station_split for y in sorted(station_split[sid])
    ]))
    if len(dates) != len(frame):
        raise ValueError(f"split has {len(dates)} days but the frame has {len(frame)} rows.")
    if not np.array_equal(dates.dayofyear.to_numpy(), frame["DOY"].to_numpy(dtype=int)):
        raise ValueError("frame DOY does not match the dates rebuilt from the split.")
    return dates


def season_rows(frame: pd.DataFrame, station_split: dict, season: str) -> np.ndarray:
    """Row positions of a fold frame that fall in a meteorological season."""
    return np.flatnonzero(split_dates(frame, station_split).month.isin(SEASON_MONTHS[season]))


def _get_phy_para(config: dict) -> list:
    """Return the process-model physical output columns, regardless of PM.

    Prefers model::io::y_cols to derive columns (matches calibration_dataloader_handoff
    training behaviour). Falls back to legacy Basfor/Preles::y_phy_para keys.
    """
    y_io_cols = config.get("model::io::y_cols")
    if y_io_cols:
        return [f"{y}_pred" for y in y_io_cols]
    pm = config.get("process_model::name", "preles").capitalize()
    return (
        config.get(f"{pm}::y_phy_para")
        or config.get("Preles::y_phy_para")
        or []
    )


def _get_x_para(config: dict) -> list:
    """Return the process-model input feature columns, regardless of PM."""
    pm = config.get("process_model::name", "preles").capitalize()
    return (
        config.get("Process::x_para")
        or config.get(f"{pm}::x_para")
        or config.get("Preles::x_para")
        or []
    )



def _unique_feature_pairs(features: Sequence[str]) -> list[list[str]]:
    seen = set()
    pairs: list[list[str]] = []
    for a, b in combinations(features, 2):
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([a, b])
    return pairs


def _pairwise_cell_counts(ale_results: dict, X: pd.DataFrame, *, fill: bool) -> dict:
    """Rows per 2D ALE grid cell, binned exactly like PyALE (pd.cut on the surface's edges).

    Lets strength summaries down-weight empty/sparse cells (filled by imputation).
    ``fill`` mirrors how the pair's rows were prepared: ffill/bfill (process model)
    or dropna (NN). GPP and ET share one grid, so the GPP surfaces are used.
    Returns {pair_key: DataFrame (n0 x n1), index/columns = upper cell edges}.
    """
    counts = {}
    for key, surface in (ale_results.get("GPP") or {}).items():
        if not isinstance(surface, pd.DataFrame):
            continue
        f0, f1 = surface.index.name, surface.columns.name
        if f0 not in X.columns or f1 not in X.columns:
            continue
        xp = X[[f0, f1]].ffill().bfill() if fill else X[[f0, f1]].dropna()
        e0 = np.asarray(surface.index, dtype=float)
        e1 = np.asarray(surface.columns, dtype=float)
        c0 = pd.cut(xp[f0], e0, include_lowest=True).cat.codes.to_numpy()
        c1 = pd.cut(xp[f1], e1, include_lowest=True).cat.codes.to_numpy()
        grid = np.zeros((len(e0) - 1, len(e1) - 1), dtype=int)
        ok = (c0 >= 0) & (c1 >= 0)
        np.add.at(grid, (c0[ok], c1[ok]), 1)
        counts[key] = pd.DataFrame(
            grid, index=pd.Index(e0[1:], name=f0), columns=pd.Index(e1[1:], name=f1)
        )
    return counts


def _drop_constant_pairs(X_df: pd.DataFrame, feature_pairs) -> list:
    """Drop pairs containing a feature with <2 distinct values (e.g. PRELES CO2).

    PyALE's 2D grid for a constant feature is empty and fails with
    "operands could not be broadcast together with shapes (20,) (0,)".
    """
    constant = {f for pair in feature_pairs for f in pair if X_df[f].nunique(dropna=True) < 2}
    if constant:
        print(f"[INFO] Skipping pairwise ALE for constant feature(s): {sorted(constant)}")
    return [pair for pair in feature_pairs if not constant.intersection(pair)]


def _resolve_feature_list(
    pred_type: str,
    config: dict,
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Return (X_df_train, X_df_test, feature_list) for ALE computation.

    For residual prediction type the base x_parameter may not include
    GPP_pred/ET_pred (and any additional pred cols) if the config was written
    before that convention was standardised.  This function detects that case
    and augments the feature selection so that all columns the model actually
    received at training time are included.
    """
    x_cols = list(config.get("data::train::x_parameter", []))
    X_train = train_dataset[x_cols]
    X_test  = test_dataset[x_cols]
    feature_list = list(X_test.columns)

    if pred_type.lower() == "residual":
        # Standard process-model pred cols + any additional ones registered in config
        y_cols = list(config.get("data::train::y_parameter", []))
        standard_pred = [f"{y}_pred" for y in y_cols]
        additional_pred = list(config.get("model::io::y_pred_additional_cols") or [])
        all_pred_cols = list(dict.fromkeys(standard_pred + additional_pred))

        missing = [
            f for f in all_pred_cols
            if f not in feature_list
            and f in train_dataset.columns
            and f in test_dataset.columns
        ]
        if missing:
            feature_list = feature_list + missing
            X_train = train_dataset[feature_list]
            X_test  = test_dataset[feature_list]

    return X_train, X_test, feature_list


def _normalise_fold_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply z-score normalisation to fold DataFrames, mirroring SpatialDataset.

    Returns (train_df, test_df, norm_stats) where norm_stats contains the
    per-feature mean and std fitted on the training split.  These are needed
    to convert z-scored ALE x-axes and effect values back to physical units
    when plotting.
    """
    if config.get("data::normalise") != "z-score":
        return train_df, test_df, {}

    x_cols = [c for c in config.get("data::train::x_parameter", []) if c in train_df.columns]
    y_cols = [c for c in config.get("data::train::y_parameter", []) if c in train_df.columns]
    _standard_pred = [f"{y}_pred" for y in config.get("data::train::y_parameter", [])]
    _additional_pred = list(config.get("model::io::y_pred_additional_cols") or [])
    _all_pred = list(dict.fromkeys(_standard_pred + _additional_pred))
    y_pred = [c for c in _all_pred if c in train_df.columns]

    x_mean = train_df[x_cols].mean().to_numpy(dtype=np.float32)
    x_std  = train_df[x_cols].std(ddof=0).replace(0, 1).to_numpy(dtype=np.float32)
    y_mean = train_df[y_cols].mean().to_numpy(dtype=np.float32)
    y_std  = train_df[y_cols].std(ddof=0).replace(0, 1).to_numpy(dtype=np.float32)

    norm_stats = {
        "x_parameters": x_cols,
        "x_mean":        x_mean.tolist(),
        "x_std":         x_std.tolist(),
        "y_parameters":  y_cols,
        "y_mean":        y_mean.tolist(),
        "y_std":         y_std.tolist(),
    }

    # y_pred columns that are already in x_cols have been normalised by x stats
    # in the x_cols pass; don't normalise them a second time with y stats.
    x_cols_set = set(x_cols)
    y_pred_only = [c for c in y_pred if c not in x_cols_set]

    def _apply(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[x_cols] = (df[x_cols].to_numpy(dtype=np.float32) - x_mean) / x_std
        df[y_cols] = (df[y_cols].to_numpy(dtype=np.float32) - y_mean) / y_std
        if y_pred_only:
            n = min(len(y_pred_only), len(y_mean))
            df[y_pred_only[:n]] = (df[y_pred_only[:n]].to_numpy(dtype=np.float32) - y_mean[:n]) / y_std[:n]
        return df

    return _apply(train_df), _apply(test_df), norm_stats

def _localise_fold_config(config: dict, summary: dict, save: str) -> dict:
    """Re-point cluster paths to the local checkout and pin the train split to the
    one the checkpoint was trained on.

    The checkpoint's train split decides which calibration bundle the *_pred inputs
    came from and which train data the z-score constants were fitted on, so it must
    win over the CV JSON when the two have drifted (e.g. a year removed later).
    """
    config["cfg::save"] = save
    config["checkpoint::path_dir"] = os.path.join(save, "checkpoints")
    config["data::train::path_dir"] = config["data::test::path_dir"] = os.path.join(save, "Data")
    config["calibration::root"] = os.path.join(save, "Data", "calibrations")

    ckpt_train = summary.get("config", {}).get("data::train::station_id")
    # Optuna_temporal_NCV.py saved the config template's {startdate, stopdate} split
    # (DK-Sor/FI-Hyy/IT-Col) instead of the fold's; that records nothing, keep the CV JSON's.
    if isinstance(ckpt_train, dict) and any(isinstance(v, dict) for v in ckpt_train.values()):
        ckpt_train = None
    if ckpt_train and ckpt_train != config["data::train::station_id"]:
        print(
            f"[WARNING] {summary.get('outer_fold')}: CV JSON train split differs from the "
            f"checkpoint's; using the checkpoint's.\n  CV JSON:    {config['data::train::station_id']}"
            f"\n  checkpoint: {ckpt_train}"
        )
        config["data::train::station_id"] = ckpt_train
    return config


def _get_test_dataloader(config):
    """
    Return the raw Dataset objects (BasforDataset / PrelesDataset) for a fold,
    suitable for passing into run_ale_for_process_model, plus the fold's
    calibrated PRELES parameters (None for BASFOR, which resolves its own).

    Uses load_fold_datasets_embedded so the process model can be called
    repeatedly with modified inputs during ALE computation.
    """
    config = {**config, "data::normalise": None}
    pm_manager = ProcessModelCalibrationManager(
        config=config,
        process_model=config.get("process_model::name", "preles"),
        auto_merge_pm_config=False,
    )
    train_ds, test_ds, _ = pm_manager.load_fold_datasets_embedded(
        train_stations=config["data::train::station_id"],
        test_stations=config["data::test::station_id"],
        val_stations=None,
    )

    params = None
    if pm_manager.process_model == "preles":
        # The embedded loader only carries default parameters; load the fold's calibration.
        paths = pm_manager._calibration_paths(config["data::train::station_id"], sparse=pm_manager._resolve_sparse("train"))
        params, source = pm_manager._load_existing_calibrations(
            paths=paths,
            train_stations=config["data::train::station_id"],
            sparse=pm_manager._resolve_sparse("train"),
            mode="require_existing",
        )
        if params is None:
            raise FileNotFoundError(f"No PRELES calibration bundle at '{paths['folder']}' (source={source}).")
    return train_ds, test_ds, params

def run_ale_for_model(model, df, feature_list, grid_size = 10, prediction_type = "naive", y_p=None, model_device="cpu"):
    """
    Runs ALE calculations for a given model and dataframe of features.
    
    Parameters:
        model: Trained PyTorch model
        df: DataFrame containing input features
        feature_list: List of features to generate ALE plots for
        prediction_type: "naive", "parallel", or "regularized"
        y_p: Optional process model predictions (numpy array, same shape as model output)
        target_idx: Which output target to plot ALE for (0 = GPP, 1 = ET)
        model_device: "cpu" or "cuda"
        save_dir: Optional path to save ALE plots
    """

    ale_results = {"GPP": {}, "ET": {}}
    y_p_aligned = pd.DataFrame(y_p, index=df.index) if y_p is not None else None
    wrapper = TorchPredictWrapper(model, model_device=model_device, prediction_type=prediction_type, y_p=y_p_aligned)
    for idx, name in enumerate(["GPP", "ET"]):
        predict_fn = wrapper.get_predict_fn(target_idx=idx)
        for feature in feature_list:
            try:
                result = ale(
                    X=df,
                    model=Wrapper(predict_fn),
                    feature=[feature],
                    include_CI=True,
                    C=0.95,
                    grid_size=grid_size,
                    impute_empty_cells=True,
                    plot=False
                )
                result = result.reset_index().rename(columns={"index": feature})
                result["feature"] = feature
                result["target"] = name
                ale_results[name][feature] = result
            except Exception as e:
                print(f"[WARNING] Skipping {feature}-{name} due to error: {e}")
                continue

    return ale_results

def run_pairwise_ale_for_model(
    model,
    X_df,
    feature_pairs,
    grid_size=10,
    model_device="cpu",
    prediction_type="naive",
    y_p=None,
    impute_empty_cells=True,
):
    """
    Run 2D ALE (interaction) analysis for selected feature pairs.
    
    Args:
        model: trained model
        X_df: input features (pandas DataFrame)
        feature_pairs: list of 2-element lists (e.g., [["PAR", "fapar"], ["VPD", "Tair"]])
        impute_empty_cells: whether to interpolate missing grid cells before computing ALE
        impute_empty_cells: whether to interpolate missing grid cells before computing ALE
    """

    feature_pairs = _drop_constant_pairs(X_df, feature_pairs)
    ale_results = {"GPP": {}, "ET": {}}
    y_p_aligned = pd.DataFrame(y_p, index=X_df.index) if y_p is not None else None
    wrapper = TorchPredictWrapper(model, model_device=model_device, prediction_type=prediction_type, y_p=y_p_aligned)
    for idx, name in enumerate(["GPP", "ET"]):
        predict_fn = wrapper.get_predict_fn(target_idx=idx)
        for pair in feature_pairs:
            try:
                # Drop rows where either feature is NaN; PyALE can't handle
                # mismatched sample counts across features in a pair.
                X_pair = X_df.dropna(subset=pair)
                result = ale(
                    X=X_pair,
                    model=Wrapper(predict_fn),
                    feature=pair,
                    include_CI=False,
                    C=0.95,
                    grid_size=grid_size,
                    impute_empty_cells=impute_empty_cells,
                    plot=False
                )
                key = f"{pair[0]}_{pair[1]}"
                ale_results[name][key] = result

            except Exception as e:
                print(f"[WARNING] Failed for pair {pair}: {e}")
    
    return ale_results

def run_pairwise_fulleffect_ale_for_model(model, X_df, feature_pairs, grid_size = 10, model_device="cpu", prediction_type="naive", y_p=None, save_dir=None):
    """
    Run 2D ALE (full effect: main effects + interaction) for selected feature pairs.
    
    Args:
        model: trained model
        X_df: input features (pandas DataFrame)
        feature_pairs: list of 2-element lists (e.g., [["PAR", "fapar"], ["VPD", "Tair"]])
    """
    feature_pairs = _drop_constant_pairs(X_df, feature_pairs)
    ale_results = {"GPP": {}, "ET": {}}
    y_p_aligned = pd.DataFrame(y_p, index=X_df.index) if y_p is not None else None
    wrapper = TorchPredictWrapper(model, model_device=model_device, prediction_type=prediction_type, y_p=y_p_aligned)
    for idx, name in enumerate(["GPP", "ET"]):
        predict_fn = wrapper.get_predict_fn(target_idx=idx)
        for pair in feature_pairs:
            try:
                X_pair = X_df.dropna(subset=pair)
                results = ale_correlated_full_effect(
                    X=X_pair,
                    model=Wrapper(predict_fn),
                    features=pair,
                    grid_size=grid_size,
                    impute_empty_cells=True
                )
                key = f"{pair[0]}_{pair[1]}"
                ale_results[name][key] = results

            except Exception as e:
                print(f"[WARNING] Failed for pair {pair}: {e}")
    
    return ale_results

def run_ale_for_process_model(dataloader, x_parameter, grid_size = 10, save_dir=None, params=None, rows=None):
    """
    Runs ALE plots for a process model based on dataloader inputs.
    Plots GPP and ET as subplots for each feature.

    rows: optional row positions to restrict the ALE to (e.g. one season); the
    sequence is still simulated in full, only these rows are perturbed and scored.
    """
    # Positional index: the wrapper patches rows back into the full sequence by position
    x_df = dataloader.data[x_parameter].reset_index(drop=True)
    if rows is not None:
        x_df = x_df.iloc[rows]
    ale_results = {"GPP": {}, "ET": {}}

    for idx, name in enumerate(["GPP", "ET"]):
        wrapper = ProcessModelPredictWrapper(dataloader, x_parameter=x_parameter, target_idx=idx, params=params) #ALE needs a predict function
        for feature in x_parameter:
            try:
                result = ale(
                    X=x_df,
                    model=wrapper,
                    feature=[feature],
                    include_CI=True,
                    C=0.95,
                    grid_size=grid_size,
                    plot=False
                )
                result = result.reset_index().rename(columns={"index": feature})
                result["feature"] = feature
                result["target"] = name
                result["prediction_type"] = "process"

                # print(f"[ALE Result] {feature} | Min x: {result.index.min()}, Max x: {result.index.max()}")
                # print(f"[x_df DEBUG] Min PAR: {x_df['PAR'].min()}, Max PAR: {x_df['PAR'].max()}")
                
                ale_results[name][feature] = result
            except Exception as e:
                print(f"[WARNING] Skipping {feature}-{name} due to error: {e}")
                continue

    return ale_results

def run_chained_ale(dataloader, x_parameter, features, model, prediction_type, nn_frame, fold_norm,
                    grid_size=10, params=None):
    """
    ALE of a hybrid's full response to the weather: each perturbed day is re-run through
    PRELES (local, from its unperturbed start-of-day state) and the network gets the perturbed
    weather plus the new PRELES outputs (ChainedPredictWrapper). x stays physical (PRELES
    driver frame); the effect is in the network's z-units of the target.
    """
    x_df = dataloader.data[x_parameter].reset_index(drop=True)  # positional index, as the process ALE
    process_wrapper = ProcessModelPredictWrapper(dataloader, x_parameter=x_parameter, params=params)
    ale_results = {"GPP": {}, "ET": {}}
    for idx, name in enumerate(["GPP", "ET"]):
        wrapper = ChainedPredictWrapper(process_wrapper, model, prediction_type, nn_frame, fold_norm,
                                        weather_cols=features, target_idx=idx)
        for feature in features:
            try:
                result = ale(X=x_df, model=wrapper, feature=[feature], include_CI=True, C=0.95,
                             grid_size=grid_size, plot=False)
                result = result.reset_index().rename(columns={"index": feature})
                result["feature"] = feature
                result["target"] = name
                ale_results[name][feature] = result
            except Exception as e:
                print(f"[WARNING] Skipping chained {feature}-{name} due to error: {e}")
                continue
    return ale_results


def run_chained_full_ale(dataloader, x_parameter, feature_pairs, model, prediction_type, nn_frame, fold_norm,
                         grid_size=10, params=None):
    """
    Full-effect 2D ALE (main + interaction, ale_correlated_full_effect) of a hybrid's response to
    pairs of weather drivers through PRELES (ChainedPredictWrapper, as run_chained_ale). x stays
    physical (PRELES driver frame); the effect is in the network's z-units of the target.
    """
    x_df = dataloader.data[x_parameter].reset_index(drop=True)  # positional index, as the process ALE
    process_wrapper = ProcessModelPredictWrapper(dataloader, x_parameter=x_parameter, params=params)
    weather = sorted({f for pair in feature_pairs for f in pair})
    ale_results = {"GPP": {}, "ET": {}}
    for idx, name in enumerate(["GPP", "ET"]):
        wrapper = ChainedPredictWrapper(process_wrapper, model, prediction_type, nn_frame, fold_norm,
                                        weather_cols=weather, target_idx=idx)
        for pair in feature_pairs:
            try:
                ale_results[name][f"{pair[0]}_{pair[1]}"] = ale_correlated_full_effect(
                    X=x_df, model=wrapper, features=pair, grid_size=grid_size, impute_empty_cells=True)
            except Exception as e:
                print(f"[WARNING] Skipping chained {pair}-{name} due to error: {e}")
                continue
    return ale_results


def _select_pairs(pairs, wanted):
    """Keep the pairs named in ``wanted`` (e.g. [["PAR", "VPD"]]), in either order; None = all."""
    if not wanted:
        return pairs
    keys = {tuple(sorted(w)) for w in wanted}
    return [p for p in pairs if tuple(sorted(p)) in keys]


def run_pairwise_ale_for_process_model(
    dataloader,
    x_parameter,
    feature_pairs,
    grid_size=10,
    save_dir=None,
    impute_empty_cells=True,
    params=None,
    ):
    """
    Run 2D ALE (interaction) analysis for selected feature pairs.
    
    Args:
        model: trained model
        X_df: input features (pandas DataFrame)
        feature_pairs: list of 2-element lists (e.g., [["PAR", "fapar"], ["VPD", "Tair"]])
    """
    x_df = dataloader.data[x_parameter].reset_index(drop=True)
    feature_pairs = _drop_constant_pairs(x_df, feature_pairs)

    ale_results = {"GPP": {}, "ET": {}}

    for idx, name in enumerate(["GPP", "ET"]):
        wrapper = ProcessModelPredictWrapper(dataloader, x_parameter=x_parameter, target_idx=idx, params=params)
        for pair in feature_pairs:
            try:
                # Impute NaN values rather than dropping rows: the process model
                # expects a fixed number of rows (eval_ndays per site), so removing
                # rows would corrupt the positional assignment into the weather matrix.
                x_pair = x_df.copy()
                x_pair[pair] = x_pair[pair].ffill().bfill()
                result = ale(
                    X=x_pair,
                    model=wrapper,
                    feature=pair,
                    include_CI=False,
                    C=0.95,
                    grid_size=grid_size,
                    impute_empty_cells=impute_empty_cells,
                    plot=False
                )
                key = f"{pair[0]}_{pair[1]}"
                ale_results[name][key] = result
            except Exception as e:
                print(f"[WARNING] Skipping {pair}-{name} due to error: {e}")
                continue

    return ale_results

def run_pairwise_full_ale_for_process_model(dataloader, x_parameter, feature_pairs, grid_size=10, save_dir=None,
                                            params=None, impute_empty_cells=True):
    """
    Run 2D ALE (full effect: main effects + interaction) for selected feature pairs.

    Args:
        model: trained model
        X_df: input features (pandas DataFrame)
        feature_pairs: list of 2-element lists (e.g., [["PAR", "fapar"], ["VPD", "Tair"]])
    """
    x_df = dataloader.data[x_parameter].reset_index(drop=True)
    feature_pairs = _drop_constant_pairs(x_df, feature_pairs)
    ale_results = {"GPP": {}, "ET": {}}

    for idx, name in enumerate(["GPP", "ET"]):
        wrapper = ProcessModelPredictWrapper(dataloader, x_parameter=x_parameter, target_idx=idx, params=params)
        for pair in feature_pairs:
            try:
                x_pair = x_df.copy()
                x_pair[pair] = x_pair[pair].ffill().bfill()
                result = ale_correlated_full_effect(
                    X=x_pair,
                    model=wrapper,
                    features=pair,
                    grid_size=grid_size,
                    impute_empty_cells=impute_empty_cells,
                )
                key = f"{pair[0]}_{pair[1]}"
                ale_results[name][key] = result
            except Exception as e:
                print(f"[WARNING] Skipping {pair}-{name} due to error: {e}")
                continue

    return ale_results

def generate_ale_top_models(
        prediction_types, 
        logs_root, 
        cv_config,
        target_list=["GPP", "ET"],
        grid_size=20,
        save_intermediate_dir=None,
        checkpoint_every=1,
        existing_results=None,
        season=None,
        preles_zero=False,
        outer_folds=None,
        chain_process=False,
        **kwargs

    ):
    """
    Generate ALE results for top models across outer folds and prediction types.

    Args:
        impute_empty_cells: when False, skip filling empty grid cells during pairwise ALE
        season: None for all days, or "DJF"/"MAM"/"JJA"/"SON" to compute the ALE on
            that season's days only (train and test split each restricted separately)
        preles_zero: set the process-model outputs to a physical 0 on every row — residual:
            the GPP_pred/ET_pred inputs (then constant, so no curve of their own); parallel:
            the added process offset y_p. Bias/naive have nothing to set and are skipped,
            and the process-model ALE (unaffected) is not computed.
        outer_folds: optional list of outer-fold ids to compute; None = all
        chain_process: ALE of the full response to the weather drivers (PRELES re-run per
            perturbed day, then the network; run_chained_ale) for residual / bias / parallel.
            Curves keep physical x; their normalisation stats carry identity x constants
            (x_units = "physical") and the real y constants. No process-model ALE.

    Returns:
        dict: {prediction_type: {outer_fold_id: ale_results, ...}, 'process': {...}}
    """
    # Initialize containers to collect all fold results
    all_model_1D_ale = defaultdict(dict)
    process_ale_results = {}
    all_norm_stats: dict = {}

    
    processed = 0  # count processed outer folds across all pred_types
    if existing_results:
        all_model_1D_ale.update(existing_results.get("model_1D_ale", {}))
        process_ale_results.update(existing_results.get("process_1D_ale", {}))
        all_norm_stats.update(existing_results.get("normalization_stats", {}))


    if chain_process and (season is not None or preles_zero):
        raise ValueError("chain_process cannot be combined with season or preles_zero.")
    for pred_type in prediction_types:
        if preles_zero and pred_type.lower() not in {"residual", "parallel"}:
            print(f"[ALE] preles_zero: skipping {pred_type} (no process output to set to 0)")
            continue
        if chain_process and pred_type.lower() not in {"residual", "bias", "parallel"}:
            print(f"[ALE] chain_process: skipping {pred_type} (does not use PRELES)")
            continue
        ncv_kind = infer_ncv_kind(logs_root, pred_type)
        logs_glob = os.path.join(logs_root, f"logs_spatial_{pred_type}*")
        log_dirs = glob.glob(logs_glob)
        if not log_dirs:
            print(f"[Warning] No log directories found for prediction type '{pred_type}' (pattern: {logs_glob})")
            continue
        
        print(f"[ALE] Generating for: {pred_type}")

        for log_path in log_dirs:
            for outer_fold in iter_summary_dirs(log_path):
                if outer_folds is not None and outer_fold not in outer_folds:
                    continue
                for config_idx, summary in enumerate(iter_fold_configs(log_path, outer_fold)):
                    config_key = f"config_{config_idx}"

                    if config_key in all_model_1D_ale[pred_type].get(outer_fold, {}):
                        print(f"[INFO] Skipping already processed fold {outer_fold} {config_key} for {pred_type}")
                        continue

                    config = _prepare_fold_config(summary, ncv_kind, cv_config)

                    config = _localise_fold_config(config, summary, PROJECT_ROOT + os.sep)

                    print("[Train Loader]")
                    train_dataloader, test_dataloader, pm_params = _get_test_dataloader(config)
                    train_dataset, test_dataset, model = _load_model_and_dataset(summary, config)
                    train_dataset, test_dataset, _fold_norm = _normalise_fold_datasets(train_dataset, test_dataset, config)
                    all_norm_stats.setdefault(pred_type, {}).setdefault(outer_fold, {})[config_key] = _fold_norm

                    X_df_train, X_df_test, feature_list = _resolve_feature_list(
                        pred_type, config, train_dataset, test_dataset
                    )

                    if pred_type.lower() in {"parallel", "mixture"}:
                        y_p_train = train_dataset[_get_phy_para(config)].values
                        y_p_test = test_dataset[_get_phy_para(config)].values
                    else:
                        y_p_train = y_p_test = None

                    if pred_type.lower() == "bias":
                        y_p_cols = [f"{t}_pred" for t in target_list]
                        X_df_test = test_dataset[y_p_cols]
                        X_df_train = train_dataset[y_p_cols]
                        feature_list = y_p_cols
                        y_p_train = y_p_test = None

                    if preles_zero:
                        # process outputs at a physical 0, in the model's z-units
                        if pred_type.lower() == "residual":
                            pcols = [c for c in (f"{t}_pred" for t in target_list) if c in X_df_train.columns]
                            for c in pcols:
                                i = _fold_norm["x_parameters"].index(c)
                                z0 = (0.0 - _fold_norm["x_mean"][i]) / _fold_norm["x_std"][i]
                                X_df_train = X_df_train.assign(**{c: z0})
                                X_df_test = X_df_test.assign(**{c: z0})
                            feature_list = [f for f in feature_list if f not in pcols]
                        else:  # parallel: y_p is scaled with the observed target's constants
                            z0 = np.array([(0.0 - _fold_norm["y_mean"][_fold_norm["y_parameters"].index(t)])
                                           / _fold_norm["y_std"][_fold_norm["y_parameters"].index(t)]
                                           for t in target_list], dtype=np.float32)
                            y_p_train = np.broadcast_to(z0, y_p_train.shape).copy()
                            y_p_test = np.broadcast_to(z0, y_p_test.shape).copy()

                    rows_train = rows_test = None
                    if season is not None:
                        # Dates from the process frames (raw DOY); the NN frames share their row order
                        rows_train = season_rows(train_dataloader.data, config["data::train::station_id"], season)
                        rows_test = season_rows(test_dataloader.data, config["data::test::station_id"], season)
                        if len(X_df_train) != len(train_dataloader.data) or len(X_df_test) != len(test_dataloader.data):
                            raise ValueError(f"{outer_fold}: NN and process frames differ in length; cannot share season rows.")
                        X_df_train, X_df_test = X_df_train.iloc[rows_train], X_df_test.iloc[rows_test]
                        if y_p_train is not None:
                            y_p_train, y_p_test = y_p_train[rows_train], y_p_test[rows_test]

                    fold_store = all_model_1D_ale[pred_type].setdefault(outer_fold, {}).setdefault(config_key, {})

                    if chain_process:
                        pm_x_para = _get_x_para(config)
                        weather = [f for f in pm_x_para if f not in ("CO2", "DOY")]
                        for split, dl, X_nn in (("train", train_dataloader, X_df_train),
                                                ("test", test_dataloader, X_df_test)):
                            if len(X_nn) != len(dl.data):
                                raise ValueError(f"{outer_fold}: NN and process frames differ in length.")
                            fold_store[split] = run_chained_ale(
                                dataloader=dl, x_parameter=pm_x_para, features=weather, model=model,
                                prediction_type=pred_type, nn_frame=X_nn, fold_norm=_fold_norm,
                                grid_size=grid_size, params=pm_params,
                            )
                        # x stays physical: identity x constants, real y constants (effect → physical)
                        all_norm_stats[pred_type][outer_fold][config_key] = {
                            "x_parameters": weather, "x_mean": [0.0] * len(weather), "x_std": [1.0] * len(weather),
                            "y_parameters": _fold_norm["y_parameters"], "y_mean": _fold_norm["y_mean"],
                            "y_std": _fold_norm["y_std"], "x_units": "physical",
                        }
                    else:
                        fold_store["train"] = run_ale_for_model(
                            model=model,
                            df=X_df_train,
                            feature_list=feature_list,
                            grid_size=grid_size,
                            model_device="cpu",
                            prediction_type=pred_type,
                            y_p=y_p_train
                        )

                        fold_store["test"] = run_ale_for_model(
                            model=model,
                            df=X_df_test,
                            feature_list=feature_list,
                            grid_size=grid_size,
                            model_device="cpu",
                            prediction_type=pred_type,
                            y_p=y_p_test
                        )

                    # Process model ALE is data-only — compute once per outer fold
                    process_save_dir = os.path.join(log_path, "process") if log_path else None
                    if not (preles_zero or chain_process) and outer_fold not in process_ale_results:
                        proc_store = process_ale_results.setdefault(outer_fold, {})
                        pm_x_para = _get_x_para(config)

                        proc_store["train"] = run_ale_for_process_model(
                            dataloader=train_dataloader,
                            x_parameter=pm_x_para,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params, rows=rows_train
                        )

                        proc_store["test"] = run_ale_for_process_model(
                            dataloader=test_dataloader,
                            x_parameter=pm_x_para,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params, rows=rows_test
                        )

                    processed += 1
                    if save_intermediate_dir and (processed % checkpoint_every == 0):
                        os.makedirs(save_intermediate_dir, exist_ok=True)
                        joblib.dump(
                            {
                                "model_1D_ale": {pred_type: dict(all_model_1D_ale[pred_type])},
                                "process_1D_ale": process_ale_results,
                                "normalization_stats": all_norm_stats,
                            },
                            os.path.join(save_intermediate_dir, f"ale_ckpt_main_{pred_type}.joblib")
                        )

    return {
        "model_1D_ale":        dict(all_model_1D_ale),
        "process_1D_ale":      process_ale_results,
        "normalization_stats": all_norm_stats,
    }

def generate_ale_pairwise_top_models(
        prediction_types, 
        logs_root, 
        cv_config,
        target_list=["GPP", "ET"],
        grid_size=20,
        impute_empty_cells=True,
        save_intermediate_dir=None, 
        checkpoint_every=1,
        existing_results=None,
        **kwargs
        
):
    """
    Generate ALE results for top models across outer folds and prediction types.

    Args:
        impute_empty_cells: when False, skip filling empty grid cells during pairwise ALE

    Returns:
        dict: {prediction_type: {outer_fold_id: ale_results, ...}, 'process': {...}}
    """
    
    all_pairwise_ale = defaultdict(dict)
    pairwise_process_ale_results = {}
    all_pairwise_counts = defaultdict(dict)
    pairwise_process_counts: dict = {}
    all_norm_stats: dict = {}

    processed = 0  # count processed outer folds across all pred_types

    if existing_results:
        all_pairwise_ale.update(existing_results.get("model_pairwise_ale", {}))
        pairwise_process_ale_results.update(existing_results.get("process_pairwise_ale", {}))
        all_pairwise_counts.update(existing_results.get("model_pairwise_cell_counts", {}))
        pairwise_process_counts.update(existing_results.get("process_pairwise_cell_counts", {}))
        all_norm_stats.update(existing_results.get("normalization_stats", {}))


    for pred_type in prediction_types:
        ncv_kind = infer_ncv_kind(logs_root, pred_type)
        logs_glob = os.path.join(logs_root, f"logs_spatial_{pred_type}*")
        log_dirs = glob.glob(logs_glob)
        if not log_dirs:
            print(f"[Warning] No log directories found for prediction type '{pred_type}' (pattern: {logs_glob})")
            continue
        print(f"[ALE] Generating for: {pred_type}")

        for log_path in log_dirs:
            for outer_fold in iter_summary_dirs(log_path):
                for config_idx, summary in enumerate(iter_fold_configs(log_path, outer_fold)):
                    config_key = f"config_{config_idx}"

                    if config_key in all_pairwise_ale[pred_type].get(outer_fold, {}):
                        print(f"[INFO] Skipping already processed fold {outer_fold} {config_key} for {pred_type}")
                        continue

                    config = _prepare_fold_config(summary, ncv_kind, cv_config)

                    config = _localise_fold_config(config, summary, PROJECT_ROOT + os.sep)

                    train_dataset, test_dataset, model = _load_model_and_dataset(summary, config)
                    train_dataset, test_dataset, _fold_norm = _normalise_fold_datasets(train_dataset, test_dataset, config)
                    all_norm_stats.setdefault(pred_type, {}).setdefault(outer_fold, {})[config_key] = _fold_norm
                    fold_store = all_pairwise_ale[pred_type].setdefault(outer_fold, {}).setdefault(config_key, {})
                    X_df_train, X_df_test, feature_list = _resolve_feature_list(
                        pred_type, config, train_dataset, test_dataset
                    )

                    if pred_type.lower() in {"parallel", "mixture"}:
                        y_p_train = train_dataset[_get_phy_para(config)].values
                        y_p_test = test_dataset[_get_phy_para(config)].values
                    else:
                        y_p_train = y_p_test = None

                    if pred_type.lower() == "bias":
                        y_p_cols = [f"{t}_pred" for t in target_list]
                        X_df_train = train_dataset[y_p_cols]
                        X_df_test = test_dataset[y_p_cols]
                        y_p_train = y_p_test = None
                        feature_list = y_p_cols

                    feature_pairs = _unique_feature_pairs(feature_list)

                    fold_store["train"] = run_pairwise_ale_for_model(
                        model,
                        X_df_train,
                        feature_pairs,
                        grid_size=grid_size,
                        prediction_type=pred_type,
                        y_p=y_p_train,
                        impute_empty_cells=impute_empty_cells,
                    )

                    fold_store["test"] = run_pairwise_ale_for_model(
                        model,
                        X_df_test,
                        feature_pairs,
                        grid_size=grid_size,
                        prediction_type=pred_type,
                        y_p=y_p_test,
                        impute_empty_cells=impute_empty_cells,
                    )
                    all_pairwise_counts[pred_type].setdefault(outer_fold, {})[config_key] = {
                        "train": _pairwise_cell_counts(fold_store["train"], X_df_train, fill=False),
                        "test": _pairwise_cell_counts(fold_store["test"], X_df_test, fill=False),
                    }

                    # Process model ALE is data-only — compute once per outer fold
                    process_save_dir = os.path.join(log_path, "process") if log_path else None
                    train_dataloader, test_dataloader, pm_params = _get_test_dataloader(config)
                    if outer_fold not in pairwise_process_ale_results:
                        proc_store = pairwise_process_ale_results.setdefault(outer_fold, {})
                        pm_x_para = _get_x_para(config)
                        base_feature_pairs = _unique_feature_pairs(pm_x_para)

                        proc_store["train"] = run_pairwise_ale_for_process_model(
                            dataloader=train_dataloader,
                            x_parameter=pm_x_para,
                            feature_pairs=base_feature_pairs,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params,
                            impute_empty_cells=impute_empty_cells,
                        )

                        proc_store["test"] = run_pairwise_ale_for_process_model(
                            dataloader=test_dataloader,
                            x_parameter=pm_x_para,
                            feature_pairs=base_feature_pairs,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params,
                            impute_empty_cells=impute_empty_cells,
                        )
                        pairwise_process_counts[outer_fold] = {
                            split: _pairwise_cell_counts(
                                proc_store[split], loader.data[pm_x_para].reset_index(drop=True), fill=True
                            )
                            for split, loader in (("train", train_dataloader), ("test", test_dataloader))
                        }

                    processed += 1
                    if save_intermediate_dir and (processed % checkpoint_every == 0):
                        os.makedirs(save_intermediate_dir, exist_ok=True)
                        joblib.dump(
                            {
                                "model_pairwise_ale": {pred_type: dict(all_pairwise_ale[pred_type])},
                                "process_pairwise_ale": pairwise_process_ale_results,
                                "model_pairwise_cell_counts": {pred_type: dict(all_pairwise_counts[pred_type])},
                                "process_pairwise_cell_counts": pairwise_process_counts,
                                "normalization_stats": all_norm_stats,
                            },
                            os.path.join(save_intermediate_dir, f"ale_ckpt_pairwise_{pred_type}.joblib")
                        )

    return {
        "model_pairwise_ale":   dict(all_pairwise_ale),
        "process_pairwise_ale": pairwise_process_ale_results,
        "model_pairwise_cell_counts":   dict(all_pairwise_counts),
        "process_pairwise_cell_counts": pairwise_process_counts,
        "normalization_stats":  all_norm_stats,
    }
    
def generate_pairwise_full_top_models(
        prediction_types, 
        logs_root, 
        cv_config,
        target_list=["GPP", "ET"],
        grid_size=20,
        save_intermediate_dir=None, 
        checkpoint_every=1,
        existing_results=None,
        chain_process=False,
        outer_folds=None,
        feature_pairs=None,
        **kwargs

):
    """
    Generate ALE results for top models across outer folds and prediction types.

    chain_process: full surface of the hybrid's response to the weather drivers through PRELES
        (run_chained_full_ale) for residual / bias / parallel; x physical, identity x constants
        in the normalisation stats; no process surfaces (those are in the plain "full" run).
    feature_pairs: optional subset, e.g. [["PAR", "VPD"]] (either order); None = all pairs.

    Returns:
        dict: {prediction_type: {outer_fold_id: ale_results, ...}, 'process': {...}}
    """
    
    # Initialize containers to collect all fold results

    all_pairwise_full_ale = defaultdict(dict)
    pairwise_process_full_ale_results = {}
    all_norm_stats: dict = {}

    processed = 0  # count processed outer folds across all pred_types

    if existing_results:
        all_pairwise_full_ale.update(existing_results.get("model_pairwise_full_ale", {}))
        pairwise_process_full_ale_results.update(existing_results.get("process_pairwise_full_ale", {}))
        all_norm_stats.update(existing_results.get("normalization_stats", {}))

    print(f"[ALE] Starting calculations for full effect (main + interaction) ")
    for pred_type in prediction_types:
        if chain_process and pred_type.lower() not in {"residual", "bias", "parallel"}:
            print(f"[ALE] chain_process: skipping {pred_type} (does not use PRELES)")
            continue
        ncv_kind = infer_ncv_kind(logs_root, pred_type)
        logs_glob = os.path.join(logs_root, f"logs_{ncv_kind}_{pred_type}*")
        log_dirs = glob.glob(logs_glob)
        print(f"[ALE] Generating for: {pred_type}")

        for log_path in log_dirs:
            for outer_fold in iter_summary_dirs(log_path):
                if outer_folds is not None and outer_fold not in outer_folds:
                    continue
                for config_idx, summary in enumerate(iter_fold_configs(log_path, outer_fold)):
                    config_key = f"config_{config_idx}"

                    if config_key in all_pairwise_full_ale[pred_type].get(outer_fold, {}):
                        print(f"[INFO] Skipping already processed fold {outer_fold} {config_key} for {pred_type}")
                        continue

                    config = _prepare_fold_config(summary, ncv_kind, cv_config)

                    config = _localise_fold_config(config, summary, PROJECT_ROOT + os.sep)

                    train_dataset, test_dataset, model = _load_model_and_dataset(summary, config)
                    train_dataset, test_dataset, _fold_norm = _normalise_fold_datasets(train_dataset, test_dataset, config)
                    all_norm_stats.setdefault(pred_type, {}).setdefault(outer_fold, {})[config_key] = _fold_norm
                    fold_store = all_pairwise_full_ale[pred_type].setdefault(outer_fold, {}).setdefault(config_key, {})
                    X_df_train, X_df_test, feature_list = _resolve_feature_list(
                        pred_type, config, train_dataset, test_dataset
                    )

                    if pred_type.lower() in {"parallel", "mixture"}:
                        y_p_train = train_dataset[_get_phy_para(config)].values
                        y_p_test = test_dataset[_get_phy_para(config)].values
                    else:
                        y_p_train = y_p_test = None

                    if pred_type.lower() == "bias":
                        y_p_cols = [f"{t}_pred" for t in target_list]
                        X_df_train = train_dataset[y_p_cols]
                        X_df_test = test_dataset[y_p_cols]
                        feature_list = y_p_cols
                        y_p_train = y_p_test = None

                    if chain_process:
                        train_dataloader, test_dataloader, pm_params = _get_test_dataloader(config)
                        pm_x_para = _get_x_para(config)
                        weather = [f for f in pm_x_para if f not in ("CO2", "DOY")]
                        pairs = _select_pairs(_unique_feature_pairs(weather), feature_pairs)
                        for split, dl, X_nn in (("train", train_dataloader, X_df_train),
                                                ("test", test_dataloader, X_df_test)):
                            if len(X_nn) != len(dl.data):
                                raise ValueError(f"{outer_fold}: NN and process frames differ in length.")
                            fold_store[split] = run_chained_full_ale(
                                dataloader=dl, x_parameter=pm_x_para, feature_pairs=pairs, model=model,
                                prediction_type=pred_type, nn_frame=X_nn, fold_norm=_fold_norm,
                                grid_size=grid_size, params=pm_params,
                            )
                        # x stays physical: identity x constants, real y constants (effect → physical)
                        all_norm_stats[pred_type][outer_fold][config_key] = {
                            "x_parameters": weather, "x_mean": [0.0] * len(weather), "x_std": [1.0] * len(weather),
                            "y_parameters": _fold_norm["y_parameters"], "y_mean": _fold_norm["y_mean"],
                            "y_std": _fold_norm["y_std"], "x_units": "physical",
                        }
                        processed += 1
                        if save_intermediate_dir and (processed % checkpoint_every == 0):
                            os.makedirs(save_intermediate_dir, exist_ok=True)
                            joblib.dump(
                                {"model_pairwise_full_ale": {pred_type: dict(all_pairwise_full_ale[pred_type])},
                                 "process_pairwise_full_ale": pairwise_process_full_ale_results,
                                 "normalization_stats": all_norm_stats},
                                os.path.join(save_intermediate_dir, f"ale_ckpt_full_{pred_type}.joblib"))
                        continue

                    feature_pairs_fold = _select_pairs(_unique_feature_pairs(feature_list), feature_pairs)

                    fold_store["train"] = run_pairwise_fulleffect_ale_for_model(
                        model,
                        X_df_train,
                        feature_pairs_fold,
                        grid_size=grid_size,
                        prediction_type=pred_type,
                        y_p=y_p_train,
                    )
                    fold_store["test"] = run_pairwise_fulleffect_ale_for_model(
                        model,
                        X_df_test,
                        feature_pairs_fold,
                        grid_size=grid_size,
                        prediction_type=pred_type,
                        y_p=y_p_test,
                    )

                    # Process model ALE is data-only — compute once per outer fold
                    process_save_dir = os.path.join(log_path, "process") if log_path else None
                    train_dataloader, test_dataloader, pm_params = _get_test_dataloader(config)
                    if outer_fold not in pairwise_process_full_ale_results and pred_type not in {'residual', 'naive'}:
                        proc_store = pairwise_process_full_ale_results.setdefault(outer_fold, {})
                        pm_x_para = _get_x_para(config)
                        base_feature_pairs = _select_pairs(_unique_feature_pairs(pm_x_para), feature_pairs)

                        proc_store["train"] = run_pairwise_full_ale_for_process_model(
                            dataloader=train_dataloader,
                            x_parameter=pm_x_para,
                            feature_pairs=base_feature_pairs,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params
                        )

                        proc_store["test"] = run_pairwise_full_ale_for_process_model(
                            dataloader=test_dataloader,
                            x_parameter=_get_x_para(config),
                            feature_pairs=base_feature_pairs,
                            grid_size=grid_size,
                            save_dir=process_save_dir, params=pm_params
                        )

                    processed += 1
                    if save_intermediate_dir and (processed % checkpoint_every == 0):
                        os.makedirs(save_intermediate_dir, exist_ok=True)
                        joblib.dump(
                            {
                                "model_pairwise_full_ale": {pred_type: dict(all_pairwise_full_ale[pred_type])},
                                "process_pairwise_full_ale": pairwise_process_full_ale_results,
                                "normalization_stats": all_norm_stats,
                            },
                            os.path.join(save_intermediate_dir, f"ale_ckpt_full_{pred_type}.joblib")
                        )

    return {
        "model_pairwise_full_ale":   dict(all_pairwise_full_ale),
        "process_pairwise_full_ale": pairwise_process_full_ale_results,
        "normalization_stats":       all_norm_stats,
    }


# ---------------------------------------------------------------------------
# High-level orchestration helpers (prediction-type batches / Mixture)
# ---------------------------------------------------------------------------

_ALE_VARIANT_FNS = {
    "main": generate_ale_top_models,
    "pairwise": generate_ale_pairwise_top_models,
    "full": generate_pairwise_full_top_models,
}


def _coerce_cv_config(cv_config):
    if isinstance(cv_config, str):
        if not os.path.exists(cv_config):
            raise FileNotFoundError(f"CV config path not found: {cv_config}")
        with open(cv_config, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(cv_config, dict):
        return cv_config
    raise TypeError("cv_config must be a dict or path to a JSON file")


def _ale_save_root(save_dir_base: str, ncv_kind: str, length: str, grid_size: int, variant: str) -> str:
    return os.path.join(save_dir_base, f"{ncv_kind}_{length}", f"grid{grid_size}", variant)


def run_prediction_type_ale(
    *,
    logs_root: str,
    cv_config,
    length: str,
    prediction_types: Optional[Sequence[str]] = None,
    variant: str = "main",
    grid_size: int = 20,
    target_list=("GPP", "ET"),
    save_dir_base: str = "XAI/results/ALE",
    checkpoint_every: Optional[int] = 0,
    overwrite: bool = False,
    save_intermediate: bool = True,
) -> dict:
    """Batch ALE generation for a set of prediction types."""

    prediction_types = list(prediction_types or ["Mixture"])
    if not prediction_types:
        raise ValueError("prediction_types must contain at least one entry")

    cv_config_dict = _coerce_cv_config(cv_config)

    if variant not in _ALE_VARIANT_FNS:
        raise ValueError(f"Unknown ALE variant '{variant}'. Expected one of {list(_ALE_VARIANT_FNS)}")

    ncv_kind = infer_ncv_kind(logs_root, prediction_types[0])
    root = _ale_save_root(save_dir_base, ncv_kind, length, grid_size, variant)
    os.makedirs(root, exist_ok=True)

    save_intermediate_dir = None
    if save_intermediate:
        save_intermediate_dir = os.path.join(root, "intermediate")
        os.makedirs(save_intermediate_dir, exist_ok=True)

    results = batch_explain(
        explanation_fn=_ALE_VARIANT_FNS[variant],
        prediction_types=prediction_types,
        logs_root=logs_root,
        cv_config=cv_config_dict,
        save_dir=root,
        file_prefix=variant,
        target_list=list(target_list),
        use_process=True,
        grid_size=grid_size,
        save_intermediate_dir=save_intermediate_dir,
        checkpoint_every=checkpoint_every or 0,
        overwrite=overwrite,
    )

    bundle_path = os.path.join(root, f"{variant}_bundle.joblib")
    joblib.dump(results, bundle_path)

    meta = {
        "variant": variant,
        "ncv_kind": ncv_kind,
        "length": length,
        "grid_size": grid_size,
        "prediction_types": prediction_types,
        "logs_root": logs_root,
        "save_dir_base": save_dir_base,
        "checkpoint_every": checkpoint_every,
        "save_intermediate": save_intermediate,
        "target_list": list(target_list),
    }
    if variant == "full":
        meta["full_effect"] = "main+interaction"
    meta_path = os.path.join(root, f"{variant}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "results": results,
        "root": root,
        "bundle_path": bundle_path,
        "meta_path": meta_path,
    }


def run_mixture_ale(**kwargs) -> dict:
    """Convenience wrapper to generate ALE artefacts for the Mixture model."""
    kwargs.setdefault("prediction_types", ["Mixture"])
    return run_prediction_type_ale(**kwargs)

def generate_ale_for_all_top_models(
        prediction_types, 
        logs_root, 
        cv_config,
        target_list=["GPP", "ET"],
        grid_size=20,
        save_intermediate_dir=None, 
        checkpoint_every=1,
        existing_results=None,
        **kwargs
        
):
    """
    Generate ALE results for top models across outer folds and prediction types.

    Returns:
        dict: {prediction_type: {outer_fold_id: ale_results, ...}, 'process': {...}}
    """
    # Initialize containers to collect all fold results
    all_model_1D_ale = defaultdict(dict)
    all_pairwise_ale = defaultdict(dict)
    all_pairwise_full_ale = defaultdict(dict)
    process_ale_results = {}
    pairwise_process_ale_results = {}
    pairwise_process_full_ale_results = {}
    all_norm_stats: dict = {}

    processed = 0  # count processed outer folds across all pred_types
    
    if existing_results:
        all_model_1D_ale.update(existing_results["model_1D_ale"])
        all_pairwise_ale.update(existing_results["pairwise_ale"])
        all_pairwise_full_ale.update(existing_results["pairwise_full_ale"])
        process_ale_results.update(existing_results["process_ale"])
        pairwise_process_ale_results.update(existing_results["process_pairwise_ale"])
        pairwise_process_full_ale_results.update(existing_results["process_pairwise_full_ale"])

    for pred_type in prediction_types:
        ncv_kind = infer_ncv_kind(logs_root, pred_type)
        logs_glob = os.path.join(logs_root, f"logs_{ncv_kind}_{pred_type}*")
        log_dirs = glob.glob(logs_glob)
        print(f"[ALE] Generating for: {pred_type}")

        for log_path in log_dirs:
            for outer_fold in iter_summary_dirs(log_path):
                
                if pred_type in all_model_1D_ale and outer_fold in all_model_1D_ale[pred_type]:
                    print(f"[INFO] Skipping already processed fold {outer_fold} for {pred_type}")
                    continue
                
                summary = _load_fold_summary(log_path, outer_fold)
                if not summary:
                    continue

                # print(f"[ALE] Currently processing: {outer_fold}")
                config = _prepare_fold_config(summary, cv_config)
                train_dataset, test_dataset, model = _load_model_and_dataset(summary, config)
                train_dataset, test_dataset, _fold_norm = _normalise_fold_datasets(train_dataset, test_dataset, config)
                all_norm_stats.setdefault(outer_fold, {})["config_0"] = _fold_norm
                X_df_train, X_df_test, feature_list = _resolve_feature_list(
                    pred_type, config, train_dataset, test_dataset
                )

                # Compute ALE for the primary model
                if pred_type.lower() in {"parallel", "mixture"}:
                    y_p_train = train_dataset[_get_phy_para(config)].values
                    y_p_test = test_dataset[_get_phy_para(config)].values
                else:
                    y_p_train = y_p_test = None

                # Determine input features based on prediction type
                if pred_type.lower() == "bias":
                    # For bias, model input is y_p
                    y_p_cols = [f"{t}_pred" for t in target_list]
                    X_df_train = train_dataset[y_p_cols]
                    X_df_test = test_dataset[y_p_cols]
                    feature_list = y_p_cols
                    y_p_train = y_p_test = None

                fold_store = all_model_1D_ale[pred_type].setdefault(outer_fold, {})

                # Model ALE
                fold_store["train"] = run_ale_for_model(
                    model=model,
                    df=X_df_train,
                    feature_list=feature_list,
                    grid_size=grid_size,
                    model_device="cpu",
                    prediction_type=pred_type,
                    y_p=y_p_train
                )
                fold_store["test"] = run_ale_for_model(
                    model=model,
                    df=X_df_test,
                    feature_list=feature_list,
                    grid_size=grid_size,
                    model_device="cpu",
                    prediction_type=pred_type,
                    y_p=y_p_test
                )

                feature_pairs = _unique_feature_pairs(feature_list)

                pair_store = all_pairwise_ale[pred_type].setdefault(outer_fold, {})
                pair_store["train"] = run_pairwise_ale_for_model(
                    model, 
                    X_df_train, 
                    feature_pairs, 
                    grid_size=grid_size,
                    prediction_type=pred_type, 
                    y_p=y_p_train
                    )
                pair_store["test"] = run_pairwise_ale_for_model(
                    model, 
                    X_df_test, 
                    feature_pairs, 
                    grid_size=grid_size,
                    prediction_type=pred_type, 
                    y_p=y_p_test
                    )
                
                full_store = all_pairwise_full_ale[pred_type].setdefault(outer_fold, {})
                full_store["train"] = run_pairwise_fulleffect_ale_for_model(
                    model, 
                    X_df_train, 
                    feature_pairs, 
                    grid_size=grid_size,
                    prediction_type=pred_type, 
                    y_p=y_p_train,
                    )
                full_store["test"] = run_pairwise_fulleffect_ale_for_model(
                    model, 
                    X_df_test, 
                    feature_pairs, 
                    grid_size=grid_size,
                    prediction_type=pred_type, 
                    y_p=y_p_test,
                    )
                    
                process_save_dir = os.path.join(log_path, "process") if log_path else None
                train_dataloader, test_dataloader, pm_params = _get_test_dataloader(config)

                if outer_fold not in process_ale_results:

                    process_store = process_ale_results.setdefault(outer_fold, {})
                    pair_proc_store = pairwise_process_ale_results.setdefault(outer_fold, {})
                    pair_full_proc_store = pairwise_process_full_ale_results.setdefault(outer_fold, {})

                    pm_x_para = _get_x_para(config)
                    base_feature_pairs = _unique_feature_pairs(pm_x_para)

                    process_store["train"] = run_ale_for_process_model(
                        dataloader=train_dataloader,
                        x_parameter=pm_x_para,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )
                    process_store["test"] = run_ale_for_process_model(
                        dataloader=test_dataloader,
                        x_parameter=pm_x_para,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )

                    pair_proc_store["train"] = run_pairwise_ale_for_process_model(
                        dataloader=train_dataloader,
                        x_parameter=pm_x_para,
                        feature_pairs=base_feature_pairs,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )
                    pair_proc_store["test"] = run_pairwise_ale_for_process_model(
                        dataloader=test_dataloader,
                        x_parameter=pm_x_para,
                        feature_pairs=base_feature_pairs,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )

                    pair_full_proc_store["train"] = run_pairwise_full_ale_for_process_model(
                        dataloader=train_dataloader,
                        x_parameter=pm_x_para,
                        feature_pairs=base_feature_pairs,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )
                    pair_full_proc_store["test"] = run_pairwise_full_ale_for_process_model(
                        dataloader=test_dataloader,
                        x_parameter=pm_x_para,
                        feature_pairs=base_feature_pairs,
                        grid_size=grid_size,
                        save_dir=process_save_dir, params=pm_params
                    )
                    
                    processed += 1
                    if save_intermediate_dir and (processed % checkpoint_every == 0):
                        os.makedirs(save_intermediate_dir, exist_ok=True)
                        joblib.dump(
                            {
                            "model_1D_ale": {pred_type: dict(all_model_1D_ale[pred_type])},
                            "pairwise_ale": {pred_type: dict(all_pairwise_ale[pred_type])},
                            "pairwise_full_ale": {pred_type: dict(all_pairwise_full_ale[pred_type])},
                            "process_ale": process_ale_results,
                            "process_pairwise_ale": pairwise_process_ale_results,
                            "process_pairwise_full_ale": pairwise_process_full_ale_results,
                            },
                            os.path.join(save_intermediate_dir, f"ale_ckpt_{pred_type}.joblib")
                        )     
                
    return {
        "model_1D_ale": dict(all_model_1D_ale),
        "pairwise_ale":              dict(all_pairwise_ale),
        "pairwise_full_ale":         dict(all_pairwise_full_ale),
        "process_ale":               process_ale_results,
        "process_pairwise_ale":      pairwise_process_ale_results,
        "process_pairwise_full_ale": pairwise_process_full_ale_results,
        "normalization_stats":       all_norm_stats,
    }
