import torch
import pandas as pd
import numpy as np
import os
import json
import glob
import joblib
import re
from typing import Sequence, Optional
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

from utilis import load_CV_datasets
from process_model_calibration_manager import ProcessModelCalibrationManager

def evaluate_model(model_path, test_loader, config):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    
    y_true_all, y_pred_all = [], []
    prediction_type = config.get("training::loss_physic_type", None)

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    X, y = batch
                elif len(batch) == 3:
                    X, y, y_p = batch  # e.g. y_p for regularized
            else:
                raise ValueError("Unknown batch format")

            preds = model(X)
            # Add process model prediction if using parallel loss
            if prediction_type == "parallel":
                assert y_p is not None, "Missing process model predictions for 'parallel'"
                preds += y_p
            
            y_true_all.append(y.numpy())
            y_pred_all.append(preds.numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    # Compute overall test loss/metric
    loss, metric = model.test_epoch(test_loader)

    mae_gpp = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_et = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    # Compute GPP and ET RMSE separately
    rmse_gpp = root_mean_squared_error(y_true[:, 0], y_pred[:, 0])
    rmse_et = root_mean_squared_error(y_true[:, 1], y_pred[:, 1])

    return loss, metric, mae_gpp, mae_et, rmse_gpp, rmse_et, model

def load_torch_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    return model
    
def infer_ncv_kind(logs_root, prediction_type):
    """
    Infers whether the NCV kind is 'temporal' or 'spatial' based on
    the directory path in logs_root containing the prediction_type.

    Returns:
        str: 'temporal' or 'spatial'
    """
    
    prediction_root = os.path.join(logs_root, prediction_type)

    # Normalize path for case-insensitive check
    path_lower = prediction_root.lower()

    if "temporal" in path_lower:
        return "temporal"
    elif "koppen" in path_lower:
        return "koppen"
    elif "spatial" in path_lower:
        return "spatial"
    else:
        print(f"[WARNING] Could not find 'temporal', 'koppen', or 'spatial' in {prediction_root}. Defaulting to 'spatial'.")
        return "spatial"


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
    from XAI.xai_utils import ale_utils as _ale_utils

    return _ale_utils.run_prediction_type_ale(
        logs_root=logs_root,
        cv_config=cv_config,
        length=length,
        prediction_types=prediction_types,
        variant=variant,
        grid_size=grid_size,
        target_list=target_list,
        save_dir_base=save_dir_base,
        checkpoint_every=checkpoint_every,
        overwrite=overwrite,
        save_intermediate=save_intermediate,
    )


def run_mixture_ale(**kwargs) -> dict:
    from XAI.xai_utils import ale_utils as _ale_utils

    return _ale_utils.run_mixture_ale(**kwargs)

def evaluate_prediction_type(pred_type):
    save_file = os.path.join(base_path, f"evaluation_seperate_{pred_type}.csv")

    # --- Check if results already exist ---
    if os.path.exists(save_file):
        print(f"[INFO] Found existing evaluation file for {pred_type}, loading instead of recalculating...")
        return pd.read_csv(save_file)

    results = []
    logs_path = os.path.join(base_path, f"logs_spatial_{pred_type}")

    print(f"Prediction: {pred_type}")

    for outer_fold in iter_summary_dirs(logs_path):
        print(f"Fold: {outer_fold}")
        summary = _load_fold_summary(logs_path, outer_fold)
        if not summary:
            continue

        config = summary['config']
        model_path = summary['best_model_path']
        if not model_path:
            continue


        # Update test set based on outer fold
        outer_id = summary["outer_id"]
        test_station = {outer_id: CV_CONFIG['all_stations'][outer_id]}
        # Remaining stations were trained on
        train_station = {
            k: v for k, v in CV_CONFIG['all_stations'].items() if k != outer_id
        }     #need the training stations to generate the associated process model predictions
        
        config['data::train::station_id'] = train_station
        config['data::test::station_id'] = test_station
        config['data::test::path_dir'] = config['data::train::path_dir']  # Reuse train path
        config['data::test::x_parameter'] = config['data::train::x_parameter']
        config['data::test::y_parameter'] = config['data::train::y_parameter']
        config['data::test::batchsize'] = config['training::batchsize']

        pm_manager = ProcessModelCalibrationManager(
            config=config,
            process_model=config.get("process_model::name", "preles"),
            auto_merge_pm_config=False,
        )
        _, test_dataset, _ = pm_manager.load_fold_datasets(
            train_stations=config['data::train::station_id'],
            test_stations=config['data::test::station_id'],
            val_stations=None,
            calibrate=False,
        )

        test_set = load_CV_datasets(config, test_dataset)
        test_loss, test_metric, model_mae_gpp, model_mae_et, model_rmse_gpp, model_rmse_et, _ = evaluate_model(model_path, test_set, config)
        
        model_residuals_gpp = model_residuals_et = process_residuals_gpp = process_residuals_et = None
        if "GPP" in test_dataset and "GPP_pred" in test_dataset:
            model_residuals_gpp = (test_dataset["GPP"] - test_dataset["GPP_pred"]).tolist()
        if "ET" in test_dataset and "ET_pred" in test_dataset:
            model_residuals_et = (test_dataset["ET"] - test_dataset["ET_pred"]).tolist()

        if pred_type == "regularized":
            df_data = test_dataset
            try:
                gpp_mae = mean_absolute_error(df_data["GPP"], df_data["GPP_pred"])
                et_mae = mean_absolute_error(df_data["ET"], df_data["ET_pred"])

                gpp_rmse = np.sqrt(mean_squared_error(df_data["GPP"], df_data["GPP_pred"]))
                et_rmse = np.sqrt(mean_squared_error(df_data["ET"], df_data["ET_pred"]))

                process_residuals_gpp = (df_data["GPP"] - df_data["GPP_pred"]).tolist()
                process_residuals_et = (df_data["ET"] - df_data["ET_pred"]).tolist()
            
            except KeyError as e:
                print(f"[WARNING] Missing column {e} in fold {outer_fold}")
                gpp_rmse, et_rmse = None, None

        else:
            gpp_mae,et_mae, gpp_rmse, et_rmse = None, None, None, None

            # save_residuals(
            #     pred_type,
            #     outer_fold,
            #     model_residuals={
            #         "GPP": model_residuals_gpp,
            #         "ET": model_residuals_et
            #     },
            #     process_residuals={
            #         "GPP": process_residuals_gpp,
            #         "ET": process_residuals_et
            #     }
            # )
                
        results.append({
            "prediction_type": pred_type,
            "outer_fold": summary["outer_fold"],
            "scenario_id": summary.get("scenario_id", outer_fold),
            "config_id": summary.get("config_id", 1),
            "test_loss": test_loss,
            "test_metric": test_metric,
            "model_mae_gpp": model_mae_gpp,
            "model_mae_et":model_mae_et,
            "model_rmse_gpp": model_rmse_gpp,
            "model_rmse_et": model_rmse_et,
            "process_mae_gpp": gpp_mae,
            "process_mae_et":et_mae,
            "process_rmse_gpp": gpp_rmse,
            "process_rmse_et": et_rmse,
            "model_path": model_path,
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_file, index=False)
    print(f"[SAVED] Results for {pred_type} saved at {save_file}")

    return results_df

def _find_summary_file(fold_path, fold_dir_name):
    candidates = [
        os.path.join(fold_path, "summary.json"),
        os.path.join(fold_path, f"outer_summary_{fold_dir_name}_1.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # koppen folds use config_1_summary.json, config_2_summary.json, ...
    config_summaries = sorted(glob.glob(os.path.join(fold_path, "config_*_summary.json")))
    if config_summaries:
        return config_summaries[0]
    return None

def iter_summary_dirs(logs_path):
    if not os.path.isdir(logs_path):
        return []

    summary_dirs = []
    for name in sorted(os.listdir(logs_path)):
        fold_path = os.path.join(logs_path, name)
        if not os.path.isdir(fold_path):
            continue
        if _find_summary_file(fold_path, name):
            summary_dirs.append(name)
    return summary_dirs

def _normalize_summary_payload(payload):
    if isinstance(payload, list):
        return payload[0] if payload else {}

    if isinstance(payload, dict) and "configs" in payload:
        summary = dict(payload)
        configs = summary.get("configs") or []
        top_config = dict(configs[0]) if configs else {}
        if top_config:
            for key in ("config", "config_id", "best_model_path", "final_train_log"):
                if key in top_config and key not in summary:
                    summary[key] = top_config[key]
        return summary

    return dict(payload)


def _load_fold_summary(logs_path, outer_fold):
    """
    Load the JSON summary for a given outer fold, return None if missing.
    Also extract outer_id to summary.
    """
    fold_path = os.path.join(logs_path, outer_fold)
    summary_file = _find_summary_file(fold_path, outer_fold)
    if not summary_file:
        return None

    with open(summary_file, 'r') as f:
        summary = _normalize_summary_payload(json.load(f))

    summary['scenario_id'] = summary.get('scenario_id', outer_fold)
    summary['outer_fold'] = summary.get('outer_fold', outer_fold)
    summary['outer_id'] = (summary['outer_fold']
                           .replace('outer_station_', '')
                           .replace('outer_fold_', ''))

    best_model_path = summary.get('best_model_path')
    if best_model_path and not os.path.isabs(best_model_path):
        summary['best_model_path'] = os.path.abspath(os.path.join(fold_path, best_model_path))
    elif best_model_path and not os.path.exists(best_model_path):
        match = re.split(r"logs_spatial_[^/]+/?", best_model_path)
        if len(match) > 1:
            subpath = match[-1].lstrip(os.sep)
            summary['best_model_path'] = os.path.join(logs_path, subpath)
    return summary

def iter_fold_configs(logs_path, outer_fold):
    """
    Yields one summary dict per training draw stored under the fold directory.

    For folds with multiple draws (config_1_summary.json, config_2_summary.json, …)
    each file is read separately and yields one entry; config_idx reflects the
    file order (0-based).  Falls back to the single-file / legacy formats
    (summary.json, outer_summary_*.json, or a single config_*_summary.json).

    Each yielded dict includes a 'config_idx' field (0 = first draw).
    """
    fold_path = os.path.join(logs_path, outer_fold)

    # Collect all per-draw summary files in draw order: config_<i>_summary.json, or the
    # legacy per-draw outer_summary_<fold>_<i>.json (e.g. PRELES len12 mixture).
    def _draw_no(path):
        m = re.search(r"_(\d+)(?:_summary)?\.json$", os.path.basename(path))
        return int(m.group(1)) if m else -1
    config_files = sorted(glob.glob(os.path.join(fold_path, "config_*_summary.json")), key=_draw_no)
    if len(config_files) <= 1:
        legacy = sorted(glob.glob(os.path.join(fold_path, f"outer_summary_{outer_fold}_*.json")), key=_draw_no)
        if len(legacy) > 1:
            config_files = legacy

    if len(config_files) > 1:
        # Multiple draws: iterate one file per draw.
        for config_idx, summary_file in enumerate(config_files):
            with open(summary_file, 'r') as f:
                payload = json.load(f)
            # Each file is a one-element list or a plain dict.
            if isinstance(payload, list):
                cfg_entry = payload[0] if payload else {}
            else:
                cfg_entry = payload

            summary = dict(cfg_entry)
            summary['config_idx'] = config_idx
            summary['scenario_id'] = summary.get('scenario_id', outer_fold)
            summary['outer_fold'] = summary.get('outer_fold', outer_fold)
            summary['outer_id'] = (summary['outer_fold']
                                   .replace('outer_station_', '')
                                   .replace('outer_fold_', ''))

            best_model_path = summary.get('best_model_path')
            if best_model_path and not os.path.isabs(best_model_path):
                summary['best_model_path'] = os.path.abspath(os.path.join(fold_path, best_model_path))
            elif best_model_path and not os.path.exists(best_model_path):
                match = re.split(r"logs_spatial_[^/]+/?", best_model_path)
                if len(match) > 1:
                    subpath = match[-1].lstrip(os.sep)
                    summary['best_model_path'] = os.path.join(logs_path, subpath)

            yield summary
        return

    # Single file (or legacy summary.json / outer_summary_*.json): original behaviour.
    summary_file = _find_summary_file(fold_path, outer_fold)
    if not summary_file:
        return

    with open(summary_file, 'r') as f:
        payload = json.load(f)

    if isinstance(payload, list):
        configs = payload
        base = {}
    elif isinstance(payload, dict) and "configs" in payload:
        base = {k: v for k, v in payload.items() if k != "configs"}
        configs = payload["configs"] or []
    else:
        configs = [payload]
        base = {}

    for config_idx, cfg_entry in enumerate(configs):
        summary = dict(base)
        summary.update(cfg_entry)
        summary['config_idx'] = config_idx
        summary['scenario_id'] = summary.get('scenario_id', outer_fold)
        summary['outer_fold'] = summary.get('outer_fold', outer_fold)
        summary['outer_id'] = (summary['outer_fold']
                               .replace('outer_station_', '')
                               .replace('outer_fold_', ''))

        best_model_path = summary.get('best_model_path')
        if best_model_path and not os.path.isabs(best_model_path):
            summary['best_model_path'] = os.path.abspath(os.path.join(fold_path, best_model_path))
        elif best_model_path and not os.path.exists(best_model_path):
            match = re.split(r"logs_spatial_[^/]+/?", best_model_path)
            if len(match) > 1:
                subpath = match[-1].lstrip(os.sep)
                summary['best_model_path'] = os.path.join(logs_path, subpath)

        yield summary


def _prepare_fold_config(summary,ncv_kind, cv_config):
    """
    Patch config for training/testing stations and normalization settings.
    """
    outer_fold_key = summary["outer_fold"]
    cfg = summary['config'].copy()

    if ncv_kind in {"spatial", "koppen"}:
        outer_split = cv_config[outer_fold_key]["outer_split"]

        cfg['data::train::station_id'] = outer_split["train"]
        cfg['data::test::station_id'] = outer_split["test"]

    if ncv_kind == "temporal":
        station_id = str(cv_config["station"])
        fold_config = cv_config[outer_fold_key]

        cfg['data::test::station_id'] = {station_id: fold_config["hold_out_years"]}
        cfg['data::train::station_id'] = {station_id: fold_config["train_hold_out"]}
                    


    # outer_id = summary['outer_id']
    # all_stations = cv_config['all_stations']
    # cfg['data::train::station_id'] = {
    #     k: v for k, v in all_stations.items() if k != outer_id
    # }
    # cfg['data::test::station_id'] = {outer_id: all_stations[outer_id]}

    # Mirror train parameters into test block
    for key in ['path_dir', 'x_parameter', 'y_parameter']:
        cfg[f"data::test::{key}"] = cfg[f"data::train::{key}"]
    cfg['data::test::batchsize'] = cfg['training::batchsize']

    return cfg

def _load_model_and_dataset(summary, config):
    """
    Load the fold's PRELES predictions (ProcessModelCalibrationManager) and trained model.
    Returns: test_dataset, model
    """
    pm_manager = ProcessModelCalibrationManager(
        config=config,
        process_model=config.get("process_model::name", "preles"),
        auto_merge_pm_config=False,
    )
    # require_existing: *_pred inputs must come from the fold calibration the model
    # was trained on (calibrate=False meant "defaults" = uncalibrated PM outputs).
    train_dataset, test_dataset, _ = pm_manager.load_fold_datasets(
        train_stations=config['data::train::station_id'],
        test_stations=config['data::test::station_id'],
        val_stations=None,
        calibrate="require_existing",
    )
    
    model = load_torch_model(
        summary['best_model_path']
    )
    
    return train_dataset, test_dataset, model

class Wrapper:
    def __init__(self, func):
        self.func = func
    def predict(self, X):
        return self.func(X)

class TorchPredictWrapper:
    def __init__(self, model, model_device, prediction_type, y_p=None, target_idx=0):
        self.model_device = torch.device(model_device) if not isinstance(model_device, torch.device) else model_device

        # Move model once and keep it in eval mode for repeated SHAP calls
        self.model = model
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.model_device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.prediction_type = (prediction_type or "").lower()
        # Keep as DataFrame when possible so _get_process_outputs can use
        # index-based row lookup (critical for correct alignment during ALE,
        # where PyALE passes row subsets rather than the full dataset).
        if y_p is None:
            self.y_p = None
        elif isinstance(y_p, pd.DataFrame):
            self.y_p = y_p
        else:
            self.y_p = np.asarray(y_p)
        self.target_idx = target_idx
        self._input_dtype = torch.float32

    def _prepare_input(self, X_input):
        """
        Convert SHAP-provided inputs to contiguous float32 tensors on the target device.
        """
        if isinstance(X_input, pd.DataFrame):
            np_input = X_input.to_numpy(dtype=np.float32, copy=False)
        else:
            np_input = np.asarray(X_input, dtype=np.float32)

        if np_input.ndim == 1:
            np_input = np_input.reshape(1, -1)

        # Ensure contiguous array before creating tensor to avoid internal copies later
        np_input = np.ascontiguousarray(np_input)
        return torch.as_tensor(np_input, dtype=self._input_dtype, device=self.model_device)

    def _get_process_outputs(self, X_input_raw):
        """Return y_p rows aligned to X_input_raw.

        When y_p is a DataFrame and X_input_raw is a DataFrame, uses
        index-based lookup so rows match correctly even after PyALE subsets
        the data into grid bins. Falls back to positional slicing when index
        information is not available (e.g. numpy array inputs).
        """
        if self.y_p is None:
            raise RuntimeError("Process outputs (y_p) must be provided for this prediction type.")
        if isinstance(self.y_p, pd.DataFrame) and isinstance(X_input_raw, pd.DataFrame):
            y_p_vals = self.y_p.loc[X_input_raw.index].to_numpy(dtype=np.float32)
        else:
            batch_size = X_input_raw.shape[0] if hasattr(X_input_raw, "shape") else len(X_input_raw)
            y_p_arr = self.y_p.values if isinstance(self.y_p, pd.DataFrame) else self.y_p
            y_p_vals = np.asarray(y_p_arr[:batch_size], dtype=np.float32)
        return torch.as_tensor(y_p_vals, dtype=self._input_dtype, device=self.model_device)

    def get_predict_fn(self, target_idx):
        def predict(X_input):
            X_tensor = self._prepare_input(X_input)

            with torch.no_grad():
                if self.prediction_type == "mixture" and hasattr(self.model, "model_mixture"):
                    mm = self.model.model_mixture
                    y_p_tensor = self._get_process_outputs(X_input)

                    torch.nn.Module.train(self.model, False)
                    torch.nn.Module.train(mm, False)
                    try:
                        mm.set_noise_mult(0.0)
                        mm.set_hard_expert_scale(1.0)
                    except AttributeError:
                        pass

                    try:
                        mm.set_precomputed_outputs(y_p_tensor)
                        pred_tensor, *rest = mm(X_tensor, anneal_noise=False)
                    except AttributeError:
                        if hasattr(self.model, "set_precomputed_outputs"):
                            self.model.set_precomputed_outputs(y_p_tensor)
                            pred_tensor = self.model(X_tensor)
                        else:
                            pred_tensor, *rest = mm(X_tensor, anneal_noise=False)
                else:
                    pred_tensor = self.model(X_tensor)

                    if self.prediction_type == "parallel":
                        parallel_offset = self._get_process_outputs(X_input)
                        pred_tensor = pred_tensor + parallel_offset

                pred_np = pred_tensor.detach().cpu().numpy()
                return pred_np[:, target_idx]

        return predict

class ProcessModelPredictWrapper:
    _BASFOR_TARGETS = ["GPP_pred", "ET_pred"]

    # preles_c.run argument positions of the drivers, and of the initial states
    _PRELES_DRIVER_ARGS = {"PAR": 0, "Tair": 1, "VPD": 2, "Precip": 3, "CO2": 4, "fapar": 5, "DOY": 9}
    _PRELES_STATE_ARGS = {"SW": 33, "Canopywater": 34, "SOG": 35, "S": 36}
    _PRELES_T0_ARG = 37

    def __init__(self, dataframe, x_parameter, target_idx=0, params=None, local=True):
        self.dataframe = dataframe
        self.x_parameter = x_parameter
        self.target_idx = target_idx
        # PRELES calibrated parameters; None -> the dataset's default_parameters
        self.params = params
        # local=True: a perturbed row is predicted from the UNPERTURBED state at the
        # start of its day (= only that day's inputs changed), matching the per-day NNs.
        # local=False: the patched rows run through the whole sequence together, so
        # ALE's all-rows-at-once edge shift rewrites the soil-water/snow history and
        # every other row sees it (e.g. Precip at the upper bin edges ≈ 3.7× the
        # site's total rain) — the effect of a shifted climate, not of the day's input.
        self.local = local
        self._baseline = None
        self._is_basfor = type(dataframe).__name__ == "BasforDataset"
        if self._is_basfor:
            from Config_Files.basfor_configs import cfg as basfor_cfg
            self._weather_cols = list(basfor_cfg.get(
                "Basfor::required_weather_columns",
                ["year", "DOY", "GR", "Tair", "Precip", "WN", "VP"],
            ))

    def predict(self, X_input):
        if isinstance(X_input, np.ndarray):
            if not self._is_basfor and len(X_input) != len(self.dataframe.data):
                raise ValueError(
                    "numpy input must cover every dataset row; pass a DataFrame indexed "
                    "by row position to predict a subset."
                )
            X_input = pd.DataFrame(X_input, columns=self.x_parameter)
        elif not isinstance(X_input, pd.DataFrame):
            raise ValueError("X_input must be a DataFrame or numpy array.")

        if self._is_basfor:
            return self._predict_basfor(X_input)
        return self._predict_preles(X_input)

    def _predict_preles(self, X_input):
        # PRELES is sequential (soil water, snow, phenology state carry over), so run
        # the full dataset sequence through run_process_model — the pathway that built
        # GPP_pred/ET_pred for training — with the X_input rows patched in.
        # X_input's index must be row positions into dataset.data (PyALE passes the
        # full frame, a 1-row check, or row subsets for discrete features).
        ds = self.dataframe
        target_col = f"{ds.y_parameter[self.target_idx]}_pred"
        pos = X_input.index.to_numpy()
        if self.local:
            return self._predict_preles_local(X_input, pos, [target_col])[target_col]
        patched = ds.data.reset_index(drop=True)
        patched.loc[pos, self.x_parameter] = X_input[self.x_parameter].to_numpy()
        preds = self._run_preles_on(patched)
        return preds[target_col].to_numpy()[pos]

    def _run_preles_on(self, frame):
        """run_process_model on `frame` in place of the dataset's rows, then restore.

        The PRELES dataset simulates `data_full` (the continuous daily series behind
        sparse thinning), not `data`, so both must be swapped; with thinning the
        two differ and a patched `data` cannot be mapped back, so that is refused.
        """
        ds = self.dataframe
        keep = getattr(ds, "_keep_mask", None)
        if keep is not None and not keep.all():
            raise ValueError("Patched PRELES runs are not supported on sparse-thinned datasets.")
        orig, orig_full = ds.data, getattr(ds, "data_full", None)
        ds.data = frame
        if orig_full is not None:
            ds.data_full = frame
            ds._keep_mask = np.ones(len(frame), dtype=bool)
        try:
            return ds.run_process_model(params=self.params)
        finally:
            ds.data = orig
            if orig_full is not None:
                ds.data_full = orig_full
                ds._keep_mask = keep

    def _preles_baseline(self):
        """Unperturbed run (cached): data, predictions and, for preles_c, run args + daily states."""
        if self._baseline is None:
            ds = self.dataframe
            data = ds.data.reset_index(drop=True)
            base = {"data": data, "pred": ds.run_process_model(params=self.params), "args": None}
            # State restart needs preles_c (returns daily states) and no hidden phenology
            # accumulator (PhenoS is not returned; t0 = -999 disables the fPheno model).
            if getattr(ds, "backend", None) == "preles_c":
                args = ds.prepare_preles_parameters_batch(data, params=self.params)
                if float(args[self._PRELES_T0_ARG]) <= -998:
                    from PRELES_dataset import preles as preles_c
                    out = preles_c.run(*args)
                    base["args"] = args
                    base["states"] = {name: np.asarray(out[name]) for name in self._PRELES_STATE_ARGS}
                    base["run"] = preles_c.run
            self._baseline = base
        return self._baseline

    def predict_outputs(self, X_input):
        """All PRELES outputs ({"GPP_pred": ..., "ET_pred": ...}) for the X_input rows, each
        perturbed row re-run once (local mode, PRELES only); index = row positions."""
        if self._is_basfor or not self.local:
            raise NotImplementedError("predict_outputs is implemented for local PRELES runs only.")
        cols = [f"{y}_pred" for y in self.dataframe.y_parameter]
        return self._predict_preles_local(X_input, X_input.index.to_numpy(), cols)

    def _predict_preles_local(self, X_input, pos, target_cols):
        """{target_col: values} for the X_input rows; one PRELES day run per perturbed row."""
        base = self._preles_baseline()
        data = base["data"]
        new_vals = X_input[self.x_parameter].to_numpy(dtype=float)
        old_vals = data.loc[pos, self.x_parameter].to_numpy(dtype=float)
        changed = ~np.all(np.isclose(new_vals, old_vals, equal_nan=True), axis=1)
        out = {c: base["pred"][c].to_numpy()[pos].astype(float) for c in target_cols}
        if not changed.any():
            return out

        if base["args"] is not None:
            # Exact: a day's outputs depend only on the state at its start and its own
            # drivers, so run that one day from the baseline end-of-previous-day state.
            args, states, run = base["args"], base["states"], base["run"]
            missing = [f for f in self.x_parameter if f not in self._PRELES_DRIVER_ARGS]
            if missing:
                raise KeyError(f"No preles_c driver slot for {missing}")
            for k in np.flatnonzero(changed):
                i = int(pos[k])
                day_args = list(args)
                for a in range(10):
                    day_args[a] = np.ascontiguousarray(np.asarray(args[a])[i : i + 1])
                for f, v in zip(self.x_parameter, new_vals[k]):
                    slot = self._PRELES_DRIVER_ARGS[f]
                    day_args[slot] = np.asarray([v], dtype=day_args[slot].dtype)
                if i > 0:
                    for name, slot in self._PRELES_STATE_ARGS.items():
                        day_args[slot] = float(states[name][i - 1])
                day = run(*day_args)
                for c in target_cols:
                    out[c][k] = float(np.asarray(day[c[: -len("_pred")]])[0])
            return out

        # Fallback (other backends / phenology on): rerun the prefix up to day i with
        # only day i patched. Exact, but one run per perturbed row.
        for k in np.flatnonzero(changed):
            i = int(pos[k])
            prefix = data.iloc[: i + 1].copy()
            prefix.loc[i, self.x_parameter] = new_vals[k]
            run_i = self._run_preles_on(prefix)
            for c in target_cols:
                out[c][k] = float(run_i[c].iloc[i])
        return out

    def _predict_basfor(self, X_input):
        ds = self.dataframe
        target_col = self._BASFOR_TARGETS[self.target_idx]

        if ds._per_site_matrices:
            # Patch each site's weather matrix using the corresponding slice of X_input,
            # then restore after running. X_input rows are ordered by site (matching self.data).
            orig_matrices = [m["weather_matrix"].copy() for m in ds._per_site_matrices]
            try:
                row_offset = 0
                for site_info in ds._per_site_matrices:
                    n = site_info["eval_ndays"]
                    site_rows = X_input.iloc[row_offset: row_offset + n]
                    eval_start = site_info["eval_start_idx"]
                    for feat in self.x_parameter:
                        if feat in self._weather_cols:
                            col_idx = self._weather_cols.index(feat)
                            site_info["weather_matrix"][
                                eval_start: eval_start + n, col_idx
                            ] = site_rows[feat].values
                    row_offset += n
                preds_df = ds.run_basfor()
                return preds_df[target_col].values if target_col in preds_df.columns else np.zeros(len(X_input))
            finally:
                for site_info, orig in zip(ds._per_site_matrices, orig_matrices):
                    site_info["weather_matrix"] = orig
        else:
            orig_matrix = ds.weather_matrix.copy()
            try:
                for feat in self.x_parameter:
                    if feat in self._weather_cols:
                        col_idx = self._weather_cols.index(feat)
                        ds.weather_matrix[
                            ds.eval_start_idx: ds.eval_start_idx + ds.eval_ndays, col_idx
                        ] = X_input[feat].values
                preds = ds.run_basfor()
                return preds.get(target_col, np.zeros(len(X_input)))
            finally:
                ds.weather_matrix = orig_matrix


class ChainedPredictWrapper:
    """Hybrid prediction through PRELES: perturbed weather → PRELES re-run for that day →
    the network (ALE of the full response, path weather → PRELES → model included).

    X_input: rows of the PRELES driver frame (physical units, index = row positions into
    the process dataset). ``nn_frame``: the network's z-scored input frame in model column
    order, same rows/order as the PRELES frame. The perturbed weather values go into the
    network's own weather inputs (z-scored with ``fold_norm`` x constants); PRELES's outputs
    go in as GPP_pred/ET_pred inputs (residual, bias; their x constants) or as the added
    process offset y_p (parallel; the observed target's constants, as in training).
    Returns the network output for ``target_idx`` in z-units.
    """

    def __init__(self, process_wrapper, model, prediction_type, nn_frame, fold_norm, weather_cols,
                 target_idx=0, model_device="cpu"):
        self.process_wrapper = process_wrapper
        self.prediction_type = (prediction_type or "").lower()
        self.nn_frame = nn_frame.reset_index(drop=True)
        self.weather_cols = list(weather_cols)
        self.target_idx = target_idx
        self.torch_wrapper = TorchPredictWrapper(model, model_device=model_device, prediction_type=self.prediction_type)
        xp, yp = fold_norm["x_parameters"], fold_norm["y_parameters"]
        self._x = {c: (fold_norm["x_mean"][i], fold_norm["x_std"][i]) for i, c in enumerate(xp)}
        self._y = {c: (fold_norm["y_mean"][i], fold_norm["y_std"][i]) for i, c in enumerate(yp)}

    def predict(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            raise ValueError("ChainedPredictWrapper needs a DataFrame indexed by row position.")
        pos = X_input.index.to_numpy()
        preles = self.process_wrapper.predict_outputs(X_input)  # physical, per row
        X_nn = self.nn_frame.iloc[pos].copy()
        for c in self.weather_cols:
            if c in X_nn.columns:
                m, s = self._x[c]
                X_nn[c] = (X_input[c].to_numpy(dtype=float) - m) / s
        if self.prediction_type == "parallel":
            y_p = np.column_stack([(preles[f"{t}_pred"] - m) / s for t, (m, s) in self._y.items()])
            self.torch_wrapper.y_p = pd.DataFrame(y_p, index=X_nn.index)
        else:
            for c, v in preles.items():
                if c in X_nn.columns:
                    m, s = self._x[c]
                    X_nn[c] = (v - m) / s
        return self.torch_wrapper.get_predict_fn(self.target_idx)(X_nn)
