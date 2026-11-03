from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Basic_dataset import Basic_dataset


HYBRID_LOSS_TYPES = {"parallel", "regularized", "error"}


def _as_dataframe(split_df: Any) -> pd.DataFrame:
    data = getattr(split_df, "data", split_df)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Expected split_df to be a pandas.DataFrame or an object exposing .data as DataFrame."
        )
    return data.copy(deep=True)

def _resolve_process_model(config: Dict[str, Any], process_model: Optional[str]) -> str:
    value = process_model or config.get("process_model::name") or "preles"
    return str(value).strip().lower()

def _resolve_prediction_type(config: Dict[str, Any]) -> str:
    value = config.get("model::prediction_type", "spatial")
    return str(value).strip().lower()

def _resolve_loss_type(config: Dict[str, Any]) -> str:
    value = config.get("training::loss_physic_type", "")
    return str(value).strip().lower()

def _resolve_x_cols(config: Dict[str, Any], split: str) -> List[str]:
    candidates = [
        config.get(f"data::{split}::x_parameter"),
        config.get("data::train::x_parameter"),
        config.get("Process::x_para"),
        config.get("Preles::x_para"),
        config.get("Basfor::x_para"),
    ]
    for cand in candidates:
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            return list(cand)
    raise KeyError("Could not resolve x columns from config.")

def _resolve_y_cols(config: Dict[str, Any], split: str) -> List[str]:
    candidates = [
        config.get(f"data::{split}::y_parameter"),
        config.get("data::train::y_parameter"),
        config.get("Process::y_para"),
        config.get("Preles::y_para"),
        config.get("Basfor::y_para"),
    ]
    for cand in candidates:
        if isinstance(cand, (list, tuple)) and len(cand) > 0:
            return list(cand)
    raise KeyError("Could not resolve y columns from config.")

def _resolve_temporal_include_y_past(config: Dict[str, Any]) -> bool:
    if "model::prediction_temporal:include_Y_past" in config:
        return bool(config["model::prediction_temporal:include_Y_past"])
    if "data::past-include" in config:
        return bool(config["data::past-include"])
    return True

def _resolve_positive_window(config: Dict[str, Any], key: str, default: int = 1) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected '{key}' to be int-like or None, got {value!r} ({type(value).__name__})."
        ) from exc
    if parsed < 1:
        raise ValueError(f"Expected '{key}' to be >= 1, got {parsed}.")
    return parsed

def _resolve_batch_size(config: Dict[str, Any], split: str) -> int:
    if split == "train":
        return int(config.get("training::batchsize", 1))
    return int(config.get("data::test::batchsize", config.get("training::batchsize", 1)))

def _resolve_normalise_flag(config: Dict[str, Any]) -> Optional[str]:
    value = config.get("data::normalise")
    if value == "z-score":
        return "z-score"
    return None

def _resolve_sparse(config: Dict[str, Any], split: str, sparse: Optional[Any] = None) -> Optional[int]:
    if sparse is not None:
        value = sparse
    else:
        candidates = [
            config.get(f"data::{split}::sparse"),
            config.get(f"data:{split}:sparse"),
            config.get("data::sparse"),
            config.get("data:sparse"),
        ]
        if split == "val":
            candidates = [
                config.get("data::val::sparse"),
                config.get("data:val:sparse"),
                config.get("data::test::sparse"),
                config.get("data:test:sparse"),
            ] + candidates
        value = next((c for c in candidates if c is not None), None)

    if value in (None, False):
        return None
    if isinstance(value, bool):
        raise ValueError("Sparse flag must be None or an integer >= 1, not bool.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Sparse flag must be int-like, got {value!r}.") from exc
    if parsed < 1:
        raise ValueError(f"Sparse flag must be >= 1, got {parsed}.")
    return parsed

def _validate_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required {label} columns in prepared DataFrame: {missing}")


@dataclass
class DatasetBuildSpec:
    process_model: str
    prediction_type: str
    loss_type: str
    x_cols: List[str]
    y_cols: List[str]
    y_pred_cols: List[str]
    include_process_predictions: bool
    is_error_mode: bool
    normalise: Optional[str]
    modeltype: Optional[str]
    temporal_in: int
    temporal_out: int
    temporal_include_y_past: bool
    sparse: Optional[int]


def _build_spec(
    config: Dict[str, Any],
    split: str,
    process_model: Optional[str],
    sparse: Optional[Any] = None,
) -> DatasetBuildSpec:
    resolved_process_model = _resolve_process_model(config, process_model)
    prediction_type = _resolve_prediction_type(config)
    loss_type = _resolve_loss_type(config)
    is_error_mode = loss_type == "error"
    x_cols = _resolve_x_cols(config, split)
    y_cols = _resolve_y_cols(config, split)
    y_pred_cols = [f"{y}_pred" for y in y_cols]
    additional_pred_cols = config.get("model::io::y_pred_additional_cols") or []
    y_pred_cols = list(dict.fromkeys(y_pred_cols + [c for c in additional_pred_cols if c not in y_pred_cols]))
    include_process_predictions = loss_type in HYBRID_LOSS_TYPES
    normalise = _resolve_normalise_flag(config)
    modeltype = config.get("model::modeltype")
    temporal_in = _resolve_positive_window(config, "model::windows_input", default=1)
    temporal_out = _resolve_positive_window(config, "model::windows_output", default=1)
    include_y_past = _resolve_temporal_include_y_past(config)
    sparse_value = _resolve_sparse(config=config, split=split, sparse=sparse)

    return DatasetBuildSpec(
        process_model=resolved_process_model,
        prediction_type=prediction_type,
        loss_type=loss_type,
        x_cols=x_cols,
        y_cols=y_cols,
        y_pred_cols=y_pred_cols,
        include_process_predictions=include_process_predictions,
        is_error_mode=is_error_mode,
        normalise=normalise,
        modeltype=modeltype,
        temporal_in=temporal_in,
        temporal_out=temporal_out,
        temporal_include_y_past=include_y_past,
        sparse=sparse_value,
    )

class SpatialDataset(Basic_dataset):
    """
    Process-model-agnostic spatial dataset for prepared split DataFrames.

    Returns:
    - `(x, y)` for non-hybrid losses
    - `(x, y, y_p)` for parallel/regularized losses
    - `((x, x_err), y, (y_p, y_err))` for error mode
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        y_pred_cols: Optional[Sequence[str]] = None,
        normalise: Optional[str] = None,
        is_error_mode: bool = False,
        device: bool = True,
        sparse=None,
        normalization: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.test_device(device)
        self.data = dataframe.copy(deep=True)
        self.x_parameter = list(x_cols)
        self.y_parameter = list(y_cols)
        self.y_phy_para = list(y_pred_cols) if y_pred_cols else []
        self.is_error_mode = bool(is_error_mode)
        self.sparse = sparse

        _validate_columns(self.data, self.x_parameter, "x")
        _validate_columns(self.data, self.y_parameter, "y")
        if self.y_phy_para:
            _validate_columns(self.data, self.y_phy_para, "process-prediction")

        if self.is_error_mode and not self.y_phy_para:
            raise ValueError("Error mode requires process-model prediction columns (<y>_pred).")

        # `normalization=None` self-fits on this split's own dataframe. For test/val
        # that leaks the split's statistics (including the observed target mean/sd)
        # into the prediction path, so callers must hand down the train-fitted stats.
        if normalise == "z-score":
            self.normalise(normalization=normalization)

        if self.is_error_mode:
            self.data, err_para = self.error_labels(
                data=self.data,
                real_label=self.y_parameter,
                preles_label=self.y_phy_para,
            )
            self.check_parameters(err_para)
            self.err_para = err_para

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        x = torch.tensor(row[self.x_parameter].values.tolist(), dtype=torch.float32).to(self.device)
        y = torch.tensor(row[self.y_parameter].values.tolist(), dtype=torch.float32).to(self.device)

        if self.y_phy_para:
            y_p = torch.tensor(row[self.y_phy_para].values.tolist(), dtype=torch.float32).to(self.device)
            if self.is_error_mode:
                x_err = torch.tensor(row[self.err_para].values.tolist(), dtype=torch.float32).to(self.device)
                y_err = torch.tensor(row[self.err_para].values.tolist(), dtype=torch.float32).to(self.device)
                return (x, x_err), y, (y_p, y_err)
            return x, y, y_p

        return x, y

class TemporalDataset(Basic_dataset):
    """
    Process-model-agnostic temporal dataset built from prepared split DataFrames.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        input_window: int,
        output_window: int,
        y_pred_cols: Optional[Sequence[str]] = None,
        normalise: Optional[str] = None,
        sparse: Optional[int] = None,
        modeltype: Optional[str] = None,
        y_past_include: bool = True,
        is_error_mode: bool = False,
        device: bool = True,
        normalization: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.test_device(device)
        self.data = dataframe.copy(deep=True)
        self.x_parameter = list(x_cols)
        self.y_parameter = list(y_cols)
        self.y_phy_para = list(y_pred_cols) if y_pred_cols else []
        self.sparse = sparse
        self.modeltype = modeltype
        self.in_wind = int(input_window)
        self.out_wind = int(output_window)
        self.is_error_mode = bool(is_error_mode)
        self.include_process_predictions = len(self.y_phy_para) > 0

        _validate_columns(self.data, self.x_parameter, "x")
        _validate_columns(self.data, self.y_parameter, "y")
        if self.y_phy_para:
            _validate_columns(self.data, self.y_phy_para, "process-prediction")

        if self.is_error_mode and not self.y_phy_para:
            raise ValueError("Error mode requires process-model prediction columns (<y>_pred).")

        # Fitted before windowing, so the constants describe the raw columns and are
        # interchangeable with the spatial ones.  See SpatialDataset for why callers
        # must hand down the train-fitted stats.
        if normalise == "z-score":
            self.normalise(normalization=normalization)

        if self.is_error_mode:
            self.data, err_para = self.error_labels(
                data=self.data,
                real_label=self.y_parameter,
                preles_label=self.y_phy_para,
            )
            self.check_parameters(err_para)
            self.err_para = err_para

        if self.include_process_predictions:
            self.create_temporal_dataset(physics=True, error=self.is_error_mode, y_past_include=y_past_include)
            if self.modeltype == "LSTM":
                if self.is_error_mode:
                    (
                        self.data_x,
                        self.data_err_x,
                        self.data_y,
                        self.data_p,
                        self.data_err_y,
                    ) = self.rewrite_LSTM(physics=True, error=True)
                else:
                    self.data_x, self.data_y, self.data_p = self.rewrite_LSTM(physics=True)
            return

        self.create_temporal_dataset(physics=False, error=False, y_past_include=y_past_include)
        if self.modeltype == "LSTM":
            self.data_x, self.data_y = self.rewrite_LSTM(physics=False)

    def __getitem__(self, index: int):
        if self.modeltype == "LSTM":
            x = torch.tensor(self.data_x[index], dtype=torch.float32).to(self.device)
            y = torch.tensor(self.data_y[index], dtype=torch.float32).to(self.device)

            if self.include_process_predictions:
                y_p = torch.tensor(self.data_p[index], dtype=torch.float32).to(self.device)
                if self.is_error_mode:
                    x_err = torch.tensor(self.data_err_x[index], dtype=torch.float32).to(self.device)
                    y_err = torch.tensor(self.data_err_y[index], dtype=torch.float32).to(self.device)
                    return (x, x_err), y, (y_p, y_err)
                return x, y, y_p

            return x, y

        row = self.data.iloc[index]
        x = torch.tensor(row[self.X_t_parameter].values.tolist(), dtype=torch.float32).to(self.device)
        y = torch.tensor(row[self.Y_t_parameter].values.tolist(), dtype=torch.float32).to(self.device)

        if self.include_process_predictions:
            y_p = torch.tensor(row[self.Y_t_physics].values.tolist(), dtype=torch.float32).to(self.device)
            if self.is_error_mode:
                x_err = torch.tensor(row[self.err_t_x].values.tolist(), dtype=torch.float32).to(self.device)
                y_err = torch.tensor(row[self.err_t_y].values.tolist(), dtype=torch.float32).to(self.device)
                return (x, x_err), y, (y_p, y_err)
            return x, y, y_p

        return x, y

def build_split_dataset(
    config: Dict[str, Any],
    split_df: Any,
    split: str = "train",
    process_model: Optional[str] = None,
    sparse: Optional[Any] = None,
    normalization: Optional[Mapping[str, Any]] = None,
):
    df = _as_dataframe(split_df)
    spec = _build_spec(config=config, split=split, process_model=process_model, sparse=sparse)

    _validate_columns(df, spec.x_cols, "x")
    _validate_columns(df, spec.y_cols, "y")

    # A non-train split that self-fits normalises (and later denormalises) with its
    # own statistics, including the observed target's mean/sd — the model's physical
    # prediction then depends on having measured the test site.  Callers that want
    # that behaviour deliberately must pass the split's stats in explicitly.
    if split != "train" and spec.normalise == "z-score" and normalization is None:
        import warnings
        warnings.warn(
            f"[{split}] Building a non-train split with z-score normalisation but no "
            "train-fitted statistics; it will self-fit. This leaks the split's own "
            "mean/sd (targets included) into the prediction path. Pass "
            "normalization=<train stats> unless you specifically intend this.",
            stacklevel=2,
        )

    if spec.include_process_predictions:
        _validate_columns(df, spec.y_pred_cols, "process-prediction")
        y_pred_cols: Optional[List[str]] = spec.y_pred_cols
    else:
        y_pred_cols = None

    # Fail early if any required column is entirely NaN — indicates a missing or
    # unpopulated column in the preprocessed CSVs.  The user should remove that
    # column from the x_parameter / y_parameter config keys.

    # To Do: Update here to exclude the naive model from looking the process model predictions as they are not relevant for it.
    required_cols = spec.x_cols + spec.y_cols
    all_nan_cols = [c for c in required_cols if c in df.columns and bool(df[[c]].isna().all(axis=None))]
    if all_nan_cols:
        raise ValueError(
            f"[{split}] The following required columns are entirely NaN in the "
            f"prepared DataFrame: {all_nan_cols}. Remove them from the x_parameter "
            f"/ y_parameter config keys (e.g. 'data::{split}::x_parameter')."
        )

    # Drop rows where any X or Y column is NaN so that failed/missing process-model
    # predictions do not propagate through the forward pass.  Log dropped rows so
    # silent data loss is visible.
    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import warnings
        warnings.warn(
            f"[{split}] Dropped {n_dropped}/{n_before} rows with NaN in X/Y columns "
            f"(likely missing process-model predictions). "
            f"{len(df)} rows remain.",
            stacklevel=2,
        )

    if spec.prediction_type == "temporal":
        return TemporalDataset(
            dataframe=df,
            x_cols=spec.x_cols,
            y_cols=spec.y_cols,
            y_pred_cols=y_pred_cols,
            normalise=spec.normalise,
            sparse=spec.sparse,
            modeltype=spec.modeltype,
            y_past_include=spec.temporal_include_y_past,
            input_window=spec.temporal_in,
            output_window=spec.temporal_out,
            is_error_mode=spec.is_error_mode,
            normalization=normalization,
        )

    return SpatialDataset(
        dataframe=df,
        x_cols=spec.x_cols,
        y_cols=spec.y_cols,
        y_pred_cols=y_pred_cols,
        normalise=spec.normalise,
        sparse=spec.sparse,
        is_error_mode=spec.is_error_mode,
        normalization=normalization,
    )


def build_split_dataloader(
    config: Dict[str, Any],
    split_df: Any,
    split: str = "train",
    process_model: Optional[str] = None,
    shuffle: bool = False,
    sparse: Optional[Any] = None,
    normalization: Optional[Mapping[str, Any]] = None,
) -> DataLoader:
    dataset = build_split_dataset(
        config=config,
        split_df=split_df,
        split=split,
        process_model=process_model,
        sparse=sparse,
        normalization=normalization,
    )
    batch_size = _resolve_batch_size(config, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def build_fold_dataloaders_from_predictions(
    config: Dict[str, Any],
    train_df: Any,
    test_df: Any,
    val_df: Any = None,
    process_model: Optional[str] = None,
    shuffle_train: bool = True,
    sparse: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    # Build train FIRST and reuse its fitted statistics for the other splits, so the
    # whole fold shares one transform.  Fitting each split separately makes the
    # test-site prediction depend on the test site's own measured mean/sd.
    train_loader = build_split_dataloader(
        config=config,
        split_df=train_df,
        split="train",
        process_model=process_model,
        shuffle=shuffle_train,
        sparse=sparse,
    )
    train_stats = getattr(train_loader.dataset, "normalization", None)

    test_loader = build_split_dataloader(
        config=config,
        split_df=test_df,
        split="test",
        process_model=process_model,
        shuffle=False,
        sparse=sparse,
        normalization=train_stats,
    )

    val_loader = None
    if val_df is not None:
        val_loader = build_split_dataloader(
            config=config,
            split_df=val_df,
            split="val",
            process_model=process_model,
            shuffle=False,
            sparse=sparse,
            normalization=train_stats,
        )

    return train_loader, test_loader, val_loader

def _resolve_station_splits(
    config: Dict[str, Any],
    cv_config_path: Optional[str],
    outer_fold: Optional[str],
    inner_loop: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    if cv_config_path:
        from process_model_calibration_manager import _resolve_from_cv

        with open(cv_config_path, "r", encoding="utf-8") as f:
            cv_cfg = json.load(f)

        train, test, val, used_outer, used_inner = _resolve_from_cv(
            cv_config=cv_cfg,
            outer_fold=outer_fold,
            inner_loop=inner_loop,
        )
        print(f"[handoff][main] CV outer={used_outer} inner={used_inner}")
        return train, test, val

    train = dict(config["data::train::station_id"])
    test = dict(config["data::test::station_id"])
    val = dict(config["data::val::station_id"]) if "data::val::station_id" in config else None
    return train, test, val



if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys

    from dotenv import load_dotenv

    from Config_Files.config_loader import merge_process_model_config
    from Config_Files.config_spatial import build_config
    from process_model_calibration_manager import ProcessModelCalibrationManager

    parser = argparse.ArgumentParser(description="Runner for calibration_dataloader_handoff diagnostics.")
    parser.add_argument("--config",         type=str, default=None, help="Optional path to frozen config JSON.")
    parser.add_argument("--cv-config",      type=str, default=None, help="Optional CV split JSON path.")
    parser.add_argument("--outer-fold",     type=str, default=None, help="Outer fold key.")
    parser.add_argument("--inner-loop",     type=str, default=None, help="Inner loop key.")
    parser.add_argument("--process-model",  type=str, default="preles", choices=["preles", "basfor"])
    parser.add_argument("--calibrate",      type=str, default="defaults", choices=["auto", "defaults", "force_calibrate"])
    parser.add_argument("--shuffle-train", action="store_true", help="Shuffle training DataLoader.")
    parser.add_argument("--sparse", type=int, default=None, help="Optional sparse stride for handoff dataloaders.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    IDE_SETTINGS = {
        "enabled": True,
        "config": None,
        "cv_config": None,
        "outer_fold": None,
        "inner_loop": None,
        "process_model": "preles",
        "calibrate": "auto",
        "shuffle_train": False,
        "sparse": None,
        "debug": False,
    }

    if len(sys.argv) == 1 and IDE_SETTINGS["enabled"]:
        print("[handoff][main] No CLI args detected, using IDE_SETTINGS.")
        args.config = IDE_SETTINGS["config"]
        args.cv_config = IDE_SETTINGS["cv_config"]
        args.outer_fold = IDE_SETTINGS["outer_fold"]
        args.inner_loop = IDE_SETTINGS["inner_loop"]
        args.process_model = IDE_SETTINGS["process_model"]
        args.calibrate = IDE_SETTINGS["calibrate"]
        args.shuffle_train = IDE_SETTINGS["shuffle_train"]
        args.sparse = IDE_SETTINGS["sparse"]
        args.debug = IDE_SETTINGS["debug"]

    load_dotenv()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = build_config()

    config = merge_process_model_config(base_config=config, process_model=args.process_model)
    if args.debug:
        config["calibration::debug"] = True
    if "calibration::root" not in config:
        config["calibration::root"] = os.getenv("CALIBRATION_ROOT", "calibrations")

    train_stations, test_stations, val_stations = _resolve_station_splits(
        config=config,
        cv_config_path=args.cv_config,
        outer_fold=args.outer_fold,
        inner_loop=args.inner_loop,
    )

    print(f"[handoff][main] process_model={args.process_model} calibrate={args.calibrate}")
    print(f"[handoff][main] train={train_stations}")
    print(f"[handoff][main] test={test_stations}")
    if val_stations is not None:
        print(f"[handoff][main] val={val_stations}")

    manager = ProcessModelCalibrationManager(config=config, process_model=args.process_model)
    train_df, test_df, val_df = manager.load_fold_datasets(
        train_stations=train_stations,
        test_stations=test_stations,
        val_stations=val_stations,
        calibrate=args.calibrate,
    )

    train_loader, test_loader, val_loader = build_fold_dataloaders_from_predictions(
        config=config,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        process_model=args.process_model,
        shuffle_train=args.shuffle_train,
        sparse=args.sparse,
    )


    print("Done")
