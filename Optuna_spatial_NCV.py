# Integration of Optuna in Ray
import torch 
import numpy as np
import copy
import socket
import os
import json
import pandas as pd
import logging
import argparse
import time
import sys
import math
from math import isfinite
import random
import importlib
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from filelock import FileLock

# Optuna
import optuna
from optuna.trial import TrialState
from optuna.storages import RDBStorage
from optuna.pruners import HyperbandPruner #hyperband for Pruning models
from optuna.samplers import TPESampler # Bayesian Optimisation sampling
import wandb

#Torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#own 
from NN_framework import NN_module
from utilis_shorts import json_open, save_frozen_config, save_normalization # conv_torch
from process_model_calibration_manager import ProcessModelCalibrationManager
from utilis import optimize_within_fold_budget
from calibration_dataloader_handoff import build_fold_dataloaders_from_predictions
from calibration_paths import (
    build_spatial_logs_root,
    build_spatial_optuna_db_path,
    build_spatial_scenario_id,
    cv_label_from_path,
    station_signature,
)


ProcessModel = Literal["preles"]
CalibrationMode = Literal["auto", "defaults", "force_calibrate", "require_existing"]


@dataclass
class RuntimeContext:
    model_config: dict[str, Any]
    cv_config: dict[str, Any]
    process_model_name: str
    training_years_len: str
    cv_label: str
    prediction_type: str
    prediction_loss: str
    artifact_namespace: str
    calibration_mode: str
    save_root: str
    cv_config_path: str
    checkpoint_root: str
    logs_root: str
    optuna_db_dir: str
    wandb_run_dir: str
    outer_fold_to_run: str
    inner_loop_to_run: str | None
    worker_id: int
    n_trials: int
    workers_per_fold: int
    top_trials: int

def _norm_path(path_value: str, base_dir: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(path_value)))
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir, expanded)
    return os.path.abspath(expanded)

def _resolve_project_root(model_config: dict) -> str:
    root_hint = os.getenv("PROJECT_ROOT") or model_config.get("cfg::save") or os.getcwd()
    return _norm_path(root_hint, os.getcwd())

def _resolve_save_root(model_config: dict, project_root: str) -> str:
    save_hint = model_config.get("cfg::save")
    if save_hint:
        return _norm_path(save_hint, project_root)
    return project_root

def _resolve_cv_config_path(
    explicit_path: str | None,
    project_root: str,
    save_root: str,
    training_years_len: str,
) -> str:
    if explicit_path:
        return _norm_path(explicit_path, project_root)
    return os.path.join(
        save_root,
        "configs",
        "CV_splits",
        f"CV_station_{training_years_len}_config.json",
    )

def _resolve_wandb_dir(save_root: str) -> str:
    return _norm_path(os.getenv("WANDB_RUN_DIR", os.path.join(save_root, "wanddir")), save_root)

def _resolve_optuna_db_dir(save_root: str) -> str:
    return _norm_path(os.getenv("OPTUNA_DB_DIR", os.path.join(save_root, "Optuna_DBs")), save_root)

def _normalize_process_model(value: str | None) -> ProcessModel:
    normalized = str(value or "preles").strip().lower()
    if normalized not in {"preles"}:
        raise ValueError(f"Unsupported process model '{value}'. Expected 'preles'.")
    return normalized  # type: ignore[return-value]

def _normalize_calibration_mode(value: str | None) -> CalibrationMode:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "defaults", "force_calibrate", "require_existing"}:
        raise ValueError(
            f"Unsupported calibration mode '{value}'. Expected 'auto', 'defaults', 'force_calibrate', or 'require_existing'."
        )
    return normalized  # type: ignore[return-value]

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_fold", type=str, help="Outer fold name, e.g. outer_station_5, out_l_1")
    parser.add_argument(
        "--inner_loop",
        "--inner_fold",
        dest="inner_loop",
        type=str,
        help="Optional inner loop name, e.g. inner_loop_1.",
    )
    parser.add_argument("--worker_id", type=int, help="Unique ID for the worker (optional)")
    parser.add_argument("--workers_per_fold", type=int, help="Number of parallel workers contributing to one fold.")
    parser.add_argument("--n_trials", type=int, help="Number of trials for inner HPO per worker")
    parser.add_argument("--top_trials", type=int, help="Number of best completed trials to finalize in outer training.")
    parser.add_argument("--config", type=str, help="Path to the frozen config file (JSON)")
    parser.add_argument(
        "--CV_config_path",
        dest="cv_config_path",
        type=str,
        help="Path to CV split JSON.",
    )
    parser.add_argument(
        "--process-model",
        dest="process_model",
        type=str,
        choices=["preles"],
        help="Override process model namespace.",
    )
    return parser


def _apply_ide_fallback(args: argparse.Namespace, ide: dict[str, Any]) -> argparse.Namespace:
    if (len(sys.argv) == 1 or os.getenv("DEBUG_MODE") == "1") and ide.get("enabled", False):
        print("[DEBUG] No CLI arguments detected — entering local/debug mode.")
        args.outer_fold = ide.get("outer_fold")
        args.inner_loop = ide.get("inner_loop", ide.get("inner_fold"))
        args.worker_id = ide.get("worker_id")
        args.workers_per_fold = ide.get("workers_per_fold")
        args.n_trials = ide.get("n_trials")
        args.top_trials = ide.get("top_trials")
        args.config = ide.get("config")
        args.cv_config_path = ide.get("cv_config_path")
        args.process_model = ide.get("process_model")
        print(
            f"[DEBUG] Outer Fold: {args.outer_fold} | Inner Loop: {args.inner_loop} "
            f"| Worker: {args.worker_id} | Trials: {args.n_trials}"
        )
    return args


def _load_base_config(args: argparse.Namespace) -> dict[str, Any]:
    from Config_Files.config_spatial import build_config

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_config()


def _resolve_model_config(base_config: dict[str, Any], process_model_arg: str | None) -> tuple[dict[str, Any], str]:
    from Config_Files.config_loader import merge_process_model_config, validate_embedding_mode_tokens

    process_model_name = _normalize_process_model(
        process_model_arg or base_config.get("process_model::name") or "preles"
    )
    base_config["process_model::name"] = process_model_name
    resolved = merge_process_model_config(
        base_config=base_config,
        process_model=process_model_name,
    )
    validate_embedding_mode_tokens(resolved)
    return resolved, process_model_name


def _resolve_runtime_context(
    args: argparse.Namespace,
    model_config: dict[str, Any],
    process_model_name: str,
) -> RuntimeContext:
    training_years_len = str(os.getenv("spatial_training_years_len") or "len12")
    project_root = _resolve_project_root(model_config)
    save_root = _resolve_save_root(model_config, project_root)

    cv_config_path = _resolve_cv_config_path(
        explicit_path=args.cv_config_path,
        project_root=project_root,
        save_root=save_root,
        training_years_len=training_years_len,
    )
    if not os.path.exists(cv_config_path):
        raise FileNotFoundError(
            f"CV config not found: {cv_config_path}. "
            "Pass --cv-config-path/--CV_config_path or set IDE_SETTINGS['cv_config_path']."
        )
    cv_config = json_open(cv_config_path)
    cv_label = cv_label_from_path(cv_config_path)

    data_sparsity = model_config.get("data:train:sparse")
    prediction_type = model_config["model::prediction_type"]
    prediction_loss = model_config["training::loss_physic_type"]
    artifact_namespace = str(
        os.getenv("EXPERIMENT_ARTIFACT_NAMESPACE")
        or model_config.get("experiment::artifact_namespace")
        or os.getenv("CALIBRATION_ARTIFACT_NAMESPACE")
        or ""
    ).strip()
    experiment_paths = build_spatial_logs_root(
        save_root=save_root,
        process_model_name=process_model_name,
        cv_label=cv_label,
        prediction_type=prediction_type,
        prediction_loss=prediction_loss,
        sparse=data_sparsity,
        artifact_namespace=artifact_namespace,
    )
    checkpoint_root = experiment_paths["checkpoint_root"]
    logs_root = experiment_paths["logs_root"]
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)

    wandb_run_dir = _resolve_wandb_dir(save_root=save_root)
    optuna_db_dir = _resolve_optuna_db_dir(save_root=save_root)
    os.makedirs(wandb_run_dir, exist_ok=True)
    os.makedirs(optuna_db_dir, exist_ok=True)

    reserved_cv_keys = {"all_stations", "site_metadata"}

    outer_fold_to_run = args.outer_fold
    if outer_fold_to_run is None:
        candidate_folds = [k for k in cv_config.keys() if k not in reserved_cv_keys]
        if not candidate_folds:
            raise ValueError("No outer folds found in CV config.")
        outer_fold_to_run = candidate_folds[0]
        print(f"[INFO] No --outer_fold provided; defaulting to {outer_fold_to_run}")
    if outer_fold_to_run in reserved_cv_keys:
        raise ValueError(
            f"'{outer_fold_to_run}' is a excluded CV config section and cannot be used as an outer fold."
        )
    if outer_fold_to_run not in cv_config:
        raise KeyError(f"Outer fold '{outer_fold_to_run}' not found in CV config.")

    inner_loop_to_run = args.inner_loop
    if inner_loop_to_run is not None:
        available_inner_loops = sorted(
            key for key in cv_config[outer_fold_to_run].keys()
            if key.startswith("inner_loop_")
        )
        if inner_loop_to_run not in available_inner_loops:
            raise KeyError(
                f"Inner loop '{inner_loop_to_run}' not found in outer fold '{outer_fold_to_run}'. "
                f"Available: {available_inner_loops}"
            )

    worker_id = args.worker_id if args.worker_id is not None else 0
    workers_per_fold = args.workers_per_fold
    if workers_per_fold is None:
        workers_per_fold = int(os.getenv("WORKERS_PER_FOLD", "1"))
    if workers_per_fold < 1:
        raise ValueError(f"workers_per_fold must be >= 1, got {workers_per_fold}.")
    # Per-worker trials: explicit --n_trials, else the process-model config's per-fold
    # budget split across the workers (so the fold total doesn't depend on worker count).
    trials_per_fold = model_config.get("hpo::trials_per_fold")
    if args.n_trials is not None:
        n_trials = args.n_trials
    elif trials_per_fold:
        n_trials = math.ceil(int(trials_per_fold) / workers_per_fold)
    else:
        n_trials = 2
    top_trials = args.top_trials
    if top_trials is None:
        top_trials = int(os.getenv("TOP_TRIALS", "5"))
    if top_trials < 1:
        raise ValueError(f"top_trials must be >= 1, got {top_trials}.")
    calibration_mode = _normalize_calibration_mode(os.getenv("calibration_flag", "auto"))

    print(f"[INFO] save_root={save_root}")
    print(f"[INFO] cv_config={cv_config_path}")
    print(f"[INFO] checkpoint_root={checkpoint_root}")
    print(f"[INFO] wandb_dir={wandb_run_dir}")
    print(f"[INFO] optuna_db_dir={optuna_db_dir}")
    print(f"[INFO] experiment_artifact_namespace={artifact_namespace or '<default>'}")
    print(f"[INFO] workers_per_fold={workers_per_fold}  top_trials={top_trials}  "
          f"n_trials/worker={n_trials} ({'--n_trials' if args.n_trials is not None else 'hpo::trials_per_fold' if trials_per_fold else 'fallback'})")

    return RuntimeContext(
        model_config=model_config,
        cv_config=cv_config,
        process_model_name=process_model_name,
        training_years_len=training_years_len,
        cv_label=cv_label,
        prediction_type=prediction_type,
        prediction_loss=prediction_loss,
        artifact_namespace=artifact_namespace,
        calibration_mode=calibration_mode,
        save_root=save_root,
        cv_config_path=cv_config_path,
        checkpoint_root=checkpoint_root,
        logs_root=logs_root,
        optuna_db_dir=optuna_db_dir,
        wandb_run_dir=wandb_run_dir,
        outer_fold_to_run=outer_fold_to_run,
        inner_loop_to_run=inner_loop_to_run,
        worker_id=worker_id,
        n_trials=n_trials,
        workers_per_fold=workers_per_fold,
        top_trials=top_trials,
    )


def _run_single_outer_fold(ctx: RuntimeContext) -> dict[str, Any]:
    return run_outer_fold_CV(
        outer_fold_key=ctx.outer_fold_to_run,
        model_config=ctx.model_config,
        data_config=ctx.cv_config,
        out_dir=ctx.logs_root,
        ctx=ctx,
        n_trials_inner=ctx.n_trials,
        top_trials=ctx.top_trials,
        outerepochs=150,
    )


### Helper functions ####
def setup_logging(checkpoints_dir, in_l=None):
    """Set up loggers for each outer & inner loop."""
    
    # Define log file paths for outer and inner loops
    log_file = os.path.join(checkpoints_dir, "training.log") if in_l is None else os.path.join(checkpoints_dir, in_l, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create a unique logger for each outer/inner loop
    logger_name = f"optuna.{checkpoints_dir}" if in_l is None else f"optuna.{checkpoints_dir}.{in_l}"
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def save_checkpoint(model, optimizer, best_epoch, model_config, checkpoint_dir, epoch, inner_loop, logger,
                    normalization=None, subdir_name=None, reason="new best val loss"):
    """
    Save model checkpoint and associated metadata.

    `subdir_name` defaults to a per-epoch directory. Callers that only ever want to
    keep the single best epoch pass a fixed name so each improvement overwrites the
    previous one instead of accumulating one directory per improving epoch.
    """
    trial_epoch_dir = os.path.join(checkpoint_dir, inner_loop, subdir_name or f"epoch_{epoch}")
    os.makedirs(trial_epoch_dir, exist_ok=True)
    checkpoint_path = os.path.join(trial_epoch_dir, "model.pt")
    optimizer_path = os.path.join(trial_epoch_dir, "optimizer.pt")
    extra_state_path = os.path.join(trial_epoch_dir, "extra_state.pt")
    config_path = os.path.join(trial_epoch_dir, "config.pt")

    torch.save(model.state_dict(), checkpoint_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save({"epoch": best_epoch}, extra_state_path)
    torch.save({"config": model_config}, config_path)
    # The train-fitted z-score constants are part of the model: predictions cannot be
    # returned to physical units without them.
    save_normalization(normalization, trial_epoch_dir)

    logger.info(f"[CHECKPOINT SAVED] {reason} at epoch {epoch}")
    return checkpoint_path

def save_inner_results(inner_results, filepath, out_dir, outer_loop_id):
    """
    Save inner loop results as JSON, but handle non-serializable objects like DataFrames.
    For each inner loop result, if a "trials_df" key exists (a DataFrame),
    save it as a CSV in out_dir and replace it with the CSV file path.
    """
    serializable_results = {}
    for key, result in inner_results.items():
        if isinstance(result, dict):
            temp = {}
            temp["outer_loop"] = outer_loop_id
            for k, v in result.items():
                if k == "trials_df":
                    # Save the DataFrame as CSV.
                    csv_filename = f"trials_df_{outer_loop_id}_{key}.csv"
                    csv_filepath = os.path.join(out_dir, csv_filename)
                    try:
                        v.to_csv(csv_filepath, index=False)
                        temp[k] = csv_filepath  # Store the file path instead.
                    except Exception as e:
                        temp[k] = f"Error saving CSV: {e}"
                else:
                    temp[k] = v
            serializable_results[key] = temp
        else:
            serializable_results[key] = result
    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=4)

######## INNER LOOP Functions #########
def _extract_inner_station_splits(
    inner_conf: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """
    Extract train/test/val station dicts from an inner loop config entry.

    inner_conf is a list of dicts structured as:
        [
            {"data::train::station_id": {station: [years], ...}},
            {"data::test::station_id":  {station: [years], ...}},
            {"data::val::station_id":   {station: [years], ...}},  # optional
        ]
    """
    merged: dict[str, Any] = {}
    for item in inner_conf:
        merged.update(item)
    train_stations = merged["data::train::station_id"]
    test_stations  = merged["data::test::station_id"]
    val_stations   = merged.get("data::val::station_id")
    return train_stations, test_stations, val_stations

def run_inner_loop(
    inner_loop: str,
    inner_conf: Any,
    out_l: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    pm_manager: ProcessModelCalibrationManager,
    trial: optuna.trial.Trial,
    model: NN_module,
    ctx: RuntimeContext,
) -> dict[str, Any]:
    """
    Run one inner-loop training fold and return its results.
    """
    train_stations, test_stations, val_stations = _extract_inner_station_splits(inner_conf)
    out_dir_full = os.path.join(ctx.logs_root, out_l)

    # Embedding mode keeps the raw-dataset path.
    # All other spatial modes use the unified ProcessModelCalibrationManager dataloader handoff.
    if model_config["model::modeltype"] == "embedding": 
        # we need the raw data for the embedding, without the process model predictions.
        train_dataset, test_dataset, val_dataset = pm_manager.load_fold_datasets_embedded(
            train_stations=train_stations,
            test_stations=test_stations,
            val_stations=val_stations,
        )

        train_set, test_set, val_set = build_fold_dataloaders_from_predictions(
            config=model_config,
            train_df=train_dataset,
            test_df=test_dataset,
            val_df=val_dataset,
            process_model=model_config.get("process_model::name"),
            shuffle_train=True,
            sparse=model_config.get("data:train:sparse"),
        )
    else:
        train_set, test_set, val_set = pm_manager.load_fold_dataloaders(
            train_stations=train_stations,
            test_stations=test_stations,
            val_stations=val_stations,
            calibrate=ctx.calibration_mode,
            shuffle_train=True,
            sparse=model_config.get("data:train:sparse"),
        )

    # Initialize tracking variables
    inner_epoch_records = []
    epoch_records = []
    best_test_loss = float("inf")
    best_test_metric = float("inf")
    best_val_loss = float("inf")
    best_val_metric = float("inf")
    best_train_loss = float("inf")
    best_train_metric = float("inf")
    best_epoch = 0
    best_checkpoint_path = None
    # Test loss/metric *at the val-selected epoch*. Unlike `best_test_loss` (a minimum
    # over all epochs, which is optimistically biased by selecting on test noise),
    # these describe the epoch the checkpoint actually comes from.
    test_loss_at_best_epoch = float("nan")
    test_metric_at_best_epoch = float("nan")
    # Consecutive epochs whose train loss was non-finite; see the divergence abort below.
    nonfinite_train_epochs = 0
    divergence_abort_enabled = bool(model_config.get("training::divergence_abort::enabled", True))
    divergence_patience = int(model_config.get("training::divergence_abort::patience", 1))
    #global_step = 0

    checkpoint_window = model_config.get("checkpoint::window_size")
    checkpoints_dir_inner = os.path.join(out_dir_full, "inner_loops")
    os.makedirs(checkpoints_dir_inner, exist_ok=True)

    logger = setup_logging(checkpoints_dir_inner, inner_loop)

    # Loop over training epochs
    for epoch in range(model_config['training::epochs']):
        train_loss, train_metric = model.train_epoch(train_set, epoch)
        test_loss, test_metric = model.test_epoch(test_set)
        
        if model_config.get("training::loss_physic_type") == "regularized":
            val_loss, val_metric = model.val_epoch(val_set)
        else:
            val_loss, val_metric = model.test_epoch(val_set)

        # Convert tensor losses/metrics to scalars
        train_loss_scalar = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        train_metric_scalar = train_metric.item() if isinstance(train_metric, torch.Tensor) else train_metric
        test_loss_scalar = test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        test_metric_scalar = test_metric.item() if isinstance(test_metric, torch.Tensor) else test_metric
        
        if not math.isfinite(test_metric_scalar):
            logger.warning(f"[{inner_loop}] Non-finite test metric detected at epoch {epoch}: {test_metric_scalar}")
    
        val_loss_scalar = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        val_metric_scalar = val_metric.item() if isinstance(val_metric, torch.Tensor) else val_metric
        if not math.isfinite(val_loss_scalar):
            logger.warning(f"[{inner_loop}] Non-finite val loss detected at epoch {epoch}: {val_loss_scalar}")
        if not math.isfinite(val_metric_scalar):
            logger.warning(f"[{inner_loop}] Non-finite val metric detected at epoch {epoch}: {val_metric_scalar}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss_scalar,
            "train_metric": train_metric_scalar,
            "test_loss": test_loss_scalar,
            "test_metric": test_metric_scalar,
            "val_loss": val_loss_scalar,
            "val_metric": val_metric_scalar,
        }
        
        inner_epoch_records.append(epoch_log)

        # Divergence abort.
        # Only the FIRST inner loop reports to the pruner (see the trial.report call
        # below), so a trial that blows up in inner loop 2 or 3 was never visible to it:
        # it ran every remaining epoch, returned a non-finite objective, and Optuna
        # recorded it as FAIL only after the whole budget had been spent. A non-finite
        # TRAIN loss means the weights themselves are non-finite, and no later epoch can
        # undo that, so end the trial as soon as it is established.
        # Pruned (not failed) is the right state: the trial was stopped early for being
        # unpromising, and Optuna's TPE still learns from pruned trials.
        if divergence_abort_enabled and not math.isfinite(train_loss_scalar):
            nonfinite_train_epochs += 1
            if nonfinite_train_epochs >= divergence_patience:
                logger.warning(
                    f"[{inner_loop}] Train loss non-finite ({train_loss_scalar}) for "
                    f"{nonfinite_train_epochs} consecutive epoch(s), latest epoch {epoch}. "
                    f"Weights cannot recover — aborting trial {trial.number}."
                )
                trial.set_user_attr("diverged_inner_loop", inner_loop)
                trial.set_user_attr("diverged_at_epoch", epoch)
                raise optuna.exceptions.TrialPruned()
        else:
            nonfinite_train_epochs = 0

        # Check and update best test loss
        if train_loss_scalar < best_train_loss:
            best_train_loss = train_loss_scalar
        if math.isfinite(train_metric_scalar) and train_metric_scalar < best_train_metric:
            best_train_metric = train_metric_scalar
        if math.isfinite(test_loss_scalar) and test_loss_scalar < best_test_loss:
            # Minimum over epochs — a DIAGNOSTIC only. Reported alongside, but not the
            # fold's test performance: see `test_loss_at_best_epoch`.
            best_test_loss = test_loss_scalar

        if math.isfinite(test_metric_scalar) and test_metric_scalar < best_test_metric:
            best_test_metric = test_metric_scalar

        val_improved = False
        if math.isfinite(val_loss_scalar) and val_loss_scalar < best_val_loss:
            best_val_loss = val_loss_scalar
            # `best_epoch` decides which epoch's checkpoint is kept. Select it on the
            # val split, not on the inner-test split: val is already computed every
            # epoch and was simply unused here.
            best_epoch = epoch
            val_improved = True
            # Read the test numbers off the same epoch the checkpoint comes from.
            test_loss_at_best_epoch = test_loss_scalar
            test_metric_at_best_epoch = test_metric_scalar
        if val_metric_scalar < best_val_metric:
            best_val_metric =val_metric_scalar


        is_last_epoch = epoch == model_config['training::epochs'] - 1

        # Log progress periodically
        if epoch % checkpoint_window == 0 or is_last_epoch:
            epoch_records.append(epoch_log)
            logger.info(f"Epoch {epoch}: {epoch_log}")

        # Checkpointing is NOT gated on the logging window: a new best epoch can fall
        # anywhere, and with window_size=10 a best epoch at e.g. 7 was silently never
        # written, while the unconditional final-epoch save left `best_checkpoint_path`
        # pointing at the last epoch in nearly every trial. Each improvement overwrites
        # one fixed directory, so only the selected epoch is kept on disk.
        if val_improved:
            best_checkpoint_path = save_checkpoint(
                model, model.optimizer, best_epoch, model_config,
                checkpoints_dir_inner, epoch, f"{inner_loop}/{trial._trial_id}", logger,
                normalization=getattr(train_set.dataset, "normalization", None),
                subdir_name="best",
                reason=f"New best val loss {best_val_loss:.5f}",
            )
        elif is_last_epoch and best_checkpoint_path is None:
            # No epoch ever produced a finite val loss — keep the final weights so the
            # trial still has an artifact, and say so. The recorded epoch must be the
            # one these weights come from (`epoch`), not `best_epoch`, which is still
            # sitting at its initial 0 because nothing ever improved.
            best_checkpoint_path = save_checkpoint(
                model, model.optimizer, epoch, model_config,
                checkpoints_dir_inner, epoch, f"{inner_loop}/{trial._trial_id}", logger,
                normalization=getattr(train_set.dataset, "normalization", None),
                subdir_name="best",
                reason="Final epoch (no finite val loss was ever observed)",
            )
            logger.warning(
                f"[{inner_loop}] No finite val loss in any epoch; saved final-epoch weights instead."
            )

        if epoch % checkpoint_window == 0 or is_last_epoch:
            # global_step += 1
            trial.set_user_attr("checkpoint_path", best_checkpoint_path)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("val_loss", val_loss_scalar)
            trial.set_user_attr("val_best_loss", best_val_loss)
            # innerloop

        # # Check for pruning if this is the first inner loop
        first_inner_loop = next((key for key in data_config[out_l].keys() if key.startswith("inner_loop_")), None)
        if inner_loop == first_inner_loop:
            # Prune on the same quantity the objective optimises (val), otherwise the
            # pruner kills trials by a criterion the search is not ranking them on.
            trial.report(val_loss_scalar, epoch)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_epoch", epoch)
                raise optuna.exceptions.TrialPruned()

    # Save epoch logs as JSON
    log_file = os.path.join(checkpoints_dir_inner, f"{inner_loop}_log.json")
    with open(log_file, "w") as f:
        json.dump(epoch_records, f, indent=4)

    return {
        "inner_epoch_records": inner_epoch_records,
        "best_train_loss": best_train_loss,
        "best_train_metric": best_train_metric,
        # Minimum over epochs — diagnostic only.
        "best_test_loss": best_test_loss,
        "best_test_metric": best_test_metric,
        # Test performance at the val-selected epoch: this is the fold's reportable
        # test number, not the min-over-epochs above.
        "test_loss_at_best_epoch": test_loss_at_best_epoch,
        "test_metric_at_best_epoch": test_metric_at_best_epoch,
        "best_val_metric": best_val_metric,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
    }

def define_search_space(trial, model_config,
                        random_model_length=True,
                        random_activation = True,
                        learning_momentum=True, 
                        dropout_tuning=True, 
                        random_training_optimiser=True):
    """
    Configures the hyperparameter search space dynamically for Optuna.
    Updates model_config with tuned hyperparameters while keeping untuned ones unchanged.

    Args:
        trial: Optuna trial object.
        model_config: Base configuration dictionary (will not be modified directly).
        random_model_length: If True, vary the number of hidden layers and sizes.
        learning_momentum: If True, tune learning rate and momentum.
        dropout_tuning: If True, include dropout tuning.
        random_training_optimiser: If True, allow random optimizer selection.

    Returns:
        updated_config: A copy of model_config with tuned hyperparameters applied.
    """
    # Create a copy of model_config to avoid modifying the original
    updated_config = copy.deepcopy(model_config)

    # Seed: FIXED, deliberately not searched.
    # `NN_module` now honours `model::seed` (it previously overwrote it with
    # `training::seed`, making it inert). Searching it would let TPE optimise the
    # random initialisation itself — "best seed" is noise, so the top-k trials would
    # risk being lucky inits rather than good architectures. Replication across seeds
    # belongs to the _otrainN draw mechanism, which varies data rather than luck.
    updated_config["model::seed"] = updated_config.get("model::seed", updated_config["training::seed"])
    trial.set_user_attr("model::seed", updated_config["model::seed"])


    # Randomize model length (number of layers & hidden sizes)
    if random_model_length:
        if updated_config['training::loss_physic_type'] == 'bias':
            layer_number = 2 # trial.suggest_int("model::layer_number", 2, 3)
            hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 5, 25) for i in range(layer_number-1)]
        else:
            layer_number = trial.suggest_int("model::layer_number", 2, 5)
            hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 15, 75) for i in range(layer_number-1)]
        
        updated_config["model::layer_number"] = layer_number
        trial.set_user_attr("model::layer_number", layer_number)

        # Suggest hidden sizes per possible layer
        updated_config["model::hidden_size"] = hidden_sizes
        trial.set_user_attr("model::hidden_size", hidden_sizes)

        # Fixed bias list: one per layer
        bias_list = [True] * layer_number
        updated_config["model::bias_list"] = bias_list
        trial.set_user_attr("model::bias_list", bias_list)

        # Activation choice
        if random_activation:
            activation = trial.suggest_categorical("model::activation", ["Relu", "Tanh", "Sigmoid"])
            activation_list = [activation] * (layer_number - 1) + [None]
            updated_config["model::activation"] = activation_list
            trial.set_user_attr("model::activation", activation_list)
        else:
            # If activation is fixed
            activation_list = updated_config["model::activation"]
            if isinstance(activation_list, list):
                # Adjust the list size to match layer_number
                activation_list = activation_list[:layer_number - 1] + [None]  # Ensure length matches layer_number
            updated_config["model::activation"] = activation_list
            trial.set_user_attr("model::activation", activation_list)

    # Optimizer: Adam family only.
    # SGD was removed from the space. It was never actually searched in any of the
    # archived paper-2 studies (`training::optimizer` appears in none of them), so every
    # historical result is Adam; there is no evidence SGD helps here, and it is the sole
    # source of divergence risk, because its usable region is a narrow diagonal in
    # lr x momentum (the effective step scales as lr/(1-momentum), so lr=0.3 with
    # momentum=0.99 is an effective step of 30 — divergence, not a hyperparameter).
    # Mapping that diagonal costs far more than the trial budget allows.
    if random_training_optimiser:
        optimizer_name = trial.suggest_categorical("training::optimizer", ["Adam", "AdamW"])
    else:
        optimizer_name = updated_config["training::optimizer"]
    updated_config["training::optimizer"] = optimizer_name
    trial.set_user_attr("training::optimizer", optimizer_name)

    # `training::momentum` is NOT searched: torch's Adam/AdamW have no momentum term
    # (Adam's equivalent is beta_1), so the value was read by nothing. It was tuned over
    # 0.0-0.3 in all 277 archived studies while having no effect on any of them.
    # Pinned here to the config default purely so the key stays well-defined.
    updated_config["training::momentum"] = updated_config.get("training::momentum", 0.0)

    # The embedding branch below suggests its own "training::learning_rate"; skip here
    # to avoid suggesting the same name twice with different distributions in one trial.
    is_embedding_loss = updated_config.get("training::loss_physic_type") == "embedding"
    if learning_momentum and not is_embedding_loss:
        learning_rate = trial.suggest_float("training::learning_rate", 1e-4, 1e-2, log=True)
        updated_config["training::learning_rate"] = learning_rate
        trial.set_user_attr("training::learning_rate", learning_rate)

    # beta_1/beta_2 are FIXED at the torch defaults rather than searched. beta_1 is
    # Adam's momentum term and 0.9 is almost always right; neither is the parameter that
    # matters next to learning rate and weight decay, and at ~48 trials the budget is
    # better spent on those. They were never searched historically either.
    updated_config["training::beta_1"] = updated_config.get("training::beta_1", 0.9)
    updated_config["training::beta_2"] = updated_config.get("training::beta_2", 0.999)

    # Weight decay was pinned at 0 and never searched. With folds training on ~3 sites,
    # regularisation strength is one of the more consequential knobs for OOD transfer,
    # so it is now tuned. The 1e-8 floor stands in for "effectively off" (a log-uniform
    # range cannot include exact zero).
    weight_decay = trial.suggest_float("training::weight_decay", 1e-8, 1e-2, log=True)
    updated_config["training::weight_decay"] = weight_decay
    trial.set_user_attr("training::weight_decay", weight_decay)

    # Tune dropout layers if enabled
    if dropout_tuning:
        if updated_config['training::loss_physic_type'] == 'naive':
            updated_config["model::dropout_list"] = trial.suggest_float("model::dropout_list", 0, 0.5)
        else:
            updated_config["model::dropout_list"] = trial.suggest_float("model::dropout_list", 0, 0.3)

    # Regularization tuning for physics-based loss
    if updated_config.get("training::loss_physic_type") == "regularized":
        updated_config["training::physical_loss_regul_term"] = trial.suggest_float("training::physical_loss_regul_term", 0.0, 1.0)

    if updated_config.get("training::loss_physic_type") == "embedding":
        # Suggest the base learning rate
        base_lr = trial.suggest_float("training::learning_rate", 1e-5, 1e-2, log=True)
        updated_config["training::learning_rate"] = base_lr

        # Ensure output MLP learning rate is at least 1.5x larger
        # (but still constrained by upper bound)
        min_output_lr = min(base_lr * 1.5, 1e-1)
        max_output_lr = 1e-1  # Set upper limit reasonably

        output_lr = trial.suggest_float(
            "output_MLP::Learning_rate",
            min_output_lr,
            max_output_lr,
            log=True
        )
        updated_config["output_MLP::Learning_rate"] = output_lr
        
    # If using LSTM, adjust input size and depth
    #     # Adjust LSTM parameters if applicable
        if updated_config.get("model::modeltype") == "LSTM":
            updated_config["model::input_size"] = len(updated_config['data::train::x_parameter']) + len(updated_config['data::train::y_parameter'])
            updated_config["model::lstm_depth"] = trial.suggest_int("model::lstm_depth", 1, 5)
            # the hidden state is a memory buffer, not one slot per input feature,
            # so its width is searched independently of the number of inputs
            updated_config["model::hidden_lstm"] = trial.suggest_int("model::hidden_lstm", 8, 64, log=True)

    return updated_config

def objective(
    trial: optuna.trial.Trial,
    out_l: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    ctx: RuntimeContext,
) -> float:
    """ 
    Objective function for Optuna trials.
    This function is called for each trial and runs the inner loops for a given outer fold.
    It trains a model, evaluates it on an inner loop, and returns the test loss.
    The model is reinitialized for each inner loop with the hyperparameters suggested by Optuna trial.
    The test loss is aggregated across all inner loops and returned as the objective value for Optuna.    
    
    Args:
        trial: Optuna trial object.
        out_l: Outer loop identifier (e.g., outer_station_1).
        model_config: Base configuration dictionary for the model.
        data_config: Configuration dictionary containing data splits.
    """  

    # Basic environment info for logging
    hostname = socket.gethostname()
    pid = os.getpid()
    print(f"[{hostname}][PID:{pid}] Starting objective for {out_l}, trial {trial.number}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"[INFO] Device found: {device}")

    model_config = define_search_space(trial, model_config)
    
    # set optuna seed
    torch.manual_seed(model_config['model::seed'])
    model= NN_module(model_config).to(device)

    #average training, validation and test loss
    best_train_loss_avg = list()
    train_metric_avg = list()
    test_loss__best_avag = list()
    test_metric_best_avag = list()
    test_loss_at_best_epoch_avg = list()
    test_metric_at_best_epoch_avg = list()
    val_metric_best_avag = list()
    val_loss_avag_best = list()
    
    # Track results
    all_epoch_records = {}

    # Create fold manager instance for this trial.
    pm_manager = ProcessModelCalibrationManager(
        config=model_config,
        process_model=model_config.get("process_model::name"),
        cv_config=data_config,
        outer_fold=out_l,
    )

    # Checkpoint directory for logging.
    out_dir_full = os.path.join(ctx.logs_root, out_l)
    
    checkpoints_dir_inner = os.path.join(out_dir_full, "inner_loops")
    os.makedirs(checkpoints_dir_inner, exist_ok=True)

    logger = setup_logging(checkpoints_dir_inner)
    logger.info(f"[{hostname}][PID:{pid}] Starting evaluation for outer fold {out_l} in trial {trial.number}")
    global_step_loop = 0

    # starts 3 inner loops and evalutates averages 
    for i, (inner_loop, inner_conf) in enumerate(data_config[out_l].items()):
        if not inner_loop.startswith("inner_loop"):
            continue  # skip metadata keys (outer_split, koppen_group, forest_type, etc.)
        # Run innerloop with configuration
        logger.info(f"[{hostname}][PID:{pid}] Processing {inner_loop} for fold {out_l}")
        inner_results = run_inner_loop(
            inner_loop,
            inner_conf,
            out_l,
            model_config,
            data_config,
            pm_manager,
            trial,
            model,
            ctx,
        )

        #which of these losses should be tracked ??
        best_train_loss_avg.append(inner_results["best_train_loss"])
        train_metric_avg.append(inner_results["best_train_metric"])
        test_loss__best_avag.append(inner_results["best_test_loss"])
        test_metric_best_avag.append(inner_results["best_test_metric"])
        test_loss_at_best_epoch_avg.append(inner_results["test_loss_at_best_epoch"])
        test_metric_at_best_epoch_avg.append(inner_results["test_metric_at_best_epoch"])
        val_metric_best_avag.append(inner_results["best_val_metric"])
        val_loss_avag_best.append(inner_results["best_val_loss"]) 
        all_epoch_records[inner_loop] = inner_results["inner_epoch_records"] # do we need this infomation ??

        if i >= 1:
            global_step_loop += 1
            
        # initilise model again
        torch.manual_seed(model_config['model::seed'])
        model= NN_module(model_config).to(device)
        logger.info(f"Model re-initialized for next inner loop")

    # The value handed to Optuna. It is the SAME quantity the top-k trials are later
    # ranked on (`val_best_loss_mean` in run_inner_loops_HPO), so the TPE sampler and
    # the Hyperband pruner steer the search towards what actually decides selection.
    # It previously returned the inner-test mean while selection used val, so the
    # search optimised one surface and the selection read off another.
    aggregated_val_loss = np.mean(val_loss_avag_best)

    # # After processing all inner loops, compute aggregated metrics
    logger.info(f"[{hostname}][PID:{pid}] Finished objective for outer fold {out_l}, trial {trial.number} with aggregated val loss: {aggregated_val_loss:.5f}")

    trial.set_user_attr("val_best_loss_mean", aggregated_val_loss) #this is the value for choosing the models for the outer fold
    trial.set_user_attr("test_best_loss_mean", np.mean(test_loss__best_avag))
    trial.set_user_attr("val_best_metric_mean", np.mean(val_metric_best_avag))
    trial.set_user_attr("test_best_metric_mean", np.mean(test_metric_best_avag))
    trial.set_user_attr("train_best_loss_mean", np.mean(best_train_loss_avg))
    trial.set_user_attr("train_best_metric_mean", np.mean(train_metric_avg))
    # Test performance at the val-selected epoch, averaged over inner loops. Report
    # THIS rather than `test_best_loss_mean`, which is a mean of per-loop minima over
    # epochs and is therefore biased low by selection on test noise.
    # NaN entries mean an inner loop never saw a finite val loss; store None rather
    # than a NaN so the attribute round-trips cleanly through the Optuna storage.
    def _mean_or_none(values):
        mean = float(np.mean(values)) if len(values) else float("nan")
        return None if not math.isfinite(mean) else mean

    trial.set_user_attr("test_loss_at_best_epoch_mean", _mean_or_none(test_loss_at_best_epoch_avg))
    trial.set_user_attr("test_metric_at_best_epoch_mean", _mean_or_none(test_metric_at_best_epoch_avg))

    return aggregated_val_loss

def _other_workers_exited(ctx) -> bool:
    """
    True once every other worker of this fold has exited, successfully or not.

    The batch script writes <WORKER_STATUS_DIR>/worker<N>.exited when worker N's process
    ends (the shell does it, so crashes and kills count too). Without WORKER_STATUS_DIR
    this always returns False and the barrier behaves as before.
    """
    status_dir = os.getenv("WORKER_STATUS_DIR")
    if not status_dir:
        return False
    return all(
        os.path.exists(os.path.join(status_dir, f"worker{w}.exited"))
        for w in range(ctx.workers_per_fold)
        if w != ctx.worker_id
    )


def run_inner_loops_HPO(
    out_l: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    ctx: RuntimeContext,
    n_trials_inner: int = 8,
    top_trials: int = 5,
    wait_for_all_workers: bool = False,
):
    """
    Run HPO for inner-loops configuration using Optuna.
    Returns a tuple: (best_params, best_metric, trial_records)
    """

    prediction_loss = model_config["training::loss_physic_type"]
    prediction_type = model_config["model::prediction_type"]
    data_sparsity = model_config.get("data:train:sparse", None)

    process_model = model_config["process_model::name"]
    os.makedirs(ctx.optuna_db_dir, exist_ok=True)
    optuna_db = build_spatial_optuna_db_path(
        optuna_db_dir=ctx.optuna_db_dir,
        process_model=process_model,
        prediction_type=prediction_type,
        prediction_loss=prediction_loss,
        outer_fold=out_l,
        cv_label=ctx.cv_label,
        sparse=data_sparsity,
        artifact_namespace=ctx.artifact_namespace,
    )
    STUDY_NAME = optuna_db["study_name"]
    db_file = optuna_db["db_file"]
    rdb_string_url = optuna_db["url"]
    
    time.sleep(random.uniform(1.0, 5.0))  # Delay each worker slightly
    # All WORKERS_PER_FOLD workers reach here for the same db_file at roughly
    # the same time. RDBStorage's schema init (alembic table creation) isn't
    # safe under concurrent first-time creation, so serialize it with a lock
    # scoped to this fold's DB file — only the first worker to grab it creates
    # the schema, the rest just attach to the now-existing DB.
    # timeout=600 so a stale lock on the NFS-backed DB dir raises instead of
    # hanging the whole array task silently until walltime.
    with FileLock(db_file + ".lock", timeout=600):
        storage_instance = RDBStorage(
            url=rdb_string_url,
            heartbeat_interval=60,
            grace_period=180,
            # The DB lives on NFS and all WORKERS_PER_FOLD workers write user
            # attrs at the same checkpoint boundaries. sqlite3's default busy
            # timeout is 5s, far too short here — workers were dying with
            # "database is locked". 300s matches Optuna_temporal_NCV.py.
            engine_kwargs={"connect_args": {"timeout": 300, "check_same_thread": False}},
        )

    study = optuna.create_study(
            study_name= STUDY_NAME, #study_name,
            storage=storage_instance, #study_storage,
            direction="minimize", 
            # max_resource must be explicit: left at "auto" it is inferred from whichever
            # trial completes first, so with several workers sharing one RDB the bracket
            # structure would be decided by a race and would shift silently whenever the
            # epoch budget changes.
            pruner=HyperbandPruner(
                min_resource=10,  # epochs run before a trial can be halved out
                max_resource=int(model_config['training::epochs']),
            ),
            # constant_liar: workers sample concurrently against one RDB; without it
            # they cannot see each other's in-flight trials and duplicate the same region.
            # multivariate: hidden_size_i only exist for a given layer_number, so the
            # dimensions are coupled and cannot be modelled independently.
            sampler=TPESampler(constant_liar=True, multivariate=True),
            load_if_exists=True
            )
    
    # The inner-loop objective function can be defined separately.
    # A transient storage error must fail one trial, not the worker process:
    # worker 0 is the leader, and killing it leaves the fold with no outer
    # training and no summary JSON at all.
    # The budget is per fold, not per worker, so a resumed fold only tops up.
    optimize_within_fold_budget(
        study,
        lambda trial: objective(trial, out_l, model_config, data_config, ctx),
        n_trials_per_worker=n_trials_inner,
        fold_budget=ctx.workers_per_fold * n_trials_inner,
        catch=(optuna.exceptions.StorageInternalError,),
    )

    if wait_for_all_workers and ctx.workers_per_fold > 1:
        expected_finished_trials = ctx.workers_per_fold * n_trials_inner
        timeout_sec = int(os.getenv("OPTUNA_SYNC_TIMEOUT_SEC", "7200"))
        poll_sec = int(os.getenv("OPTUNA_SYNC_POLL_SEC", "20"))
        # FAIL does not count: a resumed fold carries the FAILed trials its killed run
        # left behind, and the workers re-run those slots (optimize_within_fold_budget).
        # A crashed worker's shortfall is released by _other_workers_exited or the timeout.
        finished_states = {
            TrialState.COMPLETE,
            TrialState.PRUNED,
        }
        shortfall_grace_sec = int(os.getenv("OPTUNA_SYNC_SHORTFALL_GRACE_SEC", "600"))
        wait_started = time.time()
        shortfall_started = None
        while True:
            refreshed_study = optuna.load_study(study_name=STUDY_NAME, storage=storage_instance)
            finished_count = sum(1 for t in refreshed_study.trials if t.state in finished_states)
            # A fold finalized from too few COMPLETE trials silently shrinks the
            # experiment (2 outer configs instead of top_trials). So once the
            # count is met, give stragglers a bounded grace period to supply
            # enough selectable trials before finalizing anyway.
            selectable_count = sum(
                1 for t in refreshed_study.trials
                if t.state == TrialState.COMPLETE and "val_best_loss_mean" in t.user_attrs
            )
            if finished_count >= expected_finished_trials:
                if selectable_count >= top_trials:
                    study = refreshed_study
                    print(
                        f"[INFO] Fold {out_l}: collected {finished_count}/{expected_finished_trials} "
                        f"finished trials ({selectable_count} selectable). Starting outer-fold finalization."
                    )
                    break
                if shortfall_started is None:
                    shortfall_started = time.time()
                    print(
                        f"[WARNING] Fold {out_l}: all {finished_count}/{expected_finished_trials} trials "
                        f"finished but only {selectable_count} are selectable (need {top_trials}). "
                        f"Waiting up to {shortfall_grace_sec}s for stragglers."
                    )
                elif time.time() - shortfall_started >= shortfall_grace_sec:
                    study = refreshed_study
                    print(
                        f"[WARNING] Fold {out_l}: finalizing with only {selectable_count} selectable "
                        f"trials (requested {top_trials}) after {shortfall_grace_sec}s grace."
                    )
                    break

            # A crashed worker never delivers its remaining trials, so the count above
            # would only be released by the timeout. Once every other worker has exited,
            # nothing more can arrive: re-read the study (after the check, so their last
            # trials are included) and proceed.
            if _other_workers_exited(ctx):
                study = optuna.load_study(study_name=STUDY_NAME, storage=storage_instance)
                finished_count = sum(1 for t in study.trials if t.state in finished_states)
                print(
                    f"[WARNING] Fold {out_l}: all other workers have exited "
                    f"({finished_count}/{expected_finished_trials} trials finished). Proceeding."
                )
                break

            elapsed = int(time.time() - wait_started)
            if elapsed >= timeout_sec:
                study = refreshed_study
                print(
                    f"[WARNING] Fold {out_l}: timeout while waiting for workers "
                    f"({finished_count}/{expected_finished_trials} finished after {elapsed}s). "
                    "Proceeding with currently available trials."
                )
                break

            print(
                f"[INFO] Fold {out_l}: waiting for workers "
                f"({finished_count}/{expected_finished_trials} finished, elapsed={elapsed}s)."
            )
            time.sleep(poll_sec)
    
    # Get a DataFrame of all trials
    trial_df = study.trials_dataframe()
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.FAIL
    ]

    # Sort completed scored trials by lowest average validation loss across inner loops.
    sorted_trials = sorted(
        [
            t for t in completed_trials
            if "val_best_loss_mean" in t.user_attrs
        ],
        key=lambda t: t.user_attrs["val_best_loss_mean"],
    )

    # Select the top 5 best parameter sets and their corresponding objective values
    selected_trials = sorted_trials[:top_trials]
    top_params = [{**trial.params, **trial.user_attrs} for trial in selected_trials]
    top_metrics = [trial.value for trial in selected_trials]

    print(
        f"[INFO] Fold {out_l}: trials total={len(study.trials)}, "
        f"complete={len(completed_trials)}, pruned={len(pruned_trials)}, failed={len(failed_trials)}, "
        f"selectable_complete={len(sorted_trials)}, requested_top_trials={top_trials}, "
        f"selected_for_outer={len(selected_trials)}"
    )
    if len(selected_trials) < top_trials:
        print(
            f"[WARNING] Fold {out_l}: requested top {top_trials} outer configs, "
            f"but only {len(selected_trials)} completed scored trials are available."
        )

    return top_params, top_metrics, trial_df

######## OUTER LOOP Functions #########
def merge_best_params(model_config: dict[str, Any], best_result: dict[str, Any]) -> dict[str, Any]:
    """
    Merges a best_result dictionary (from Optuna/Ray) into the base model_config.
    Supports pattern-matching keys like hidden_size_0, dropout_1, etc., and merges them into lists.
    """
    best_model = copy.deepcopy(model_config)

    # First, handle best_result.params (the ones Optuna suggests)
    for key, value in best_result.items():
        # Check if the key is in model_config for direct matching
        if key in model_config:
            if key == "model::activation":
                # Ensure activation list is properly expanded if it's just a string
                if isinstance(value, str) and isinstance(model_config[key], list):
                    layer_number = best_result.get("model::layer_number", model_config.get("model::layer_number"))
                    best_model[key] = [value] * (layer_number - 1) + [None]
                    continue

            if isinstance(value, type(model_config[key])):
                best_model[key] = value
            elif isinstance(value, (int, float)) and isinstance(model_config[key], (int, float)):
                best_model[key] = value
            else:
                raise Exception(f"[merge_best_params] Type mismatch for key '{key}': expected {type(model_config[key])}, got {type(value)}")
        else:
            # Unrecognized new key, add directly
            best_model[key] = value

    return best_model

def run_outer_training_fold(
    out_l: str,
    best_params: dict[str, Any],
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    config_id: int,
    outerepochs: int,
    ctx: RuntimeContext,
) -> dict[str, Any]:
    """
    Using the aggregated best inner-loop parameters, merge them into the model config
    and run the outer-loop training once.
    """
    hostname = socket.gethostname()
    pid = os.getpid()
    print(f"[{hostname}][PID:{pid}] Running outer training for fold {out_l} with config_id {config_id}")

    # Merge the aggregated inner-loop params into the base model configuration.
    best_config = merge_best_params(model_config, best_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(best_config['model::seed'])
    
    model = NN_module(best_config).to(device)

    wandb.init(project="HPO_S_NCV_Outer_Fold", 
                name=f"{out_l}_config_{config_id}",
                group=f"{ctx.cv_label}" + (f"_{ctx.artifact_namespace}" if ctx.artifact_namespace else ""),#f"trial_number_{trial_number}",
                reinit=True,
                dir=ctx.wandb_run_dir,
                settings=wandb.Settings(start_method="thread"),
                config=best_config
    )

    outer_results = dict()

    # Determine outer-loop train/test splits from data_config["all_stations"]
    train_config = copy.deepcopy(best_config)
    test_config = copy.deepcopy(best_config)
    
    outer_split = data_config[out_l]["outer_split"]
    train_config['data::train::station_id'] = outer_split["train"]
    test_config['data::test::station_id'] = outer_split["test"]
    
    best_config['data::train::station_id'] = copy.deepcopy(outer_split["train"])
    best_config['data::test::station_id']  = copy.deepcopy(outer_split["test"])

    if best_config["model::modeltype"] == "embedding":
        pm_manager = ProcessModelCalibrationManager(
            config=best_config,
            process_model=best_config.get("process_model::name"),
            cv_config=data_config,
            outer_fold=out_l,
        )
        train_dataset, test_dataset, _ = pm_manager.load_fold_datasets_embedded(
            train_stations=train_config['data::train::station_id'],
            test_stations=test_config['data::test::station_id'],
            val_stations=None,
        )

        train_set, test_set, _ = build_fold_dataloaders_from_predictions(
            config=model_config,
            train_df=train_dataset,
            test_df=test_dataset,
            val_df=None,
            process_model=model_config.get("process_model::name"),
            shuffle_train=True,
            sparse=model_config.get("data:train:sparse"),
        )
    else:
        pm_manager = ProcessModelCalibrationManager(
            config=best_config,
            process_model=best_config.get("process_model::name"),
            cv_config=data_config,
            outer_fold=out_l,
        )
        train_set, test_set, _ = pm_manager.load_fold_dataloaders(
            train_stations=train_config['data::train::station_id'],
            test_stations=test_config['data::test::station_id'],
            val_stations=None,
            calibrate=ctx.calibration_mode,
            shuffle_train=True,
            sparse=best_config.get("data:train:sparse"),
        )

    # get final results of HPO
    final_train_log = []
    best_test= np.inf
    best_metric= np.inf
    best_epoch = -1

    # Outer early stopping — deliberately on TRAINING loss.
    # This loop is a retrain of the architecture the inner HPO already chose, so it
    # needs a convergence criterion, not model selection. It must never look at the
    # test split: the previous version saved whichever epoch minimised test loss,
    # which over `outerepochs` epochs made the saved weights a function of test
    # performance and destroyed the fold as an out-of-sample estimate.
    # Caveat: training loss falls near-monotonically, so this is a convergence guard,
    # not an overfitting guard. `outerepochs` remains the hard backstop.
    es_patience = int(model_config.get("training::outer_early_stop::patience", 20))
    es_min_delta = float(model_config.get("training::outer_early_stop::min_delta", 1e-4))
    es_enabled = bool(model_config.get("training::outer_early_stop::enabled", True))
    best_train_loss = np.inf
    bad_epochs = 0
    stopped_epoch = None

    # LR schedule — outer loop only, and deliberately ReduceLROnPlateau rather than a
    # horizon-based schedule (cosine): this loop stops at a variable epoch, so a fixed
    # T_max would either skip its own annealed tail or suppress the early stop above.
    # At a constant LR a converged run oscillates in a noise band whose width scales
    # with the LR, so what trips `es_min_delta` is partly step size rather than genuine
    # convergence. Halving the LR on a plateau lets that criterion mean what it says.
    # The inner loop is left unscheduled: with Hyperband pruning at min_resource, a
    # decayed model looks artificially poor in exactly the window the pruner judges it.
    sched_enabled = bool(model_config.get("training::outer_lr_schedule::enabled", True))
    sched_factor = float(model_config.get("training::outer_lr_schedule::factor", 0.5))
    sched_patience = int(model_config.get("training::outer_lr_schedule::patience", 8))
    sched_min_lr = float(model_config.get("training::outer_lr_schedule::min_lr", 1e-7))
    scheduler = None
    lr_reduction_count = 0
    if sched_enabled:
        if es_enabled and sched_patience >= es_patience:
            # The stop would fire before a reduction ever had a chance to help, making
            # the schedule dead code.
            logger_msg = (
                f"[WARNING] Outer {out_l}: lr-schedule patience ({sched_patience}) >= early-stop "
                f"patience ({es_patience}); the run will stop before a reduction can take effect."
            )
            print(logger_msg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer,
            mode="min",
            factor=sched_factor,
            patience=sched_patience,
            min_lr=sched_min_lr,
        )

    # Set up logging for outer loop
    out_dir_outer = os.path.join(ctx.logs_root, out_l)
    model_dir = os.path.join(out_dir_outer, "models")
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logging(out_dir_outer, out_l)
    logger.info(f"[{hostname}][PID:{pid}] Starting outer training for fold {out_l}")

    # Run outer training for the specified number of epochs                 
    for epoch in range(outerepochs):                                                 
        train_loss, train_metric = model.train_epoch(train_set, epoch)
        test_loss, test_metric = model.test_epoch(test_set)

        # Convert tensor losses to float if needed
        train_loss_value = train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        train_metric_value = train_metric.item() if isinstance(train_metric, torch.Tensor) else train_metric
        test_loss_value  = test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss
        test_metric_value  = test_metric.item() if isinstance(test_metric, torch.Tensor) else test_metric
        
        final_train_log.append({
            "epoch": epoch,
            "train_loss": train_loss_value,
            "train_metric": train_metric_value,
            "test_loss": test_loss_value,
            "test_metric": test_metric_value,
        })
        logger.info(f"[{hostname}][PID:{pid}] Outer {out_l} Epoch {epoch} - train_loss: {train_loss_value:.4f}, test_loss: {test_loss_value:.4f}")

        # Step the LR schedule on the same signal the early stop watches. Log the LR
        # only when it actually changes, so the drop is visible in the training log.
        if scheduler is not None and math.isfinite(train_loss_value):
            lr_before = model.optimizer.param_groups[0]["lr"]
            scheduler.step(train_loss_value)
            lr_after = model.optimizer.param_groups[0]["lr"]
            if lr_after != lr_before:
                lr_reduction_count += 1
                logger.info(
                    f"[{hostname}][PID:{pid}] Outer {out_l} epoch {epoch}: "
                    f"lr reduced {lr_before:.3e} -> {lr_after:.3e} (train-loss plateau)."
                )

        # Update best test loss if improved
        # NOTE: this loop runs for `outerepochs`, not `training::epochs` (the inner-loop
        # budget) — the last-epoch term has to track the loop it is actually inside.
        if (epoch % model_config['checkpoint::window_size'] == 0 or epoch == outerepochs - 1):
            wandb.log({"epoch": epoch,
                       "fold": out_l,
                       "test_loss": test_loss_value,
                       "config_ID": config_id,
                       "lr": model.optimizer.param_groups[0]["lr"]})
        
        if math.isfinite(test_metric_value) and test_metric_value < best_metric:
            best_metric = test_metric_value

        # Track the best test loss/epoch as DIAGNOSTICS ONLY. They are still reported
        # in outer_summary, but no longer decide which weights are kept — see the
        # early-stopping note above.
        if math.isfinite(test_loss_value) and test_loss_value < best_test:
            best_test = test_loss_value
            best_epoch = epoch

        # Convergence check on training loss. Relative threshold so it stays
        # meaningful across the different loss scales the naive/hybrid configs give.
        # The first finite epoch must seed `best_train_loss` unconditionally: with the
        # np.inf sentinel the relative threshold is itself inf, so `inf - loss > inf`
        # is False, the sentinel never gets replaced, and every epoch counts as bad —
        # which stopped every outer run at exactly `es_patience` regardless of whether
        # it had converged.
        if es_enabled and math.isfinite(train_loss_value):
            if not math.isfinite(best_train_loss):
                best_train_loss = train_loss_value
                bad_epochs = 0
            elif (best_train_loss - train_loss_value) > es_min_delta * max(abs(best_train_loss), 1.0):
                best_train_loss = train_loss_value
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= es_patience:
                    stopped_epoch = epoch
                    logger.info(
                        f"[{hostname}][PID:{pid}] Outer {out_l} early stop at epoch {epoch}: "
                        f"train loss has not improved by >{es_min_delta} (relative) for "
                        f"{es_patience} epochs (best train loss {best_train_loss:.4f})."
                    )
                    break

    # Save the model reached at the end of training. The epoch is chosen by the
    # training-loss criterion above, never by test performance.
    best_model_path = os.path.join(model_dir, f"best_model_{config_id}.pth")
    torch.save(model, best_model_path)
    # Persist the fold's train-fitted z-score constants beside the weights: without
    # them the checkpoint cannot be returned to physical units later.
    save_normalization(getattr(train_set.dataset, "normalization", None), model_dir)
    logger.info(
        f"[{hostname}][PID:{pid}] [MODEL SAVED] Outer {out_l} final model at epoch "
        f"{stopped_epoch if stopped_epoch is not None else outerepochs - 1} "
        f"(saved at {best_model_path})"
    )

    logger.info(f"[{hostname}][PID:{pid}] Outer training completed for {out_l}. Final train log: {final_train_log}")
    
    final_train_log_path = os.path.join(out_dir_outer, f"final_train_log_{out_l}_{config_id}.json")
    with open(final_train_log_path, "w") as f:
        json.dump(final_train_log, f, indent=4)

    outer_summary = []
    outer_summary.append({
        "outer_fold": out_l,
        'final_train_loss': train_loss_value, 
        'final_train metric': train_metric_value,
        'final_test loss':  test_loss_value, 
        # `best_test_loss` / `best_test_metric` / `best_epoch` are DIAGNOSTICS: the
        # best test loss seen during training and the epoch it occurred at. They no
        # longer describe the saved checkpoint, which is the epoch training stopped at
        # (`saved_epoch`) as decided by the train-loss criterion.
        'best_test_loss': best_test,
        'best_test_metric':  best_metric,
        "final_train_log": final_train_log_path,
        "best_epoch": best_epoch,
        "saved_epoch": stopped_epoch if stopped_epoch is not None else outerepochs - 1,
        "early_stopped": stopped_epoch is not None,
        # Whether the plateau schedule ever fired, and where the LR ended up. If this
        # is always 0, the schedule is doing nothing and its patience is too high.
        "lr_reductions": lr_reduction_count,
        "final_lr": model.optimizer.param_groups[0]["lr"],
        "best_model_path": best_model_path,
        "config_id": config_id,
        "config": best_config
        })
        
    summary_path = os.path.join(out_dir_outer, f"config_{config_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(outer_summary, f, indent=4)
                

    wandb.define_metric( "test_loss", summary='min', step_metric='outer')
    wandb.define_metric( "test_metric", summary='min', step_metric='outer')  
    wandb.log({
            "epoch": epoch,
            "fold": out_l,
            "train_loss": train_loss_value,
            "train_metric": train_metric_value,
            "test_loss": test_loss_value,
            "test_metric": test_metric_value,
            "best_test_loss": best_test,
            "Config_ID":config_id
    })

    wandb.run.summary["best test accuracy"] = best_test
    wandb.run.summary["state"] = f"completed {out_l}"
    wandb.finish()
    
    #Save best model _state
    logger.info(f"[{hostname}][PID:{pid}] Final_best_test_loss for {out_l}: {best_test:.4f} ")
    logger.info(f"[{hostname}][PID:{pid}] Final_best_metric_loss for {out_l}: {best_metric:.4f} ")
    
    outer_results[out_l] = {
        'train_loss': train_loss_value, 
        'train metric': train_metric_value,
        'test loss':  test_loss_value,
        'best_test_loss': best_test,  
        'test metric':  test_metric_value 
    }

    return {
        "outer_fold": out_l,
        "process_model": ctx.process_model_name,
        "validation_available": False,
        "config_id": config_id,
        "final_train_loss": train_loss_value,
        "final_train_metric": train_metric_value,
        "final_test_loss": test_loss_value,
        # See outer_summary above: these three are diagnostics, not the saved epoch.
        "best_test_loss": best_test,
        "best_test_metric": best_metric,
        "best_epoch": best_epoch,
        "saved_epoch": stopped_epoch if stopped_epoch is not None else outerepochs - 1,
        "early_stopped": stopped_epoch is not None,
        # Whether the plateau schedule ever fired, and where the LR ended up. If this
        # is always 0, the schedule is doing nothing and its patience is too high.
        "lr_reductions": lr_reduction_count,
        "final_lr": model.optimizer.param_groups[0]["lr"],
        "final_train_log": final_train_log_path,
        "best_model_path": best_model_path,
        "config": best_config,
        "outer_result": outer_results[out_l],
    }

def _canonical_fold_key(outer_fold_key: str) -> str | None:
    """
    Return the canonical (HPO) fold key for an _otrainN draw, or None if this
    is already a canonical fold.  E.g. 'outer_fold_s_otrain2' → 'outer_fold_s'.
    """
    import re
    m = re.match(r"^(.+)_otrain\d+$", outer_fold_key)
    return m.group(1) if m else None


def _load_top_params_from_summary(summary_path: str) -> tuple[list, list]:
    """
    Load (top_params, top_metrics) from a canonical outer-fold summary JSON.
    Each entry in the summary has 'config' (the merged param dict) and 'metric'.
    """
    with open(summary_path, "r") as f:
        entries = json.load(f)
    top_params = [e["config"] for e in entries]
    top_metrics = [e["metric"] for e in entries]
    return top_params, top_metrics


def run_outer_fold_CV(
    outer_fold_key: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    out_dir: str,
    ctx: RuntimeContext,
    n_trials_inner: int = 8,
    outerepochs: int = 50,
    top_trials: int = 5,
):
    """
    Runs a single outer fold in nested CV as an independent job.

    For _otrainN draw folds (e.g. outer_fold_s_otrain2), HPO is skipped and the
    best configs are loaded from the canonical fold's summary JSON.  The canonical
    fold (e.g. outer_fold_s) must have been completed first.
    """
    if outer_fold_key == "all_stations":
        return None

    elif outer_fold_key == "site_metadata":
        return None

    print(f"[INFO] Starting {outer_fold_key}")

    summary_path = os.path.join(out_dir, f"outer_summary_{outer_fold_key}.json")
    if os.path.exists(summary_path):
        print(f"[INFO] Outer fold {outer_fold_key} already evaluated. Skipping.")
        with open(summary_path, "r") as f:
            return {outer_fold_key: json.load(f)}

    # --- _otrainN draw: reuse canonical HPO result, skip inner loop entirely ---
    canonical_key = _canonical_fold_key(outer_fold_key)
    if canonical_key is not None:
        canonical_summary_path = os.path.join(out_dir, f"outer_summary_{canonical_key}.json")
        if not os.path.exists(canonical_summary_path):
            raise FileNotFoundError(
                f"Canonical fold summary not found: {canonical_summary_path}. "
                f"Run '{canonical_key}' before '{outer_fold_key}'."
            )
        print(
            f"[INFO] {outer_fold_key} is an outer-train draw — loading HPO params "
            f"from canonical fold '{canonical_key}'."
        )
        top_params, top_metrics = _load_top_params_from_summary(canonical_summary_path)
        outer_fold_results = []
        for i, config in enumerate(top_params):
            print(f"[INFO] Outer {outer_fold_key} | Top {i+1}: metric = {top_metrics[i]:.5f}")
            result = run_outer_training_fold(
                outer_fold_key,
                config,
                model_config,
                data_config,
                config_id=i + 1,
                outerepochs=outerepochs,
                ctx=ctx,
            )
            outer_fold_results.append({
                # Use the config returned by run_outer_training_fold so that
                # data::train::station_id reflects this draw's sites, not the canonical's.
                "config": result["config"],
                "metric": top_metrics[i],
                "outer_result": result["outer_result"],
                "best_model_path": result["best_model_path"],
                "best_test_loss": result["best_test_loss"],
                "best_epoch": result["best_epoch"],
                "config_id": result["config_id"],
            })
        with open(summary_path, "w") as f:
            json.dump(outer_fold_results, f, indent=4)
        return {outer_fold_key: outer_fold_results}

    # --- canonical fold: run HPO as normal ---
    is_leader_worker = (ctx.worker_id == 0)
    if ctx.workers_per_fold > 1 and not is_leader_worker:
        # Worker-only mode: contribute HPO trials and exit without fold finalization.
        run_inner_loops_HPO(
            outer_fold_key,
            model_config,
            data_config,
            ctx=ctx,
            n_trials_inner=n_trials_inner,
            top_trials=top_trials,
            wait_for_all_workers=False,
        )
        print(
            f"[INFO] Fold {outer_fold_key}: worker {ctx.worker_id} finished its HPO contribution "
            f"({n_trials_inner} trials). Leader worker 0 will run outer training."
        )
        return {
            outer_fold_key: {
                "status": "hpo_worker_complete",
                "worker_id": ctx.worker_id,
                "trials_run": n_trials_inner,
            }
        }

    # Run inner loop HPO
    top_params, top_metrics, trial_df = run_inner_loops_HPO(
        outer_fold_key,
        model_config,
        data_config,
        ctx=ctx,
        n_trials_inner=n_trials_inner,
        top_trials=top_trials,
        wait_for_all_workers=(ctx.workers_per_fold > 1 and is_leader_worker),
    )

    if len(top_params) == 0:
        raise RuntimeError(
            f"No completed Optuna trials with 'val_best_loss_mean' found for fold '{outer_fold_key}'."
        )
    if len(top_params) < top_trials:
        print(
            f"[WARNING] Fold {outer_fold_key}: finalizing only {len(top_params)} outer configs "
            f"although top_trials={top_trials}."
        )

    outer_fold_results = []
    for i, config in enumerate(top_params):
        print(f"[INFO] Outer {outer_fold_key} | Top {i+1}: metric = {top_metrics[i]:.5f}")
        result = run_outer_training_fold(
            outer_fold_key,
            config,
            model_config,
            data_config,
            config_id=i+1,
            outerepochs=outerepochs,
            ctx=ctx,
        )
        outer_fold_results.append({
            "config": config,
            "metric": top_metrics[i],
            "outer_result": result["outer_result"],
            "best_model_path": result["best_model_path"],
            "best_test_loss": result["best_test_loss"],
            "best_epoch": result["best_epoch"],
            "config_id": result["config_id"],
        })

    # Save summary
    with open(summary_path, "w") as f:
        json.dump(outer_fold_results, f, indent=4)

    return {outer_fold_key: outer_fold_results}






if __name__ == "__main__":
    
    load_dotenv()

    parser = _build_arg_parser()
    args = parser.parse_args()

    ide_settings = {
        "enabled": True, # Use for when running locally
        "outer_fold": "outer_station_5",
        "inner_fold": "inner_loop_1",
        "worker_id": 0,
        "workers_per_fold": 1,
        "n_trials": 2,
        "top_trials": 5,
        "config": None,
        "cv_config_path": 'configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json',
        "process_model": "preles",
        # Artifact namespaces for local testing (leave as None to use cluster defaults)
        "experiment_artifact_namespace": None,      # e.g., "local_test_v1" or "debug"
        "calibration_artifact_namespace":None,      #"misspec_pure_tpe_base",  # e.g., "local_test_v1" or "debug"
    }

    args = _apply_ide_fallback(args, ide_settings)

    # Set artifact namespace environment variables if provided in ide_settings
    if ide_settings.get("enabled", False):
        if ide_settings.get("experiment_artifact_namespace"):
            os.environ["EXPERIMENT_ARTIFACT_NAMESPACE"] = ide_settings["experiment_artifact_namespace"]
        if ide_settings.get("calibration_artifact_namespace"):
            os.environ["CALIBRATION_ARTIFACT_NAMESPACE"] = ide_settings["calibration_artifact_namespace"]

    base_config = _load_base_config(args)
    model_config, process_model_name = _resolve_model_config(base_config, args.process_model)
    # Guardrail: fail before trial orchestration if legacy embedding token is provided.
    from Config_Files.config_loader import validate_embedding_mode_tokens
    validate_embedding_mode_tokens(model_config)
    ctx = _resolve_runtime_context(args, model_config=model_config, process_model_name=process_model_name)

    _run_single_outer_fold(ctx)
