#%%%%%%%%%%% uncomment if not in cluster debug modus
# import debugpy

# # Listen on all interfaces, port 5678
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for VS Code debugger to attach...")
# debugpy.wait_for_client()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import torch 
import numpy as np
import copy
import socket
import os
import time
import math
import random
import json
import logging
import argparse
import re
import sys
from collections import defaultdict

from dotenv import load_dotenv
from filelock import FileLock

# Optuna
import optuna
from optuna.pruners import HyperbandPruner #hyperband for Pruning models
from optuna.samplers import TPESampler #baysian Optimisation sampling
from optuna.trial import TrialState
#import wandb

#own
from NN_framework import NN_module
from utilis_shorts import json_open, save_frozen_config, save_normalization #, conv_torch
from process_model_calibration_manager import ProcessModelCalibrationManager
from utilis import optimize_within_fold_budget
from calibration_paths import build_spatial_logs_root, cv_label_from_path
# v2: fixed gate-column -> expert routing, schedule end points, post-warmup entropy term.
# The 5th value of train_epoch/test_epoch is the mean routing weight per expert.
from noisy_mixture_v2 import MixtureModel


# Searched warmup, as a fraction of the training budget. Shared by the search space and
# the pruner, whose resource axis starts when warmup ends.
WARMUP_FRAC_RANGE = (0.3, 0.6)

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

def save_checkpoint(mixture_of_experts, optimizer, best_epoch, model_config, checkpoint_dir, epoch, inner_loop, logger,
                    normalization=None, reason="new best val loss"):
    """
    Save model checkpoint and associated metadata.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "inner_loop": inner_loop,
        "mixture_of_experts": mixture_of_experts.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config,
    }

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)
    # The train-fitted z-score constants are part of the model: predictions cannot be
    # returned to physical units without them.
    save_normalization(normalization, checkpoint_dir)

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
def run_inner_loop(inner_loop, inner_conf, out_l, general_config, data_config, pm_manager, trial, checkpoint_path, model):
    """
    Run one inner-loop training fold and return its results.
    """
    # Setup log directories and logger
    out_dir_full = os.path.join(out_dir_logs, out_l)

    # Extract station IDs from inner_conf
    train_stations = {data_config['station']:inner_conf['train']}
    test_stations  = {data_config['station']:inner_conf['test']}
    val_stations   = {data_config['station']:inner_conf['val']}

    if general_config["model::modeltype"] != "MLP":
        raise Exception('model_type should be MLP')

    # Same loader path as Optuna_temporal_NCV.py. It replaces CrossValidationCalibrationManager
    # + load_CV_datasets, which (a) z-scored test/val with their OWN statistics — the
    # normalization leak — (b) never shuffled the train batches, and (c) could not
    # require an existing calibration.
    train_set, test_set, val_set = pm_manager.load_fold_dataloaders(
        train_stations=train_stations,
        test_stations=test_stations,
        val_stations=val_stations,
        # Calibrated process model only, loaded from the existing bundle; fail rather than
        # calibrate on the fly or fall back to default parameters.
        calibrate="require_existing",
        shuffle_train=True,
        sparse=general_config.get("data:train:sparse"),
    )

    # Size the schedule to this inner loop's real step count. The model is built before
    # the loader exists, from a years-based budget; these are plain attributes read by
    # MixtureModel.train_epoch each step, and global_step is still 0 here.
    inner_budget = len(train_set) * general_config['training::epochs']
    model.total_budget = inner_budget
    model.warmup_steps = int(general_config["schedule::warmup_frac"] * inner_budget)
    model.ramp_steps = int(general_config["schedule::ramp_frac"] * inner_budget)
    model.noise_steps = int(general_config["schedule::noise_frac"] * inner_budget)

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
    # First epoch that ended with the process model in the mixture; origin of the pruner's steps.
    pm_start_epoch = None
    divergence_abort_enabled = bool(general_config.get("training::divergence_abort::enabled", True))
    divergence_patience = int(general_config.get("training::divergence_abort::patience", 1))

    checkpoint_window = general_config.get("checkpoint::window_size")
    checkpoints_dir_inner = os.path.join(out_dir_full, "checkpoints")
    os.makedirs(checkpoints_dir_inner, exist_ok=True)

    logger = setup_logging(checkpoints_dir_inner, inner_loop)

    # Only the first inner loop reports to the pruner. The temporal CV schema nests the
    # loops under "inner_loops", next to fold metadata such as "hold_out_years" — so the
    # fold's first key is NOT an inner loop, and pruning never fired.
    first_inner_loop = next(iter(data_config[out_l]["inner_loops"]))

    # Loop over training epochs
    for epoch in range(general_config['training::epochs']):
        train_loss, train_metric, _, _, _  = model.train_epoch(train_set)
        test_loss, test_metric, test_us, test_entropy_av, test_share = model.test_epoch(test_set)
        val_loss, val_metric, val_us, val_entropy_av, val_share = model.test_epoch(val_set)

        # Convert tensor losses/metrics to scalars
        train_loss_scalar = train_loss
        train_metric_scalar = train_metric
        
        test_loss_scalar = test_loss
        test_metric_scalar =test_metric
        test_usage_avag=test_us
        test_entropy_avag= test_entropy_av
        test_routing_share = test_share

        
        val_loss_scalar = val_loss
        val_metric_scalar = val_metric
        val_usage_avag=val_us
        val_entropy_avag= val_entropy_av
        val_routing_share = val_share

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss_scalar,
            "train_metric": train_metric_scalar,
            "test_loss": test_loss_scalar,
            "test_metric": test_metric_scalar,
            "test_usage": test_usage_avag.tolist(),
            "test_entropy_avag" : test_entropy_avag,
            "test_routing_share": test_routing_share.tolist(),
            "val_loss": val_loss_scalar,
            "val_metric": val_metric_scalar,
            "val_usage_avag": val_usage_avag.tolist(),
            "val_entropy_avag" : val_entropy_avag,
            "val_routing_share": val_routing_share.tolist(),
        }
        print("train_loss:", train_loss_scalar,
              "train_metric:", train_metric_scalar,
              "test_loss:", test_loss_scalar,
              "test_metric:", test_metric_scalar,
              "test_usage:", test_usage_avag)
        
        inner_epoch_records.append(epoch_log)

        # Divergence abort. A non-finite TRAIN loss means the weights themselves are
        # non-finite and no later epoch can undo that. Inner loops 2+ never report to
        # the pruner, so without this a diverged trial would run its full budget and
        # only then FAIL. Pruned (not failed): TPE still learns from pruned trials.
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

        if train_loss_scalar < best_train_loss:
            best_train_loss = train_loss_scalar
        if math.isfinite(train_metric_scalar) and train_metric_scalar < best_train_metric:
            best_train_metric = train_metric_scalar
        # Minimum test loss/metric over epochs — DIAGNOSTICS only, see `test_loss_at_best_epoch`.
        if math.isfinite(test_loss_scalar) and test_loss_scalar < best_test_loss:
            best_test_loss = test_loss_scalar
        if math.isfinite(test_metric_scalar) and test_metric_scalar < best_test_metric:
            best_test_metric = test_metric_scalar

        # Only epochs after warmup are selectable. During warmup the hard expert is
        # masked, so the "mixture" is a plain NN; selecting (and ranking trials) on
        # those epochs would reward configurations that never use the process model. Gated on the
        # warmup end rather than the ramp end: warmup+ramp can exceed the budget, so
        # the ramp may never complete within training.
        pm_in_mixture = model.global_step >= model.warmup_steps
        if pm_in_mixture and pm_start_epoch is None:
            pm_start_epoch = epoch
        val_improved = False
        if pm_in_mixture and math.isfinite(val_loss_scalar) and val_loss_scalar < best_val_loss:
            best_val_loss = val_loss_scalar
            # The kept checkpoint is selected on val, never on the inner-test split.
            best_epoch = epoch
            val_improved = True
            test_loss_at_best_epoch = test_loss_scalar
            test_metric_at_best_epoch = test_metric_scalar
        if pm_in_mixture and val_metric_scalar < best_val_metric:
            best_val_metric = val_metric_scalar

        is_last_epoch = epoch == general_config['training::epochs'] - 1

        # Log progress periodically
        if epoch % checkpoint_window == 0 or is_last_epoch:
            epoch_records.append(epoch_log)
            logger.info(f"Epoch {epoch}: {epoch_log}")

        # Checkpointing is NOT gated on the logging window: a best epoch can fall
        # between window epochs. Each improvement overwrites one fixed `best/`
        # directory, so only the selected epoch is kept on disk.
        best_dir = os.path.join(checkpoints_dir_inner, inner_loop, f"{trial._trial_id}", "best")
        if val_improved:
            best_checkpoint_path = save_checkpoint(
                model, model.optimizer, best_epoch, general_config,
                best_dir, epoch, inner_loop, logger,
                normalization=getattr(train_set.dataset, "normalization", None),
                reason=f"New best val loss {best_val_loss:.5f}",
            )
        elif is_last_epoch and best_checkpoint_path is None:
            # No epoch ever produced a finite val loss — keep the final weights so the
            # trial still has an artifact. Record `epoch` (where these weights come
            # from), not `best_epoch`, which never moved off 0.
            best_checkpoint_path = save_checkpoint(
                model, model.optimizer, epoch, general_config,
                best_dir, epoch, inner_loop, logger,
                normalization=getattr(train_set.dataset, "normalization", None),
                reason="Final epoch (no finite val loss was ever observed)",
            )
            logger.warning(
                f"[{inner_loop}] No finite val loss in any epoch; saved final-epoch weights instead."
            )

        if epoch % checkpoint_window == 0 or is_last_epoch:
            trial.set_user_attr("checkpoint_path", best_checkpoint_path)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("val_loss", val_loss_scalar)
            trial.set_user_attr("val_best_loss", best_val_loss)

        # Report only post-warmup epochs, for the same reason as the selection gate above:
        # otherwise Hyperband's early rungs judge the plain NN before the process model exists.
        if inner_loop == first_inner_loop and pm_in_mixture:
            # Prune on the same quantity the objective optimises (val). The step counts
            # epochs since the process model was enabled, not absolute epochs: warmup is searched, so
            # absolute first reports land anywhere in 0.3-0.6 of the budget and Hyperband
            # would fill every rung already passed with that one value, comparing trials
            # at different training stages.
            trial.report(val_loss_scalar, epoch - pm_start_epoch)
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
        # Test performance at the val-selected epoch: the fold's reportable test number.
        "test_loss_at_best_epoch": test_loss_at_best_epoch,
        "test_metric_at_best_epoch": test_metric_at_best_epoch,
        "best_val_metric": best_val_metric,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
    }

def define_search_space(trial, model_config, prefix="",
                        random_activation = True,
                        dropout_tuning=True):
    """
    Configures the hyperparameter search space dynamically for the modesl.
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

    # Optuna names and user attrs carry `prefix` so each expert gets its own dimensions
    # (merge_best_params splits them back out for the outer run). The config keys are
    # written UNprefixed: NN_module reads only those, so prefixed keys left every
    # inner-loop expert on the untuned base architecture.

    # Randomize model length (number of layers & hidden sizes)

    layer_number = trial.suggest_int(f"{prefix}model::layer_number", 2, 3)
    hidden_sizes = [trial.suggest_int(f"{prefix}hidden_size_{i}", 6, 64) for i in range(layer_number-1)]

    updated_config["model::layer_number"] = layer_number
    trial.set_user_attr(f"{prefix}model::layer_number", layer_number)

    ### Suggest hidden sizes per possible layer
    updated_config["model::hidden_size"] = hidden_sizes
    trial.set_user_attr(f"{prefix}model::hidden_size", hidden_sizes)


    ### Fixed bias list: one per layer
    bias_list = [True] * layer_number
    updated_config["model::bias_list"] = bias_list
    trial.set_user_attr(f"{prefix}model::bias_list", bias_list)

    # Activation choice
    if random_activation:
        activation = trial.suggest_categorical(f"{prefix}model::activation", ["Relu", "Tanh", "Sigmoid"])
        activation_list = [activation] * (layer_number - 1) + [None]
        updated_config["model::activation"] = activation_list
        trial.set_user_attr(f"{prefix}model::activation", activation_list)
    else:
        # If activation is fixed
        activation_list = updated_config["model::activation"]
        if isinstance(activation_list, list):
            # Adjust the list size to match layer_number
            activation_list = activation_list[:layer_number - 1] + [None]  # Ensure length matches layer_number
        updated_config["model::activation"] = activation_list
        trial.set_user_attr(f"{prefix}model::activation", activation_list)

    # Tune dropout layers if enabled
    if dropout_tuning:
        updated_config["model::dropout_list"] = trial.suggest_float(f"{prefix}model::dropout_list", 0.1, 0.4)


    return updated_config

def define_mixture_search_space(trial, general_config, total_train_size):
    """Define hyperparameters for the mixture layer itself."""
    
    updated_general_config= copy.deepcopy(general_config)

    # Seeds: FIXED, deliberately not searched. MixtureModel seeds from `training::seed`
    # and NN_module from `model::seed`; searching them lets TPE optimise the random
    # initialisation itself, so the top-k trials risk being lucky inits rather than
    # good configurations. Same policy as Optuna_temporal_NCV.py.
    updated_general_config["model::seed"] = updated_general_config.get("model::seed", updated_general_config["training::seed"])
    trial.set_user_attr("model::seed", updated_general_config["model::seed"])
    trial.set_user_attr("training::seed", updated_general_config["training::seed"])

    updated_general_config[f"training::learning_rate"] = trial.suggest_float("training::learning_rate", 1e-6, 1e-3, log=True)

    # `training::momentum` is NOT searched: AdamW has no momentum term (beta_1 plays
    # that role), so the value was read by nothing. Pinned so the key stays defined.
    updated_general_config["training::momentum"] = updated_general_config.get("training::momentum", 0.0)

    updated_general_config["training::optimizer"] ="AdamW"

    # beta_1/beta_2 FIXED at the torch defaults rather than searched (beta_2 was
    # searched over 0.85-0.95, far from its usual 0.999).
    updated_general_config["training::beta_1"] = updated_general_config.get("training::beta_1", 0.9)
    updated_general_config["training::beta_2"] = updated_general_config.get("training::beta_2", 0.999)

    # Weight decay is tuned; the 1e-8 floor stands in for "effectively off" (a
    # log-uniform range cannot include exact zero).
    weight_decay = trial.suggest_float("training::weight_decay", 1e-8, 1e-2, log=True)
    updated_general_config["training::weight_decay"] = weight_decay
    trial.set_user_attr("training::weight_decay", weight_decay)

    updated_general_config["num_trainable_experts"]=trial.suggest_int("num_experts", 1, 3)
    updated_general_config["init_noise_scale"]=1.0

    # The schedule params are searched as FRACTIONS of the training budget; the model
    # needs STEP counts. Here they are scaled to the placeholder budget used at model
    # construction; each inner loop and the outer run rescale them to their real step
    # counts from `schedule::*_frac` (the outer run used to read the raw fraction back
    # as a step count, warmup ≈ 0.5 steps).
    warmup_step= trial.suggest_float("warmup_steps", *WARMUP_FRAC_RANGE)
    updated_general_config["warmup_steps"]= int(warmup_step*total_train_size)

    ramp_step= trial.suggest_float("ramp_steps",warmup_step, 0.7)
    updated_general_config["ramp_steps"]= int(ramp_step*total_train_size)

    noise_step= trial.suggest_float("noise_steps", ramp_step, 0.9)
    updated_general_config["noise_steps"]= int(noise_step*total_train_size)

    for key, frac in (("warmup", warmup_step), ("ramp", ramp_step), ("noise", noise_step)):
        updated_general_config[f"schedule::{key}_frac"] = frac
        trial.set_user_attr(f"schedule::{key}_frac", frac)

    updated_general_config["alpha_entropy"]= trial.suggest_float("alpha_entropy", 0.005, 0.02)
    
    return updated_general_config

def objective(trial, out_l, general_config, data_config,  wanddir): 
    
    '''objective for optuna Hyperparameter study
        trial: optuna trial
        out_l: outer station key
        data_config: Config of nested loops
        model_config: config of the model
    '''  

    # Basic environment info for logging
    hostname = socket.gethostname()
    pid = os.getpid()
    print(f"[{hostname}][PID:{pid}] Starting objective for outer fold {out_l}, trial {trial.number}")

    # device
    if general_config['training::cuda'] == True:
            if  torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
 
    # Create Search Space for hyperparameters
    #mixture of experts config
    general_config=define_mixture_search_space(trial, general_config,general_config['train_budget'])
    # set optuna seed
    torch.manual_seed(general_config['model::seed'])
    
    #model_configs
    model_configs = list()
    for nr_ex in range(general_config['num_trainable_experts']):
        expert_config = define_search_space(trial, general_config, prefix=f"expert{nr_ex}_")
        # Own init seed per expert: NN_module reseeds from `model::seed` when built, so
        # experts sharing one seed and one architecture started from identical weights.
        expert_config["model::seed"] = general_config["model::seed"] + nr_ex + 1
        model_configs.append(expert_config)

    #add model_configs to general config
    general_config["trainable_models"] = model_configs
    
    # Initialize the model with the updated configuration
    model= MixtureModel(config=general_config,
                        model_configs=model_configs,
                        total_train_budget= general_config['train_budget'],
                        hard_expert_id=general_config['hard_expert_id'], warmup_steps=general_config["warmup_steps"], 
                        ramp_steps=general_config["ramp_steps"], 
                        noise_steps=general_config["noise_steps"], 
                        alpha_entropy=general_config["alpha_entropy"]).to(device)
  

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
    # all_epoch_records = {}

    # Create fold manager instance
    pm_manager = ProcessModelCalibrationManager(
        config=general_config,
        process_model=general_config.get("process_model::name"),
    )
  

    # Checkpoint directory for logging.
    out_dir_full = os.path.join(out_dir_logs, out_l)
    
    checkpoints_dir_inner = os.path.join(out_dir_full, "checkpoints")
    os.makedirs(checkpoints_dir_inner, exist_ok=True)
    print(f"[INFO] Checkpoint inner path: {checkpoints_dir_inner}")

    logger = setup_logging(checkpoints_dir_inner)
    logger.info(f"[{hostname}][PID:{pid}] Starting evaluation for fold {out_l} in trial {trial.number}")
    global_step_loop = 0
    
    # ['inner_loops']
    for i, (inner_loop, inner_conf) in enumerate(data_config[out_l]['inner_loops'].items()):
        # Run innerloop with configuration
        logger.info(f"[{hostname}][PID:{pid}] Processing {inner_loop}, split {inner_conf} for fold {out_l}")

        inner_results = run_inner_loop(inner_loop, inner_conf, out_l, general_config, data_config, pm_manager, trial, checkpoint_path, model)
        

        best_train_loss_avg.append(inner_results["best_train_loss"])
        train_metric_avg.append(inner_results["best_train_metric"])
        test_loss__best_avag.append(inner_results["best_test_loss"])
        test_metric_best_avag.append(inner_results["best_test_metric"])
        test_loss_at_best_epoch_avg.append(inner_results["test_loss_at_best_epoch"])
        test_metric_at_best_epoch_avg.append(inner_results["test_metric_at_best_epoch"])
        val_metric_best_avag.append(inner_results["best_val_metric"])
        val_loss_avag_best.append(inner_results["best_val_loss"]) 
        # all_epoch_records[inner_loop] = inner_results["inner_epoch_records"] # do we need this infomation ??

        if i >= 1:
            global_step_loop += 1

        #initilise model again
        torch.manual_seed(general_config['model::seed'])
        model= MixtureModel(config=general_config,
                        model_configs=model_configs, 
                        total_train_budget=general_config['train_budget'],
                        hard_expert_id=general_config['hard_expert_id'], warmup_steps=general_config["warmup_steps"],
                        ramp_steps=general_config["ramp_steps"], 
                        noise_steps=general_config["noise_steps"], 
                        alpha_entropy=general_config["alpha_entropy"]).to(device)
        logger.info(f"Model re-initialized for next inner loop")
    
    # The value handed to Optuna. It is the SAME quantity the top-k trials are later
    # ranked on (`val_best_loss_mean` in run_inner_loops_HPO), so TPE and the pruner
    # steer towards what actually decides selection.
    aggregated_val_loss = np.mean(val_loss_avag_best)

    # # After processing all inner loops, compute aggregated metrics
    logger.info(f"[{hostname}][PID:{pid}] Finished objective for outer fold {out_l}, trial {trial.number} with aggregated val loss: {aggregated_val_loss:.5f}")
    trial.set_user_attr("val_best_loss_mean", aggregated_val_loss) #this is the value for choosing the models for the outer fold
    trial.set_user_attr("test_best_loss_mean",np.mean(test_loss__best_avag))
    trial.set_user_attr("val_best_metric_mean",np.mean(val_metric_best_avag))
    trial.set_user_attr("test_best_metric_mean",np.mean(test_metric_best_avag))
    trial.set_user_attr("train_best_loss_mean",np.mean(best_train_loss_avg))
    trial.set_user_attr("train_best_metric_mean", np.mean(train_metric_avg))
    # Test performance at the val-selected epoch, averaged over inner loops. Report THIS
    # rather than `test_best_loss_mean` (mean of per-loop minima, biased low).
    # Stored as None when non-finite so it round-trips cleanly through Optuna storage.
    def _mean_or_none(values):
        mean = float(np.mean(values)) if len(values) else float("nan")
        return None if not math.isfinite(mean) else mean

    trial.set_user_attr("test_loss_at_best_epoch_mean", _mean_or_none(test_loss_at_best_epoch_avg))
    trial.set_user_attr("test_metric_at_best_epoch_mean", _mean_or_none(test_metric_at_best_epoch_avg))

    return aggregated_val_loss

def _other_workers_exited() -> bool:
    """
    True once every other worker of this fold has exited, successfully or not.

    The batch script writes <WORKER_STATUS_DIR>/worker<N>.exited when worker N's process
    ends (the shell does it, so crashes and kills count too). Without WORKER_STATUS_DIR
    this always returns False and the leader waits for the trial count or the timeout.
    """
    status_dir = os.getenv("WORKER_STATUS_DIR")
    if not status_dir:
        return False
    return all(
        os.path.exists(os.path.join(status_dir, f"worker{w}.exited"))
        for w in range(workers_per_fold)
        if w != worker_id
    )

def _wait_for_worker_trials(study, study_name, storage_instance, out_l, n_trials_inner, top_trials):
    """
    Leader barrier: poll the shared study until the whole fold budget (workers_per_fold x
    n_trials_inner) has finished, then return the refreshed study. Ported from
    Optuna_spatial_NCV.py; the same OPTUNA_SYNC_* env vars tune it.
    """
    expected_finished_trials = workers_per_fold * n_trials_inner
    timeout_sec = int(os.getenv("OPTUNA_SYNC_TIMEOUT_SEC", "7200"))
    poll_sec = int(os.getenv("OPTUNA_SYNC_POLL_SEC", "20"))
    shortfall_grace_sec = int(os.getenv("OPTUNA_SYNC_SHORTFALL_GRACE_SEC", "600"))
    # FAIL does not count: a resumed fold carries the FAILed trials its killed run left
    # behind, and the workers re-run those slots (optimize_within_fold_budget). A crashed
    # worker's shortfall is released by _other_workers_exited or the timeout.
    finished_states = {TrialState.COMPLETE, TrialState.PRUNED}
    wait_started = time.time()
    shortfall_started = None
    while True:
        refreshed_study = optuna.load_study(study_name=study_name, storage=storage_instance)
        finished_count = sum(1 for t in refreshed_study.trials if t.state in finished_states)
        # A fold finalized from too few selectable trials silently shrinks the
        # experiment, so once the count is met
        # give stragglers a bounded grace period to supply top_trials selectable ones.
        selectable_count = sum(
            1 for t in refreshed_study.trials
            if t.state == TrialState.COMPLETE and "val_best_loss_mean" in t.user_attrs
        )
        if finished_count >= expected_finished_trials:
            if selectable_count >= top_trials:
                print(
                    f"[INFO] Fold {out_l}: collected {finished_count}/{expected_finished_trials} "
                    f"finished trials ({selectable_count} selectable). Starting outer-fold finalization."
                )
                return refreshed_study
            if shortfall_started is None:
                shortfall_started = time.time()
                print(
                    f"[WARNING] Fold {out_l}: all {finished_count}/{expected_finished_trials} trials "
                    f"finished but only {selectable_count} are selectable (need {top_trials}). "
                    f"Waiting up to {shortfall_grace_sec}s for stragglers."
                )
            elif time.time() - shortfall_started >= shortfall_grace_sec:
                print(
                    f"[WARNING] Fold {out_l}: finalizing with only {selectable_count} selectable "
                    f"trials (requested {top_trials}) after {shortfall_grace_sec}s grace."
                )
                return refreshed_study

        # A crashed worker never delivers its remaining trials. Once every other worker has
        # exited nothing more can arrive: re-read the study (after the check, so their last
        # trials are included) and proceed.
        if _other_workers_exited():
            refreshed_study = optuna.load_study(study_name=study_name, storage=storage_instance)
            finished_count = sum(1 for t in refreshed_study.trials if t.state in finished_states)
            print(
                f"[WARNING] Fold {out_l}: all other workers have exited "
                f"({finished_count}/{expected_finished_trials} trials finished). Proceeding."
            )
            return refreshed_study

        elapsed = int(time.time() - wait_started)
        if elapsed >= timeout_sec:
            print(
                f"[WARNING] Fold {out_l}: timeout while waiting for workers "
                f"({finished_count}/{expected_finished_trials} finished after {elapsed}s). "
                "Proceeding with currently available trials."
            )
            return refreshed_study

        print(
            f"[INFO] Fold {out_l}: waiting for workers "
            f"({finished_count}/{expected_finished_trials} finished, elapsed={elapsed}s)."
        )
        time.sleep(poll_sec)

def run_inner_loops_HPO(out_l, general_config, data_config, n_trials_inner=8, top_trials=5,
                        wait_for_all_workers=False):
    """
    Run HPO for inner-loops configuration using Optuna.
    Returns a tuple: (best_params, best_metric, trial_records)

    All workers of a fold share one study. With wait_for_all_workers (the leader), return
    only once the other workers' trials have finished too, so the top-k is taken over the
    whole fold budget.
    """

    # prediction_loss = general_config["training::loss_physic_type"] #naive, bias etc.
    prediction_type = general_config["model::prediction_type"] #spatial -effectly default.
    data_sparcity = general_config.get("data:train:sparse", None)

    sparsity_suffix = f"_sparse{data_sparcity}" if data_sparcity else "_full"


    #todo naming
    # `_v2`: objective switched from inner-test to val, and the search space changed.
    # Studies resume by name, so the old name would mix incomparable trials in one ranking.
    # Process model and station are in the name: temporal folds are `out_l_N` for every
    # station, so without them a BASFOR run, or another station, resumed this study.
    process_model_name = general_config["process_model::name"]
    station_id = data_config["station"]
    # Namespace in the key too, so a namespaced run (e.g. a smoke test) never resumes the real study.
    ns_suffix = f"_{artifact_namespace}" if artifact_namespace else ""
    study_key = f"{process_model_name}_station_{station_id}_Mixture_{out_l}{sparsity_suffix}_{training_years_len}_v2{ns_suffix}"
    STUDY_NAME = f"study_{prediction_type}_{study_key}"
    # Same DB directory as Optuna_temporal_NCV.py (OPTUNA_DB_DIR in .env, on /work).
    optuna_db_dir = os.getenv("OPTUNA_DB_DIR", os.path.join(os.getcwd(), "Optuna_DBs"))
    os.makedirs(optuna_db_dir, exist_ok=True)
    db_path = os.path.join(optuna_db_dir, f"Optuna_{prediction_type}_{study_key}.db")
    rdb_string_url = f"sqlite:///{db_path}"

    time.sleep(random.uniform(1.0, 5.0))  # Delay each worker slightly

    # All workers of the fold reach here for the same DB at roughly the same time, and
    # RDBStorage's first-time schema creation is not safe under concurrency: serialize it
    # per DB file. timeout=600 so a stale lock on NFS raises instead of hanging the task.
    with FileLock(db_path + ".lock", timeout=600):
        storage_instance = optuna.storages.RDBStorage(
            url=rdb_string_url,
            heartbeat_interval=60,
            grace_period=180,
            # The DB is on NFS and shared by all workers; sqlite's default 5 s busy
            # timeout gives "database is locked". 300 s as in Optuna_temporal_NCV.py.
            engine_kwargs={"connect_args": {"timeout": 300, "check_same_thread": False}},
        )

    study = optuna.create_study(
            study_name= STUDY_NAME,         #study_name,
            storage=storage_instance,       #study_storage,
            direction="minimize",
            # max_resource must be explicit: left at "auto" it is inferred from whichever
            # trial completes first, so with several workers sharing one RDB the bracket
            # structure would be decided by a race.
            # Resources are epochs since the process model was enabled (see run_inner_loop), so the
            # most any trial can reach is the budget left after the shortest warmup.
            pruner=HyperbandPruner(
                min_resource=5,  # post-warmup epochs run before a trial can be halved out
                max_resource=int(general_config['training::epochs'] * (1 - WARMUP_FRAC_RANGE[0])),
            ),
            # constant_liar: workers sample concurrently against one RDB; without it
            # they cannot see each other's in-flight trials and duplicate the same region.
            # multivariate + group: model parameters jointly, per group of parameters that
            # always co-occur. The space is conditional (expert1_*/expert2_* exist only for
            # num_experts >= 2/3, hidden_size_1 only for 3 layers), and plain multivariate
            # models jointly only the parameters present in EVERY trial — 9 of 22 here, with
            # all extra experts and ramp/noise falling back to one-at-a-time TPE. group=True
            # gives one joint model per group (core + expert0, expert1, expert2, ...).
            sampler=TPESampler(constant_liar=True, multivariate=True, group=True),
            load_if_exists=True
            )
    

    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    # The inner-loop objective function can be defined separately.
    # A transient storage error must fail one trial, not the worker: killing the leader
    # leaves the fold with no outer training and no summary JSON.
    # The budget is per fold, not per worker, so a resumed fold only tops up.
    optimize_within_fold_budget(
        study,
        lambda trial: objective(trial, out_l, general_config, data_config, wanddir= "test"),
        n_trials_per_worker=n_trials_inner,
        fold_budget=workers_per_fold * n_trials_inner,
        catch=(optuna.exceptions.StorageInternalError,),
    )

    if wait_for_all_workers and workers_per_fold > 1:
        study = _wait_for_worker_trials(study, STUDY_NAME, storage_instance, out_l,
                                        n_trials_inner, top_trials)

    # Get a DataFrame of all trials
    trial_df = study.trials_dataframe()
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == TrialState.FAIL]

    # Sort completed scored trials by lowest average validation loss across inner loops.
    sorted_trials = sorted(
        [t for t in completed_trials if "val_best_loss_mean" in t.user_attrs],
        key=lambda t: t.user_attrs["val_best_loss_mean"],
    )

    # Select the top parameter sets and their corresponding objective values
    top_trials_list = sorted_trials[:top_trials]
    top_params = [{**trial.params, **trial.user_attrs} for trial in top_trials_list]
    top_metrics = [trial.value for trial in top_trials_list]

    print(
        f"[INFO] Fold {out_l}: trials total={len(study.trials)}, "
        f"complete={len(completed_trials)}, pruned={len(pruned_trials)}, failed={len(failed_trials)}, "
        f"selectable_complete={len(sorted_trials)}, requested_top_trials={top_trials}, "
        f"selected_for_outer={len(top_trials_list)}"
    )

    return top_params, top_metrics, trial_df

######## OUTER LOOP Functions #########
def merge_best_params(model_config, best_result):
    """
        Merge best_result into model_config, split expert-specific configs,
        and add global model parameters to each expert model.
        
        Returns:
        - global_model: dict (non-expert configs)
        - expert_models: dict { "expert0": {...}, "expert1": {...}, ... }
    """
    best_model = copy.deepcopy(model_config)

    # First merge global (non-expert) overrides
    for key, value in best_result.items():
        if re.match(r"^expert\d+_", key):
            continue
        best_model[key] = value

    # Collect expert-specific params
    expert_params = defaultdict(dict)
    suffix_groups = defaultdict(lambda: defaultdict(dict))

    for key, value in best_result.items():
        m = re.match(r"^(expert\d+)_(.+?)(?:_(\d+))?$", key)
        if m:
            expert, base, idx = m.groups()
            if idx is not None:
                suffix_groups[expert][base][int(idx)] = value
            else:
                expert_params[expert][base] = value

    # Convert suffix groups into ordered lists
    for expert, bases in suffix_groups.items():
        for base, vals in bases.items():
            ordered = [v for _, v in sorted(vals.items())]
            expert_params[expert][base] = ordered

    # Now merge global_model into each expert model
    global_model = best_model
    expert_models = {}
    for expert, params in expert_params.items():
        expert_models[expert] = copy.deepcopy(global_model)
        expert_models[expert].update(params)  # expert overrides win
    return best_model, expert_models

def run_outer_training_fold(out_l, best_params,general_config, data_config, config_id, outerepochs):
    """
    Using the aggregated best inner-loop parameters, merge them into the model config
    and run the outer-loop training once.
    """
    hostname = socket.gethostname()
    pid = os.getpid()
    print(f"[{hostname}][PID:{pid}] Running outer training for fold {out_l} with config_id {config_id}")

    # Merge the aggregated inner-loop params into the base model configuration.
    best_model_mixture, trainable_model_configs = merge_best_params(general_config, best_params)
    # Same rule as the inner loops (`training::cuda`), so MixtureModel.device agrees.
    device = torch.device("cuda" if best_model_mixture['training::cuda'] and torch.cuda.is_available() else "cpu")
    # Experts in index order, each with the same seed offset it had in the inner loops.
    expert_keys = sorted(trainable_model_configs, key=lambda k: int(k[len("expert"):]))
    model_configs = []
    for nr_ex, key in enumerate(expert_keys):
        expert_config = trainable_model_configs[key]
        expert_config["model::seed"] = best_model_mixture["model::seed"] + nr_ex + 1
        model_configs.append(expert_config)

    # .init(project="HPO_Outer_Fold_Temporal", 
    #             name=f"{out_l}_config_{config_id}",
    #             group=f"{training_years_len}",#f"trial_number_{trial_number}",  
    #             reinit=True,
    #             dir="wanddir",
    #             settings=wandb.Settings(start_method="thread"),
    #             config=best_config
    # )

    outer_results = dict()

    # Determine outer-loop train/test splits from data_config["all_stations"]
    train_config = copy.deepcopy(best_model_mixture)
    test_config = copy.deepcopy(best_model_mixture)
    train_config['data::train::station_id'] = {data_config['station']: data_config[out_l]['train_hold_out']}
    test_config['data::test::station_id'] = {data_config['station']:data_config[out_l]['hold_out_years']}
    
    best_model_mixture['data::train::station_id'] = copy.deepcopy({data_config['station']: data_config[out_l]['train_hold_out']})
    best_model_mixture['data::test::station_id']  = copy.deepcopy({data_config['station']:data_config[out_l]['hold_out_years']})

    pm_manager = ProcessModelCalibrationManager(
        config=best_model_mixture,
        process_model=best_model_mixture.get("process_model::name"),
    )
    train_set, test_set, _ = pm_manager.load_fold_dataloaders(
        train_stations=train_config['data::train::station_id'],
        test_stations=test_config['data::test::station_id'],
        val_stations=None,
        # Same the calibrated process model as the inner loops (was `False` = default parameters,
        # so HPO selected with the calibrated process model but the final model used uncalibrated).
        calibrate="require_existing",
        shuffle_train=True,
        sparse=best_model_mixture.get("data:train:sparse"),
    )

    # The outer run has its own budget (it trains on the data; the inner loops only
    # rank architectures). Its schedule is the searched fractions scaled to that budget,
    # counted in real optimizer steps now that the loader is known.
    outer_budget = len(train_set) * outerepochs
    best_model_mixture["train_budget"] = outer_budget
    for key in ("warmup", "ramp", "noise"):
        best_model_mixture[f"{key}_steps"] = int(best_model_mixture[f"schedule::{key}_frac"] * outer_budget)

    torch.manual_seed(best_model_mixture['model::seed'])
    model= MixtureModel(config=best_model_mixture,
                        model_configs=model_configs,
                        total_train_budget=outer_budget,
                        hard_expert_id=best_model_mixture['hard_expert_id'],
                        warmup_steps=best_model_mixture["warmup_steps"],
                        ramp_steps=best_model_mixture["ramp_steps"],
                        noise_steps=best_model_mixture["noise_steps"],
                        alpha_entropy=best_model_mixture["alpha_entropy"]).to(device)

    # get final results of HPO
    final_train_log = []
    best_test= np.inf
    best_test_metric =np.inf
    best_test_routing_share = None
    best_usage=np.inf
    best_epoch = -1

    # Set up logging for outer loop
    out_dir_outer = os.path.join(out_dir_logs, out_l)
    model_dir = os.path.join(out_dir_outer, "models")
    os.makedirs(model_dir, exist_ok=True)

    trial_epoch_dir = os.path.join(model_dir)
    os.makedirs(trial_epoch_dir, exist_ok=True)

    logger = setup_logging(out_dir_outer, out_l)
    logger.info(f"[{hostname}][PID:{pid}] Starting outer training for fold {out_l}")

    # Run outer training for the specified number of epochs
    for epoch in range(outerepochs):
        train_loss, train_metric, train_us, train_entropy_av, train_share = model.train_epoch(train_set)
        test_loss, test_metric, test_us, test_entropy_av, test_share = model.test_epoch(test_set)

        # Convert tensor losses to float if needed
        final_train_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "test_loss": test_loss,
            "test_metric": test_metric,
        })
        logger.info(f"[{hostname}][PID:{pid}] Outer {out_l} Epoch {epoch} - train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

        # Update best test loss if improved
        #if (epoch % general_config['checkpoint::window_size'] == 0 or epoch == general_config['training::epochs'] - 1):
            # wandb.log({"epoch": epoch, 
            #            "fold": out_l, 
            #            "test_loss": test_loss_value,
            #            "config_ID": config_id})
        
        # Best test loss/metric/epoch are DIAGNOSTICS ONLY; they no longer decide which
        # weights are kept. Saving the test-argmin epoch made the saved weights a
        # function of test performance and biased the fold's test score low.
        if math.isfinite(test_loss) and test_loss < best_test:
            best_test = test_loss
            best_test_metric=test_metric
            best_test_routing_share = test_share.tolist()
            best_epoch = epoch
            best_usage =test_us

    # Save the model at the end of the full outer budget. The MoE schedules (warmup,
    # hard-expert ramp, noise annealing) are defined over that budget, so the final
    # epoch is the one where they have played out, and it is chosen without the test split.
    saved_epoch = outerepochs - 1
    best_model_path = os.path.join(trial_epoch_dir, f"best_model_{config_id}.pth")
    torch.save(model, best_model_path)
    # Persist the fold's train-fitted z-score constants beside the weights.
    save_normalization(getattr(train_set.dataset, "normalization", None), trial_epoch_dir)
    logger.info(f"[{hostname}][PID:{pid}] [MODEL SAVED] Outer {out_l} final model at epoch {saved_epoch} (saved at {best_model_path})")

    logger.info(f"[{hostname}][PID:{pid}] Outer training completed for {out_l}. Final train log: {final_train_log}")

    final_train_log_path = os.path.join(out_dir_outer, f"final_train_log_{out_l}_{config_id}.json")
    with open(final_train_log_path, "w") as f:
        json.dump(final_train_log, f, indent=4)

    outer_summary = []
    outer_summary.append({
        "outer_fold": out_l,
        'final_train_loss': train_loss, 
        'final_train metric': train_metric,
        'final_train_usage': train_us.tolist(),
        'final_train_entropy': train_entropy_av,
        'final_train_routing_share': train_share.tolist(),
        'final_test loss':  test_loss,
        # `best_test_*` / `best_epoch` are DIAGNOSTICS: the best test numbers seen
        # during training. The saved checkpoint is `saved_epoch` (the final epoch).
        'best_test_loss': best_test,
        'final_test_entropy': test_entropy_av,
        'final_test_routing_share': test_share.tolist(),
        'best_test_routing_share': best_test_routing_share,
        'best_test_metric': best_test_metric,
        'best_test_usage': best_usage.tolist() if isinstance(best_usage, np.ndarray) else None,
        'final_test metric':  test_metric,
        'final_test_usage': test_us.tolist(),
        "final_train_log": final_train_log_path,
        "best_epoch": best_epoch,
        "saved_epoch": saved_epoch,
        "best_model_path": best_model_path,
        "config_id": config_id,
        "config": best_model_mixture
        })
        
    summary_path = os.path.join(out_dir_outer, f"outer_summary_{out_l}_{config_id}.json")
    with open(summary_path, "w") as f:
        json.dump(outer_summary, f, indent=4)
                

    # wandb.define_metric( "test_loss", summary='min', step_metric='outer')
    # wandb.define_metric( "test_metric", summary='min', step_metric='outer')  
    # wandb.log({
    #         "outer_fold": out_l,
    #         "epoch": epoch,
    #         "train_loss": train_loss_value,
    #         "train_metric": train_metric_value,
    #         "test_loss": test_loss_value,
    #         "test_metric": test_metric_value,
    #         "best_test_loss": best_test,
    #         "Config_ID":config_id
    # })

    # wandb.run.summary["best test accuracy"] = best_test
    # wandb.run.summary["state"] = f"completed {out_l}"
    logger.info(f"[{hostname}][PID:{pid}] Final best test loss for {out_l}: {best_test:.4f}")
    
    #Save best model _state
    print(f"[{hostname}][PID:{pid}] Final_best_test_loss for {out_l}: {best_test:.4f}")
    outer_results[out_l] = {
        'train_loss': train_loss, 
        'train metric': train_metric,
        'test loss':  test_loss,
        'best_test_loss': best_test,  
        'best_test_metric': best_test_metric,
        'test metric':  test_metric 
    }

    return outer_results

def run_outer_fold_CV(outer_fold_key, general_config, data_config, out_dir, n_trials_inner=8, outerepochs=500, top_trials=5):
    """
    Runs a single outer fold in nested CV as an independent job.
    """
    if outer_fold_key == "all_years":
        return None

    print(f"[INFO] Starting outer fold {outer_fold_key}")

    summary_path = os.path.join(out_dir, f"outer_summary_{outer_fold_key}.json")
    if os.path.exists(summary_path):
        print(f"[INFO] Outer fold {outer_fold_key} already evaluated. Skipping.")
        with open(summary_path, "r") as f:
            return {outer_fold_key: json.load(f)}

    # Only the leader (worker 0) finalizes the fold; the other workers contribute their
    # HPO trials to the shared study and exit.
    is_leader_worker = (worker_id == 0)
    if workers_per_fold > 1 and not is_leader_worker:
        run_inner_loops_HPO(outer_fold_key, general_config, data_config,
                            n_trials_inner=n_trials_inner, top_trials=top_trials)
        print(
            f"[INFO] Fold {outer_fold_key}: worker {worker_id} finished its HPO contribution "
            f"({n_trials_inner} trials). Leader worker 0 will run outer training."
        )
        return {outer_fold_key: {"status": "hpo_worker_complete", "worker_id": worker_id,
                                 "trials_run": n_trials_inner}}

    # Run inner loop HPO
    top_params, top_metrics, trial_df = run_inner_loops_HPO(outer_fold_key, general_config, data_config, n_trials_inner=n_trials_inner, top_trials=top_trials,
                                                            wait_for_all_workers=(workers_per_fold > 1))

    # Without this an empty summary JSON would be written, and the resume skip above
    # would then treat the fold as finished on every rerun.
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
        result = run_outer_training_fold(outer_fold_key, config, general_config,  data_config, config_id=i+1, outerepochs=outerepochs)
        outer_fold_results.append({
            "config": config,
            "metric": top_metrics[i],
            "outer_result": result[outer_fold_key]
        })

    # Save summary
    with open(summary_path, "w") as f:
        json.dump(outer_fold_results, f, indent=4)

    return {outer_fold_key: outer_fold_results}

def calculate_total_training_budget(train_years, train_epochs, batch_size, sparse=None):
    train_years = int(train_years.strip('len'))
    training_budget= (train_years*train_epochs*365)/batch_size 
    if sparse is not None:
        training_budget=training_budget/sparse
    print(f'training_budget {training_budget}')
    return int(training_budget)

if __name__ == "__main__":
    from Config_Files.config_spatial_mixture import build_config     ## explicitly set which experiment configuration file you wish to use
    from Config_Files.config_loader import merge_process_model_config, SUPPORTED_PROCESS_MODELS
    
    # Load environment variables from .env file
    load_dotenv()

    # Calibration
    training_years_len = str(os.getenv("temporal_training_years_len")) 

    # === Argument parser for bash script on cluster ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_fold", type=str,       help="Outer fold name, e.g. outer_station_5, out_l_1")
    parser.add_argument("--worker_id", type=int,        help="Unique ID for the worker (optional)")
    parser.add_argument("--n_trials", type=int,         help="Number of trials for inner HPO per worker")
    parser.add_argument("--config", type=str,           help="Path to the frozen config file (JSON)")
    parser.add_argument("--CV_config_path", type=str,    help="Name of the config module (e.g., config_temporal)")
    parser.add_argument("--process-model", type=str, choices=SUPPORTED_PROCESS_MODELS, help="Override process model namespace.")
    parser.add_argument("--top_trials", type=int, default=5, help="Number of best inner trials retrained in the outer loop")
    parser.add_argument("--workers_per_fold", type=int, help="Number of parallel workers contributing to one fold.")
    args = parser.parse_args()
    
    # ================== Detect if running locally or in debug mode ==================
    if len(sys.argv) == 1 or os.getenv("DEBUG_MODE") == "1":
        print("[DEBUG] No CLI arguments detected — entering local/debug mode.")

        args.outer_fold = "out_l_3"
        args.worker_id = 0
        args.n_trials = 1
        args.CV_config_path = os.path.join(
            os.getcwd(), "configs/CV_splits/Paper_1_preles_ncvs/temporal_cv_train_len1_station_5.json"
        )
        print(f"[DEBUG] Outer Fold: {args.outer_fold} | Worker: {args.worker_id} | Trials: {args.n_trials}")

    # ================== Config: frozen JSON on the cluster, else build_config ==================
    # As in the spatial driver, build_config() is the default. The debug branch used to
    # build it and then discard it by opening a hard-coded JSON from another user's home.
    if args.config:
        with open(args.config, "r") as f:
            general_config = json.load(f)
        print(f"[INFO] Using frozen config: {args.config}")
    else:
        general_config = build_config()

    # ProcessModelCalibrationManager requires the process-model keys (Process::x_para, ...);
    # merge them into the frozen config as Optuna_temporal_NCV.py does.
    process_model_name = str(
        args.process_model or general_config.get("process_model::name") or "preles"
    ).strip().lower()
    general_config["process_model::name"] = process_model_name
    general_config = merge_process_model_config(
        base_config=general_config,
        process_model=process_model_name,
    )

    # ================== Workers and trial budget ==================
    # Module-level, read by run_outer_fold_CV / run_inner_loops_HPO. Per-worker trials:
    # explicit --n_trials, else the process-model config's per-fold budget split across
    # the workers, so the fold total doesn't depend on the worker count.
    worker_id = args.worker_id if args.worker_id is not None else 0
    workers_per_fold = args.workers_per_fold
    if workers_per_fold is None:
        workers_per_fold = int(os.getenv("WORKERS_PER_FOLD", "1"))
    if workers_per_fold < 1:
        raise ValueError(f"workers_per_fold must be >= 1, got {workers_per_fold}.")
    trials_per_fold = general_config.get("hpo::trials_per_fold")
    if args.n_trials is not None:
        n_trials = args.n_trials
    elif trials_per_fold:
        n_trials = math.ceil(int(trials_per_fold) / workers_per_fold)
    else:
        n_trials = 2
    print(f"[INFO] worker {worker_id}/{workers_per_fold}  n_trials/worker={n_trials}")

    cv_config = json_open(args.CV_config_path)

    save = general_config["cfg::save"]
    print(f"[INFO] Using {training_years_len}")
    
    prediction_type = general_config["model::prediction_type"]
    Station_ID = cv_config["station"]
    prediction_loss = general_config["training::loss_physic_type"]
    data_sparcity = general_config['data:train:sparse']

    general_config['train_budget'] = calculate_total_training_budget(training_years_len, 
                                                                     general_config['training::epochs'], 
                                                                     general_config['training::batchsize'], 
                                                                     sparse=general_config['data:train:sparse'])


    
    # Same layout as the other paper-1 architectures (calibration_paths.build_spatial_logs_root):
    #   <save>/checkpoints/<pm>/<full|sparseN>/<cv_label>/[<namespace>/]logs_<type>_mixture
    # "mixture", not the config's loss type: the MoE runs with loss "parallel", which would
    # land in the parallel architecture's logs_spatial_parallel directory.
    artifact_namespace = str(
        os.getenv("EXPERIMENT_ARTIFACT_NAMESPACE")
        or general_config.get("experiment::artifact_namespace")
        or os.getenv("CALIBRATION_ARTIFACT_NAMESPACE")
        or ""
    ).strip()
    experiment_paths = build_spatial_logs_root(
        save_root=save,
        process_model_name=process_model_name,
        cv_label=cv_label_from_path(args.CV_config_path),
        prediction_type=prediction_type,
        prediction_loss="mixture",
        sparse=data_sparcity,
        artifact_namespace=artifact_namespace,
    )
    checkpoint_path = experiment_paths["checkpoint_root"]
    print(f"[INFO] Checkpoint path set to {checkpoint_path}")
    print(f"[INFO] artifact_namespace={artifact_namespace or '<default>'}")
    
    outer_fold_to_run = args.outer_fold

    out_dir_logs = experiment_paths["logs_root"]
    os.makedirs(out_dir_logs, exist_ok=True)

    # initialize wanb 
    #wandb.login()  # Ensure you're logged in
    outer_result = run_outer_fold_CV(
        outer_fold_to_run, 
        general_config, 
        cv_config, 
        out_dir_logs, 
        n_trials_inner=n_trials,
        # training::outer_epochs lets a smoke-test config shorten the outer run.
        outerepochs=int(general_config.get("training::outer_epochs", 150)),
        top_trials=args.top_trials)
    #wandb.finish()

    print('blub')
