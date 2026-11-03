
import torch 
import torch.nn as nn
# from pysr import PySRRegressor
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Optional, Tuple, Union
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from Dataloader import spatial_dataset, Preles_data, temporal_dataset
from utilis_shorts import json_open

def call_symb( x,y, binary_operators = ["+", "*", "-", "/"], unary_operators =  ["cos","exp","sin","square", "cube"], #"inv(x) = 1/x"
                    nested_constraints = {"sin": {"sin": 2}, "cos": {"cos": 2}}, optimizer='SGD', equation_file=None):
    '''calls symbolic regression and fits it to data'''

    x = np.array(x)
    #create Symb model
    symb_model = PySRRegressor(
        niterations= 450,  # < Increase me for better results
        binary_operators= binary_operators,
        unary_operators= unary_operators,
        batching=True,
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        nested_constraints= nested_constraints,
            # ^ Define operator for SymPy as well
        elementwise_loss="L2DistLoss()",
        model_selection= 'accuracy',
        maxsize=20,
        equation_file= equation_file)
        
    #fit and convert to pytorch model
    symb_model.fit(x,y)
    symb_model = symb_model.pytorch()

    # if optimizer == 'SGD':
    #     symb_optimizer = SGD(symb_model.parameters(), lr=0.001)
    symb_optimizer = 1
        
    print(symb_model)        
    return symb_model, symb_optimizer

def train_symb(symb_model, epochs, x, y):
    symb_model = symb_model.pytorch()

    symb_optimizer = SGD(symb_model.parameters(), lr=0.001)
    loss_ = nn.MSELoss()

    for _ in range(epochs):
        pred = symb_model(torch.tensor(x))
        symb_optimizer.zero_grad()
        loss = loss_(torch.tensor(y).squeeze(),pred)
        loss.backward()
        symb_optimizer.step()

    return symb_model

def get_symbolic_data(symb_config):
    ''' gets and convert data for symbolic regression '''
    symb_train_data = Preles_data(symb_config['data::train::path_dir'], 
                                station_id=symb_config['data::train::station_id'],
                                x_para=symb_config['Preles::x_para'], 
                                y_para=symb_config['Preles::y_para'],
                                y_phy_para =symb_config['Preles::y_phy_para'],
                                normalise= symb_config['data::normalise'],
                                dataset=True )
    
    symb_test_data = Preles_data(symb_config['data::train::path_dir'], 
                                station_id=symb_config['data::test::station_id'],
                                x_para=symb_config['Preles::x_para'], 
                                y_para=symb_config['Preles::y_para'],
                                y_phy_para =symb_config['Preles::y_phy_para'],
                                normalise= symb_config['data::normalise'],
                                dataset=True )
    
    x_train = np.array(symb_train_data.data[symb_config['Preles::x_para']])
    y_train = np.array(symb_train_data.data[symb_config['Preles::y_para']])
    #test
    x_test = np.array(symb_test_data.data[symb_config['Preles::x_para']])
    y_test = np.array(symb_test_data.data[symb_config['Preles::y_para']])

    return x_train, y_train, x_test, y_test

def load_datasets(config):
    """
    Load datasets based on the model configuration.
    """
    if config['training::loss_physic_type'] not in ['naive', 'bias'] and config['model::prediction_type'] == 'temporal':
        print(f"Loading Preles dataset: {config['model::prediction_type']}")
        if config['training::loss_physic_type'] == 'error':
            error = True
        else:
            error = False
        train_dataset = Preles_data(
                config['data::train::path_dir'],
                station_id=config['data::train::station_id'],
                x_para=config['Preles::x_para'],
                y_para=config['Preles::y_para'],
                y_phy_para=config['Preles::y_phy_para'],
                normalise=config['data::normalise'],
                modeltype=config['model::modeltype'],
                temporal=config['model::prediction_type'],
                window_input=config['model::windows_input'],
                window_output=config['model::windows_output'],
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                dataset=True,
                error= error,
                previously_calibrated=config['data::previously_calibrated'],
                sparse=False,   #['data::sparse']           #use full or sparse dataset
                error_norm=config['data::error_norm'],
                    )
        test_dataset = Preles_data(
                config['data::test::path_dir'],
                station_id=config['data::test::station_id'],
                x_para=config['Preles::x_para'],
                y_para=config['Preles::y_para'],
                y_phy_para=config['Preles::y_phy_para'],
                normalise=config['data::normalise'],
                modeltype=config['model::modeltype'],
                temporal=config['model::prediction_type'],
                window_input=config['model::windows_input'],
                window_output=config['model::windows_output'],
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                dataset=True,
                error= error,
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=False,   #['data::sparse']           #use full or sparse dataset
                error_norm=config['data::error_norm'],
            )
    
    elif config['training::loss_physic_type'] in ['naive', 'bias', 'embedding'] and config['model::prediction_type'] == 'spatial':
            print("Loading spatial dataset...")
            train_dataset = spatial_dataset(
                config['data::train::path_dir'],
                station_id=config['data::train::station_id'],
                x_para=config['data::train::x_parameter'],
                y_para=config['data::train::y_parameter'],
                normalise=config['data::normalise'],
                dataset=True,
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=config['data::sparse']           #use full or sparse dataset
            )

            test_dataset = spatial_dataset(
                config['data::test::path_dir'],
                station_id=config['data::test::station_id'],
                x_para=config['data::test::x_parameter'],
                y_para=config['data::test::y_parameter'],
                normalise=config['data::normalise'],
                dataset=True,
                previously_calibrated=config['data::previously_calibrated'], # config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=config['data::sparse']           #use full or sparse dataset
            )
    elif config['training::loss_physic_type'] not in ['naive', 'bias', 'embedding'] and config['model::prediction_type'] == 'spatial':
        if config['training::loss_physic_type'] == 'error':
            error = True
        else:
            error = False
        train_dataset = Preles_data(
                config['data::train::path_dir'],
                station_id=config['data::train::station_id'],
                x_para=config['Preles::x_para'],
                y_para=config['Preles::y_para'],
                y_phy_para=config['Preles::y_phy_para'],
                normalise=config['data::normalise'],
                modeltype=config['model::modeltype'],
                temporal=config['model::prediction_type'],
                dataset=True,
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=False,                    #['data::sparse']           # use full or sparse dataset
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                error= error,
                error_norm=config['data::error_norm'],
                    )
        
        test_dataset = Preles_data(
                config['data::test::path_dir'],
                station_id=config['data::test::station_id'],
                x_para=config['Preles::x_para'],
                y_para=config['Preles::y_para'],
                y_phy_para=config['Preles::y_phy_para'],
                normalise=config['data::normalise'],
                modeltype=config['model::modeltype'],
                temporal=config['model::prediction_type'],
                dataset=True,
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=False,   #['data::sparse']           #use full or sparse dataset
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                error= error,
                error_norm=config['data::error_norm'],
            )
    
    elif config['training::loss_physic_type'] in ['naive', 'bias', 'embedding'] and config['model::prediction_type'] == 'temporal':
            print("Loading temporal dataset...")
            train_dataset = temporal_dataset(
                config['data::train::path_dir'],
                station_id=config['data::train::station_id'],
                x_para=config['data::train::x_parameter'],
                y_para=config['data::train::y_parameter'],
                normalise=config['data::normalise'],
                dataset=True,
                input_window=config['model::windows_input'],
                output_window=config['model::windows_output'],
                modeltype=config['model::modeltype'],
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=False,   #['data::sparse']           #use full or sparse dataset
            )

            test_dataset = temporal_dataset(
                config['data::test::path_dir'],
                station_id=config['data::test::station_id'],
                x_para=config['data::test::x_parameter'],
                y_para=config['data::test::y_parameter'],
                normalise=config['data::normalise'],
                dataset=True,
                input_window=config['model::windows_input'],
                output_window=config['model::windows_output'],
                modeltype=config['model::modeltype'],
                y_past_include = config['model::prediction_temporal:include_Y_past'],
                previously_calibrated=config['data::previously_calibrated'], # use data based on calibrated preles parameters
                sparse=False,   #['data::sparse']           #use full or sparse dataset
            )
    else:
        raise ValueError("Invalid version type specified. Use 'spatial' or 'temporal'.")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training::batchsize'])
    test_loader = DataLoader(test_dataset, batch_size=config['data::test::batchsize'])

    return train_loader, test_loader

def load_CV_datasets(config, dataset, shuffle=False, normalization=None):
    """
    Dynamically loads a dataset for cross-validation using an existing DataFrame
    (`dataset`) that already contains the calibrated/predicted columns.

    Args:
        config (dict): Your model + data configuration dictionary. Must include keys:
            'training::loss_physic_type', 'model::prediction_type', 'data::train::path_dir',
            'data::train::station_id', etc.

        dataset (pd.DataFrame or similar): A "calibrated" or "augmented" dataset
            (e.g. from ProcessModelCalibrationManager.load_fold_datasets) containing:
              - x-parameters
              - y-parameters
              - Possibly y_pred or other columns appended
        shuffle (bool): Whether to shuffle the DataLoader.
        normalization (dict or None): Pre-fitted z-score constants to apply. Build the
            TRAIN loader first and pass its `loader.dataset.normalization` here when
            building the test/val loaders — otherwise each split self-fits and the
            test-site prediction depends on the test site's own measured mean/sd.
            `None` self-fits, which is only correct for the train split.

    Returns:
        DataLoader: A PyTorch DataLoader wrapping the appropriate Dataset class
                    (Preles_data, spatial_dataset, or temporal_dataset).
    """
    prediction_type = config['model::prediction_type']
    loss_type = config['training::loss_physic_type']

    is_hybrid = loss_type in ['parallel', 'regularized' ,'error']  # Add others if needed
    is_embed = loss_type == 'embedding'
    is_error = loss_type == 'error'
    is_spatial = prediction_type == 'spatial'
    is_temporal = prediction_type == 'temporal'

    # Override x/y_para for embedding - requires 7 inputs
    x_para = config['Preles::x_para'] if is_embed else config['data::train::x_parameter']
    y_para = config['Preles::y_para'] if is_embed else config['data::train::y_parameter']

    df_input = getattr(dataset, 'data', dataset)
    if isinstance(df_input, pd.DataFrame):
        df_input = df_input.copy(deep=True)

    shared_kwargs = dict(
        station_id=config['data::train::station_id'],
        x_para=x_para,
        y_para=y_para,
        normalise=config['data::normalise'],
        dataset=True,
        calibrated_df=df_input,
        normalization=normalization,
    )

    if is_hybrid:
        dataset_class = Preles_data
        specific_kwargs = dict(
            y_phy_para=config['Preles::y_phy_para'],
            modeltype=config['model::modeltype'],
            temporal=prediction_type,
            y_past_include=config['model::prediction_temporal:include_Y_past'],
            error=is_error,
            error_norm=config['data::error_norm'],
        )

        if is_temporal:
            specific_kwargs.update({
                'window_input': config['model::windows_input'],
                'window_output': config['model::windows_output'],
            })

    elif is_spatial:
        dataset_class = spatial_dataset
        specific_kwargs = dict(
            # sparse=config['data:train:sparse']
        )

    # elif is_temporal:
    #     dataset_class = temporal_dataset
    #     specific_kwargs = dict(
    #         input_window=config['model::windows_input'],
    #         output_window=config['model::windows_output'],
    #         modeltype=config['model::modeltype'],
    #         y_past_include=config['model::prediction_temporal:include_Y_past'],
    #         sparse=config['data::sparse'],
    #     )

    else:
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    dataset = dataset_class("", **shared_kwargs, **specific_kwargs)
    
    dataset_loader = DataLoader(dataset, batch_size=config['training::batchsize'], shuffle=shuffle)

    return dataset_loader


######## Fold-level HPO budget (shared by the Optuna_*_NCV entrypoints) #########
# Trials that count toward a fold's budget. FAIL does not: a trial killed by preemption or
# a crashed worker produced nothing, so its slot is run again.
def _budget_states():
    from optuna.trial import TrialState
    return (TrialState.COMPLETE, TrialState.PRUNED)


def count_budget_trials(study):
    """Number of trials in the study that count toward the fold budget."""
    return len(study.get_trials(deepcopy=False, states=_budget_states()))


def optimize_within_fold_budget(study, func, n_trials_per_worker, fold_budget, catch=()):
    """
    study.optimize for one worker of a fold, stopping once the shared study holds
    fold_budget trials that count (see _budget_states).

    Without this a requeued or resubmitted fold runs every worker's full
    n_trials_per_worker again on top of the trials it already finished. A fresh fold is
    unaffected: each worker is still capped at n_trials_per_worker, so the total cannot
    pass fold_budget = workers x n_trials_per_worker before the caps stop it anyway.
    Workers mid-trial when the budget is reached still finish that trial, so a resumed
    fold can end up to workers-1 trials over.
    """
    from optuna.study import MaxTrialsCallback

    done = count_budget_trials(study)
    if done >= fold_budget:
        print(f"[INFO] {study.study_name}: fold budget already met ({done}/{fold_budget}); running no new trials.")
        return
    if done > 0:
        print(f"[INFO] {study.study_name}: resuming with {done}/{fold_budget} trials done; "
              f"running until the fold reaches {fold_budget}.")
    study.optimize(
        func,
        n_trials=n_trials_per_worker,
        catch=catch,
        callbacks=[MaxTrialsCallback(fold_budget, states=_budget_states())],
    )
