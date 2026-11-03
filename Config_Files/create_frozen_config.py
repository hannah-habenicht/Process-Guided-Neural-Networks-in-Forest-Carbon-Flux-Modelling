# create_frozen_config.py
import argparse
import os
import importlib
from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1])) 

from utilis_shorts import save_frozen_config

if __name__ == "__main__":
    # Hardcoded values
    config_module_name = "config_spatial_mixture"  # Module name for the config (also the output dir); "config_spatial" for the other architectures
    prediction_loss = ["parallel"] #"embed" "naive", "bias", "parallel", "regularized"; the MoE runs with "parallel"
    sparsities = [None, 50, 30, 7]  # None = full resolution

    for pred, sparsity in ((p, s) for p in prediction_loss for s in sparsities):
        sparsity_part = f"_sparse{sparsity}" if sparsity else ""
        filename = f"{pred}{sparsity_part}.json"
        save_path = str(Path(__file__).resolve().parents[1] / "configs" / config_module_name / filename)

        config_module = importlib.import_module(f"{config_module_name}")
        config = config_module.build_config()

        config["training::loss_physic_type"] = pred
        if pred == 'bias':
            config['model::input_size']= 2
            config['data::train::x_parameter']= ['GPP_pred', 'ET_pred'] #, 'SW_pred'
            config['data::test::x_parameter']= ['GPP_pred', 'ET_pred']
        
        elif pred == 'residual':
            config['model::input_size']= 7
            config['data::train::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar','GPP_pred', 'ET_pred'] #, 'SW_pred'
            config['data::test::x_parameter']= ['Precip', 'Tair', 'PAR', 'VPD', 'fapar','GPP_pred', 'ET_pred']
            
        # config["training::length"] = training_length
        config["data:train:sparse"] = sparsity

        save_frozen_config(config, output_path=save_path)
        print(f"[OK] Saved frozen config to {save_path}")
