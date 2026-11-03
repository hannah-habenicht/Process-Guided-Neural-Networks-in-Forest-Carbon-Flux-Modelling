import copy
import json
import math
import os
import re
import sys
import time
from typing import Dict, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.pruners import NopPruner  # Disable pruning (recommended for deterministic models)
from optuna.samplers import CmaEsSampler, TPESampler
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
# import torch

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_PRELES_C_BUILD_DIRS = (
    os.path.join(_ROOT_DIR, "PM_Extension", "prelesR", "src"),
    os.path.join(
        _ROOT_DIR,
        "PM_Extension",
        "prelesR",
        "src",
        "build",
        f"lib.linux-x86_64-cpython-{sys.version_info.major}{sys.version_info.minor}",
    ),
)
for _path in _PRELES_C_BUILD_DIRS:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import preles_c as preles  # batched PRELES C extension
except Exception:
    preles = None

# Differentiable torch rewrite (module name `preles`), called row-by-row.
# Kept under a distinct alias so it never collides with the batched `preles_c`.
_PRELES_REWRITE_BUILD_DIRS = (
    os.path.join(_ROOT_DIR, "PM_Extension", "PRELES_rewrite"),
    os.path.join(
        _ROOT_DIR,
        "PM_Extension",
        "PRELES_rewrite",
        "build",
        f"lib.linux-x86_64-cpython-{sys.version_info.major}{sys.version_info.minor}",
    ),
)
for _path in _PRELES_REWRITE_BUILD_DIRS:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
try:
    import torch as _torch  # must load torch's shared libs (libc10.so) before the extension
    import preles as preles_rewrite  # torch differentiable PRELES rewrite (row-by-row)
except Exception:
    preles_rewrite = None
from Basic_dataset import Basic_dataset

from calibration_paths import (
    build_calibration_metadata,
    build_calibration_paths,
    build_expected_metadata_context,
    is_metadata_compatible,
    load_calibration_bundle,
    save_calibration_bundle,
)


def _legacy_generate_calibration_filename(path_dir, station_id, sparse=None):
        """
        Legacy PRELES calibration filepath builder retained for backwards compatibility.
        """
        station_entries = []

        for station, info in station_id.items():
            if isinstance(info, list) and all(isinstance(year, int) for year in info):
                sorted_years = sorted(info)
                years_str = "-".join(map(str, sorted_years))
                station_entries.append(f"{station}_{years_str}")

            elif isinstance(info, dict) and 'startdate' in info and 'stopdate' in info:
                date_ranges = []
                for start, stop in zip(info['startdate'], info['stopdate']):
                    if start[0] == stop[0]:
                        date_ranges.append(
                            f"{start[0]}_{start[1]:02d}_{start[2]:02d}-{stop[1]:02d}_{stop[2]:02d}"
                        )
                    else:
                        date_ranges.append(f"{start[0]}-{stop[0]}")
                date_str = "+".join(date_ranges)
                station_entries.append(f"{station}_{date_str}")

            else:
                raise ValueError(f"Unrecognized station ID format for {station}: {info}")

        station_part = "__".join(station_entries)
        sparse_suffix = f"_sparse{sparse}" if isinstance(sparse, int) and sparse > 1 else ""
        filename = f"{station_part}{sparse_suffix}.json"
        if len(filename) > 255:
            filename = filename[:250] + ".json"
        return os.path.join(path_dir, 'Preles', os.getenv("CALIBRATION_PATH") or "", filename)


def generate_calibration_filename(path_dir, station_id, sparse=None):
        """
        Canonical PRELES calibration params path based on shared calibration path policy.

        :param path_dir: Base directory where the calibration file will be saved.
        :param station_id: Dictionary containing station IDs and their respective years.
        :param sparse: Optional integer indicating sparse sampling (e.g., 7 = every 7th entry)
        :return: Full path to canonical params.json in the calibration bundle folder.
        """
        cfg = {
            "process_model::name": "preles",
            "calibration::cv_type": os.getenv("CALIBRATION_CV_TYPE", "default"),
            # "calibration::training_years_len": (
            #     os.getenv("spatial_training_years_len")
            #     or os.getenv("temporal_training_years_len1")
            #     or "unspecified_len"
            # ),
            "calibration::root": os.getenv("CALIBRATION_ROOT"),
            "data:train:sparse": sparse,
            "model::prediction_type": os.getenv("CALIBRATION_CV_TYPE", "default"),
            "calibration::artifact_namespace": os.getenv("CALIBRATION_ARTIFACT_NAMESPACE"),
        }
        paths = build_calibration_paths(
            config=cfg,
            station_id=station_id,
            process_model="preles",
            sparse=sparse,
            data_root=path_dir,
        )
        return paths["params_file"]

class Preles_dataset(Basic_dataset):
    """
    Initializes a raw dataset instance for PRELES by loading data, constructing file paths,
    and setting up for default or calling previously calibrated PRELES parameters.

    Args:
        path_dir (str): Base directory path containing preprocessed data.
        station_id (Dict): Dictionary mapping station IDs to their associated date ranges.
        x_para (list): List of input feature names.
        y_para (list): List of target output variable names.
        default_parameters (dict, optional): Default parameter values for PRELES if calibration data is unavailable.
        control (dict, optional): Dictionary of control parameters for the PRELES model.
        preintialised_output (dict, optional): Precomputed outputs for initializing model states.
        device (bool, optional): Whether to use GPU (`True`) or CPU (`False`) for tensor storage.
        dataset (bool, optional): Whether to immediately load the dataset.
        previously_calibrated (bool, optional): Flag indicating whether calibration data should be loaded.
        cv_config_path (str, optional): Path to the CV configuration to determine file naming.
    """
    def __init__(self,path_dir:str, 
                 station_id:Dict,
                 x_para: list, 
                 y_para:list,
                 default_parameters:dict=None, 
                 control=None,
                 preintialised_output=None, 
                 device:bool=True,
                 dataset=True,
                 previously_calibrated = False, 
                 sparse=None,
                 calibration_namespace: str = None,
                 calibration_context: dict = None,
                 backend: str = "preles_c",
                ) -> None:
        
        self.path_dir = os.path.join(path_dir, "preprocessed_data")
        self.data_root = path_dir
        
        super(Preles_dataset,self).__init__()
        # File Paths
        self.station_id_map = station_id
        self.station_id = list(station_id.keys())
        self.sparse = sparse 

        # Canonical calibration bundle paths (new policy) + legacy fallback filepath.
        # Prefer values from the caller-supplied config so the bundle paths match the
        # ones the ProcessModelCalibrationManager computed; only fall back to env vars
        # when the caller did not provide a context (e.g. standalone use).
        ctx = calibration_context or {}
        cv_type = (
            ctx.get("calibration::cv_type")
            or ctx.get("model::prediction_type")
            or os.getenv("CALIBRATION_CV_TYPE", "spatial")
        )
        self.calibration_context_config = {
            "process_model::name": "preles",
            "calibration::cv_type": cv_type,
            "calibration::root": ctx.get("calibration::root") or os.getenv("CALIBRATION_ROOT"),
            "data:train:sparse": sparse,
            "model::prediction_type": ctx.get("model::prediction_type") or cv_type,
            "calibration::mode": ctx.get("calibration::mode") or os.getenv("calibration_flag", "auto"),
            "calibration::artifact_namespace": (
                calibration_namespace
                if calibration_namespace is not None
                else ctx.get("calibration::artifact_namespace")
                or os.getenv("CALIBRATION_ARTIFACT_NAMESPACE")
            ),
        }
        self.calibration_paths = build_calibration_paths(
            config=self.calibration_context_config,
            station_id=station_id,
            process_model="preles",
            sparse=sparse,
            data_root=self.data_root,
        )
        self.calibration_path = self.calibration_paths["folder"]
        self.calibration_file = self.calibration_paths["params_file"]
        self.calibration_metadata_file = self.calibration_paths["metadata_file"]
        self.legacy_calibration_file = _legacy_generate_calibration_filename(
            self.path_dir, station_id, sparse=sparse
        )
        os.makedirs(self.calibration_path, exist_ok=True)

        #device
        self.test_device(device)
        self.dataset = dataset
        self.x_parameter = x_para
        self.y_parameter = y_para
        self.backend = str(backend or "preles_c").strip().lower()
        # Optional extra PRELES outputs (e.g. ["SWC"]) appended as "<name>_pred" by
        # run_process_model. Not returned by call_preles_batch, so calibration is unaffected.
        self.extra_outputs = [o for o in (ctx.get("Preles::extra_outputs") or []) if o not in y_para]
        self._last_raw_outputs = None

        # If previously calibrated, load the parameters from json file
        if previously_calibrated:
            self.default_parameters = self.load_calibrated_params()
        else: 
            self.default_parameters=None

        if self.default_parameters is None:
            self.default_parameters = default_parameters

        self.control = control #control variables
        self.preintialised_output = preintialised_output #output of preles
        self.previously_calibrated = previously_calibrated

        ## get data
        self.get_all_data(self.path_dir, 
                          station_id, 
                          x_para, 
                          y_para, 
                          dataset=dataset,  
                          previously_calibrated=previously_calibrated,
                          sparse=sparse,
                          defer_sparse=True,
                          )
        # Ensure site_id is always included
        if "site_id" not in self.data.columns:
            self.data["site_id"] = self.station_id[0]  # Assign first station ID

        # PRELES is stateful (soil water, acclimation, snow): sparse runs simulate the
        # continuous daily series (data_full) and only expose the kept rows as self.data.
        self.data_full = self.data
        self._keep_mask = np.ones(len(self.data_full), dtype=bool)
        if "_sparse_keep" in self.data_full.columns:
            self._keep_mask = self.data_full.pop("_sparse_keep").to_numpy(dtype=bool)
            self.data = self.data_full.loc[self._keep_mask].reset_index(drop=True)
        
    def load_calibrated_params(self):
        params, metadata = load_calibration_bundle(self.calibration_paths)
        if params is not None:
            if metadata is None:
                print(f"[DEBUG] Loading calibrated parameters from {self.calibration_file} (metadata missing).")
                self.default_parameters = params
                return self.default_parameters
            
            expected = build_expected_metadata_context(
                config=self.calibration_context_config,
                station_id=self.station_id_map,
                process_model="preles",
                sparse=self.sparse,
            )
            compatible, reason = is_metadata_compatible(metadata, expected)
            if compatible:
                print(f"[DEBUG] Loading calibrated parameters from {self.calibration_file}")
                self.default_parameters = params
                return self.default_parameters
            print(f"[DEBUG] Found calibration bundle but metadata is incompatible ({reason}); falling back.")

        # Backward compatibility for legacy flat PRELES calibration files.
        if os.path.exists(self.legacy_calibration_file):
            print(f"[DEBUG] Loading legacy calibrated parameters from {self.legacy_calibration_file}")
            with open(self.legacy_calibration_file, 'r', encoding='utf-8') as file:
                self.default_parameters = json.load(file)
            return self.default_parameters

        print("[DEBUG] No calibrated parameters found. Using default parameters.")
        return None

    def save_calibration_artifacts(self, params: Dict, extra_metadata: Dict = None):
        """
        Save PRELES calibration params + metadata to the canonical calibration bundle.
        """
        metadata = build_calibration_metadata(
            config=self.calibration_context_config,
            station_id=self.station_id_map,
            process_model="preles",
            sparse=self.sparse,
            calibration_mode=self.calibration_context_config.get("calibration::mode", "auto"),
            train_station_id=self.station_id_map,
            extra=extra_metadata or {},
        )
        save_calibration_bundle(self.calibration_paths, params=params, metadata=metadata)
        return self.calibration_paths
        
    def __len__(self):
        return len(self.data)

    def read_all_data(self):
        #restrict dataset, for huge dataset we should work differently e.g checkout Hp5
        x_list = self.data[self.x_parameter].values.astype(np.float32)
        y_list = self.data[self.y_parameter].values.astype(np.float32)

        self.x = x_list
        self.y = y_list

        return self.x, self.y #, self.y_p

    def __getitem__(self, index):
        row = self.data.iloc[index]

        site_id = row["site_id"]

        # x and y as 1D numpy arrays
        x_i = row[self.x_parameter].values.astype(np.float32)
        y_i = row[self.y_parameter].values.astype(np.float32) 

        return x_i, y_i, site_id   
    
    
    # Fixed positional order of the 60 arguments the torch rewrite (`preles.preles`)
    # expects. Identical to the legacy New_datasets.py `preles_order`.
    _REWRITE_PRELES_ORDER = [
        'PAR', 'Tair', 'VPD', 'Precip', 'CO2', 'fapar', 'GPPmeas', 'ETmeas', 'SWmeas',
        'GPP', 'ET', 'SW', 'SOG', 'fS', 'fD', 'fW', 'fE', 'Throughfall',
        'Interception', 'Snowmelt', 'Drainage', 'Canopywater', 'S', 'soildepth', 'ThetaFC',
        'ThetaPWP', 'tauDrainage', 'beta', 'tau', 'S0', 'Smax', 'kappa', 'gamma', 'soilthres',
        'bCO2', 'xCO2', 'ETbeta', 'ETkappa', 'ETchi', 'ETsoilthres', 'ETnu', 'MeltCoef', 'I0',
        'CWmax', 'SnowThreshold', 'T_0', 'SWinit', 'CWinit', 'SOGinit', 'Sinit', 't0', 'tcrit',
        'tsumcrit', 'etmodel', 'LOGFLAG', 'NofDays', 'DOY', 'transp', 'evap', 'fWE',
    ]
    # Output-buffer slots whose data/config column uses a different name.
    _REWRITE_KEY_ALIASES = {'SW': 'SWC'}
    # The rewrite returns 14 outputs in this fixed order (no transp/evap/fWE).
    _REWRITE_OUTPUT_ORDER = {
        'GPP': 0, 'ET': 1, 'SW': 2, 'SOG': 3, 'fS': 4, 'fD': 5, 'fW': 6, 'fE': 7,
        'Throughfall': 8, 'Interception': 9, 'Snowmelt': 10, 'Drainage': 11,
        'Canopywater': 12, 'S': 13,
    }

    def prepare_preles_parameters(self, index, params: dict = None, data_subset=None):
        """
        Legacy New_datasets.py row-wise input builder for the torch rewrite.

        Resolves each of the 60 ordered arguments for a SINGLE row, in this precedence:
        data column -> x_para -> y_para -> default_parameters (params override) ->
        control -> preintialised_output. This is a verbatim port of the legacy behaviour:
        in particular GPPmeas/ETmeas/SWmeas and transp/evap/fWE come from
        default_parameters (-999), the GPP/ET buffers are seeded from the observed data,
        the remaining output buffers from preintialised_output (0), NofDays/etmodel/LOGFLAG
        from control, and SWinit stays the constant default every row (NO carry-forward).
        """
        if data_subset is None:
            data = self.data.iloc[index]
        else:
            data = data_subset.iloc[0]

        _MISSING = object()
        input_preles = []
        for order in self._REWRITE_PRELES_ORDER:
            # SW is an output-buffer slot; the current data/config name it SWC.
            candidates = [order] + ([self._REWRITE_KEY_ALIASES[order]]
                                    if order in self._REWRITE_KEY_ALIASES else [])
            val = _MISSING
            for key in candidates:
                if key in data.index:
                    val = data[key]
                elif key in self.x_parameter:
                    val = self.data.iloc[index][key]
                elif key in self.y_parameter:
                    val = self.data.iloc[index][key]
                elif key in self.default_parameters.keys():
                    val = params[key] if (params and key in params) else self.default_parameters[key]
                elif key in self.control.keys():
                    val = self.control[key]
                elif key in self.preintialised_output.keys():
                    val = self.preintialised_output[key]
                if val is not _MISSING:
                    break
            if val is _MISSING:
                raise KeyError(
                    f"[prepare_preles_parameters] Key '{order}' not found in data, "
                    "default parameters, control, or pre-initialised output."
                )
            input_preles.append(val)
        return input_preles

    def _rewrite_predict_rowwise(self, params: dict = None):
        """
        Run the torch rewrite (`preles.preles`) one row at a time over self.data.

        Each day is independent: initial states are re-seeded from the constant
        calibration scalars every row (no state carried from the previous day),
        reproducing the legacy New_datasets.py path. Returns a list of length-N
        numpy arrays aligned to self.y_parameter, matching call_preles_batch.
        """
        if preles_rewrite is None:
            raise RuntimeError(
                "backend='preles_rewrite' requested, but the `preles` rewrite extension "
                "could not be imported (build it under PM_Extension/PRELES_rewrite)."
            )
        import torch

        N = len(self.data)
        preds = {y: np.empty(N, dtype=np.float64) for y in self.y_parameter}
        for i in range(N):
            input_preles = self.prepare_preles_parameters(i, params=params)
            in_tensor = torch.tensor(input_preles, dtype=torch.float32, requires_grad=False)
            res = preles_rewrite.preles(*in_tensor)
            for y in self.y_parameter:
                preds[y][i] = float(
                    np.asarray(res[self._REWRITE_OUTPUT_ORDER[y]].detach().cpu().numpy()).squeeze()
                )
        return [preds[y] for y in self.y_parameter]

    def prepare_preles_parameters_batch(self, data_subset: pd.DataFrame, params: dict = None):
        """
        Build the argument tuple for preles_c.run(...) where:
        - time-series inputs + measurements + DOY are arrays of length N
        - all parameters (site/GPP/ET/snow/initials/phenology) are scalars
        - etmodel, LOGFLAG are plain ints
        """
        if not isinstance(data_subset, pd.DataFrame):
            raise TypeError("data_subset must be a pandas DataFrame")
        df = data_subset
        N = len(df)
        if N == 0:
            raise ValueError("data_subset is empty")

        # --- array helpers (float64 for drivers/measurements, int32 for DOY) ---
        def arr(col, default=None):
            if col in df.columns:
                return df[col].to_numpy(dtype=np.float64)
            if default is not None:
                return np.full(N, default, dtype=np.float64)
            raise KeyError(f"Missing column '{col}' and no default provided.")

        # time-series inputs
        PAR    = arr("PAR")
        TAir   = arr("Tair")
        VPD    = arr("VPD")
        Precip = arr("Precip")
        CO2    = arr("CO2")
        fAPAR  = arr("fapar")
        
        def meas(*cols):
            for col in cols:
                if col in df.columns:
                    a = df[col].to_numpy(dtype=np.float64)
                    # replace NaN with -999 sentinel
                    a = np.where(np.isnan(a), -999.0, a)
                    return a
            a = np.full(N, -999.0, dtype=np.float64)
            return a

        # measurements (fill with -999 if absent)
        GPPmeas = meas("GPP")
        ETmeas  = meas("ET")
        SWmeas  = meas("SWC", "SW")

        # DOY (int)
        day = (df["DOY"].to_numpy(dtype=np.int32)
            if "DOY" in df.columns else np.arange(1, N+1, dtype=np.int32))

        # --- scalar params from config (defaults < control < params override) ---
        P = {}
        if getattr(self, "default_parameters", None): P.update(self.default_parameters)
        if getattr(self, "control", None):            P.update(self.control)
        if params:                                    P.update(params)

        # Normalize alias if your config uses "I_0"
        if "I_0" in P and "I0" not in P:
            P["I0"] = P["I_0"]

        def p(name: str) -> float:
            if name not in P:
                raise KeyError(f"Missing parameter '{name}'")
            return float(P[name])

        # scalars only below
        soildepth   = p("soildepth");  
        ThetaFC     = p("ThetaFC");  
        ThetaPWP    = p("ThetaPWP");  
        tauDrainage = p("tauDrainage")
        beta = p("beta"); tau = p("tau"); S0 = p("S0"); Smax = p("Smax"); kappa = p("kappa"); gamma = p("gamma")
        soilthres = p("soilthres"); bCO2 = p("bCO2"); xCO2 = p("xCO2")
        ETbeta = p("ETbeta"); ETkappa = p("ETkappa"); ETchi = p("ETchi"); ETsoilthres = p("ETsoilthres"); ETnu = p("ETnu")
        MeltCoef = p("MeltCoef"); I0 = p("I0"); CWmax = p("CWmax"); SnowThreshold = p("SnowThreshold"); T_0 = p("T_0")
        
        # --- initial states ---
        # SWinit precedence: explicit override in params -> first df['SWC'/'SW'] value (if present & finite) -> config
        if params and "SWinit" in params:
            SWinit = float(params["SWinit"])
        elif "SWC" in df.columns and np.isfinite(df["SWC"].iloc[0]):
            SWinit = float(df["SWC"].iloc[0])
        elif "SW" in df.columns and np.isfinite(df["SW"].iloc[0]):
            SWinit = float(df["SW"].iloc[0])
        else:
            SWinit = p("SWinit")

        # other initials: keep config unless you also want df-derived seeds
        CWinit  = p("CWinit")
        SOGinit = p("SOGinit")
        Sinit   = p("Sinit")
        
        t0 = p("t0"); tcrit = p("tcrit"); tsumcrit = p("tsumcrit")

        etmodel = int(P.get("etmodel", 0))
        LOGFLAG = int(P.get("LOGFLAG", 0))

        # ---- final tuple for preles_c.run ----
        return (
            PAR, TAir, VPD, Precip, CO2, fAPAR,
            GPPmeas, ETmeas, SWmeas,
            day,
            soildepth, ThetaFC, ThetaPWP, tauDrainage,
            beta, tau, S0, Smax, kappa, gamma, soilthres, bCO2, xCO2,
            ETbeta, ETkappa, ETchi, ETsoilthres, ETnu,
            MeltCoef, I0, CWmax, SnowThreshold, T_0,
            SWinit, CWinit, SOGinit, Sinit,
            t0, tcrit, tsumcrit,
            etmodel, LOGFLAG,
        )
    
    def call_preles_batch(self, batch_inputs, collect_diag=False):
        if self.backend == "preles_c":
            if preles is None:
                raise RuntimeError(
                    "backend='preles_c' requested, but `preles_c` could not be imported."
                )
            out = preles.run(*batch_inputs)  # Returns a dict of output arrays.
            if "SWC" not in out and "SW" in out:
                out["SWC"] = out["SW"]
            output_parameter_order = {
                'GPP':0, 'ET':1, 'SWC':2, 'SOG':3, 'fS':4, 'fD':5, 'fW':6, 'fE':7,
                'Throughfall':8, 'Interception':9, 'Snowmelt':10, 'Drainage':11,
                'Canopywater':12, 'S':13, 'transp':14, 'evap':15, 'fWE':16
            }
        elif self.backend == "enzyme":
            from PM_Extension.prelesR.ADPRELES.call import OUTPUT_NAMES, call_PRELES_C

            weather, meas, day, params_vec, etmodel, logflag = self._to_enzyme_inputs(batch_inputs)
            y = call_PRELES_C(
                weather=weather,
                meas=meas,
                day=day,
                params=params_vec,
                ndays=weather.shape[0],
                etmodel=etmodel,
                logflag=logflag,
            )
            output_parameter_order = {name: idx for idx, name in enumerate(OUTPUT_NAMES)}
            out = {name: y[:, idx] for name, idx in output_parameter_order.items()}
        else:
            raise ValueError(
                f"Unsupported PRELES backend '{self.backend}'. Expected 'preles_c' or 'enzyme'."
            )

        if "SWC" not in out and "SW" in out:
            out["SWC"] = out["SW"]
        self._last_raw_outputs = out

        # If you have multiple outputs (e.g., GPP, ET), stack them
        outputs = [out[y] for y in self.y_parameter]
        
        # Stack outputs as (N_samples, N_outputs)
        # out_results = np.stack(outputs, axis=0)  # shape: (N_samples, N_outputs)
        if not collect_diag: 
            return outputs
        

    @staticmethod
    def _to_enzyme_inputs(batch_inputs):
        """
        Convert tuple from `prepare_preles_parameters_batch` to ADPRELES wrapper inputs:
        weather (N,6), meas (N,3), day (N,), params (30,), etmodel, logflag.
        """
        if len(batch_inputs) != 42:
            raise ValueError(
                f"Expected 42 PRELES batch inputs, got {len(batch_inputs)}."
            )

        PAR, TAir, VPD, Precip, CO2, fAPAR = [np.asarray(v, dtype=np.float64) for v in batch_inputs[0:6]]
        GPPmeas, ETmeas, SWmeas = [np.asarray(v, dtype=np.float64) for v in batch_inputs[6:9]]
        day = np.asarray(batch_inputs[9], dtype=np.int32)

        weather = np.column_stack([PAR, TAir, VPD, Precip, CO2, fAPAR]).astype(np.float64, copy=False)
        meas = np.column_stack([GPPmeas, ETmeas, SWmeas]).astype(np.float64, copy=False)

        params_vec = np.asarray(batch_inputs[10:40], dtype=np.float64)
        etmodel = int(batch_inputs[40])
        logflag = int(batch_inputs[41])
        return weather, meas, day, params_vec, etmodel, logflag

    def _simulate(self, params: dict = None):
        """
        Run preles_c/enzyme on the continuous daily series and return the outputs
        (in y_parameter order) at the kept rows only, aligned with self.data.
        """
        data_full = getattr(self, "data_full", self.data)
        keep = getattr(self, "_keep_mask", None)
        batch_inputs = self.prepare_preles_parameters_batch(data_full, params=params)
        outputs = [np.asarray(o).ravel() for o in self.call_preles_batch(batch_inputs)]
        if keep is not None:
            outputs = [o[keep] for o in outputs]
        return outputs

    def run_process_model(self, params: dict = None) -> pd.DataFrame:
        """
        Canonical process-model execution hook used by process-model-agnostic managers.
        Returns a DataFrame with x, y, y_pred, and site_id columns.
        """
        model_params = params if params is not None else self.default_parameters
        if self.backend in ("preles_rewrite", "torch"):
            outputs = self._rewrite_predict_rowwise(params=model_params)
        else:
            outputs = self._simulate(params=model_params)

        pred_map = {f"{k}_pred": np.asarray(v).ravel() for k, v in zip(self.y_parameter, outputs)}
        if self.extra_outputs:
            raw = self._last_raw_outputs
            if raw is None:
                raise ValueError(
                    f"Preles::extra_outputs={self.extra_outputs} is not supported for backend '{self.backend}'."
                )
            keep = getattr(self, "_keep_mask", None)
            for k in self.extra_outputs:
                v = np.asarray(raw[k]).ravel()
                pred_map[f"{k}_pred"] = v[keep] if keep is not None else v
        x_df = self.data[self.x_parameter].reset_index(drop=True)
        y_df = self.data[self.y_parameter].reset_index(drop=True)
        pred_df = pd.DataFrame(pred_map)
        site_col = (
            self.data["site_id"].reset_index(drop=True)
            if "site_id" in self.data.columns
            else pd.Series([self.station_id[0]] * len(self.data), name="site_id")
        )
        out = pd.concat([x_df, y_df, pred_df, site_col], axis=1)
        out = out.loc[:, ~out.columns.duplicated(keep="last")]
        return out

    # Backward-compatible name for local scripts
    def run_preles(self, params: dict = None) -> pd.DataFrame:
        return self.run_process_model(params=params)
        
     
class Preles_Calibrator:

    """Handles process model parameter calibration for preles"""

    def __init__(self, config,  train_dataset, test_dataset, pbounds, loss_log_file=None, train_calibration_file=None):
        """
        Initialize with train and test dataloaders.

        Parameters:
            train_dataset (New_Preles_data): Training data loader.
            pbounds (dict): Parameter bounds for calibration.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.pbounds = pbounds
        self.loss_log_file = loss_log_file
        self.train_calibration_file = train_calibration_file


        # Use attributes from train_dataset object
        self.default_parameters = train_dataset.default_parameters
        self.x_parameter = train_dataset.x_parameter
        self.y_parameter = train_dataset.y_parameter

        self.path_dir = train_dataset.path_dir
        self.device = train_dataset.device

        self.predictions = {}

        # 2x2 GPP/ET residual covariance for the MVN objective. Estimated ONCE at a
        # reference parameter set (see estimate_fixed_cov) and held fixed across all
        # trials. Re-estimating it per trial (the old behaviour) made the loss reduce
        # to minimizing log|cov|, which is scale-free per target and rewards a
        # degenerate GPP-ET correlation direction -> ET could be left unfit (~0).
        self._fixed_cov = None

        # Define `pbounds` only if not already set
        self.pbounds = {param_name: (param_info['min'], param_info['max'])
                        for param_name, param_info in pbounds.items()}
        
        years_map = self.config.get('data::train::station_id', None)
        if years_map:
            # Prefer station id from dataset if available
            sid = getattr(self.train_dataset, "station_id", None)
            if sid is None:
                # fallback: first key in the mapping
                sid = next(iter(years_map))
            sid = str(sid)
            years = years_map.get(sid, None)
            if years:
                self.time_index = self.build_daily_index_from_years(years)

    @staticmethod
    def _sanitize_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")

    def _build_study_name(self) -> str:
        """
        Build a stable Optuna study/database name from the calibration file path.
        Example:
        .../calibrations/preles/spatial/single_site/full/5_2013/params.json
        -> preles_spatial_single_site_full_5_2013
        """
        if not self.train_calibration_file:
            return "preles_default"

        cal_dir = os.path.dirname(os.path.abspath(self.train_calibration_file))
        split_sig = os.path.basename(cal_dir) or "split"
        sparse_lbl = os.path.basename(os.path.dirname(cal_dir)) or "full"
        scope = os.path.basename(os.path.dirname(os.path.dirname(cal_dir))) or "scope"
        cv_type = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(cal_dir)))) or "cv"
        pm_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cal_dir))))) or "preles"

        raw_name = f"{pm_name}_{cv_type}_{scope}_{sparse_lbl}_{split_sig}"
        return self._sanitize_name(raw_name)[:180]
        
    
    # -------- prediction packing / slicing loss --------
    def _pack_predictions(self):
        """
        Ensure predictions dict holds 1D numpy arrays aligned to df.index order.
        Call after run_preles_predictions.
        """
        for k in ["GPP_pred", "ET_pred", "obs_GPP", "obs_ET"]:
            arr = np.asarray(self.predictions[k])
            # handle shapes like (1, N) → (N,)
            self.predictions[k] = arr.ravel()

    def _resolve_cov(self, residuals):
        """
        Covariance to use in the MVN objective. Prefer the fixed covariance estimated
        once at a reference point; fall back to a per-call estimate only when no fixed
        covariance has been set (keeps behaviour safe if calibrate_with_optuna isn't
        the entry point).
        """
        if getattr(self, "_fixed_cov", None) is not None:
            return self._fixed_cov
        return np.cov(residuals, rowvar=False) + 1e-6 * np.eye(2)

    def estimate_fixed_cov(self, reference_params):
        """
        Estimate the 2x2 GPP/ET residual covariance ONCE at `reference_params` and
        cache it on self._fixed_cov. Holding the covariance fixed turns the MVN loss
        into a proper scale-anchored weighted SSE (Mahalanobis term now varies with
        fit quality), so neither target -- ET in particular -- can be ignored by the
        optimizer re-fitting the covariance each trial.
        """
        ref = copy.deepcopy(self)
        ref._fixed_cov = None
        ref.run_preles_predictions(reference_params)
        g_pred = np.asarray(ref.predictions['GPP_pred'], dtype=float).ravel()
        e_pred = np.asarray(ref.predictions['ET_pred'],  dtype=float).ravel()
        g_obs  = np.asarray(ref.predictions['obs_GPP'],  dtype=float).ravel()
        e_obs  = np.asarray(ref.predictions['obs_ET'],   dtype=float).ravel()
        m = np.isfinite(g_pred) & np.isfinite(e_pred) & np.isfinite(g_obs) & np.isfinite(e_obs)
        if m.sum() < 3:
            raise ValueError("estimate_fixed_cov: fewer than 3 finite reference rows.")
        resid = np.stack([g_obs[m] - g_pred[m], e_obs[m] - e_pred[m]], axis=1)
        cov = np.cov(resid, rowvar=False) + 1e-6 * np.eye(2)
        if not np.all(np.isfinite(cov)):
            raise ValueError(f"estimate_fixed_cov: non-finite covariance {cov}.")
        self._fixed_cov = cov
        print(f"[Calibration] Fixed residual covariance (var_GPP={cov[0,0]:.4g}, "
              f"var_ET={cov[1,1]:.4g}, rho={cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):.3f})")
        return cov

    def _objective_from_arrays(self, y_pred_gpp, y_pred_et, y_obs_gpp, y_obs_et):
        """
        Same math as objective_function, but taking arrays directly
        (so we can evaluate on sub-windows without re-simulating).
        Returns the scalar loss (negative total log-likelihood).
        """
        # Keep only rows where both predictions and observations are finite.
        finite_mask = (
            np.isfinite(y_pred_gpp) &
            np.isfinite(y_pred_et) &
            np.isfinite(y_obs_gpp) &
            np.isfinite(y_obs_et)
        )
        if finite_mask.sum() < 3:
            return np.inf

        y_pred_gpp = np.asarray(y_pred_gpp)[finite_mask]
        y_pred_et = np.asarray(y_pred_et)[finite_mask]
        y_obs_gpp = np.asarray(y_obs_gpp)[finite_mask]
        y_obs_et = np.asarray(y_obs_et)[finite_mask]

        # Stack into arrays of shape (N, 2)
        y_pred = np.stack([y_pred_gpp, y_pred_et], axis=1)
        y_obs  = np.stack([y_obs_gpp,  y_obs_et],  axis=1)

        # Residuals and covariance estimation
        residuals = y_obs - y_pred
        if not np.all(np.isfinite(residuals)):
            return np.inf
        cov_matrix = self._resolve_cov(residuals)
        if not np.all(np.isfinite(cov_matrix)):
            return np.inf

        total_log_likelihood = multivariate_normal.logpdf(
            residuals, mean=np.zeros(2), cov=cov_matrix, allow_singular=True
        ).sum()
        if not np.isfinite(total_log_likelihood):
            return np.inf

        # Monitor (optional): mimic main objective bookkeeping
        rmse_gpp = np.sqrt(np.mean((y_obs_gpp - y_pred_gpp)**2))
        rmse_et  = np.sqrt(np.mean((y_obs_et  - y_pred_et )**2))
        nrmse_gpp = rmse_gpp / (np.mean(y_obs_gpp) + 1e-6)
        nrmse_et  = rmse_et  / (np.mean(y_obs_et ) + 1e-6)
        rho = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

        # Optionally store window checks (comment out if noisy)
        # if not hasattr(self, "_window_checks"): self._window_checks = []
        # self._window_checks.append(dict(rmse_gpp=rmse_gpp, rmse_et=rmse_et, rho=rho))

        return -total_log_likelihood

    def compute_loss_from_arrays(self, preds_dict, window_slice):
        """
        Compute the scalar loss on a given integer slice (start:end+1).
        'preds_dict' should contain 1D arrays: GPP_pred, ET_pred, obs_GPP, obs_ET
        """
        if isinstance(window_slice, slice):
            sl = window_slice
        else:
            s, e = window_slice
            sl = slice(s, e + 1)

        y_pred_gpp = preds_dict["GPP_pred"][sl]
        y_pred_et  = preds_dict["ET_pred"][sl]
        y_obs_gpp  = preds_dict["obs_GPP"][sl]
        y_obs_et   = preds_dict["obs_ET"][sl]

        return self._objective_from_arrays(y_pred_gpp, y_pred_et, y_obs_gpp, y_obs_et)    

    def update_datasets(self, new_train_data): #, new_val_data
        """
        Updates the internal dataset references for a new iteration of cross-validation.

        Parameters:
            new_train_data (DataFrame): The new training dataset.
            new_val_data (DataFrame): The new validation dataset.
        """
        self.train_dataset.data = new_train_data
        # self.test_dataset.data = new_val_data
        print(f" Updated datasets: Train ({len(new_train_data)})") #, Val ({len(new_val_data)})

    def run_preles_predictions(self, params):
        """
        Runs PRELES for all data points using new BO parameters
        and stores the predictions.
        """
        # Store predictions externally
        self.predictions = {'GPP_pred': [], 'ET_pred': [], 'obs_GPP': [], 'obs_ET': []}

        y_obs = self.train_dataset.data[self.y_parameter]

        # Run PRELES (row-by-row rewrite, or batched preles_c on the continuous daily
        # series, scored on the kept rows only — see PrelesDataset._simulate)
        if getattr(self.train_dataset, "backend", "preles_c") in ("preles_rewrite", "torch"):
            preles_output = self.train_dataset._rewrite_predict_rowwise(params=params)
        else:
            preles_output = self.train_dataset._simulate(params=params)

        # Extract predictions
        y_pred_gpp = preles_output[0]
        y_pred_et = preles_output[1]

        # print(f"GPP_pred: {y_pred_gpp}")
        # print(f"ET_pred: {y_pred_et}")
        
        # Store results
        self.predictions['GPP_pred'] = np.asarray(y_pred_gpp).ravel()
        self.predictions['ET_pred']  = np.asarray(y_pred_et).ravel()
        self.predictions['obs_GPP']  = np.asarray(y_obs['GPP']).ravel()
        self.predictions['obs_ET']   = np.asarray(y_obs['ET']).ravel()

        # normalize internal shape
        self._pack_predictions()

    def objective_function(self, **params):
        """
        Objective function using joint probability distribution (multivariate normal)
        for calibrating correlated outputs.

        Args:
            predictions (dict): Dictionary of predicted values for each process.
            observations (dict): Dictionary of observed values for each process.

        Returns:
            float: Negative Total log-likelihood.
        """
        # Ensure predictions exist
        if not hasattr(self, 'predictions') or len(self.predictions['GPP_pred']) == 0:
            raise ValueError("Predictions have not been generated! Run `run_preles_predictions()` first.")

        # Extract predictions and observed values
        y_pred_gpp = np.asarray(self.predictions['GPP_pred'], dtype=float).ravel()
        y_pred_et  = np.asarray(self.predictions['ET_pred'],  dtype=float).ravel()
        y_obs_gpp  = np.asarray(self.predictions['obs_GPP'],  dtype=float).ravel()
        y_obs_et   = np.asarray(self.predictions['obs_ET'],   dtype=float).ravel()

        N = len(y_pred_gpp)
        if any(len(a) != N for a in (y_pred_et, y_obs_gpp, y_obs_et)):
            raise ValueError("Length mismatch among prediction/observation")

        finite_mask = (
            np.isfinite(y_pred_gpp) &
            np.isfinite(y_pred_et) &
            np.isfinite(y_obs_gpp) &
            np.isfinite(y_obs_et)
        )
        if finite_mask.sum() < 3:
            return np.inf

        y_pred_gpp = y_pred_gpp[finite_mask]
        y_pred_et = y_pred_et[finite_mask]
        y_obs_gpp = y_obs_gpp[finite_mask]
        y_obs_et = y_obs_et[finite_mask]
                         
        # Stack into arrays of shape (N, 2)
        y_pred = np.stack([y_pred_gpp, y_pred_et], axis=1)
        y_obs = np.stack([y_obs_gpp, y_obs_et], axis=1)

        ## Residuals and covariance estimation
        residuals = y_obs - y_pred
        if not np.all(np.isfinite(residuals)):
            return np.inf
        cov_matrix = self._resolve_cov(residuals)  # fixed cov if set; else per-call estimate
        if not np.all(np.isfinite(cov_matrix)):
            return np.inf

        # Compute multivariate log-likelihoods
        total_log_likelihood = multivariate_normal.logpdf(residuals, mean=np.zeros(2), cov=cov_matrix, allow_singular=True).sum()
        if not np.isfinite(total_log_likelihood):
            return np.inf

        # Compute Mean Squared Error (MSE) for GPP and ET
        mse_gpp = np.mean((y_obs_gpp - y_pred_gpp) ** 2)
        mse_et = np.mean((y_obs_et - y_pred_et) ** 2)

        # Compute RMSE for GPP and ET
        rmse_gpp = np.sqrt(np.mean((y_obs_gpp - y_pred_gpp) ** 2))
        rmse_et = np.sqrt(np.mean((y_obs_et - y_pred_et) ** 2))

        # Normalize RMSE (so it scales similarly to log-likelihood)
        mean_gpp = np.mean(y_obs_gpp) + 1e-6  
        mean_et = np.mean(y_obs_et) + 1e-6

        nrmse_gpp = rmse_gpp / mean_gpp  
        nrmse_et = rmse_et / mean_et  

        # monititor empirical calibration
        rho = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

        # # Store loss history for tracking
        if not hasattr(self, "loss_history"):
            self.loss_history = []
        self.loss_history.append({
            "params": params,
            "rho": rho,
            "covariance": cov_matrix,
            "total_log_likelihood": total_log_likelihood,
            # "log_likelihood_gpp": log_likelihood_gpp,
            # "log_likelihood_et": log_likelihood_et,
            "mse_gpp": mse_gpp,
            "mse_et": mse_et,
            "rmse_gpp": rmse_gpp,
            "rmse_et": rmse_et,
            "nrmse_gpp": nrmse_gpp,
            "nrmse_et": nrmse_et,

        })

        # This BO maximizes
        return -total_log_likelihood

    def calibrate_with_optuna(self, n_trials=1000, timeout=None):
        """
        Uses Optuna's CmaEsSampler to calibrate parameters.
        """
        
        def _save_dataframe_csv(df, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=True)

        def _oaat_sensitivity(model_prototype, best_params, 
                            rel_step=0.1, 
                            abs_step_small=("ThetaFC","ThetaPWP","soilthres","ETsoilthres"),
                            warmup_days=0):
            """
            Returns a DataFrame with mean |ΔMSE| per parameter.
            Uses a deep copy of model_prototype to avoid side effects.
            Optionally, you can add a warm-up if your model supports it.
            """

            # Work on a fresh model copy to avoid contaminating outer state
            model = copy.deepcopy(model_prototype)

            # Base loss at best params
            model.run_preles_predictions(best_params)
            base_loss = model.objective_function(**best_params)

            out = []
            for k, v in best_params.items():
                # Skip non-finite or derived params
                if not np.isfinite(v): 
                    continue
                # Choose step size
                step = rel_step * abs(v) if abs(v) > 0 else rel_step
                if k in abs_step_small:
                    step = max(step, 0.02)  # small absolute for fractional in [0,1]
                # Perturb ±
                for sign in (+1, -1):
                    p = dict(best_params)
                    p[k] = max(1e-9, v + sign*step)  # keep positive-only params positive
                    # Fresh model copy for each run prevents state carry-over
                    m = copy.deepcopy(model_prototype)
                    m.run_preles_predictions(p)
                    loss = m.objective_function(**p)
                    out.append((k, sign, loss - base_loss))

            df = pd.DataFrame(out, columns=["param","sign","+dMSE"])
            s = df.groupby("param")["+dMSE"].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
            return s.to_frame("mean_|ΔMSE|")
        
        def _get_bounds(pbounds, name, *, required=True):
            if name not in pbounds:
                if required:
                    raise KeyError(f"'{name}' not found in model.pbounds")
                return None
            b = pbounds[name]
            if isinstance(b, dict):
                return float(b["min"]), float(b["max"])
            elif isinstance(b, (tuple, list)) and len(b) == 2:
                return float(b[0]), float(b[1])
            else:
                raise ValueError(f"Unexpected bounds format for '{name}': {b}")

        def objective(trial):
            """
            One Optuna trial: sample params → run PRELES → return scalar loss.
            Thread/process-safe and robust against NaNs/Inf.
            """
            model = copy.deepcopy(self)          # cheap: only Python objects, no data files

            params = {}
            LOG_PARAMS = {"gamma", "ETbeta", "ETkappa", "ETnu", "tauDrainage"}
            pbounds = model.pbounds
            
            # skip = {"ThetaFC", "soildepth"} #"Smax"
            for name, bounds in pbounds.items():
                # if name in skip:
                #     continue
                low, high = _get_bounds(pbounds, name)
                # params[name] = trial.suggest_float(name, low, high)
                use_log = name in LOG_PARAMS
                params[name] = trial.suggest_float(name, low, high, log=use_log)
            
            # Custom metadata for tracking
            trial.set_user_attr("params", params)  # optional, for easier inspection/debugging

            try:
                model.run_preles_predictions(params)
                loss = model.objective_function(**params)
            except Exception as exc:
                trial.set_user_attr("failure_reason", repr(exc))
                raise optuna.exceptions.TrialPruned()

            if not np.isfinite(loss):
                raise optuna.exceptions.TrialPruned()
            
            trial.set_user_attr("params", params)
            # trial.set_user_attr("PAW", float(PAW))
            trial.set_user_attr("rho", model.loss_history[-1]["rho"])

            return loss # You minimize RMSE or -log_likelihood
        
        def save_best_callback(study, trial):
            # Save best trial params so far
            if trial.number % 100 != 0:
                return
            try:
                if not study.best_trials:
                    return
                best_trial = min(study.best_trials, key=lambda t: sum(t.values))

                # Original path to JSON file
                file_name = self.train_calibration_file

                # Extract base folder and filename
                base_folder = os.path.dirname(file_name)
                filename = os.path.basename(file_name).replace(".json", "_best_params_temp.json")

                # Define subfolder for history
                loss_subfolder = os.path.join(base_folder, "loss_history")
                loss_log_file = os.path.join(loss_subfolder, filename)
                os.makedirs(loss_subfolder, exist_ok=True)

                with open(loss_log_file, 'w') as f:
                    json.dump(best_trial.params, f, indent=4)
                    print(f"[INFO] Callback saved current best params at trial {trial.number} to: {loss_log_file}")
            except Exception as e:
                print(f"[WARNING] Callback failed to save best trial at trial {trial.number}: {e}")

        STUDY_NAME = self._build_study_name()
        project_root = os.path.dirname(os.path.abspath(__file__))
        db_dir = os.path.join(project_root, "Optuna_DBs", "preles")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, f"{STUDY_NAME}.db")
        rdb_string_url = f"sqlite:///{db_file}"

        # The study is opened with load_if_exists=True, so an existing DB means these
        # trials are ADDED to an old study and best_trial is picked across both, even
        # when the objective changed in between. Recalibration campaigns set this to
        # refuse that instead of silently mixing.
        if os.getenv("CALIBRATION_REQUIRE_FRESH_STUDY") == "1" and os.path.exists(db_file):
            raise RuntimeError(
                f"CALIBRATION_REQUIRE_FRESH_STUDY=1 but an Optuna study already exists: {db_file}. "
                "Move it aside to start this calibration from scratch."
            )
    
        storage_instance = optuna.storages.RDBStorage(
                url=rdb_string_url,
                heartbeat_interval=60,
                grace_period=180,
                engine_kwargs={"connect_args": {"timeout": 300}})

        # Fixed seed -> reproducible CMA-ES trajectory for the (deterministic) model.
        seed = int(self.config.get("calibration::seed", 42))

        # Warm-start point: the literature/default parameter values, clipped into the
        # tuning bounds. CMA-ES samples around x0 instead of the box centre, so the
        # search starts from a known-good point and won't end up worse than defaults.
        defaults = getattr(self, "default_parameters", None) or {}
        x0 = {}
        for name, (low, high) in self.pbounds.items():
            val = defaults.get(name)
            if val is None or not np.isfinite(val):
                val = 0.5 * (low + high)
            x0[name] = float(min(max(float(val), low), high))

        # Estimate the MVN covariance ONCE at the warm-start (default) parameters and
        # hold it fixed for every trial. This is what keeps ET from collapsing: with a
        # fixed covariance the loss is a proper scale-anchored weighted SSE rather than
        # a re-fit log|cov| that ignores absolute ET error. Each trial deep-copies self,
        # so setting it here propagates to all trials.
        self.estimate_fixed_cov(x0)

        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage_instance,
            direction="minimize",
            sampler=CmaEsSampler(x0=x0, seed=seed, n_startup_trials=1),
            pruner=NopPruner(),
            load_if_exists=True,
        )

        # Always evaluate the defaults explicitly so the best trial can't be worse than them.
        study.enqueue_trial(x0, skip_if_exists=True)

        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[save_best_callback], gc_after_trial=True)
        except Exception as e:
            print(f"[WARNING] Optimization interrupted: {e}")
            
        completed_trials = [trial for trial in study.trials if trial.value is not None]
        if not completed_trials:
            failure_reasons = [
                trial.user_attrs.get("failure_reason")
                for trial in study.trials
                if trial.user_attrs.get("failure_reason")
            ]
            first_reason = failure_reasons[0] if failure_reasons else "unknown"
            raise RuntimeError(
                "Optuna finished without any completed trials. "
                f"First recorded failure reason: {first_reason}"
            )

        # Store results
        best_trial = min(completed_trials, key=lambda t: sum(t.values))
        best_params = best_trial.params
        best_loss = best_trial.value

        print(f"Best Calibration combined loss: {best_loss:.5f}")
        self.study = study  # Store study for later analysis (e.g., plots)
        
        
        # # --- Sensitivity on best params ---
        sens_df = _oaat_sensitivity(self, best_params, rel_step=0.1)
        
        
        # Save next to your loss_history
        history_dir = os.path.join(os.path.dirname(self.train_calibration_file), "loss_history")
        os.makedirs(history_dir, exist_ok=True)
        
        sens_path_csv = os.path.join(history_dir, os.path.basename(self.train_calibration_file).replace(".json", "_oaat_sensitivity.csv"))
        
        # sens_win_csv = os.path.join(history_dir,
        #     os.path.basename(self.train_calibration_file).replace(".json", "_oaat_windowed.csv"))
        
       
        # _save_dataframe_csv(sens_wind_df, sens_win_csv)
        print("[Sensitivity] Top leverage parameters:\n", sens_df.head(10))
        print(f"[Sensitivity] Saved OAAT sensitivity to: {sens_path_csv}")
         # _save_dataframe_csv(sens_df, sens_path_csv)

        # --- Global parameter importance (fANOVA over the trial history) ---
        # Two complementary views, because they answer different questions:
        #   * full_study  -> which parameter, set badly, dominates the loss (ruin-the-run)
        #   * best_<q>pct -> which parameters matter for fine-tuning near the optimum
        # Importance is conditional on the sampled distribution and the objective: a low
        # score means "did not drive loss variance here", NOT "physically irrelevant".
        def _param_importance(study, best_frac=0.25):
            import optuna.importance as _imp
            evaluator = _imp.FanovaImportanceEvaluator(seed=seed)
            done = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
            frames = []

            def _rank(trials, label):
                # fANOVA needs a few trials and >1 varying param to be meaningful.
                if len(trials) < 4 or len({p for t in trials for p in t.params}) < 2:
                    return
                sub = optuna.create_study(direction="minimize")
                sub.add_trials(trials)
                ranking = _imp.get_param_importances(sub, evaluator=evaluator)
                frames.append(pd.DataFrame(
                    {"param": list(ranking), "importance": list(ranking.values()), "scope": label}
                ))

            _rank(done, "full_study")
            if done:
                cutoff = np.percentile([t.value for t in done], 100 * best_frac)
                _rank([t for t in done if t.value <= cutoff], f"best_{int(best_frac * 100)}pct")
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["param", "importance", "scope"]
            )

        try:
            imp_df = _param_importance(study)
            if not imp_df.empty:
                imp_path_csv = os.path.join(
                    history_dir,
                    os.path.basename(self.train_calibration_file).replace(".json", "_param_importance.csv"),
                )
                imp_df.to_csv(imp_path_csv, index=False)
                _top = (imp_df[imp_df["scope"] == "full_study"]
                        .sort_values("importance", ascending=False).head(5))
                print("[Importance] Top fANOVA parameters (full study):")
                for _, _r in _top.iterrows():
                    print(f"    {_r['param']:14s} {_r['importance']:.3f}")
                print(f"[Importance] Saved parameter importance to: {imp_path_csv}")
        except Exception as exc:
            print(f"[WARNING] Parameter-importance computation failed: {exc!r}")


        # Extract Optuna trial history
        history_records = []
        for trial in study.trials:
            if trial.value is None:
                continue  # skip failed or pruned trials
            record = trial.params.copy()
            record["loss"] = trial.value
            record["trial_number"] = trial.number
            record["state"] = trial.state.name
            history_records.append(record)

        # Convert to DataFrame
        history_df = pd.DataFrame(history_records)

        # Original path to JSON file
        history_log_file = self.train_calibration_file

        # Extract base folder and filename
        base_folder = os.path.dirname(history_log_file)
        filename = os.path.basename(history_log_file).replace(".json", "_loss_history.xlsx")

        # Define subfolder for history
        history_subfolder = os.path.join(base_folder, "loss_history")
        os.makedirs(history_subfolder, exist_ok=True)  # Create the subfolder if it doesn't exist

        # Full path to new history log file
        history_log_file = os.path.join(history_subfolder, filename)
        
        history_df.to_excel(history_log_file, index=False)
        print(f"[Optuna] Saved full calibration history to: {history_log_file}")

        return {
            "params": best_params,
            "loss": best_loss,
            "n_trials_requested": n_trials,
            "n_trials_in_study": len(completed_trials),
            "seed": seed,
        }


# Canonical adapter-style names for the ongoing process-model refactor.
# Keep existing class names for backward compatibility.
class PrelesDataset(Preles_dataset):
    pass


class PrelesCalibrator(Preles_Calibrator):
    pass
    

      
def plot_preles_predictions(data_object, title_prefix="Train", save_path=None, x_para=None, y_para=None):
    """
    Plot actual vs predicted targets along with selected input variables (x_para).
    
    Parameters:
    - data_object: Dataset object with `.data` and `.calibrated_df`
    - title_prefix: String to prefix in plot titles (e.g., 'Train', 'Test')
    - save_path: Optional path to save the figure as PNG
    - x_para: List of strings indicating which input parameters to plot (e.g., ['PAR', 'TAir'])
    - y_para: List of strings indicating which output parameters to plot (e.g., ['GPP', 'ET'])
    """
    df = data_object.data.copy()
    y_para = list(y_para or ["GPP", "ET"])
    missing_y_cols = [
        col
        for tgt in y_para
        for col in (tgt, f"{tgt}_pred")
        if col not in df.columns
    ]
    if missing_y_cols:
        raise KeyError(f"Missing prediction columns for plotting: {missing_y_cols}")

    # Use Date if available
    x_axis = df.index

    y_true = np.stack([df[tgt].values for tgt in y_para], axis=1)
    y_pred = np.stack([df[f"{tgt}_pred"].values for tgt in y_para], axis=1)

    # Flatten to match NN.MSELoss() behavior (average over all elements)
    combined_mse = mean_squared_error(y_true.flatten(), y_pred.flatten())

    # Plotting
    per_target_mse = {
        tgt: mean_squared_error(df[tgt], df[f"{tgt}_pred"])
        for tgt in y_para
    }
    mse_summary = ", ".join(f"MSE {tgt}: {mse:.4f}" for tgt, mse in per_target_mse.items())
    print(f"[{title_prefix}] {mse_summary}, Combined MSE: {combined_mse:.4f}")

    # Prepare figure layout
    n_xpara = len(x_para) if x_para else 0
    n_targets = len(y_para)
    total_plots = n_targets + n_xpara
    fig, axs = plt.subplots(total_plots, 1, figsize=(12, 2 * total_plots), sharex=True)
    axs = np.atleast_1d(axs)

    for idx, tgt in enumerate(y_para):
        y_true_tgt = df[tgt]
        y_pred_tgt = df[f"{tgt}_pred"]
        mse_tgt = per_target_mse[tgt]
        axs[idx].plot(x_axis, y_true_tgt, label=f"{tgt} True", alpha=0.7)
        axs[idx].plot(x_axis, y_pred_tgt, label=f"{tgt} Preles", alpha=0.7, linestyle="--")
        axs[idx].set_ylabel(tgt)
        axs[idx].legend()
        axs[idx].set_title(f"{title_prefix} {tgt} (MSE: {mse_tgt:.4f})")
    
    for idx, para in enumerate(x_para or []):
        if para in df.columns:
            axs[idx + n_targets].plot(x_axis, df[para], label=f"{para}", color="tab:gray")
            axs[idx + n_targets].set_ylabel(para)
            axs[idx + n_targets].legend()
        else:
            axs[idx + n_targets].text(0.5, 0.5, f"{para} not found", transform=axs[idx + n_targets].transAxes,
                                      ha="center", va="center", color="red")

    # Optional: Add combined MSE to the figure's supertitle
    fig.suptitle(f"{title_prefix} Predictions (Combined MSE: {combined_mse:.4f})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300)

    # plt.show()

def get_ncv_version(cv_config):
    keys = cv_config.keys()
    if any(k.startswith("out_l_") for k in keys) and "station" in cv_config:
        return 'temporal'
    elif any(k.startswith("outer_station_") for k in keys):
        return 'spatial'
    else:
        raise ValueError("Could not infer NCV version from cv_config keys")

def get_num_train_years(data_split):
    """Count unique training years in this split."""
    years = data_split.get("train", [])
    return len(set(years))

def get_base_dir(base_dir, NCV_version, n_train_years=None, data_sparsity=None):
    if NCV_version == "temporal" and n_train_years is not None:
        suffix = f"years{n_train_years}"
    else:
        suffix = ""
    sparsity_str = f"sparse{data_sparsity}" if data_sparsity is not None else "full"
    return os.path.join(base_dir, f"{NCV_version}NCV_{suffix}_{sparsity_str}")


if __name__=='__main__':
    # from Config_Files.config_spatial import build_config     ## explicitly set which experiment configuration file you wish to use
    
    # config = build_config()
    # # Load environment variables from .env file
    # load_dotenv()

    cfg = {
        "process_model::name":"preles",
        "model::prediction_type":"spatial",
        "calibration::cv_type":"spatial",
        "calibration::training_years_len":"len3",
        "calibration::root":"calibrations",
        "data:train:sparse":None
    }
    station_id = {"12":[2008,2009]}
    paths = build_calibration_paths(cfg, station_id, process_model="preles")
    meta = build_calibration_metadata(cfg, station_id, process_model="preles")
    save_calibration_bundle(paths, {"beta":0.7}, meta)
    p,m = load_calibration_bundle(paths)
    ctx = build_expected_metadata_context(cfg, station_id, process_model="preles")
    print(paths["params_file"])
    print(is_metadata_compatible(m, ctx))

    
    # # Calibration
    # N_TRIALS = int(os.getenv("N_trials"))
    # training_years_len = str(os.getenv("temporal_training_years_len1")) 

    # save = config["cfg::save"]
    # sparse = config["data:train:sparse"]
    # print(f"[INFO] Sparcity set to {sparse}")

    # CV_CONFIG = save +'configs/CV_splits/CV_station_len9_config.json'                                   # spatial NCV config
    # CV_CONFIG = save + 'configs/CV_splits/temporal_cv_train_len1_station_5.json'
    # CV_CONFIG = save + 'new_temporal_cv_train_len1_station_12.json' # temporal NCV config

    ## ------------ Only relevant when wanting to start this file via the cluster - see batch_calibration.sh file ------------ 
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--outer_fold', type=str)
    # parser.add_argument('--inner_loop', type=str)
    # parser.add_argument("--worker_id",  type=int,        help="Unique ID for the worker (optional)")
    # parser.add_argument('--config',     type=str,        help="Path to frozen config JSON file")
    # parser.add_argument('--cv_config',  type=str,        help="Path to CV config JSON file")
    # args = parser.parse_args()

    # ## Load config files    
    # if not os.path.exists(args.config):
    #     raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # with open(args.config, "r") as f:
    #     config = json.load(f)
        
    # save = config["cfg::save"]
    # sparse = config["data:train:sparse"]
    # print(f"[INFO] Sparcity set to {sparse}")

    # manager = Dataset_Calibration_Manager(config, cv_config_path=args.cv_config, calibrate=True)
    # manager.run_single_fold_calibration(args.outer_fold, args.inner_loop)
    # train_data, test_data = manager.get_train_test_set(config, return_raw=True)
    # # -----------------------------------------------------------------------------------------------------------------------
    
    # ## ------------ If you want to run/load a single fold split from the NCV config for analysis --------------------------
    # outer_fold_key = "outer_station_14"                  # set which fold you want 'outer_station_14' or 'out_l_1
    # inner_loop_key = None                      # 'inner_loop_3' for spatial or 'inner_loop1' for temporal

    # manager = Preles_Calibration_Manager(config, cv_config_path=CV_CONFIG, calibrate=True)
    # manager.run_single_fold_calibration(outer_fold_key, inner_loop_key)
    # train_data, test_data = manager.get_train_test_set(config, return_raw=True)
    
    # train_plot_path = os.path.join(save, f"{outer_fold_key}_train.png")
    # test_plot_path = os.path.join(save, f"{outer_fold_key}_test.png")
    
    # plot_preles_predictions(train_data, title_prefix=f"{outer_fold_key}| {inner_loop_key} (Train)", save_path=train_plot_path)
    # plot_preles_predictions(test_data, title_prefix=f"{outer_fold_key}| {inner_loop_key} (Test)", save_path=test_plot_path,)
    # ## ---------------------------------------------------------------------------------------------------


    # ## ------------ If you want to run/load a dataset from the config file --------------------------
    # dataset_manager = Dataset_Calibration_Manager(config, cv_config_path=CV_CONFIG, calibrate=True)  # Initialize DatasetManager for calibration
    # dataset_manager.load_next_split()
    # train_data, test_data = dataset_manager.get_train_test_set(config, return_raw=True)
    

    # train_data.data.to_csv('[Name].csv', index=False)
    ## ---------------------------------------------------------------------------------------------------
    # manager = Dataset_Calibration_Manager(config, cv_config_path=CV_CONFIG, calibrate=True)
    # manager.plot_parameter_calibration_metrics(
    #     calibration_log_path="Data/new_calibrations/loss_history/5_2009_loss_history.xlsx",
    #     save_path=save + "5_2009_plot_parameter.png"
    # )
