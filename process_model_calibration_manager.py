import argparse
import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from PRELES_dataset import plot_preles_predictions
from Config_Files.config_loader import merge_process_model_config, SUPPORTED_PROCESS_MODELS
from calibration_paths import (
    build_calibration_metadata,
    build_calibration_paths,
    build_expected_metadata_context,
    is_metadata_compatible,
    load_calibration_bundle,
    save_calibration_bundle,
)

_BASFOR_VARIANT_ALIASES = {
    "conifer": "conifer",
    "conif": "conifer",
    "coniferous": "conifer",
    "needleleaf": "conifer",
    "evergreen_needleleaf": "conifer",
    "deciduous": "deciduous",
    "decid": "deciduous",
    "broadleaf": "deciduous",
    "deciduous_broadleaf": "deciduous",
}


class ProcessModelCalibrationManager:
    """
    Process-model-agnostic calibration/data manager.

    Uses process-model adapters internally (PRELES/BASFOR/ other) and unified calibration
    checks via metadata + path policy.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        process_model: Optional[str] = None,
        auto_merge_pm_config: bool = False,
        pm_config_module: Optional[str] = None,
        cv_config: Optional[Dict[str, Any]] = None,
        cv_config_path: Optional[str] = None,
        outer_fold: Optional[str] = None,
        inner_loop: Optional[str] = None,
    ):
        self.process_model = (process_model or config.get("process_model::name") or "preles").lower()
        self.config = (
            merge_process_model_config(
                base_config=config,
                process_model=self.process_model,
                module_path=pm_config_module,
            )
            if auto_merge_pm_config
            else dict(config)
        )
        artifact_namespace = os.getenv("CALIBRATION_ARTIFACT_NAMESPACE")
        if artifact_namespace and "calibration::artifact_namespace" not in self.config:
            self.config["calibration::artifact_namespace"] = artifact_namespace
        self.debug = bool(self.config.get("calibration::debug", False))
        self.train_path = self.config["data::train::path_dir"]
        self.test_path = self.config["data::test::path_dir"]
        self.val_path = self.config.get("data::val::path_dir", self.test_path)

        if self.process_model not in SUPPORTED_PROCESS_MODELS:
            raise ValueError(
                f"Unsupported process_model '{self.process_model}'. Expected one of {SUPPORTED_PROCESS_MODELS}."
            )
        self._validate_required_config()
        self._bootstrap_station_context(
            cv_config=cv_config,
            cv_config_path=cv_config_path,
            outer_fold=outer_fold,
            inner_loop=inner_loop,
        )

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[PM Manager][{self.process_model}] {msg}")

    @staticmethod
    def _normalize_mode(calibrate: Any) -> str:
        if isinstance(calibrate, bool):
            return "auto" if calibrate else "defaults"
        if isinstance(calibrate, str):
            mode = calibrate.lower()
            if mode in {"auto", "defaults", "force_calibrate", "require_existing"}:
                return mode
        raise ValueError("calibrate must be bool or one of {'auto','defaults','force_calibrate','require_existing'}")

    def _resolve_sparse(self, split: str) -> Any:
        # split: "train" | "test"
        return self.config.get(f"data:{split}:sparse", self.config.get("data:train:sparse"))

    @staticmethod
    def _normalize_basfor_variant(value: Any, fallback: Optional[str] = "conifer") -> Optional[str]:
        if value is None:
            return fallback
        raw = str(value).strip().lower()
        mapped = _BASFOR_VARIANT_ALIASES.get(raw)
        if mapped is not None:
            return mapped
        return fallback

    def _resolve_basfor_variant_from_stations(self, station_ids: Dict[str, Any]) -> Optional[str]:
        if self.process_model != "basfor":
            return None
        lib_cfg = dict(self.config.get("Basfor::library", {}))
        mode = str(lib_cfg.get("selection_mode", "site_map")).strip().lower()
        default_variant = self._normalize_basfor_variant(lib_cfg.get("default_variant", "conifer"))
        mixed_variant = self._normalize_basfor_variant(lib_cfg.get("mixed_variant", "conifer"), fallback=default_variant)

        if mode != "site_map":
            return default_variant

        site_map = dict(self.config.get("Basfor::site_forest_type", {}))
        types = set()
        unknown_sites = []
        for station in station_ids:
            ftype = self._normalize_basfor_variant(site_map.get(station), fallback=None)
            if ftype is None:
                unknown_sites.append(str(station))
            else:
                types.add(ftype)

        if unknown_sites:
            self._dbg(
                f"missing/unknown Basfor::site_forest_type for stations {unknown_sites}; "
                f"using default_variant='{default_variant}'"
            )
            return default_variant
        if len(types) == 1:
            return next(iter(types))
        if len(types) > 1:
            return mixed_variant
        return default_variant

    @staticmethod
    def _normalize_basfor_site_type(value: Any, fallback: Optional[str] = None) -> Optional[str]:
        if value is None:
            return fallback
        raw = str(value).strip().lower()
        if raw in {"mixed", "mixed_forest", "mf"}:
            return "mixed"
        return _BASFOR_VARIANT_ALIASES.get(raw, fallback)

    def _resolve_basfor_variant_from_site_metadata(self, station_ids: Dict[str, Any]) -> Optional[str]:
        if self.process_model != "basfor":
            return None

        lib_cfg = dict(self.config.get("Basfor::library", {}))
        default_variant = self._normalize_basfor_variant(lib_cfg.get("default_variant", "conifer"))
        mixed_variant = self._normalize_basfor_variant(
            lib_cfg.get("mixed_variant", "conifer"),
            fallback=default_variant,
        )
        metadata_map = dict(self.config.get("Basfor::site_metadata", {}))

        types = set()
        unknown_sites = []
        for station in station_ids:
            metadata = metadata_map.get(station) or metadata_map.get(str(station))
            raw_site_type = metadata.get("variant") if isinstance(metadata, dict) else None
            site_type = self._normalize_basfor_site_type(raw_site_type, fallback=None)
            if site_type is None:
                unknown_sites.append(str(station))
            else:
                types.add(site_type)

        if unknown_sites:
            self._dbg(
                f"missing/unknown Basfor::site_metadata variant for stations {unknown_sites}; "
                f"using default_variant='{default_variant}'"
            )
            return default_variant
        if len(types) == 1:
            resolved = next(iter(types))
            if resolved == "mixed":
                return mixed_variant
            return resolved
        if "mixed" in types or len(types) > 1:
            return mixed_variant
        return default_variant

    def _resolve_basfor_split_variant_overrides(
        self,
        train_stations: Dict[str, Any],
        test_stations: Dict[str, Any],
        val_stations: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        if self.process_model != "basfor":
            return None, None, None, None

        lib_cfg = dict(self.config.get("Basfor::library", {}))
        selection_mode = str(lib_cfg.get("selection_mode", "site_map")).strip().lower()
        cv_override = getattr(self, "cv_forest_type_override", None)

        if selection_mode == "outer_fold_test_site":
            fold_variant = self._resolve_basfor_variant_from_site_metadata(test_stations)
            self._dbg(
                f"BASFOR variant resolved from outer-fold test site metadata -> "
                f"'{fold_variant}' (applied to train/test/val)."
            )
            return fold_variant, fold_variant, fold_variant, fold_variant

        train_model_variant_override = None
        test_model_variant_override = None
        val_model_variant_override = None

        if cv_override:
            train_model_variant_override = self._normalize_basfor_variant(cv_override)
            test_model_variant_override = train_model_variant_override
            val_model_variant_override = train_model_variant_override
            self._dbg(
                f"BASFOR variant from CV config forest_type='{cv_override}' -> "
                f"'{train_model_variant_override}' (applied to train/test/val)."
            )
        else:
            train_model_variant_override = self._resolve_basfor_variant_from_stations(train_stations)
            if self._should_lock_basfor_variant_to_train():
                test_model_variant_override = train_model_variant_override
                val_model_variant_override = train_model_variant_override

        if self._should_lock_basfor_variant_to_train():
            self._dbg(
                f"BASFOR variant lock enabled -> using train variant "
                f"'{train_model_variant_override}' for train/test/val."
            )
        else:
            self._dbg(
                "BASFOR variant lock disabled -> train uses the training-resolved "
                f"variant '{train_model_variant_override}', while test/val resolve "
                "from their own station sets unless a CV forest_type override is set."
            )

        return (
            train_model_variant_override,
            test_model_variant_override,
            val_model_variant_override,
            train_model_variant_override,
        )

    def _should_lock_basfor_variant_to_train(self) -> bool:
        if self.process_model != "basfor":
            return False
        lib_cfg = dict(self.config.get("Basfor::library", {}))
        return bool(lib_cfg.get("lock_variant_to_train", False))

    @staticmethod
    def _as_station_map(value: Any, key: str) -> Dict[str, Any]:
        """
        Validate and normalize a station mapping payload.

        Expected shape: {station_id: years_or_date_ranges}.
        """
        if not isinstance(value, dict):
            raise TypeError(f"Expected '{key}' to be a dict mapping station -> years/date-ranges. Got: {type(value)}")
        return dict(value)

    def _set_station_context(
        self,
        train_stations: Dict[str, Any],
        test_stations: Dict[str, Any],
        val_stations: Optional[Dict[str, Any]],
    ) -> None:
        """
        Persist active train/test/val station splits on the manager and in `self.config`.

        This ensures dataset loading and calibration metadata use the same split context.
        """
        self.train_stations = dict(train_stations)
        self.test_stations = dict(test_stations)
        self.val_stations = dict(val_stations) if val_stations is not None else None

        self.config["data::train::station_id"] = self.train_stations
        self.config["data::test::station_id"] = self.test_stations
        if self.val_stations is None:
            self.config.pop("data::val::station_id", None)
        else:
            self.config["data::val::station_id"] = self.val_stations

    def _bootstrap_station_context(
        self,
        cv_config: Optional[Dict[str, Any]],
        cv_config_path: Optional[str],
        outer_fold: Optional[str],
        inner_loop: Optional[str],
    ) -> None:
        """
        Initialize station context from CV config (if provided) or base config.

        Resolution order:
        1) `cv_config` dict
        2) `cv_config_path` JSON
        3) `config['data::train::station_id']` / `config['data::test::station_id']`
        """
        if cv_config is None and cv_config_path:
            with open(cv_config_path, "r", encoding="utf-8") as f:
                cv_config = json.load(f)

        if cv_config is not None:
            train, test, val, used_outer, used_inner, cv_forest_type = _resolve_from_cv(
                cv_config=cv_config,
                outer_fold=outer_fold,
                inner_loop=inner_loop,
            )
            self.cv_outer_fold = used_outer
            self.cv_inner_loop = used_inner
            self.cv_forest_type_override = cv_forest_type
            self._set_station_context(train, test, val)
            self._dbg(f"station context from cv_config outer={used_outer} inner={used_inner}")
            return

        train = self._as_station_map(self.config["data::train::station_id"], "data::train::station_id")
        test = self._as_station_map(self.config["data::test::station_id"], "data::test::station_id")
        val_raw = self.config.get("data::val::station_id")
        val = self._as_station_map(val_raw, "data::val::station_id") if val_raw is not None else None
        self.cv_outer_fold = None
        self.cv_inner_loop = None
        self.cv_forest_type_override = None
        self._set_station_context(train, test, val)

    def _calibration_paths(self, station_id: Dict[str, Any], sparse: Any) -> Dict[str, str]:
        cfg = dict(self.config)
        cfg["process_model::name"] = self.process_model
        return build_calibration_paths(
            config=cfg,
            station_id=station_id,
            process_model=self.process_model,
            sparse=sparse,
            data_root=self.train_path,
        )

    def _load_existing_calibrations(
        self,
        paths: Dict[str, str],
        train_stations: Dict[str, Any],
        sparse: Any,
        mode: str,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if mode == "defaults":
            return None, "defaults"
        # A forced recalibration starts from the model defaults, not from the bundle it is
        # about to replace: the existing params would become the optimiser's warm start
        # and the anchor of the PRELES fixed covariance, tying the result to the old run.
        if mode == "force_calibrate":
            return None, "forced"

        params, metadata = load_calibration_bundle(paths)
        if params is None:
            return None, "missing"

        # If metadata missing, accept for backward compatibility.
        if metadata is None:
            return params, "file_no_metadata"

        expected = build_expected_metadata_context(
            config={**self.config, "process_model::name": self.process_model},
            station_id=train_stations,
            process_model=self.process_model,
            sparse=sparse,
        )
        compatible, reason = is_metadata_compatible(metadata, expected)
        if compatible:
            return params, "file"

        print(f"[PM Manager] Found params but metadata is incompatible ({reason}).")
        return None, "incompatible"

    def _default_calibrations(self):
        if self.process_model == "preles":
            return dict(self.config["Preles::call:in"])
        if self.process_model == "basfor":
            # Let BasforDataset resolve variant-specific defaults internally.
            return None
        raise RuntimeError("unreachable")

    def _create_dataset(
        self,
        *,
        split: str,
        station_ids: Dict[str, Any],
        calibrations: Any,
        model_variant_override: Optional[str] = None,
    ):
        path = self.train_path if split == "train" else (self.test_path if split == "test" else self.val_path)
        sparse = self._resolve_sparse("train" if split == "train" else "test")
        self._dbg(f"_create_dataset split={split} station_ids={station_ids}")

        if self.process_model == "preles":
            from PRELES_dataset import PrelesDataset
            ds = PrelesDataset(
                path_dir=path,
                station_id=station_ids,
                x_para=self.config["Process::x_para"],
                y_para=self.config["Process::y_para"],
                default_parameters=calibrations,
                control=self.config["Preles::control_arg"],
                preintialised_output=self.config["Preles::preintialised_output"],
                sparse=sparse,
                dataset=True,
                previously_calibrated=False,
                calibration_context=self.config,
                backend=self.config.get("Preles::backend", "preles_c"),
            )
            return ds

        # basfor
        from basfor_dataset import BasforDataset
        # Build x_para as union of Basfor physics columns and any extra NN input columns.
        # prepare_weather_matrix uses Basfor::required_weather_columns independently,
        # so adding extra columns here does not affect the physics run.
        #
        # Always read ML input columns from data::<split>::x_parameter (or train fallback),
        # filtering out _pred columns that come from BASFOR output rather than the raw CSV.
        # This lets the user freely set data::train::x_parameter to any CSV columns without
        # affecting calibration — BASFOR only sees its own required_weather_columns.
        nn_x_cols = [
            c for c in (
                self.config.get(f"data::{split}::x_parameter")
                or self.config.get("data::train::x_parameter")
                or []
            )
            if not c.endswith("_pred")
        ]
        # ordered union: physics columns first, then the extra NN inputs
        x_para = list(dict.fromkeys(list(self.config.get("Basfor::x_para", [])) + nn_x_cols))
        ds = BasforDataset(
            path_dir=path,
            station_id=station_ids,
            x_para=x_para,
            y_para=self.config["Basfor::y_para"],
            param_vector=calibrations,
            calendar_config=self.config.get("Basfor::calendar_config"),
            warmup_config=self.config.get("Basfor::warmup"),
            model_variant_override=model_variant_override,
            sparse=sparse,
            dataset=True,
            previously_calibrated=False,
            y_pred_additional_cols=self.config.get("model::io::y_pred_additional_cols"),
        )
        return ds

    def _run_calibration(self, train_dataset, paths: Dict[str, str]) -> Any:
        n_trials = int(self.config.get("calibration::n_trials", os.getenv("N_TRIALS", 2000)))
        timeout = self.config.get("calibration::timeout")

        if self.process_model == "preles":
            from PRELES_dataset import PrelesCalibrator
            calibrator = PrelesCalibrator(
                self.config,
                train_dataset=train_dataset,
                test_dataset=None,
                pbounds=self.config["Preles::parameters:to:tune"],
                train_calibration_file=paths["params_file"],
            )
            result = calibrator.calibrate_with_optuna(n_trials=n_trials, timeout=timeout)
            # Recorded in metadata.json so the budget, and any resumed study, is visible.
            self._last_calibration_info = {
                k: result[k] for k in ("n_trials_requested", "n_trials_in_study", "seed") if k in result
            }
            merged = dict(self.config["Preles::call:in"])
            merged.update(result["params"])
            return merged

        from basfor_dataset import BasforCalibrator
        basfor_pbounds = self.config.get("Basfor::parameters:to:tune")
        by_variant = self.config.get("Basfor::parameters:to:tune_by_variant")
        if isinstance(by_variant, dict) and hasattr(train_dataset, "model_variant"):
            variant = str(getattr(train_dataset, "model_variant", "")).strip().lower()
            if variant in by_variant and isinstance(by_variant[variant], dict):
                basfor_pbounds = by_variant[variant]                                # FALLBACK
        calibrator = BasforCalibrator(
            train_dataset=train_dataset,
            pbounds=basfor_pbounds,
            loss_log_file=paths["loss_history_csv"],
            train_calibration_file=paths["params_file"],
            sensitivity_config=self.config.get("Basfor::sensitivity", {}),
        )
        result = calibrator.calibrate_with_optuna(n_trials=n_trials, timeout=timeout)
        return result["params"]

    def _predict_dataset(self, dataset, calibrations: Any) -> pd.DataFrame:
        if dataset is None:
            return None
        if hasattr(dataset, "run_process_model"):
            return dataset.run_process_model(params=calibrations)
        raise AttributeError(
            f"Dataset class '{type(dataset).__name__}' does not implement run_process_model(params=...)."
        )

    def _prepare_split_context(
        self,
        train_stations: Optional[Dict[str, Any]],
        test_stations: Optional[Dict[str, Any]],
        val_stations: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fill omitted splits from the active context and persist the result on the manager."""
        train_stations = dict(train_stations if train_stations is not None else self.train_stations)
        test_stations = dict(test_stations if test_stations is not None else self.test_stations)
        if val_stations is None:
            val_stations = self.val_stations
        val_stations = dict(val_stations) if val_stations is not None else None
        self._set_station_context(train_stations, test_stations, val_stations)
        return train_stations, test_stations, val_stations

    def _resolve_split_variants(
        self,
        train_stations: Dict[str, Any],
        test_stations: Dict[str, Any],
        val_stations: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Per-split BASFOR module overrides (None for PRELES); also sets or clears the
        active `Basfor::variant` in the config, which the calibration path keys on.
        """
        if self.process_model != "basfor":
            return None, None, None
        train_v, test_v, val_v, active_v = self._resolve_basfor_split_variant_overrides(
            train_stations=train_stations,
            test_stations=test_stations,
            val_stations=val_stations,
        )
        if active_v:
            self.config["Basfor::variant"] = active_v
        else:
            self.config.pop("Basfor::variant", None)
        return train_v, test_v, val_v

    def _create_split_datasets(
        self,
        stations: Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]],
        variants: Tuple[Optional[str], Optional[str], Optional[str]],
        calibrations: Any,
    ):
        """Build the train/test/(val) datasets; val is None when it has no stations."""
        return tuple(
            self._create_dataset(
                split=split,
                station_ids=station_ids,
                calibrations=calibrations,
                model_variant_override=variant,
            )
            if (split != "val" or station_ids)
            else None
            for split, station_ids, variant in zip(("train", "test", "val"), stations, variants)
        )

    def load_fold_datasets(
        self,
        train_stations: Optional[Dict[str, Any]] = None,
        test_stations: Optional[Dict[str, Any]] = None,
        val_stations: Optional[Dict[str, Any]] = None,
        calibrate: Any = "auto",
        sparse: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load one fold, optionally calibrate, and return process-model predictions.

        If station mappings are omitted, the manager uses the active context set during
        initialization (from CV config or base config). The `sparse` argument is
        unused: sparsity comes from `data:<split>:sparse` in the config.
        """
        stations = self._prepare_split_context(train_stations, test_stations, val_stations)
        train_stations, test_stations, val_stations = stations

        mode = self._normalize_mode(calibrate)
        self._dbg(f"load_fold_datasets mode={mode} train={train_stations} test={test_stations} val={val_stations}")
        sparse = self._resolve_sparse("train")
        variants = self._resolve_split_variants(*stations)
        paths = self._calibration_paths(train_stations, sparse=sparse)
        self._dbg(f"calibration folder={paths['folder']}")

        calibrations, source = self._load_existing_calibrations(
            paths=paths,
            train_stations=train_stations,
            sparse=sparse,
            mode=mode,
        )
        if mode == "require_existing" and source not in {"file", "file_no_metadata"}:
            raise FileNotFoundError(
                f"Required calibration bundle not found or incompatible for process_model='{self.process_model}' "
                f"at '{paths['folder']}' (source={source}). "
                "Run the calibration job first or switch calibration mode."
            )
        if calibrations is None:
            calibrations = self._default_calibrations()

        should_calibrate = (
            mode == "force_calibrate"
            or (mode == "auto" and source not in {"file", "file_no_metadata"})
        )
        if should_calibrate:
            self._dbg("calibration required -> running calibrator")
            train_temp = self._create_dataset(
                split="train",
                station_ids=train_stations,
                calibrations=calibrations,
                model_variant_override=variants[0],
            )
            self._last_calibration_info = {}
            calibrations = self._run_calibration(train_temp, paths=paths)
            extra_metadata = dict(self._last_calibration_info)
            # Where the optimiser started: "forced"/"missing"/"incompatible" = model defaults.
            extra_metadata["warm_start_source"] = source
            if self.process_model == "basfor" and hasattr(train_temp, "model_variant"):
                variant = getattr(train_temp, "model_variant", None)
                if variant:
                    extra_metadata["basfor_variant"] = str(variant)

            metadata = build_calibration_metadata(
                config={**self.config, "process_model::name": self.process_model},
                station_id=train_stations,
                process_model=self.process_model,
                sparse=sparse,
                calibration_mode=mode,
                train_station_id=train_stations,
                test_station_id=test_stations,
                val_station_id=val_stations,
                extra=extra_metadata or None,
            )
            save_calibration_bundle(paths=paths, params=calibrations, metadata=metadata)
            print(f"[PM Manager] Saved calibration bundle -> {paths['folder']}")

        datasets = self._create_split_datasets(stations, variants, calibrations)

        self._dbg("running process model predictions")
        return tuple(self._predict_dataset(ds, calibrations=calibrations) for ds in datasets)

    def load_fold_datasets_embedded(
        self,
        train_stations: Optional[Dict[str, Any]] = None,
        test_stations: Optional[Dict[str, Any]] = None,
        val_stations: Optional[Dict[str, Any]] = None,
    ):
        """
        Return raw split datasets without process-model prediction handoff.

        This is intended for embed-style training paths that need access to the
        underlying split datasets instead of prediction DataFrames.
        """
        stations = self._prepare_split_context(train_stations, test_stations, val_stations)
        calibrations = self._default_calibrations()
        variants = self._resolve_split_variants(*stations)
        return self._create_split_datasets(stations, variants, calibrations)

    def load_fold_dataloaders(
        self,
        train_stations: Optional[Dict[str, Any]] = None,
        test_stations: Optional[Dict[str, Any]] = None,
        val_stations: Optional[Dict[str, Any]] = None,
        calibrate: Any = "auto",
        shuffle_train: bool = True,
        sparse: Optional[Any] = None,
    ):
        """
        Convenience API: run fold calibration/prediction and directly return
        train/test/(optional val) PyTorch DataLoaders via the agnostic handoff.
        """
        from calibration_dataloader_handoff import build_fold_dataloaders_from_predictions

        train_pred, test_pred, val_pred = self.load_fold_datasets(
            train_stations=train_stations,
            test_stations=test_stations,
            val_stations=val_stations,
            calibrate=calibrate,
        )
        return build_fold_dataloaders_from_predictions(
            config=self.config,
            train_df=train_pred,
            test_df=test_pred,
            val_df=val_pred,
            process_model=self.process_model,
            shuffle_train=shuffle_train,
            sparse=sparse,
        )
    
    def _validate_required_config(self) -> None:
        base_required = ["data::train::path_dir", "data::test::path_dir", "Process::x_para", "Process::y_para"]
        if self.process_model == "preles":
            required = base_required + [
                "Preles::call:in",
                "Preles::control_arg",
                "Preles::preintialised_output",
                "Preles::parameters:to:tune",
            ]
        else:
            required = base_required + [
                "Basfor::parameters:to:tune",
                # TODO update with warm up and calendars; b
            ]

        missing = [k for k in required if k not in self.config]
        if missing:
            raise KeyError(
                f"Missing required config keys for process_model='{self.process_model}': {missing}. "
                "Use `config_loader.merge_process_model_config(...)` or pass "
                "`auto_merge_pm_config=True` to the manager."
            )


def _to_station_dict(station: Any, years: Any) -> Dict[str, Any]:
    return {str(station): list(years)}

def _resolve_from_cv(
    cv_config: Dict[str, Any],
    outer_fold: Optional[str],
    inner_loop: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], str, Optional[str], Optional[str]]:
    """
    Resolve (train, test, val, outer_fold, inner_loop, forest_type) station maps from a
    temporal ('out_l_*' + 'station') or spatial ('outer_station_*' / 'outer_fold_*') CV config.
    Without `inner_loop` the outer split is returned and val is None.
    """
    temporal_keys = sorted([k for k in cv_config.keys() if k.startswith("out_l_")])
    spatial_keys = sorted([k for k in cv_config.keys() if k.startswith("outer_station_") or k.startswith("outer_fold_")])

    if temporal_keys and "station" in cv_config:
        picked_outer = outer_fold or temporal_keys[0]
        if picked_outer not in cv_config:
            raise KeyError(f"Outer fold '{picked_outer}' not found in temporal CV config.")
        fold = cv_config[picked_outer]
        station = cv_config["station"]
        picked_inner = inner_loop

        if inner_loop:
            inner = fold["inner_loops"].get(inner_loop)
            if inner is None:
                available = sorted(fold["inner_loops"].keys())
                raise KeyError(f"Inner loop '{inner_loop}' not found. Available: {available}")
            train = _to_station_dict(station, inner["train"])
            test = _to_station_dict(station, inner["test"])
            val = _to_station_dict(station, inner.get("val", [])) if "val" in inner else None
        else:
            train = _to_station_dict(station, fold["train_hold_out"])
            test = _to_station_dict(station, fold["hold_out_years"])
            val = None
        return train, test, val, picked_outer, picked_inner, None

    if spatial_keys:
        picked_outer = outer_fold or spatial_keys[0]
        if picked_outer not in cv_config:
            raise KeyError(f"Outer fold '{picked_outer}' not found in spatial CV config.")
        fold = cv_config[picked_outer]
        picked_inner = inner_loop
        cv_forest_type = fold.get("forest_type")

        if inner_loop:
            inner = fold.get(inner_loop)
            if inner is None:
                available = sorted([k for k in fold.keys() if k.startswith("inner_loop_")])
                raise KeyError(f"Inner loop '{inner_loop}' not found. Available: {available}")
            train = dict(inner[0]["data::train::station_id"])
            test = dict(inner[1]["data::test::station_id"])
            val = dict(inner[2]["data::val::station_id"])
        else:
            train = dict(fold["outer_split"]["train"])
            test = dict(fold["outer_split"]["test"])
            val = None
        return train, test, val, picked_outer, picked_inner, cv_forest_type

    raise ValueError("Could not infer CV format. Expected temporal ('out_l_*') or spatial ('outer_station_*') keys.")

def _slug(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")

def _station_signature(stations: Optional[Dict[str, Any]]) -> str:
    """Short signature for plot folder names (abbreviates long year lists); distinct
    from calibration_paths.station_signature, which names calibration bundles."""
    if not stations:
        return "none"

    parts = []
    for station, years_payload in sorted(stations.items(), key=lambda x: str(x[0])):
        station_tag = _slug(station)

        if isinstance(years_payload, list):
            years = [int(y) for y in years_payload if isinstance(y, (int, float))]
            years = sorted(set(years))
            if not years:
                year_tag = "na"
            elif len(years) <= 3:
                year_tag = "-".join(str(y) for y in years)
            else:
                year_tag = f"{years[0]}-{years[-1]}"
        elif isinstance(years_payload, dict) and "startdate" in years_payload and "stopdate" in years_payload:
            year_ranges = []
            for start, stop in zip(years_payload.get("startdate", []), years_payload.get("stopdate", [])):
                if isinstance(start, (list, tuple)) and isinstance(stop, (list, tuple)) and start and stop:
                    y0, y1 = int(start[0]), int(stop[0])
                    year_ranges.append(f"{y0}-{y1}" if y0 != y1 else str(y0))
            year_tag = "+".join(year_ranges) if year_ranges else "na"
        else:
            year_tag = _slug(years_payload)

        parts.append(f"{station_tag}_{year_tag}")

    return "__".join(parts)

def _build_plot_paths(process_model: str, train_stations: Dict[str, Any], test_stations: Dict[str, Any], cv_config_path: Optional[str] = None, module_variant: Optional[str] = None) -> Dict[str, str]:
    train_sig = _station_signature(train_stations)
    test_sig = _station_signature(test_stations)
    cv_slug = _slug(os.path.splitext(os.path.basename(cv_config_path))[0]) if cv_config_path else "no_cv"
    root = os.path.join("results", "calibrations", cv_slug, f"train_{train_sig}_test_{test_sig}")
    os.makedirs(root, exist_ok=True)
    variant_suffix = f"_{_slug(module_variant)}" if module_variant else ""
    prefix = _slug(process_model.lower())
    return {
        "train": os.path.join(root, f"{prefix}{variant_suffix}_train.png"),
        "test": os.path.join(root, f"{prefix}{variant_suffix}_test.png"),
        "val": os.path.join(root, f"{prefix}{variant_suffix}_val.png"),
    }


if __name__ == "__main__":
    from Config_Files.config_spatial import build_config

    parser = argparse.ArgumentParser(description="Runner for ProcessModelCalibrationManager.")
    parser.add_argument("--config", type=str, default=None,         help="Optional path to frozen config JSON.")
    parser.add_argument("--cv-config", type=str, default=None,      help="Optional CV split JSON path.")
    parser.add_argument("--outer-fold", type=str, default=None,     help="Outer fold key, e.g. out_l_1 or outer_station_12.")
    parser.add_argument("--inner-loop", type=str, default=None,     help="Optional inner loop key, e.g. inner_loop1 or inner_loop_1.")
    parser.add_argument("--process-model", type=str, default="preles", choices=["preles"])
    parser.add_argument("--calibrate", type=str, default="auto", choices=["auto", "defaults", "force_calibrate", "require_existing"])
    parser.add_argument("--as-dataloaders", action="store_true",    help="Return/train-test-val DataLoaders instead of DataFrames.")
    parser.add_argument("--shuffle-train", action="store_true",     help="Shuffle training DataLoader in --as-dataloaders mode.")
    parser.add_argument("--sparse", type=int, default=None,         help="Optional sparse stride for dataloader handoff (e.g. 7 keeps every 7th row).")
    parser.add_argument("--no-plot", action="store_true",           help="Disable plotting in DataFrame mode.")
    parser.add_argument("--flue-csv", type=str, default=None, dest="flue_csv", help="Path to flue_stocker18nphyt.csv; when set, FLUE drought periods are shaded on BASFOR plots.")
    parser.add_argument("--debug", action="store_true",             help="Enable manager debug logging.")
    parser.add_argument("--calibration-artifact", type=str, default=None, dest="calibration_artifact", help="Calibration artifact namespace.")
    parser.add_argument("--experiment-artifact",  type=str, default=None, dest="experiment_artifact",  help="Experiment artifact namespace.")
    args = parser.parse_args()

    # IDE quick-run settings.
    IDE_SETTINGS = {
        "enabled": True,
        "config": None,             # e.g. "configs/frozen_config.json"
        "cv_config": 'configs/CV_splits/CV_dfb_module_misspec_mixed_train_6folds_config_v2.json', #
        "outer_fold": None, #"outer_fold_DE-Tha_deciduous-module", #"outer_station_FR-LBr", #'outer_fold_Cfb-mixed', #
        "inner_loop": "inner_loop_3",         # inner_loop
        "process_model": "basfor",
        "calibrate": "require_existing",            # auto | defaults | force_calibrate | require_existing
        "as_dataloaders": False,
        "shuffle_train": False,
        "sparse": None,                             # 7, 30, 50
        "no_plot": False,
        "flue_csv": None, #"Data/flue_stocker18nphyt.csv",
        "debug": False,
        "loop_all_folds": True,         # when True and outer_fold is None, iterate all outer folds
        # Artifact namespaces for local testing (leave as None to use cluster defaults)
        "experiment_artifact_namespace": None,                      #"additional_inputs_v1",  # e.g., "local_test_v1" or "debug"
        "calibration_artifact_namespace": None, #"misspec_pure_tpe_base", #"misspec_pure_tpe_base",  # e.g., "local_test_v1" or "debug"
    }


    loop_all_folds = False
    if len(sys.argv) == 1 and IDE_SETTINGS["enabled"]:
        print("[PM Manager][main] No CLI args detected, using IDE_SETTINGS.")
        args.config = IDE_SETTINGS["config"]
        args.cv_config = IDE_SETTINGS["cv_config"]
        args.outer_fold = IDE_SETTINGS["outer_fold"]
        args.inner_loop = IDE_SETTINGS["inner_loop"]
        args.process_model = IDE_SETTINGS["process_model"]
        args.calibrate = IDE_SETTINGS["calibrate"]
        args.as_dataloaders = IDE_SETTINGS["as_dataloaders"]
        args.shuffle_train = IDE_SETTINGS["shuffle_train"]
        args.sparse = IDE_SETTINGS["sparse"]
        args.no_plot = IDE_SETTINGS["no_plot"]
        args.flue_csv = IDE_SETTINGS.get("flue_csv")
        args.debug = IDE_SETTINGS["debug"]
        args.experiment_artifact = IDE_SETTINGS["experiment_artifact_namespace"]
        args.calibration_artifact = IDE_SETTINGS["calibration_artifact_namespace"]
        loop_all_folds = IDE_SETTINGS.get("loop_all_folds", False)

    load_dotenv()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = build_config()

    # Call the relevant process model you wish to use
    config = merge_process_model_config(base_config=config, process_model=args.process_model)

    artifact_namespace = os.getenv("CALIBRATION_ARTIFACT_NAMESPACE") or args.calibration_artifact
    if artifact_namespace:
        config["calibration::artifact_namespace"] = artifact_namespace

    if args.debug:
        config["calibration::debug"] = True
    if "calibration::root" not in config:
        config["calibration::root"] = os.getenv("CALIBRATION_ROOT", "calibrations")


    # Collect all outer fold keys to iterate over
    if args.cv_config:
        with open(args.cv_config, "r", encoding="utf-8") as f:
            cv_cfg = json.load(f)
        spatial_keys = sorted([k for k in cv_cfg.keys() if k.startswith("outer_station_") or k.startswith("outer_fold_")])
        temporal_keys = sorted([k for k in cv_cfg.keys() if k.startswith("out_l_")])
        all_outer_keys = spatial_keys or temporal_keys

        if loop_all_folds and args.outer_fold is None and all_outer_keys:
            outer_folds_to_run = all_outer_keys
        else:
            outer_folds_to_run = [args.outer_fold]  # single fold (or None → _resolve_from_cv picks first)
    else:
        outer_folds_to_run = [None]

    for fold_key in outer_folds_to_run:
        if args.cv_config:
            train_stations, test_stations, val_stations, used_outer, used_inner, _ = _resolve_from_cv(
                cv_config=cv_cfg,
                outer_fold=fold_key,
                inner_loop=args.inner_loop,
            )
            print(f"\n[PM Manager][main] === CV outer={used_outer} inner={used_inner} ===")
        else:
            train_stations = dict(config["data::train::station_id"])
            test_stations = dict(config["data::test::station_id"])
            val_stations = dict(config["data::val::station_id"]) if "data::val::station_id" in config else None

        print(f"[PM Manager][main] process_model={config['process_model::name']} | calibrate={args.calibrate}")
        print(f"[PM Manager][main] train={train_stations}")
        print(f"[PM Manager][main] test={test_stations}")
        if val_stations is not None:
            print(f"[PM Manager][main] val={val_stations}")

        manager = ProcessModelCalibrationManager(
            config=config,
            process_model=args.process_model,
            cv_config_path=args.cv_config if args.cv_config else None,
            outer_fold=fold_key,
            inner_loop=args.inner_loop,
        )
        if args.sparse is not None:
            manager.config["data:train:sparse"] = args.sparse

        if args.as_dataloaders:
            train_loader, test_loader, val_loader = manager.load_fold_dataloaders(
                train_stations=train_stations,
                test_stations=test_stations,
                val_stations=val_stations,
                calibrate=args.calibrate,
                shuffle_train=args.shuffle_train,
                sparse=args.sparse,
            )
            print(
                f"[PM Manager][main] built DataLoaders "
                f"(train_bs={getattr(train_loader, 'batch_size', None)}, "
                f"test_bs={getattr(test_loader, 'batch_size', None)}, "
                f"val_bs={getattr(val_loader, 'batch_size', None) if val_loader is not None else None})"
            )

        else:
            train_df, test_df, val_df = manager.load_fold_datasets(
                train_stations=train_stations,
                test_stations=test_stations,
                val_stations=val_stations,
                calibrate=args.calibrate,
                sparse=args.sparse,
            )

            if not args.no_plot:
                _basfor_variant = manager.config.get("Basfor::variant") if args.process_model == "basfor" else None
                plot_paths = _build_plot_paths(
                    process_model=args.process_model,
                    train_stations=train_stations,
                    test_stations=test_stations,
                    cv_config_path=args.cv_config if args.cv_config else None,
                    module_variant=_basfor_variant,
                )
                required_plot_cols = {"GPP", "ET", "GPP_pred", "ET_pred"}
                if required_plot_cols.issubset(train_df.columns) and required_plot_cols.issubset(test_df.columns):
                    if args.process_model == "preles":
                        x_para = config.get("Process::x_para")
                        plot_preles_predictions(
                            SimpleNamespace(data=train_df),
                            title_prefix="PRELES Bayesian Train",
                            save_path=plot_paths["train"],
                            x_para=x_para,
                        )
                        print(f"[PM Manager][main] saved: {plot_paths['train']}")

                        plot_preles_predictions(
                            SimpleNamespace(data=test_df),
                            title_prefix="PRELES Bayesian Test",
                            save_path=plot_paths["test"],
                            x_para=x_para,
                        )
                        print(f"[PM Manager][main] saved: {plot_paths['test']}")
                    else:
                        from basfor_dataset import plot_basfor_predictions, _load_station_flags
                        _train_flags = _load_station_flags(manager.train_path, train_stations, ["drought_flag"])
                        _test_flags  = _load_station_flags(manager.test_path,  test_stations,  ["drought_flag"])
                        _flue_csv = getattr(args, "flue_csv", None)
                        plot_basfor_predictions(train_df, title_prefix="Train", save_path=plot_paths["train"], module_variant=_basfor_variant, extra_flags_df=_train_flags, flue_csv=_flue_csv, station_id=train_stations)
                        plot_basfor_predictions(test_df, title_prefix="Test", save_path=plot_paths["test"], module_variant=_basfor_variant, extra_flags_df=_test_flags, flue_csv=_flue_csv, station_id=test_stations)
                        if val_df is not None and required_plot_cols.issubset(val_df.columns):
                            _val_flags = _load_station_flags(manager.val_path, val_stations, ["drought_flag"])
                            plot_basfor_predictions(val_df, title_prefix="Val", save_path=plot_paths["val"], module_variant=_basfor_variant, extra_flags_df=_val_flags, flue_csv=_flue_csv, station_id=val_stations)
                        print(f"[PM Manager][main] Saved {args.process_model.upper()} plots under: {os.path.dirname(plot_paths['train'])}")
                else:
                    print("[PM Manager][main] Skipped plotting: expected columns missing for prediction plots.")
        print(f"[PM Manager][main] Done: {fold_key or 'single fold'}")
