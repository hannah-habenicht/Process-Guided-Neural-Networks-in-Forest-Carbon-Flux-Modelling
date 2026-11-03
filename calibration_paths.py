from datetime import datetime, timezone
import json
import os
import re
from typing import Any, Dict, Optional, Tuple


def _normalize_sparse(sparse: Any) -> str:
    if sparse is None:
        return "full"
    try:
        s = int(sparse)
        return f"sparse{s}" if s > 1 else "full"
    except (TypeError, ValueError):
        return str(sparse)


def _sparse_mode(process_model: str, sparse_label: str) -> Optional[str]:
    """
    How a sparse PRELES calibration was simulated. Since 2026-10 PRELES runs on the
    continuous daily series and is scored on the kept rows; older sparse bundles ran
    PRELES on the thinned rows as if consecutive and are rejected by this tag.
    """
    if process_model == "preles" and sparse_label != "full":
        return "daily_run_then_thin"
    return None


def _normalize_basfor_variant(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"conifer", "conif", "coniferous"}:
        return "conifer"
    if raw in {"deciduous", "decid"}:
        return "deciduous"
    return None


def _normalize_artifact_namespace(value: Any) -> str:
    """
    Normalize optional calibration artifact namespace used for folder segregation.
    Empty/default-like values resolve to "" (no extra folder segment).
    """
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if raw in {"", "default", "standard", "none", "null"}:
        return ""
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-"}:
            safe.append(ch)
        elif ch.isspace():
            safe.append("_")
    return "".join(safe).strip("_-")


def _station_signature(station_id: Dict[str, Any]) -> str:
    """
    Build a stable readable signature from station/year/date-range mappings.
    """
    station_entries = []
    for station, info in sorted(station_id.items(), key=lambda kv: str(kv[0])):
        if isinstance(info, list) and all(isinstance(year, int) for year in info):
            years_str = "-".join(map(str, sorted(info)))
            station_entries.append(f"{station}_{years_str}")
            continue

        if isinstance(info, dict) and "startdate" in info and "stopdate" in info:
            date_ranges = []
            for start, stop in zip(info["startdate"], info["stopdate"]):
                if start[0] == stop[0]:
                    date_ranges.append(
                        f"{start[0]}_{start[1]:02d}_{start[2]:02d}-{stop[1]:02d}_{stop[2]:02d}"
                    )
                else:
                    date_ranges.append(f"{start[0]}-{stop[0]}")
            station_entries.append(f"{station}_{'+'.join(date_ranges)}")
            continue

        raise ValueError(f"Unrecognized station ID format for {station}: {info}")

    return "__".join(station_entries)


def _find_reordered_split_folder(parent: str, split_sig: str) -> Optional[str]:
    """
    Return an existing sibling folder holding params.json whose signature has the same
    station entries as `split_sig` in a different order, else None.

    Legacy bundles (e.g. copied in from Data/new_calibrations) list stations in numeric
    order ("5_2006__14_2005__21_2006"), while _station_signature sorts them as strings
    ("14_2005__21_2006__5_2006"), so an exact-name lookup misses them.
    """
    if not os.path.isdir(parent):
        return None
    wanted = sorted(split_sig.split("__"))
    for name in sorted(os.listdir(parent)):
        if name == split_sig or sorted(name.split("__")) != wanted:
            continue
        candidate = os.path.join(parent, name)
        if os.path.isfile(os.path.join(candidate, "params.json")):
            return candidate
    return None


def _resolve_calibration_root(config: Dict[str, Any], data_root: Optional[str] = None) -> str:
    """
    Resolve calibration root with this precedence:
    1) env CALIBRATION_PATH
    2) config['calibration::root']
    3) <data_root>/calibrations
    """
    root =  os.getenv("CALIBRATION_PATH") or config.get("calibration::root")
    if root:
        if os.path.isabs(root):
            return root
        base = data_root or config.get("data::train::path_dir") or os.getcwd()
        return os.path.join(base, root)

    base = data_root or config.get("data::train::path_dir") or os.getcwd()
    return os.path.join(base, "calibrations")


def build_calibration_paths(
    config: Dict[str, Any],
    station_id: Dict[str, Any],
    process_model: Optional[str] = None,
    sparse: Any = None,
    split_key: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build canonical calibration directory and standard file paths.

    Folder layout:
    <root>/<process_model>/<cv_type>/[<artifact_namespace>/]<calibration_scope>/<sparsity>/[<basfor_variant>/]<split_signature>/

    The <basfor_variant> segment is inserted for BASFOR runs when the variant is known,
    so that calibrations for different module assignments (conifer vs deciduous) on the
    same train split are stored in separate directories.
    """
    pm = (process_model or config.get("process_model::name") or "preles").lower()
    cv_type = str(config.get("calibration::cv_type") or config.get("model::prediction_type") or "spatial") #
    calibration_scope = "single_site" if len(station_id) == 1 else "multi_site"
    sparsity = _normalize_sparse(sparse if sparse is not None else config.get("data:train:sparse"))
    split_sig = split_key or _station_signature(station_id)
    artifact_namespace = _normalize_artifact_namespace(config.get("calibration::artifact_namespace"))
    basfor_variant = _normalize_basfor_variant(config.get("Basfor::variant")) if pm == "basfor" else None

    root = _resolve_calibration_root(config, data_root=data_root)
    parts = [root, pm, cv_type]
    if artifact_namespace:
        parts.append(artifact_namespace)
    parts.append(calibration_scope)
    parts.append(sparsity)
    if basfor_variant:
        parts.append(basfor_variant)
    parts.append(split_sig)
    folder = os.path.join(*parts)
    if not os.path.isfile(os.path.join(folder, "params.json")):
        folder = _find_reordered_split_folder(os.path.dirname(folder), split_sig) or folder

    return {
        "root": root,
        "folder": folder,
        "artifact_namespace": artifact_namespace,
        "params_file": os.path.join(folder, "params.json"),
        "metadata_file": os.path.join(folder, "metadata.json"),
        "loss_history_csv": os.path.join(folder, "loss_history.csv"),
        "loss_history_xlsx": os.path.join(folder, "loss_history.xlsx"),
    }


def build_calibration_metadata(
    config: Dict[str, Any],
    station_id: Dict[str, Any],
    process_model: Optional[str] = None,
    sparse: Any = None,
    calibration_mode: Optional[str] = None,
    train_station_id: Optional[Dict[str, Any]] = None,
    test_station_id: Optional[Dict[str, Any]] = None,
    val_station_id: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standard metadata for a calibration artifact bundle.
    """
    pm = (process_model or config.get("process_model::name") or "preles").lower()
    cv_type = str(config.get("calibration::cv_type") or config.get("model::prediction_type") or "spatial")
    #train_len = str(config.get("calibration::training_years_len") or "unspecified_len")
    sparse_label = _normalize_sparse(sparse if sparse is not None else config.get("data:train:sparse"))
    mode = calibration_mode or str(config.get("calibration::mode") or "auto")
    basfor_variant = _normalize_basfor_variant(
        config.get("Basfor::variant", config.get("basfor_variant"))
    )
    artifact_namespace = _normalize_artifact_namespace(config.get("calibration::artifact_namespace"))

    meta = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "process_model": pm,
        "prediction_type": str(config.get("model::prediction_type", cv_type)),
        "cv_type": cv_type,
        "sparse": sparse_label,
        "calibration_mode": mode,
        "split_signature": _station_signature(station_id),
        "train_station_id": train_station_id if train_station_id is not None else station_id,
        "test_station_id": test_station_id,
        "val_station_id": val_station_id,
        "objective": config.get("training::loss_type"),
        "artifact_namespace": artifact_namespace,
    }
    if pm == "basfor" and basfor_variant is not None:
        meta["basfor_variant"] = basfor_variant
    sparse_mode = _sparse_mode(pm, sparse_label)
    if sparse_mode:
        meta["sparse_mode"] = sparse_mode
    if extra:
        meta.update(extra)
    return meta


def save_calibration_bundle(
    paths: Dict[str, str],
    params: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """
    Save params + metadata as the canonical calibration artifact bundle.
    """
    folder = paths["folder"]
    os.makedirs(folder, exist_ok=True)

    with open(paths["params_file"], "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    with open(paths["metadata_file"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_calibration_bundle(paths: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load params + metadata from canonical bundle.

    Returns:
        (params, metadata) if params file exists, otherwise (None, None).
        metadata can be None when params exist but metadata file is missing.
    """
    if not os.path.exists(paths["params_file"]):
        print("Calibration file not found, cant load")
        return None, None

    with open(paths["params_file"], "r", encoding="utf-8") as f:
        params = json.load(f)

    metadata = None
    if os.path.exists(paths["metadata_file"]):
        with open(paths["metadata_file"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return params, metadata


def is_metadata_compatible(
    metadata: Optional[Dict[str, Any]],
    expected: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Validate whether stored metadata is compatible with the current run context.
    """
    if metadata is None:
        return False, "missing_metadata"

    checks = (
        ("process_model", str(expected.get("process_model", "")).lower()),
        ("cv_type", str(expected.get("cv_type", ""))),
        ("sparse", str(expected.get("sparse", ""))),
        ("split_signature", str(expected.get("split_signature", ""))),
        ("basfor_variant", str(expected.get("basfor_variant", ""))),
        ("artifact_namespace", str(expected.get("artifact_namespace", ""))),
        ("sparse_mode", str(expected.get("sparse_mode", "") or "")),
    )

    for key, want in checks:
        have = str(metadata.get(key, ""))
        if want and have != want:
            return False, f"mismatch:{key}"

    return True, "ok"


def station_signature(station_id: Dict[str, Any]) -> str:
    """Public alias for _station_signature."""
    return _station_signature(station_id)


def cv_label_from_path(cv_config_path: str) -> str:
    """
    Derive a short, filesystem-safe CV label from the config filename.

    Examples:
        "CV_koppen_ood_16folds_config.json"  ->  "koppen_ood_16folds"
        "CV_station_len3_config.json"        ->  "station_len3"
        "/path/to/CV_koppen_ood_17folds_config.json"  ->  "koppen_ood_17folds"
    """
    stem = os.path.splitext(os.path.basename(cv_config_path))[0]  # drop .json
    stem = re.sub(r"^CV_", "", stem)                               # drop leading "CV_"
    stem = re.sub(r"_config$", "", stem)                           # drop trailing "_config"
    return stem or "cv"


def build_spatial_logs_root(
    save_root: str,
    process_model_name: str,
    cv_label: str,
    prediction_type: str,
    prediction_loss: str,
    sparse: Any = None,
    artifact_namespace: Any = None,
) -> Dict[str, str]:
    """
    Build checkpoint and logs directory paths for a spatial NCV experiment.

    Returns a dict with keys:
        "checkpoint_root" : .../checkpoints/<pm>/<sparsity>/<cv_label>/[<artifact_namespace>/]
        "logs_root"       : checkpoint_root/logs_<prediction_type>_<prediction_loss>
    """
    sparse_str = _normalize_sparse(sparse)
    namespace = _normalize_artifact_namespace(artifact_namespace)
    parts = [save_root, "checkpoints", process_model_name, sparse_str, cv_label]
    if namespace:
        parts.append(namespace)
    checkpoint_root = os.path.join(*parts)
    logs_root = os.path.join(checkpoint_root, f"logs_{prediction_type}_{prediction_loss}")
    return {"checkpoint_root": checkpoint_root, "logs_root": logs_root}


def build_spatial_scenario_id(
    outer_fold: str,
    inner_loop: Optional[str] = None,
) -> str:
    """
    Build a stable, filesystem-safe scenario identifier for a spatial fold.

    Examples:
        outer_fold="outer_station_FR-LBr"                  -> "outer_station_FR-LBr"
        outer_fold="outer_station_FR-LBr", inner_loop="inner_loop_1"
                                                           -> "outer_station_FR-LBr__inner_loop_1"
    """
    if inner_loop:
        return f"{outer_fold}__{inner_loop}"
    return outer_fold


def build_spatial_optuna_db_path(
    optuna_db_dir: str,
    process_model: str,
    prediction_type: str,
    prediction_loss: str,
    outer_fold: str,
    cv_label: str,
    sparse: Any = None,
    artifact_namespace: Any = None,
) -> Dict[str, str]:
    """
    Build the Optuna SQLite database path and study name for a spatial NCV run.

    Returns a dict with keys:
        "db_file"    : absolute path to the .db file
        "study_name" : Optuna study name string
        "url"        : SQLite URL suitable for RDBStorage
    """
    sparsity_suffix = f"_sparse{int(sparse)}" if sparse is not None else "_full"
    namespace = _normalize_artifact_namespace(artifact_namespace)
    namespace_prefix = f"{namespace}_" if namespace else ""
    study_name = (
        f"study_{namespace_prefix}{process_model}_{prediction_type}_{prediction_loss}"
        f"_{outer_fold}{sparsity_suffix}_{cv_label}"
    )
    db_file = os.path.join(
        optuna_db_dir,
        f"Optuna_{namespace_prefix}{prediction_type}_{prediction_loss}_{outer_fold}{sparsity_suffix}_{cv_label}.db",
    )
    return {"db_file": db_file, "study_name": study_name, "url": f"sqlite:///{db_file}"}


def build_expected_metadata_context(
    config: Dict[str, Any],
    station_id: Dict[str, Any],
    process_model: Optional[str] = None,
    sparse: Any = None,
) -> Dict[str, Any]:
    """
    Build the expected context used by `is_metadata_compatible`.
    """
    basfor_variant = _normalize_basfor_variant(
        config.get("Basfor::variant", config.get("basfor_variant"))
    )
    expected = {
        "process_model": (process_model or config.get("process_model::name") or "preles").lower(),
        "cv_type": str(config.get("calibration::cv_type") or config.get("model::prediction_type") or "spatial"),
        "training_years_len": str(config.get("calibration::training_years_len") or "unspecified_len"),
        "sparse": _normalize_sparse(sparse if sparse is not None else config.get("data:train:sparse")),
        "split_signature": _station_signature(station_id),
        "artifact_namespace": _normalize_artifact_namespace(config.get("calibration::artifact_namespace")),
    }
    if expected["process_model"] == "basfor" and basfor_variant is not None:
        expected["basfor_variant"] = basfor_variant
    sparse_mode = _sparse_mode(expected["process_model"], expected["sparse"])
    if sparse_mode:
        expected["sparse_mode"] = sparse_mode
    return expected
