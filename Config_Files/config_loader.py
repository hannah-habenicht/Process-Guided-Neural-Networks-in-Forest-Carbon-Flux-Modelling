import copy
import importlib
from typing import Any, Dict, List, Optional


# Process-model registry. Adding a model (e.g. medfate) takes three steps:
#   1. Config_Files/<pm>_configs.py exposing `cfg`, with data::{train,test}::x_parameter /
#      y_parameter and <Prefix>::x_para / y_para / y_phy_para.
#   2. An entry here and in PROCESS_MODEL_KEY_PREFIX.
#   3. A _create_dataset / _run_calibration branch in ProcessModelCalibrationManager whose
#      dataset implements run_process_model(params=...).
# Drivers take their --process-model choices from SUPPORTED_PROCESS_MODELS.
PROCESS_MODEL_CONFIG_MODULES = {
    "preles": "Config_Files.preles_configs",
}

# Key prefix of each model's own config keys (<Prefix>::x_para -> Process::x_para).
PROCESS_MODEL_KEY_PREFIX = {
    "preles": "Preles",
}

SUPPORTED_PROCESS_MODELS = tuple(PROCESS_MODEL_CONFIG_MODULES)


def load_process_model_config(process_model: str, module_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load process-model-specific config payload from a Python module exposing `cfg`.
    """
    pm = process_model.lower()
    mod_name = module_path or PROCESS_MODEL_CONFIG_MODULES.get(pm)
    if not mod_name:
        raise ValueError(f"No default config module registered for process model '{process_model}'.")

    module = importlib.import_module(mod_name)
    payload = getattr(module, "cfg", None)
    if payload is None:
        raise AttributeError(f"Module '{mod_name}' does not define a `cfg` dictionary.")
    if not isinstance(payload, dict):
        raise TypeError(f"`cfg` in module '{mod_name}' must be a dict.")
    return copy.deepcopy(payload)


def _apply_process_aliases(cfg: Dict[str, Any], process_model: str) -> None:
    prefix = PROCESS_MODEL_KEY_PREFIX.get(process_model.lower())
    if prefix is None:
        return
    for suffix in ("x_para", "y_para", "y_phy_para"):
        key = f"{prefix}::{suffix}"
        if key in cfg:
            cfg[f"Process::{suffix}"] = copy.deepcopy(cfg[key])


def _as_non_empty_list(value: Any) -> Optional[List[str]]:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return [str(v) for v in value]
    return None


def validate_embedding_mode_tokens(cfg: Dict[str, Any]) -> None:
    """
    Enforce canonical embedding naming in active pipeline configs.
    Legacy token `embed` is not accepted.
    """
    for key in ("model::modeltype", "training::loss_physic_type"):
        if key not in cfg:
            continue
        raw_value = cfg.get(key)
        normalized = str(raw_value).strip().lower()
        if normalized == "embed":
            raise ValueError(
                f"Invalid embedding mode token 'embed' in '{key}' (got {raw_value!r}). "
                "Use 'embedding'."
            )


INPUT_ROUTING_TARGETS = ("experts", "gate", "both")


def _resolve_input_routing(cfg: Dict[str, Any], x_cols: List[str]) -> None:
    """
    Split the model inputs between the MoE gate and its trainable experts.

    `model::io::input_routing` maps an input column to "experts", "gate" or "both";
    unlisted columns are "both". Every routed column stays in x_parameter, so it is
    loaded and z-scored like any other input; the mixture only picks which columns of
    x each part reads. Sets:
      model::io::gate_cols / model::io::expert_cols   (in x order)
      model::gate_input_size / model::expert_input_size
    model::input_size is left at len(x): the non-MoE architectures ignore the routing.
    """
    routing = cfg.get("model::io::input_routing")
    if not isinstance(routing, dict):
        raise TypeError(
            f"model::io::input_routing must be a dict of column -> {INPUT_ROUTING_TARGETS}, got {type(routing)}."
        )
    unknown = sorted(set(routing) - set(x_cols))
    if unknown:
        raise KeyError(
            f"model::io::input_routing names columns that are not model inputs: {unknown}. "
            f"Inputs are: {x_cols}. Add them to data::train/test::x_parameter first."
        )
    bad = {col: target for col, target in routing.items() if target not in INPUT_ROUTING_TARGETS}
    if bad:
        raise ValueError(f"model::io::input_routing targets must be one of {INPUT_ROUTING_TARGETS}, got {bad}.")

    gate_cols = [c for c in x_cols if routing.get(c, "both") in ("gate", "both")]
    expert_cols = [c for c in x_cols if routing.get(c, "both") in ("experts", "both")]
    if not gate_cols or not expert_cols:
        raise ValueError(
            f"model::io::input_routing leaves the {'gate' if not gate_cols else 'experts'} with no inputs."
        )
    cfg["model::io::gate_cols"] = gate_cols
    cfg["model::io::expert_cols"] = expert_cols
    cfg["model::gate_input_size"] = len(gate_cols)
    cfg["model::expert_input_size"] = len(expert_cols)


def resolve_model_io_config(cfg: Dict[str, Any], split: str = "train") -> Dict[str, Any]:
    """
    Resolve spatial model input/output dimensions from config columns + loss mode.

    Rules:
    - y_cols: data::<split>::y_parameter -> data::train::y_parameter -> Process::y_para
    - x_cols: data::<split>::x_parameter -> data::train::x_parameter -> Process::x_para
    - y_pred_target_cols: [f"{y}_pred" for y in y_cols]
    - y_pred_additional_cols: manual additional prediction columns from config
    - y_pred_cols: y_pred_target_cols + y_pred_additional_cols
    - bias: x_effective = y_pred_cols
    - residual: x_effective = x_base + y_pred_cols
    - parallel/regularized/naive: x_effective = x_base
    - embedding: keep embedding behavior (no automatic remap/sizing)
    """
    prediction_type = str(cfg.get("model::prediction_type", "")).strip().lower()
    if prediction_type != "spatial":
        return cfg

    loss_type = str(cfg.get("training::loss_physic_type", "")).strip().lower()
    model_type = str(cfg.get("model::modeltype", "")).strip().lower()

    y_cols = (
        _as_non_empty_list(cfg.get(f"data::{split}::y_parameter"))
        or _as_non_empty_list(cfg.get("data::train::y_parameter"))
        or _as_non_empty_list(cfg.get("Process::y_para"))
    )
    if not y_cols:
        raise KeyError(
            "Could not resolve supervised targets for spatial sizing. "
            "Expected one of: data::<split>::y_parameter, data::train::y_parameter, Process::y_para."
        )

    x_base_cols = (
        _as_non_empty_list(cfg.get(f"data::{split}::x_parameter"))
        or _as_non_empty_list(cfg.get("data::train::x_parameter"))
        or _as_non_empty_list(cfg.get("Process::x_para"))
    )
    if not x_base_cols:
        raise KeyError(
            "Could not resolve input features for spatial sizing. "
            "Expected one of: data::<split>::x_parameter, data::train::x_parameter, Process::x_para."
        )

    y_pred_target_cols = [f"{y}_pred" for y in y_cols]
    y_pred_additional_cols = _as_non_empty_list(cfg.get("model::io::y_pred_additional_cols")) or []
    y_pred_cols = list(dict.fromkeys(y_pred_target_cols + y_pred_additional_cols))

    if loss_type == "bias":
        x_effective_cols = list(y_pred_cols)
    elif loss_type == "residual":
        x_effective_cols = list(x_base_cols) + list(y_pred_cols)
    else:
        x_effective_cols = list(x_base_cols)

    cfg["model::io::x_base_cols"] = list(x_base_cols)
    cfg["model::io::x_effective_cols"] = list(x_effective_cols)
    cfg["model::io::y_cols"] = list(y_cols)
    cfg["model::io::y_pred_target_cols"] = list(y_pred_target_cols)
    cfg["model::io::y_pred_additional_cols"] = list(y_pred_additional_cols)
    cfg["model::io::y_pred_cols"] = list(y_pred_cols)

    if model_type == "embedding":
        return cfg

    cfg["model::input_size"] = len(x_effective_cols)
    cfg["model::output_size"] = len(y_cols)

    # Only when a config asks for it; without the key nothing below changes.
    if "model::io::input_routing" in cfg:
        _resolve_input_routing(cfg, x_effective_cols)

    if loss_type in {"bias", "residual"}:
        for split_name in ("train", "test", "val"):
            key = f"data::{split_name}::x_parameter"
            if key in cfg or split_name in {"train", "test"}:
                cfg[key] = list(x_effective_cols)

    return cfg


def merge_process_model_config(
    base_config: Dict[str, Any],
    process_model: str,
    pm_config: Optional[Dict[str, Any]] = None,
    module_path: Optional[str] = None,
    resolve_model_io: bool = True,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Merge base experiment config + process-model config into a single resolved config.
    """
    cfg = copy.deepcopy(base_config)
    pm_payload = copy.deepcopy(pm_config) if pm_config is not None else load_process_model_config(
        process_model=process_model,
        module_path=module_path,
    )
    cfg.update(pm_payload)
    cfg["process_model::name"] = process_model.lower()
    validate_embedding_mode_tokens(cfg)
    _apply_process_aliases(cfg, process_model=process_model)
    if resolve_model_io:
        resolve_model_io_config(cfg, split=split)
    return cfg
