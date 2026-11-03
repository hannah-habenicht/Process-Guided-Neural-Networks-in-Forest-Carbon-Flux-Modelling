from __future__ import annotations
import os, joblib
from collections.abc import Mapping

def batch_explain(
    explanation_fn,
    prediction_types,
    logs_root,
    cv_config,
    save_dir,
    file_prefix,
    target_list=("GPP", "ET"),
    use_process=True,
    overwrite=False,
    **explanation_kwargs,
):
    """
    Runs explanation_fn once per prediction_type, caches per-type results,
    accumulates global/process blocks (only when non-empty), and returns:

      {
        "<pred_type>": { ... per-type blocks only ... },
        "process": { ... merged process_* blocks ... }   # if present
      }

    The explainer may return:
      - dict with mixed blocks (some per-type mappings, some global/process_*),
      - or (per_type_mapping, global_block1, global_block2, ...).
    """

    os.makedirs(save_dir, exist_ok=True)

    # NEVER call explainer for "process"
    prediction_types = [pt for pt in prediction_types if pt != "process"]

    def get_save_path(pred_type: str) -> str:
        return os.path.join(save_dir, f"{file_prefix}_{pred_type}.joblib")

    def is_predtype_mapping(val) -> bool:
        return isinstance(val, Mapping) and any(k in prediction_types for k in val.keys())

    def coerce_results_to_dict(results):
        if isinstance(results, Mapping):
            return dict(results)
        if isinstance(results, (list, tuple)):
            merged = {}
            for i, part in enumerate(results):
                if isinstance(part, Mapping):
                    if i == 0:
                        merged["__per_type__"] = dict(part)   # first = per-type mapping
                    else:
                        for k, v in part.items():
                            merged[k] = v if k not in merged else merged[k]
                else:
                    merged[f"part{i}"] = part
            return merged
        return {"__result__": results}

    # Accumulate process/global blocks here (skip empties, deep-merge mappings)
    global_results: dict[str, object] = {}

    def merge_global_block(key: str, val):
        if not val:  # skip empties ({} or falsy arrays)
            return
        if key not in global_results:
            global_results[key] = val
            return
        existing = global_results[key]
        
        # Shallow/deep merge for mappings
        if isinstance(existing, Mapping) and isinstance(val, Mapping):
            for subk, subv in val.items():
                if (subv is None) or (isinstance(subv, Mapping) and not subv):
                    continue
                if subk not in existing:
                    existing[subk] = subv
                else:
                    if isinstance(existing[subk], Mapping) and isinstance(subv, Mapping):
                        # one level deeper merge (outer_fold -> target map, etc.)
                        for k2, v2 in subv.items():
                            if v2 is not None:
                                if k2 not in existing[subk]:
                                    existing[subk][k2] = v2
                                elif isinstance(existing[subk][k2], Mapping) and isinstance(v2, Mapping):
                                    #existing[subk][k2].update({kk: vv for kk, vv in v2.items() if vv})
                                    existing[subk][k2].update({kk: vv for kk, vv in v2.items() if vv is not None})
                                elif not existing[subk][k2]:
                                    existing[subk][k2] = v2
                    elif not existing[subk]:
                        existing[subk] = subv
            return
        # For non-mapping: prefer first truthy
        if not existing:
            global_results[key] = val

    all_model_results: dict[str, dict] = {}
    # {pred_type: {outer_fold: {config_k: stats}}}; kept out of the process merge,
    # which would let later prediction types overwrite earlier ones' constants.
    norm_stats_all: dict[str, object] = {}

    for pred_type in prediction_types:
        result_path = get_save_path(pred_type)

        # --- load or compute per-type raw result ---
        if os.path.exists(result_path) and not overwrite:
            results_raw = joblib.load(result_path)
        else:
            partial_results = None
            ckpt_dir = explanation_kwargs.get("save_intermediate_dir")
            # Resume from a crashed run only when not overwriting; otherwise a rerun
            # would silently keep every already-checkpointed fold/config.
            if ckpt_dir and not overwrite:
                ckpt_path = os.path.join(ckpt_dir, f"ale_ckpt_{file_prefix}_{pred_type}.joblib")
                if os.path.exists(ckpt_path):
                    partial_results = joblib.load(ckpt_path)

            results_raw = explanation_fn(
                prediction_types=[pred_type],
                logs_root=logs_root,
                cv_config=cv_config,
                target_list=list(target_list),
                save_dir=save_dir,
                existing_results=partial_results,
                **explanation_kwargs,
            )
            joblib.dump(results_raw, result_path)

        # --- normalize structure & slice for this pred_type ---
        coerced = coerce_results_to_dict(results_raw)
        per_type_view: dict[str, object] = {}

        for key, val in coerced.items():
            if key == "normalization_stats":
                if isinstance(val, Mapping) and pred_type in val:
                    norm_stats_all[pred_type] = val[pred_type]
                elif val:  # legacy layout without a prediction-type level
                    norm_stats_all[pred_type] = val
                continue
            # 1) Hoist content for this pred_type from a top-level per-type block
            if key == "__per_type__" and is_predtype_mapping(val):
                entry = val.get(pred_type)
                if isinstance(entry, Mapping):
                    per_type_view.update(entry)
                else:
                    per_type_view["value"] = entry
                continue

            # 2) Blocks that are themselves per-type mappings (e.g., {"model_1D_ale": {"naive": ..., ...}})
            if is_predtype_mapping(val):
                per_type_view[key] = val.get(pred_type)
                continue

            # 3) Global/process blocks (e.g., "process_1D_ale", "process_pairwise_ale", ...)
            if use_process:
                merge_global_block(key, val)

        all_model_results[pred_type] = per_type_view

    # Expose merged process blocks once.
    # normalization_stats is promoted to the top level so run_ale.py can
    # extract and save it separately; it does not belong in the process block.
    if use_process and global_results:
        all_model_results["process"] = global_results
    if norm_stats_all:
        all_model_results["normalization_stats"] = norm_stats_all

    return all_model_results
