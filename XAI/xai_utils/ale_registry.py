# XAI/xai_utils/ale_registry.py
from __future__ import annotations

import os, json, joblib
from typing import Dict, Tuple, List
from XAI.xai_utils.ale_utils import (
    generate_ale_top_models,
    generate_ale_pairwise_top_models,
    generate_pairwise_full_top_models,
)

from XAI.xai_utils.model_utils import infer_ncv_kind  # already in your repo

# Which generator to call for each “variant”
VARIANT_FN = {
    "main": generate_ale_top_models,                    # returns model_1D_ale + process_1D_ale
    "pairwise": generate_ale_pairwise_top_models,       # returns model_pairwise_ale + process_pairwise_ale
    "full": generate_pairwise_full_top_models,          # returns model_pairwise_full_ale + process_pairwise_full_ale
}

# Which keys to expect in the saved outputs for plotting
VARIANT_KEYS = {
    "main": ("model_1D_ale", "process_1D_ale"),
    "pairwise": ("model_pairwise_ale", "process_pairwise_ale"),
    "full": ("model_pairwise_full_ale", "process_pairwise_full_ale"),
}

def save_root(save_dir_base: str, ncv_kind: str, length: str, grid_size: int) -> str:
    # e.g. XAI/results/ALE/spatial_len12/grid20
    return os.path.join(save_dir_base, f"{ncv_kind}_{length}", f"grid{grid_size}")

def variant_paths(root: str, variant: str):
    # Directory + helpers for consistent filenames
    vdir = os.path.join(root, variant)
    os.makedirs(vdir, exist_ok=True)
    def per_type(pred_type: str) -> str:
        # e.g. .../main/main_naive.joblib
        return os.path.join(vdir, f"{variant}_{pred_type}.joblib")
    bundle = os.path.join(vdir, f"{variant}_bundle.joblib")
    meta = os.path.join(vdir, f"{variant}_meta.json")
    return vdir, per_type, bundle, meta

def run_and_save_ale_variant(
    *,
    variant: str,
    prediction_types: List[str],
    logs_root: str,
    cv_config: dict,
    save_dir_base: str,
    length: str,
    grid_size: int,
    target_list=("GPP", "ET"),
    use_process=True,
    overwrite=False,
    pairwise_impute_empty_cells=True,
) -> Dict:
    assert variant in VARIANT_FN, f"Unknown variant {variant}"
    ncv_kind = infer_ncv_kind(logs_root, prediction_types[0])
    root = save_root(save_dir_base, ncv_kind, length, grid_size)
    vdir, per_type_path, bundle_path, meta_path = variant_paths(root, variant)

    # Use your batch runner; keep a simple file_prefix (variant) so per-type files are predictable
    from XAI.xai_utils.batch_utils import batch_explain
    batch_kwargs = dict(grid_size=grid_size, overwrite=overwrite)
    if variant == "pairwise":
        batch_kwargs["impute_empty_cells"] = pairwise_impute_empty_cells

    results = batch_explain(
        explanation_fn=VARIANT_FN[variant],
        prediction_types=prediction_types,
        logs_root=logs_root,
        cv_config=cv_config,
        save_dir=vdir,
        file_prefix=variant,               # <— drives per-type cache names
        target_list=list(target_list),
        use_process=use_process,
        **batch_kwargs,
    )
    joblib.dump(results, bundle_path)

    # Save lightweight metadata to make reloading self-describing
    meta = dict(
        variant=variant, ncv_kind=ncv_kind, length=length, grid_size=grid_size,
        prediction_types=prediction_types, logs_root=logs_root
    )
    if variant == "full":
        meta["full_effect"] = "main+interaction"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return results

def load_ale_results(*, save_dir_base: str, variant: str, length: str, grid_size: int):
    assert variant in VARIANT_KEYS, f"Unknown variant {variant}"
    # Try the bundle first; fall back to per-type
    # We don’t need ncv_kind at load time if we scan subdirs
    # but it’s simpler to require the same path logic used at save time:
    for ncv_kind in ("spatial","temporal"):
        root = save_root(save_dir_base, ncv_kind, length, grid_size)
        vdir, _per_type, bundle_path, _meta_path = variant_paths(root, variant)
        if os.path.exists(bundle_path):
            return joblib.load(bundle_path)
    raise FileNotFoundError(f"No ALE bundle found for {variant=} {length=} grid{grid_size} under {save_dir_base}")
