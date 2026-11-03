# scripts/run_ale.py
"""
Batch runner for ALE analysis.

See `XAI/scripts/README_ALE.md` for usage details and project conventions.
"""
import os
import sys
import json
import joblib
import glob
import warnings
warnings.filterwarnings("ignore", message="Series.ravel is deprecated", category=FutureWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Config_Files.config_spatial import build_config
from XAI.xai_utils.batch_utils import batch_explain
from XAI.xai_utils.ale_registry import VARIANT_FN
from XAI.xai_utils.experiment_registry import EXPERIMENT_CONFIGS
from XAI.xai_utils.model_utils import infer_ncv_kind

# Marks "full" bundles that hold main ALE(x0) + main ALE(x1) + centred interaction
FULL_EFFECT = "main+interaction"

def save_root(save_dir_base: str, ncv_kind: str, length: str, grid_size: int, variant: str) -> str:
    # e.g. XAI/results/ALE/paper1_spatial/spatial_len12/grid20/main
    return os.path.join(save_dir_base, f"{ncv_kind}_{length}", f"grid{grid_size}", variant)

if __name__ == "__main__":
    config = build_config()
    save = config["cfg::save"]

    print("Starting...")
    
    # ----- CONFIG (EDIT HERE) -----
    experiment = os.getenv("ALE_EXPERIMENT", "spatial")  # key from EXPERIMENT_CONFIGS
    variant = os.getenv("ALE_VARIANT", "pairwise")                 # {"main", "pairwise", "full"}
    prediction_types = ["naive", "bias", "parallel", "regularized", "residual"]
    
    if os.getenv("ALE_PREDICTION_TYPES"):
        prediction_types = [p.strip() for p in os.environ["ALE_PREDICTION_TYPES"].split(",") if p.strip()]
    
    
    # Process outputs set to a physical 0 (residual inputs / parallel offset); "main" only
    preles_zero = os.getenv("ALE_PRELES_ZERO", "").lower() in {"1", "true", "yes"}
    if preles_zero and variant != "main":
        raise ValueError("ALE_PRELES_ZERO is only implemented for the 'main' variant.")
    
    # Full response through PRELES (perturbed day re-run, then the network); "main" and "full"
    chain_process = os.getenv("ALE_CHAIN_PROCESS", "").lower() in {"1", "true", "yes"}
    if chain_process and variant not in {"main", "full"}:
        raise ValueError("ALE_CHAIN_PROCESS is only implemented for the 'main' and 'full' variants.")
    

    overwrite = False

    # Resolve experiment defaults; individual env vars still override
    _exp = EXPERIMENT_CONFIGS[experiment]
    save_dir_base = os.getenv("ALE_SAVE_DIR_BASE", _exp["save_dir_base"])
    length        = os.getenv("ALE_LENGTH",        _exp["length"])
    # None in the registry = derived from `length` (paper-1 spatial layout)
    logs_root     = os.getenv("ALE_LOGS_ROOT",     _exp["logs_root"] or f"checkpoints/preles/full/station_{length}")
    cv_config_name = _exp["cv_config_name"]
    if cv_config_name is None:
        cv_config_name = f"Paper_1_preles_ncvs/CV_station_{length}_config.json"

    # Optional: use ALE_GRID_SIZE from env if set
    grid_size = int(os.getenv("ALE_GRID_SIZE", 20))

    # Optional: checkpoint every N prediction types (set None/0 to disable)
    checkpoint_every = 1
    save_intermediate = True  # writes to ".../<variant>/intermediate/"
    
    pairwise_impute_empty_cells = True  # set False to skip filling empty ALE bins for pairwise grids
    env_impute = os.getenv("ALE_IMPUTE_EMPTY_CELLS")
    if env_impute is not None:
        pairwise_impute_empty_cells = env_impute.lower() not in {"0", "false", "no"}
    # ------------------------------

    print("[CONFIG] experiment:", experiment)
    print("[CONFIG] variant:", variant)
    print("[CONFIG] length:", length)
    print("[CONFIG] prediction_types:", prediction_types)
    print("[CONFIG] logs_root:", logs_root)
    print("[CONFIG] save_dir_base:", save_dir_base)
    print("[CONFIG] grid_size:", grid_size)
    print("[CONFIG] checkpoint_every:", checkpoint_every)
    print("[CONFIG] save_intermediate:", save_intermediate)
    print("[CONFIG] overwrite:", overwrite)
    print("[CONFIG] preles_zero:", preles_zero)
    print("[CONFIG] chain_process:", chain_process)


    if variant == "pairwise":
        print("[CONFIG] pairwise_impute_empty_cells:", pairwise_impute_empty_cells)

    # CV config path is resolved from the experiment registry
    cv_config_path = os.path.join(save, "configs", "CV_splits", cv_config_name)
    
    if not os.path.exists(cv_config_path):
        raise FileNotFoundError(f"CV config not found: {cv_config_path}")
    print("[CONFIG] cv_config_path:", cv_config_path)
    with open(cv_config_path, "r") as f:
        CV_CONFIG = json.load(f)

    # Preflight: warn if any prediction type has no matching logs
    for pred_type in prediction_types:
        logs_glob = os.path.join(logs_root, f"logs_spatial_{pred_type}*")
        if not glob.glob(logs_glob):
            print(f"[WARNING] No log directories found for prediction type '{pred_type}' (pattern: {logs_glob})")
            break

    ncv_kind = infer_ncv_kind(logs_root, prediction_types[0])
    # PRELES-at-0 runs go to .../grid20/main_preles0, next to "main"
    variant_dir = f"{variant}"
    if preles_zero:
        variant_dir += "_preles0"
    if chain_process:
        variant_dir += "_chain"
    root = save_root(save_dir_base, ncv_kind, length, grid_size, variant_dir)
    os.makedirs(root, exist_ok=True)

    # Optional intermediate checkpoint directory per variant
    save_intermediate_dir = os.path.join(root, "intermediate") if save_intermediate else None
    if save_intermediate_dir:
        os.makedirs(save_intermediate_dir, exist_ok=True)

    # Old "full" caches hold the uncentred interaction only (pre 2026-10-14); never reload them
    if variant == "full":
        meta_path = os.path.join(root, f"{variant}_meta.json")
        old_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if (glob.glob(os.path.join(root, f"{variant}_*.joblib"))
                and old_meta.get("full_effect") != FULL_EFFECT):
            raise RuntimeError(f"{root} holds interaction-only 'full' caches; use a new ALE_SAVE_DIR_BASE.")

    # Run
    batch_kwargs = dict(
        grid_size=grid_size,
        save_intermediate_dir=save_intermediate_dir,
        checkpoint_every=checkpoint_every if checkpoint_every else 0,
        overwrite=bool(overwrite),
    )
    if variant == "pairwise":
        batch_kwargs["impute_empty_cells"] = pairwise_impute_empty_cells
    if preles_zero:
        batch_kwargs["preles_zero"] = True
    if chain_process:
        batch_kwargs["chain_process"] = True

    print("Batch variant")
    results = batch_explain(
        explanation_fn=VARIANT_FN[variant],
        prediction_types=prediction_types,
        logs_root=logs_root,
        cv_config=CV_CONFIG,
        save_dir=root,
        file_prefix=variant,  # drives per-type cache names
        target_list=["GPP", "ET"],
        use_process=True,
        **batch_kwargs,
    )

    # Save normalization stats (mean/std per fold) so ALE x-axes can be
    # converted back to physical units at plot time.
    norm_stats = results.pop("normalization_stats", {})
    if norm_stats:
        norm_path = os.path.join(root, f"{variant}_normalization_stats.json")
        with open(norm_path, "w") as _f:
            json.dump(norm_stats, _f, indent=2)
        print(f"[OK] Saved normalization stats to: {norm_path}")

    # Bundle for simple reload
    bundle_path = os.path.join(root, f"{variant}_bundle.joblib")
    joblib.dump(results, bundle_path)

    # Metadata for reproducibility
    meta = dict(
        experiment=experiment,
        variant=variant, season=None, preles_zero=preles_zero, chain_process=chain_process, outer_folds=None, ncv_kind=ncv_kind,
        length=length, grid_size=grid_size,
        prediction_types=prediction_types, feature_pairs=None,
        logs_root=logs_root,
        save_dir_base=save_dir_base,
        cv_config_path=cv_config_path,
    )
    if variant == "full":
        meta["full_effect"] = FULL_EFFECT
    with open(os.path.join(root, f"{variant}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved ALE to: {bundle_path}")
    print(f"[OK] Per-type caches are under: {root}")

