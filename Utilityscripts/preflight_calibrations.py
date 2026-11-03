"""
Check that every PRELES calibration bundle a CV split needs exists and is current.

Called by batch_scripts/metaschedule_paper1.sh before submitting, once per
(CV config, sparsity). Every outer train split and every inner-loop train split needs
its own bundle (params.json + metadata.json). Sparse bundles must carry
sparse_mode == "daily_run_then_thin"; older sparse bundles ran PRELES on the thinned
rows and are rejected by the training drivers.

Usage:
    python Utilityscripts/preflight_calibrations.py \
        --cv-config configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json \
        [--sparsity 30] [--calibration-root Data/calibrations] [--show-paths]

Exit code 0 when all bundles are present and current, 1 otherwise. The summary line
reports "<n> missing" for the caller to parse.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration_paths import _sparse_mode, _normalize_sparse, build_calibration_paths  # noqa: E402


def train_splits(cv: dict) -> list:
    """All train-station mappings a CV config calibrates on (outer and inner)."""
    splits = []
    if "station" in cv:  # temporal schema: out_l_* with train_hold_out + inner_loops
        station = str(cv["station"])
        for key, fold in cv.items():
            if not key.startswith("out_l_") or "_otrain" in key:
                continue
            splits.append({station: fold["train_hold_out"]})
            for inner in fold["inner_loops"].values():
                splits.append({station: inner["train"]})
    else:  # spatial schema: outer_station_* / outer_fold_* with outer_split + inner_loop_*
        for key, fold in cv.items():
            if not key.startswith(("outer_station_", "outer_fold_")) or "_otrain" in key:
                continue
            splits.append(fold["outer_split"]["train"])
            for inner_key, inner in fold.items():
                if inner_key.startswith("inner_loop"):
                    splits.append(inner[0]["data::train::station_id"])
    return [{str(s): [int(y) for y in years] for s, years in split.items()} for split in splits]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cv-config", required=True)
    parser.add_argument("--sparsity", type=int, default=None)
    parser.add_argument("--calibration-root", default=os.path.join("Data", "calibrations"))
    parser.add_argument("--show-paths", action="store_true")
    args = parser.parse_args()

    with open(args.cv_config, "r", encoding="utf-8") as f:
        cv = json.load(f)

    config = {
        "calibration::root": os.path.abspath(args.calibration_root),
        "calibration::cv_type": "spatial",  # temporal PRELES calibrations are stored under spatial/ too
    }
    want_mode = _sparse_mode("preles", _normalize_sparse(args.sparsity))

    seen, missing, stale = set(), [], []
    for split in train_splits(cv):
        folder = build_calibration_paths(config, split, "preles", sparse=args.sparsity)["folder"]
        if folder in seen:
            continue
        seen.add(folder)
        params = os.path.join(folder, "params.json")
        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.isfile(params) or not os.path.isfile(meta_path):
            missing.append(folder)
            continue
        if want_mode:
            with open(meta_path, "r", encoding="utf-8") as f:
                if json.load(f).get("sparse_mode") != want_mode:
                    stale.append(folder)

    for label, folders in (("MISSING", missing), ("STALE", stale)):
        if args.show_paths:
            for folder in folders:
                print(f"{label} {folder}")

    bad = len(missing) + len(stale)
    print(
        f"{os.path.basename(args.cv_config)} [{_normalize_sparse(args.sparsity)}]: "
        f"{len(seen)} bundles, {bad} missing ({len(missing)} absent, {len(stale)} stale sparse_mode)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
