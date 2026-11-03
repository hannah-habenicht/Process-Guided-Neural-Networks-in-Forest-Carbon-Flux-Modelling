"""
CV config inspection utility.

Usage
-----
# Inspect a single config
python inspect_cv_config.py configs/CV_splits/CV_dfb_module_misspec_6folds_config.json

# Compare two configs side-by-side (summary tables only)
python inspect_cv_config.py configs/CV_splits/CV_dfb_module_misspec_6folds_config.json \
                            configs/CV_splits/CV_dfb_module_misspec_mixed_train_6folds_config.json

# Show only the summary table, skip per-fold detail
python inspect_cv_config.py <path> --summary-only

# Inspect specific folds by name (substring match)
python inspect_cv_config.py <path> --folds DE-Tha US-PFa
"""

import argparse
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DIVIDER  = "=" * 72
SUBDIV   = "-" * 72

def _fmt_stations(station_dict: Dict, site_meta: Dict = None) -> str:
    """Format {site: [years]} compactly, optionally annotating variant."""
    parts = []
    for site, years in station_dict.items():
        variant = (site_meta or {}).get(site, {}).get("variant", "")
        tag = f"({variant[:3]})" if variant else ""
        parts.append(f"{site}{tag}:{years}")
    return "  " + "\n  ".join(parts) if parts else "  (none)"


def _detect_format(config: Dict) -> str:
    """Return 'temporal', 'spatial_old', or 'ncv'."""
    if "station" in config:
        return "temporal"
    if any(k.startswith("outer_fold_") for k in config):
        return "ncv"
    if any(k.startswith("outer_station_") for k in config):
        return "spatial_old"
    return "unknown"


def _fold_keys(config: Dict, prefix: str, filter_substrings: Optional[List[str]] = None) -> List[str]:
    keys = sorted(k for k in config if k.startswith(prefix))
    if filter_substrings:
        keys = [k for k in keys if any(f in k for f in filter_substrings)]
    return keys


# ---------------------------------------------------------------------------
# Format: temporal (out_l_*)
# ---------------------------------------------------------------------------
def print_temporal(config: Dict, **_):
    print(f"\nStation:   {config.get('station')}")
    print(f"All years: {config.get('all_years')}")
    for outer_key in sorted(k for k in config if k.startswith("out_l_")):
        outer = config[outer_key]
        print(f"\n  {outer_key}:")
        print(f"    Hold-out years:      {outer.get('hold_out_years')}")
        print(f"    Train on hold-out:   {outer.get('train_hold_out')}")
        print(f"    Inner years:         {outer.get('inner_years')}")
        for inner_k, inner_v in outer.get("inner_loops", {}).items():
            print(f"    {inner_k}: train={inner_v['train']}  test={inner_v['test']}  val={inner_v['val']}")


# ---------------------------------------------------------------------------
# Format: old spatial (outer_station_*)
# ---------------------------------------------------------------------------
def print_spatial_old(config: Dict, filter_folds: Optional[List[str]] = None,
                       summary_only: bool = False, **_):
    pprint({"All stations": config.get("all_stations")})
    for outer_key in _fold_keys(config, "outer_station_", filter_folds):
        print(f"\n{outer_key}:")
        outer = config[outer_key]
        print("  Outer train:")
        pprint(outer["outer_split"]["train"], indent=4)
        print("  Outer test:")
        pprint(outer["outer_split"]["test"], indent=4)
        if summary_only:
            continue
        for loop_k in sorted(k for k in outer if k.startswith("inner_loop_")):
            print(f"  {loop_k}:")
            for step in outer[loop_k]:
                for role, stations in step.items():
                    print(f"    {role}:")
                    pprint(stations, indent=6)


# ---------------------------------------------------------------------------
# Format: NCV (outer_fold_*)
# ---------------------------------------------------------------------------
def _ncv_detect_columns(folds: List[Dict]) -> List[str]:
    """Return the list of optional column names that are actually present across folds."""
    optional = [
        ("forest_type",                       "module"),
        ("koppen_group",                      "koppen"),
        ("precip_group",                      "precip"),
        ("test_site_natural_variant",         "natural"),
        ("is_matched",                        "matched"),
        ("train_pool_type",                   "pool"),
        ("outer_train_composition_pct_conifer", "%con"),
    ]
    present = []
    for field, col_name in optional:
        if any(field in fold for fold in folds):
            present.append((field, col_name))
    return present


def _ncv_summary_table(config: Dict, label: str = "", filter_folds: Optional[List[str]] = None):
    """Print a compact summary table for an NCV config."""
    fold_keys = _fold_keys(config, "outer_fold_", filter_folds)
    folds = [config[fk] for fk in fold_keys]

    if label:
        print(f"\n{DIVIDER}")
        print(f"  {label}")
    print(DIVIDER)
    print(f"  Folds: {len(fold_keys)}   |   Sites with data: {len(config.get('all_stations', {}))}")
    print(DIVIDER)

    # Build dynamic columns based on what's actually in this config
    optional_cols = _ncv_detect_columns(folds)
    col_widths = [10] * len(optional_cols) + [14]  # last col = test_yrs
    fmt_parts = ["{:<38}"] + [f"{{:>{w}}}" for w in col_widths]
    col_fmt = " ".join(fmt_parts)

    header_vals = ["Fold"] + [name for _, name in optional_cols] + ["test_yrs"]
    print(col_fmt.format(*header_vals))
    print(SUBDIV)

    for fk, fold in zip(fold_keys, folds):
        name = fk.replace("outer_fold_", "")
        row = [name[:38]]
        for field, _ in optional_cols:
            val = fold.get(field)
            if field == "is_matched":
                cell = str(val) if val is not None else "mixed"
            else:
                cell = str(val) if val is not None else "?"
            row.append(cell[:10])
        test_yrs = list(fold.get("outer_split", {}).get("test", {}).values())
        row.append(str(test_yrs[0]) if test_yrs else "?")
        print(col_fmt.format(*row))
    print()


def _ncv_fold_detail(fold_key: str, fold: Dict, site_meta: Dict):
    """Print detailed breakdown for one NCV outer fold."""
    print(f"\n{DIVIDER}")
    print(f"  {fold_key}")
    # Show whatever metadata fields are present
    meta_fields = [
        ("forest_type",                       "module"),
        ("koppen_group",                      "koppen"),
        ("precip_group",                      "precip"),
        ("train_pool_type",                   "pool"),
        ("test_site_natural_variant",         "natural"),
        ("is_matched",                        "matched"),
        ("outer_train_composition_pct_conifer", "outer_%con"),
    ]
    parts = []
    for field, label in meta_fields:
        if field in fold:
            val = fold[field]
            if field == "is_matched" and val is None:
                val = "mixed/None"
            parts.append(f"{label}={val}")
    if parts:
        print("  " + "  ".join(parts))
    print(SUBDIV)

    outer = fold.get("outer_split", {})
    print("  Outer train:")
    print(_fmt_stations(outer.get("train", {}), site_meta))
    print("  Outer test:")
    print(_fmt_stations(outer.get("test", {}), site_meta))

    for ik in sorted((k for k in fold if k.startswith("inner_loop_")),
                     key=lambda k: int(k.split("_")[-1])):
        train_cfg, test_cfg, val_cfg = fold[ik]
        meta = train_cfg.get("inner_loop_meta", {})
        train_sites = train_cfg.get("data::train::station_id", {})
        inner_test  = test_cfg.get("data::test::station_id", {})
        val_sites   = val_cfg.get("data::val::station_id", {})

        def fmt_inline(sd):
            parts = []
            for s, yrs in sd.items():
                v = site_meta.get(s, {}).get("variant", "")
                tag = f"({v[:3]})" if v else ""
                parts.append(f"{s}{tag}:{yrs}")
            return ", ".join(parts)

        meta_parts = []
        if "draw" in meta:
            meta_parts.append(f"draw={meta['draw']}")
        if "composition_pct_conifer" in meta:
            meta_parts.append(f"%con={meta['composition_pct_conifer']}")
        meta_str = " ".join(meta_parts)
        print(f"  {ik}" + (f" [{meta_str}]" if meta_str else "") + ":")
        print(f"    train: {fmt_inline(train_sites)}")
        print(f"    test:  {fmt_inline(inner_test)}")
        print(f"    val:   {fmt_inline(val_sites)}")


def print_ncv(config: Dict, filter_folds: Optional[List[str]] = None,
              summary_only: bool = False, label: str = "", **_):
    site_meta = config.get("site_metadata", {})
    _ncv_summary_table(config, label=label, filter_folds=filter_folds)
    if summary_only:
        return
    for fk in _fold_keys(config, "outer_fold_", filter_folds):
        _ncv_fold_detail(fk, config[fk], site_meta)


# ---------------------------------------------------------------------------
# Comparison mode (two NCV configs)
# ---------------------------------------------------------------------------
def print_ncv_comparison(cfg_a: Dict, cfg_b: Dict,
                          label_a: str, label_b: str,
                          filter_folds: Optional[List[str]] = None,
                          summary_only: bool = False):
    """Print summary tables for both configs, then paired fold details."""
    _ncv_summary_table(cfg_a, label=label_a, filter_folds=filter_folds)
    _ncv_summary_table(cfg_b, label=label_b, filter_folds=filter_folds)

    if summary_only:
        return

    # Pair folds by the part after "outer_fold_"
    meta_a = cfg_a.get("site_metadata", {})
    meta_b = cfg_b.get("site_metadata", {})

    keys_a = {k.replace("outer_fold_", ""): k for k in _fold_keys(cfg_a, "outer_fold_", filter_folds)}
    keys_b = {k.replace("outer_fold_", ""): k for k in _fold_keys(cfg_b, "outer_fold_", filter_folds)}
    all_names = sorted(set(keys_a) | set(keys_b))

    for name in all_names:
        print(f"\n{'#' * 72}")
        print(f"  FOLD: {name}")
        if name in keys_a:
            _ncv_fold_detail(keys_a[name], cfg_a[keys_a[name]], meta_a)
        else:
            print(f"  (not present in {label_a})")
        if name in keys_b:
            _ncv_fold_detail(keys_b[name], cfg_b[keys_b[name]], meta_b)
        else:
            print(f"  (not present in {label_b})")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
PRINTERS = {
    "temporal":    print_temporal,
    "spatial_old": print_spatial_old,
    "ncv":         print_ncv,
}


def inspect(path: str, filter_folds=None, summary_only=False, label=""):
    with open(path) as f:
        config = json.load(f)
    fmt = _detect_format(config)
    label = label or os.path.basename(path)
    print(f"\n{'#' * 72}")
    print(f"  Config: {label}  [{fmt}]")
    PRINTERS.get(fmt, lambda **kw: print("Unrecognized format."))(
        config=config, filter_folds=filter_folds,
        summary_only=summary_only, label=label,
    )
    return config, fmt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inspect and compare CV split configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("configs", nargs="+", help="Path(s) to config JSON (1 or 2).")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary table only, skip per-fold detail.")
    parser.add_argument("--folds", nargs="*", metavar="SUBSTR",
                        help="Only show folds whose name contains any of these substrings.")
    args = parser.parse_args()

    if len(args.configs) == 1:
        inspect(args.configs[0], filter_folds=args.folds, summary_only=args.summary_only)

    elif len(args.configs) == 2:
        path_a, path_b = args.configs
        with open(path_a) as f:
            cfg_a = json.load(f)
        with open(path_b) as f:
            cfg_b = json.load(f)
        fmt_a = _detect_format(cfg_a)
        fmt_b = _detect_format(cfg_b)

        if fmt_a == "ncv" and fmt_b == "ncv":
            print_ncv_comparison(
                cfg_a, cfg_b,
                label_a=os.path.basename(path_a),
                label_b=os.path.basename(path_b),
                filter_folds=args.folds,
                summary_only=args.summary_only,
            )
        else:
            # Fall back to printing each independently
            inspect(path_a, filter_folds=args.folds, summary_only=args.summary_only)
            inspect(path_b, filter_folds=args.folds, summary_only=args.summary_only)
    else:
        parser.error("Provide 1 or 2 config paths.")


if __name__ == "__main__":
    main()
