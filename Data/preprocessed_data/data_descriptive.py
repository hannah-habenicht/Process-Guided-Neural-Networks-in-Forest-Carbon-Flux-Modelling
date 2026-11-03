from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Quick one-off script to collect basic metadata and climate variable stats for the key sites.

BASE_DIR = Path(__file__).resolve().parent
SITE_IDS = [5, 12, 14, 21]
SITE_META_DIR = BASE_DIR / "site_meta"
STATION_TEMPLATE = "station_{site_id}.csv"
VAR_MAP: Dict[str, str] = {
    "PAR": "PAR",
    "tair": "Tair",
    "VPD": "VPD",
    "fapar": "fapar",
    "precip": "Precip",
    "GPP": "GPP",
    "ET": "ET"
}


def climate_snippet(description_raw: str) -> str:
    """Pick the first sentence mentioning climate (fallback to first sentence)."""
    sentences = re.split(r"(?<=[.!?])\s+", description_raw)
    sentences = [s.strip() for s in sentences if s.strip()]
    for sentence in sentences:
        if "climate" in sentence.lower():
            return sentence
    return sentences[0] if sentences else ""

def load_meta(site_id: int) -> dict:
    meta_path = SITE_META_DIR / f"site_{site_id}_meta.json"
    with meta_path.open() as f:
        return json.load(f)

def summarize_site(site_id: int) -> dict:
    meta = load_meta(site_id)
    station_path = BASE_DIR / STATION_TEMPLATE.format(site_id=site_id)
    df = pd.read_csv(station_path, parse_dates=["date"])

    years: List[int] = sorted(df["date"].dt.year.unique().tolist())
    desc_raw = meta.get("description", {}).get("description_raw", "")

    row = {
        "site_id": site_id,
        "site_name": meta.get("description", {}).get("site_name", ""),
        "years": years,
        "n_years": len(years),
        "tree_type": meta.get("species")
        or ("conifer" if meta.get("is_conifer") else "broadleaf"),
        "climate_info": climate_snippet(desc_raw),
        "nat_veg": meta.get("description", {}).get("natVegetation_code1", ""),
    }

    for label, col in VAR_MAP.items():
        if col not in df.columns:
            row[f"{label} (mean+-std)"] = "nan +- nan"
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mean = series.mean()
        std = series.std()
        row[f"{label} (mean+-std)"] = f"{mean:.3f} +- {std:.3f}"

    return row


def main() -> None:
    rows = [summarize_site(site_id) for site_id in SITE_IDS]
    summary_df = pd.DataFrame(rows)

    stats_cols = [
        "site_id",
        "site_name",
        "years",
        "n_years",
        "tree_type",
        "climate_info",
        "nat_veg",
        *[f"{v} (mean+-std)" for v in VAR_MAP],
    ]
    summary_df = summary_df[stats_cols]

    pd.set_option("display.max_colwidth", 120)
    print(summary_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:0.3f}"))


if __name__ == "__main__":
    main()
