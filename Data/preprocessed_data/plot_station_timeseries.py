from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
STATION_TEMPLATE = "station_{site_id}.csv"
OUT_DIR = BASE_DIR / "plots"

# Order of variables to plot (top to bottom).
VAR_ORDER: List[str] = ["PAR", "Tair", "VPD", "fapar", "Precip", "GPP", "ET"]


def plot_site(site_id: int, variables: Sequence[str] = VAR_ORDER) -> Path:
    """Plot a column of time series (one variable per row) for the given site."""
    station_path = BASE_DIR / STATION_TEMPLATE.format(site_id=site_id)
    df = pd.read_csv(station_path, parse_dates=["date"])

    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 14), sharex=True)
    fig.suptitle(f"Site {site_id} time series", y=0.995)

    for ax, var in zip(axes, variables):
        if var not in df.columns:
            ax.text(0.5, 0.5, f"{var} missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.plot(df["date"], df[var], linewidth=0.5)
        ax.set_ylabel(var)
        ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUT_DIR / f"site_{site_id}_{len(variables)}x1.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    site_ids = [5, 12, 14, 21]
    outputs = [plot_site(sid) for sid in site_ids]
    print("Saved plots:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
