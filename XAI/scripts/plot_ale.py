import os
import json
import joblib
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import numpy as np
from matplotlib import cm
from itertools import combinations
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# from XAI.xai_utils.batch_utils import batch_explain
# from XAI.xai_utils.ale_utils import generate_ale_for_all_top_models, generate_ale_top_models


def _resolve_pairwise_key(pred_results: dict, pred_type: str) -> Optional[str]:
    """
    Locate the correct pairwise ALE block for the requested prediction type.

    Supports both the classic pairwise keys and the full-effect variant and
    returns the first matching key that exists, or None if no block is found.
    """
    if not pred_results:
        return None

    if pred_type == "process":
        candidates = ("process_pairwise_ale", "process_pairwise_full_ale")
    else:
        candidates = ("model_pairwise_ale", "model_pairwise_full_ale")

    for key in candidates:
        if key in pred_results:
            return key
    return None

def _is_target_block(payload: object) -> bool:
    return isinstance(payload, dict) and ("GPP" in payload or "ET" in payload)

def _resolve_station_payload(station_payload: object, split: str = "test") -> dict:
    """
    Support both layouts:
      legacy: station -> target -> feature -> df
      new:    station -> {train,test} -> target -> feature -> df
      draws:  station -> config_<i> -> {train,test} -> target -> feature -> df
              (one entry per trained draw; config_0 is the best HPO configuration)
    """
    if not isinstance(station_payload, dict):
        return {}
    station_payload = station_payload.get("config_0", station_payload)
    if _is_target_block(station_payload):
        return station_payload
    if split in station_payload and _is_target_block(station_payload.get(split)):
        return station_payload[split]
    for candidate in ("test", "train"):
        if candidate in station_payload and _is_target_block(station_payload.get(candidate)):
            return station_payload[candidate]
    for value in station_payload.values():
        if _is_target_block(value):
            return value
    return {}

def _first_station_payload(station_results: dict, split: str = "test") -> dict:
    if not isinstance(station_results, dict):
        return {}
    for station in station_results:
        payload = _resolve_station_payload(station_results.get(station), split=split)
        if payload:
            return payload
    return {}


def _physical_frame(df: pd.DataFrame, target: str, stats: dict) -> pd.DataFrame:
    """One ALE frame in physical units. ALE is a difference, so the effect gets no mean shift."""
    xp, yp = stats["x_parameters"], stats["y_parameters"]
    y_std = stats["y_std"][yp.index(target)]
    def to_x(values, feature):
        if feature not in xp:
            return values
        i = xp.index(feature)
        return values * stats["x_std"][i] + stats["x_mean"][i]
    df = df.copy()
    if "eff" in df.columns:  # 1D: feature, eff, size, CI columns
        df[df.columns[0]] = to_x(df[df.columns[0]].to_numpy(dtype=float), df.columns[0])
        for col in ("eff", "lowerCI_95%", "upperCI_95%"):
            if col in df.columns:
                df[col] = df[col] * y_std
        return df
    # 2D: rows = grid edges of one feature, columns = grid edges of the other
    df.index = pd.Index(to_x(df.index.to_numpy(dtype=float), df.index.name), name=df.index.name)
    df.columns = pd.Index(to_x(df.columns.to_numpy(dtype=float), df.columns.name), name=df.columns.name)
    return df * y_std


def to_physical_units(ale_results: dict, norm_stats: dict) -> dict:
    """
    Copy of an ALE bundle with the model curves in physical units.

    Model ALEs are computed in the z-scored model space; `norm_stats` holds the train-fitted
    constants saved next to the bundle (<variant>_normalization_stats.json). Process-model
    ALEs are physical already and stay as they are; chained ALEs carry identity x constants,
    so only their effect is scaled.
    """
    out = {}
    for pred_type, blocks in ale_results.items():
        out[pred_type] = dict(blocks)
        if pred_type == "process":
            continue
        for block, folds in blocks.items():
            if not block.startswith("model_") or "cell_counts" in block:
                continue
            out[pred_type][block] = {
                fold: {
                    cfg: {
                        split: {
                            target: {f: _physical_frame(df, target, norm_stats[pred_type][fold][cfg])
                                     for f, df in feats.items()}
                            for target, feats in targets.items()
                        }
                        for split, targets in splits.items()
                    }
                    for cfg, splits in draws.items()
                }
                for fold, draws in folds.items()
            }
    return out


def plot_ale_results(
    ale_results,
    plot_mode="feature_vs_output",
    features=None,
    stations=None,
    output_type="GPP",
    prediction_type=None,
    save_dir=None,
    show=True,
    n_cols=2,
    figsize=(10, 5),
    split="train",
):
    """
    Flexible ALE plotting function for multiple configurations.

    Parameters:
        ale_results: dict of dicts — {station_id: {"GPP": {feature: df}, "ET": {...}}}
        plot_mode: 
            - "feature_vs_output": One feature, GPP/ET subplots, lines by station
            - "feature_vs_station": One station, GPP/ET subplots, lines by feature
            - "single_output_all_features": One output (GPP or ET), one station, subplots by feature
        features: list of features to include (default: all)
        stations: list of stations to include (default: all)
        output_type: "GPP" or "ET" (used in single-output mode)
        prediction_type: optional string for plot titles
        save_dir: if provided, saves plots to this directory
        show: if True, shows plots
        n_cols: number of subplot columns (when plotting many features)
        figsize: base figure size per plot or subplot
        split: which dataset split to plot when station payload includes
               nested {"train": ..., "test": ...}; defaults to "test"
    """
    prediction_types = (
        prediction_type if isinstance(prediction_type, list) 
        else [prediction_type] if prediction_type 
        else list(ale_results.keys())
                  )
            
    selected_stations = None
    # for pred in prediction_types:
    #     if pred in ale_results:
    #         station_results = ale_results[pred]
    #         break

    # selected_stations = stations or list(station_results.keys())        
    feature_units = {
        "PAR": "µmol m⁻² s⁻¹",
        "Tair": "°C",
        "VPD": "kPa",
        "Precip": "mm d⁻¹",
        #"CO2": "ppm",
        "fapar": "-",
        #"DOY": "day",
        "GPP": "gC m⁻² d⁻¹",
        "ET": "mm d⁻¹",
        "GPP_pred": "gC m⁻² d⁻¹",
        "ET_pred": "mm d⁻¹",
        # Add any additional feature names here
    }

    # Generate all unordered feature pairs
    for pred_type in prediction_types:
        if pred_type == 'residual':
            feature_pairs = [list(pair) for pair in combinations(['PAR', 'Tair','VPD', 'Precip', 'fapar', 'GPP_pred', 'ET_pred'], 2)]
        else: 
            feature_pairs = [list(pair) for pair in combinations(['PAR', 'Tair','VPD', 'Precip', 'fapar'], 2)]

    if plot_mode == "feature_vs_output":
        for pred_type in prediction_types:
            if pred_type not in ale_results:
                continue
        
            # Get station list for this pred_type
            if pred_type == "process":
                station_results = ale_results[pred_type].get("process_1D_ale", {})
                sample_payload = _first_station_payload(station_results, split=split)
                sample_gpp = sample_payload.get("GPP", {})
                selected_features = features or list(sample_gpp.keys())
            elif pred_type == "bias":
                selected_features = ["GPP_pred", "ET_pred"]
                station_results = ale_results[pred_type].get("model_1D_ale", {})
            else:
                station_results = ale_results[pred_type].get("model_1D_ale", {})
                sample_payload = _first_station_payload(station_results, split=split)
                sample_gpp = sample_payload.get("GPP", {})
                selected_features = features or list(sample_gpp.keys())
            
            selected_stations = stations or list(station_results.keys())  
 
                
            for feature in selected_features:
                fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
                
                for i, out in enumerate(["GPP", "ET"]):
                    for station in selected_stations:
                        try:
                            station_payload = _resolve_station_payload(station_results[station], split=split)
                            df = station_payload[out][feature]
                            
                            # x_col = feature if feature in df.columns else "feature"
                            axs[i].plot(df[feature], df["eff"], label=f"{station}")
                            
                            if 'lowerCI_95%' in df.columns and 'upperCI_95%' in df.columns:
                                    axs[i].fill_between(
                                        df[feature],
                                        df['lowerCI_95%'],
                                        df['upperCI_95%'],
                                        alpha=0.2
                                    )
                        except KeyError:
                            continue
                    axs[i].set_title(f"{out} - {feature} ({pred_type})")
                    axs[i].set_xlabel(feature)
                    axs[i].grid(True)
                    axs[i].legend()
                    if i == 0:
                        axs[i].set_ylabel("Accumulated Local Effects")

                # fig.tight_layout()
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fname = f"ALE_{feature}_vs_outputs_{pred_type}.png"
                    plt.savefig(os.path.join(save_dir, fname))
                if show:
                    plt.show()
                plt.close(fig)

    if plot_mode == "station_vs_prediction":
        # Pick a reference pred_type to get station keys from
        ref_pred_type = prediction_types[0] if prediction_types else next(iter(ale_results))
        if ref_pred_type == "process":
            station_results = ale_results[ref_pred_type].get("process_1D_ale", {})
        else:
            station_results = ale_results[ref_pred_type].get("model_1D_ale", {})

        selected_stations = stations or list(station_results.keys())

        for station in selected_stations:
            for pred_type in prediction_types:
                # Determine the correct feature set for this prediction type
                if pred_type == "process":
                    station_results = ale_results[pred_type].get("process_1D_ale", {})
                    sample_payload = _first_station_payload(station_results, split=split)
                    sample_gpp = sample_payload.get("GPP", {})
                    selected_features = features or list(sample_gpp.keys())
                else:
                    station_results = ale_results[pred_type].get("model_1D_ale", {})
                    sample_payload = _first_station_payload(station_results, split=split)
                    sample_gpp = sample_payload.get("GPP", {})
                    selected_features = features or list(sample_gpp.keys())
                    
                # # Pick features from first available station + prediction type
                # if ref_pred_type == "process":
                #     sample_data = ale_results[ref_pred_type]["process_1D_ale"][selected_stations[0]]
                # else:
                #     sample_data = ale_results[ref_pred_type]["model_1D_ale"][selected_stations[0]]
                # selected_features = features or list(sample_data["GPP"].keys())

                for feature in selected_features:
                    n_rows = len(selected_stations)
                    n_cols = 2  # GPP and ET
                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), sharex=True, sharey=True)

                    for row_idx, station in enumerate(selected_stations):
                        for col_idx, out in enumerate(["GPP", "ET"]):
                            ax = axs[row_idx, col_idx] if n_rows > 1 else axs[col_idx]
                            for pt in prediction_types:
                                try:
                                    if pt == "process":
                                        station_payload = _resolve_station_payload(
                                            ale_results[pt]["process_1D_ale"][station], split=split
                                        )
                                    else:
                                        station_payload = _resolve_station_payload(
                                            ale_results[pt]["model_1D_ale"][station], split=split
                                        )
                                    df = station_payload[out][feature]
                                    
                                    ax.plot(df[feature], df["eff"], label=pt)
                                    # sns.rugplot(x=x_df[feature], ax=axs[i], color='gray', height=0.05, alpha=0.3)
                                    if "lowerCI_95%" in df.columns and "upperCI_95%" in df.columns:
                                        ax.fill_between(df[feature], df["lowerCI_95%"], df["upperCI_95%"], alpha=0.2)
                                except KeyError:
                                    continue
                            ax.set_title(f"{out} | Station {station}", fontsize=16)
                            if row_idx == n_rows - 1:
                                unit = feature_units.get(feature, "")
                                ax.set_xlabel(f"{feature}({unit})", fontsize=14)
                            if col_idx == 0:
                                unit = feature_units.get(out, "")
                                ax.set_ylabel(f"ALE ({unit})", fontsize=14)
                            if col_idx == 1:
                                unit = feature_units.get(out, "")
                                ax.set_ylabel(f"ALE ({unit})", fontsize=14)

                            ax.grid(True)
                            if row_idx == 0 and col_idx == 1:
                                ax.legend(loc="upper right", fontsize=14)

                    fig.suptitle(f"ALE effect of {feature} across stations and prediction types", fontsize=22)
                    fig.tight_layout(rect=[0, 0, 1, 0.95])

                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        fname = f"ALE_grid_{feature}_station_vs_prediction.png"
                        plt.savefig(os.path.join(save_dir, fname))
                    plt.close(fig)

    elif plot_mode == "single_output_all_features":
        for station in stations:
            n_features = len(features)
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
            axs = axs.flatten()

            for i, feature in enumerate(features):
                ax = axs[i]
                try:
                    df = ale_results[station][output_type][feature]
                    ax.plot(df.index, df['eff'])
                    ax.set_title(feature)
                    ax.set_xlabel(feature)
                    ax.set_ylabel("ALE")
                    ax.grid(True)
                except KeyError:
                    ax.set_visible(False)

            for j in range(i + 1, len(axs)):
                axs[j].set_visible(False)

            fig.suptitle(f"{output_type} ALE for Station {station} ({prediction_type})")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"ALE_{output_type}_all_features_station_{station}_{prediction_type or ''}.png"
                plt.savefig(os.path.join(save_dir, fname))
            if show:
                plt.show()
            plt.close()
    
    # elif plot_mode == "residual_vs_station":
    #     selected_features = candidate_keys
    #     for feature in selected_features:
    #         fig, axs = plt.subplots(2, 2, figsize=figsize, sharey='row')

    #         for col_idx, out in enumerate(["GPP", "ET"]):
    #             ax_ale = axs[0, col_idx]
    #             ax_res = axs[1, col_idx]
    #             for pred_type in prediction_types:
    #                 try:
    #                     df = ale_results[pred_type][out][feature]
    #                     df = df.reset_index()
    #                     # ALE Plot
    #                     ax_ale.plot(df.index, df["eff"], label=pred_type)

    #                     # Add support and optional residuals (if available)
    #                     if 'lowerCI_95%' in df.columns and 'upperCI_95%' in df.columns:
    #                         ax_ale.fill_between(
    #                             df.index,
    #                             df['lowerCI_95%'],
    #                             df['upperCI_95%'],
    #                             alpha=0.2
    #                         )
    #                     #ax_ale.set_xlabel(f"{feature} ({feature_units.get(feature, 'unit')})")
    #                 except KeyError:
    #                     continue

    #             ax_ale.set_title(f"{out} - {feature}")
    #             ax_ale.set_ylabel("ALE Effect")
    #             ax_ale.grid(True)
    #             ax_ale.legend()

    #             # Bin based on ALE bin centers
    #             n_bins = len(df)
    #             bin_edges = np.linspace(x_df[feature].min(), x_df[feature].max(), n_bins + 1)       

    #             # Assign each residual to a bin center for plotting
    #             bin_indices = np.digitize(x_df[feature].values, bin_edges) - 1

    #             ax_res.scatter(
    #                 bin_indices,
    #                 df_results[f"{out}_residual"], #metric results for model
    #                 alpha=0.3,
    #                 s=10,
    #                 color="black",
    #                 label="Residuals"
    #             )
    #             ax_res.axhline(0, linestyle=":", color="black", linewidth=0.8)
    #             ax_res.set_xlabel(f'{feature} (bins)')
    #             ax_res.set_ylabel("Residual")
    #             ax_res.grid(True)
    #             ax_res.legend()

    #         # fig.tight_layout()
    #         fig.subplots_adjust(hspace=0.3, right=0.88)  # make room for secondary y-axis labels

    #         if save_dir:
    #             os.makedirs(save_dir, exist_ok=True)
    #             fname = f"ALE_with_residuals_{feature}_{'_'.join(prediction_types)}.png"
    #             plt.savefig(os.path.join(save_dir, fname))
    #         if show:
    #             plt.show()
    #         plt.close()

    #     print(f"Figures Saved")

    elif plot_mode == "feature_pair_interaction":
        threshold = 0.05  # set your threshold here
        
        for pred_type in prediction_types:
            # Get station results for this pred_type
            pred_payload = ale_results.get(pred_type, {})
            pairwise_block_key = _resolve_pairwise_key(pred_payload, pred_type)
            if not pairwise_block_key:
                continue
            station_results = pred_payload.get(pairwise_block_key, {})

            selected_stations = stations or list(station_results.keys())
            
            for station in selected_stations:
                for pair in feature_pairs:  # 'features' now expected to be list of (f1, f2) pairs
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        print(f"[WARNING] Skipping invalid feature pair: {pair}")
                        continue

                    f1, f2 = pair

                    for pred_type in prediction_types:
                        for target in ["GPP", "ET"]:
                            try:
                                pair_key = f"{f1}_{f2}"
                                station_payload = _resolve_station_payload(station_results[station], split=split)
                                df = station_payload[target][pair_key]
                                # Threshold check: only plot if max ALE effect exceeds threshold
                                max_effect = np.abs(df.values).max()
                                # if max_effect < threshold:
                                #     print(f"[INFO] Skipping {f1} x {f2} ({pred_type}, {target}) — effect ({max_effect:.4f}) below threshold.")
                                #     continue

                                total_effect = df.values.max() - df.values.min()

                                fig, ax = plt.subplots(figsize=(6, 5))
                                c = ax.contourf(df.columns, df.index, df.values, cmap="coolwarm", levels=20)
                                fig.colorbar(c, ax=ax, label="ALE effect")

                                ax.set_xlabel(f2)
                                ax.set_ylabel(f1)
                                ax.set_title(f"2D ALE: {f1} x {f2} ({pred_type}, {target})\n(ALE) Max Effect = {max_effect:.3f}, Global Range: {total_effect:.3f}")

                                if save_dir:
                                    os.makedirs(save_dir, exist_ok=True)
                                    fname = f"ALE_2D_{station}_{f1}_{f2}_{target}_{pred_type}.png"
                                    plt.savefig(os.path.join(save_dir, fname))
                                if show:
                                    plt.show()
                                plt.close()

                            except Exception as e:
                                print(f"[WARNING] Failed 2D ALE plot for {pair} in {pred_type} ({target}): {e}")

    elif plot_mode == "feature_pair_3D":
        for pred_type in prediction_types:
            if pred_type not in ale_results:
                continue

            pred_payload = ale_results.get(pred_type, {})
            pairwise_block_key = _resolve_pairwise_key(pred_payload, pred_type)
            if not pairwise_block_key:
                continue
            station_results = pred_payload.get(pairwise_block_key, {})

            selected_stations = stations or list(station_results.keys())

            for station in selected_stations:
                for pair in feature_pairs:
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        print(f"[WARNING] Skipping invalid feature pair: {pair}")
                        continue

                    f1, f2 = pair
                    pair_key = f"{f1}_{f2}"

                    for target in ["GPP", "ET"]:
                        try:
                            station_payload = _resolve_station_payload(station_results[station], split=split)
                            df = station_payload[target][pair_key]

                            X, Y = np.meshgrid(df.columns.astype(float), df.index.astype(float))
                            Z = df.values.astype(float)

                            max_effect = np.abs(Z).max()
                            total_effect = Z.max() - Z.min()

                            fig = plt.figure(figsize=(8, 6))
                            ax = fig.add_subplot(111, projection="3d")
                            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor="none", antialiased=True)

                            ax.set_xlabel(f"{f2} ({feature_units.get(f2, '')})")
                            ax.set_ylabel(f"{f1} ({feature_units.get(f1, '')})")
                            ax.set_zlabel("ALE effect")
                            ax.set_title(
                                f"3D ALE: {f1} × {f2} | {pred_type} | {target}\n"
                                # f"Max |ALE| = {max_effect:.3f} • Range = {total_effect:.3f}"
                            )
                            ax.text2D(0.05, 0.95, f"Split: {station}", transform=ax.transAxes)

                            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

                            if save_dir:
                                os.makedirs(save_dir, exist_ok=True)
                                fname = f"ALE_3D_{station}_{f1}_{f2}_{target}_{pred_type}.png"
                                plt.savefig(os.path.join(save_dir, fname))
                            if show:
                                plt.show()
                            plt.close(fig)
                        except KeyError:
                            continue
                        except Exception as e:
                            print(f"[WARNING] Failed 3D ALE plot for {pair} in {pred_type} ({target}): {e}")

    elif plot_mode in ("feature_pair_3D_subplots", "feature_pair_3D_subplots_individual"):
        threshold = 0.05  # set your threshold here
        share_global_z_axis = plot_mode == "feature_pair_3D_subplots"
        # Determine available features from the first valid prediction type
        for pred in prediction_types:
            if pred not in ale_results:
                continue

            pred_payload = ale_results.get(pred, {})
            pairwise_block_key = _resolve_pairwise_key(pred_payload, pred)
            if not pairwise_block_key:
                continue

            station_results = pred_payload.get(pairwise_block_key, {})

            selected_stations = stations or list(station_results.keys())
            
            for station in selected_stations:
                station_payload = _resolve_station_payload(station_results.get(station, {}), split=split)
                gpp_data = station_payload.get("GPP", {})
                candidate_keys = list(gpp_data.keys())
                available_pairs = [
                    key.split("_")
                    for key in candidate_keys
                    if "_" in key and isinstance(gpp_data[key], pd.DataFrame)
                ]
                if available_pairs:
                    break
            if available_pairs:
                break

        if not available_pairs:
            print("[ERROR] No valid prediction types with 2D ALE GPP data found.")
            return

        selected_features = features or available_pairs

        for pair in selected_features:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                print(f"[WARNING] Skipping invalid feature pair: {pair}")
                continue

            f1, f2 = pair
            pair_key = f"{f1}_{f2}"
            for target in ["GPP", "ET"]:
                # Pre-compute global min/max for consistent z-axis
                z_mins, z_maxs = [], []

                data_dict = {}
                for pred_type in prediction_types:
                    for station in selected_stations:
                        try:
                            pred_payload = ale_results.get(pred_type, {})
                            pairwise_block_key = _resolve_pairwise_key(pred_payload, pred_type)
                            if not pairwise_block_key:
                                continue

                            pairwise_ale_root = pred_payload.get(pairwise_block_key, {})
                            
                            # pairwise_ale_root = (
                            #     ale_results[pred_type].get("process_full_ale", {})
                            #     if pred_type == "process"
                            #     else ale_results[pred_type].get("model_full_ale", {})
                            # )
                            

                            station_payload = _resolve_station_payload(pairwise_ale_root[station], split=split)
                            df = station_payload[target][pair_key]
                            
                            # Convert axis values
                            X, Y = np.meshgrid(df.columns.astype(float), df.index.astype(float))
                            Z = df.values.astype(float)
                            Z_masked = np.ma.masked_invalid(Z)  # ← Skip NaNs

                            z_mins.append(np.nanmin(Z))
                            z_maxs.append(np.nanmax(Z))
                            data_dict[(pred_type, station)] = (df, Z_masked)
                            # print(f"[DEBUG] Z stats – min: {np.nanmin(Z)}, max: {np.nanmax(Z)}, any NaN: {np.isnan(Z).any()}")

                        except Exception as e:
                            print(f"[WARNING] Skipping {pred_type} due to error: {e}")
                    
                if not data_dict:
                    continue  # skip if no valid data
                
                global_zmin = min(z_mins)
                global_zmax = max(z_maxs)

                abs_max = max(abs(global_zmin), abs(global_zmax))

                if abs_max > 0:
                    vmin, vmax = -abs_max, abs_max
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                else:
                    vmin = global_zmin
                    vmax = global_zmax
                    norm = None  # fallback if surface is completely flat
                cmap = cm.coolwarm  # coolwarm has white at its center

                num_rows = len(selected_stations)  # e.g., 4
                num_cols = len(prediction_types)  # e.g., 4
                
                fig = plt.figure(figsize=(4 * num_cols, 3.5 * num_rows))
                axs = []

                for row_idx, station in enumerate(selected_stations):
                    for col_idx, pred_type in enumerate(prediction_types):
                        ax_idx = row_idx * num_cols + col_idx + 1
                        ax = fig.add_subplot(num_rows, num_cols, ax_idx, projection='3d')
                        
                        key = (pred_type, station)
                        if key not in data_dict:
                            ax.set_visible(False)  # Hide unused subplot
                            continue

                        try:
                            df, Z_masked = data_dict[key]
                            X, Y = np.meshgrid(df.columns.astype(float), df.index.astype(float))
                            Z = Z_masked

                            surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm, edgecolor='none', antialiased=True)

                            ax.set_title(f"{pred_type} | Split {station}", fontsize=8)
                            ax.set_xlabel(f"{f2} ({feature_units.get(f2, '')})", fontsize=8)
                            ax.set_ylabel(f"{f1} ({feature_units.get(f1, '')})", fontsize=8)
                            ax.set_zlabel("ALE Effect", fontsize=8)
                            if share_global_z_axis and vmin is not None and vmax is not None:
                                ax.set_zlim(vmin, vmax)

                            last_surf = surf
                            axs.append(ax)
                        except Exception as e:
                            print(f"[WARNING] Skipping {pred_type} | Split {station} due to: {e}")
            
                fig.subplots_adjust(wspace=0.1, hspace=0.4)
                fig.suptitle(f"3D ALE | {f1} × {f2} | Target: {target}", fontsize=14)

                # Shared colorbar
                cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(mappable, cax=cbar_ax, label="ALE Effect")

                # fig.tight_layout(rect=[0, 0, 1, 0.95])

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    fname = f"ALE_3D_{f1}_{f2}_{target}_pred_all.png"
                    plt.savefig(os.path.join(save_dir, fname))
                plt.close(fig)

def bundle_path_for(save_dir_base: str, variant: str, length: str, grid_size: int) -> str:
    # Try both spatial/temporal roots; load the first that exists
    for ncv_kind in ("spatial", "temporal"):
        root = os.path.join(save_dir_base, f"{ncv_kind}_{length}", f"grid{grid_size}", variant)
        candidate = os.path.join(root, f"{variant}_bundle.joblib")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No bundle found for {variant=} {length=} grid{grid_size} under {save_dir_base}")

if __name__ == "__main__":
    # ----- CONFIG (EDIT HERE) -----
    variant = "full"   # {"main", "pairwise", "full"}
    length = "len12"
    grid_size = int(os.getenv("ALE_GRID_SIZE", 20))
    save_dir_base = "XAI/results/ALE/paper1_spatial"
    prediction_types = ["naive", "residual", "process"]

    # Optional plotting filters
    features = []   # e.g., ["Tair", "VPD"] #if empty uses all available
    stations = []   # e.g., [5, 12, 14, 21]
    split = "test"  # {"test", "train"} when bundles store per-split results

    # Where to save figures
    plots_out_dir = None  # default: XAI/results/ALE/plots_<variant>_<length>_grid<grid>
    show = False          # set True to display instead of headless save
    # ------------------------------

    bundle_path = bundle_path_for(save_dir_base, variant, length, grid_size)
    ale_results = joblib.load(bundle_path)
    with open(os.path.join(os.path.dirname(bundle_path), f"{variant}_normalization_stats.json")) as f:
        ale_results = to_physical_units(ale_results, json.load(f))

    # Map default plot modes to variant
    plot_mode = {
        "main":     "station_vs_prediction",
        "pairwise": "feature_pair_3D", #"feature_pair_interaction", #feature_pair_3D_subplots, feature_pair_interaction
        "full":     "feature_pair_3D_subplots_individual",
    }[variant]

    # Default output dir
    out_dir = plots_out_dir or os.path.join(save_dir_base, f"plots_{variant}_{length}_grid{grid_size}")
    os.makedirs(out_dir, exist_ok=True)

    plot_ale_results(
        ale_results=ale_results,
        plot_mode=plot_mode,
        features=features,
        stations=stations,
        prediction_type=prediction_types,   # your plotter accepts list
        save_dir=out_dir,
        show=show,
        split=split,
    )

    print(f"[OK] Plots written to: {out_dir}")
