import os
from pathlib import Path
import seaborn as sns
import pandas as pd



#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

import numpy as np
import pandas as pd

from powerlaw_utils import (
    power_law,
    compute_uncertainty,
    fit_power_law,
    errors_data_outer_fold_generic
    )
from utilis_shorts import errotype_to_string

STATION_NAME_MAP = {
    "outer_station_5": "Collelongo (Italy)",
    "outer_station_12": "Hyytiala (Finland)",
    "outer_station_14": "Le Bray (France)",
    "outer_station_21": "Soro (Denmark)",
}

CANONICAL_LENGTH_LABELS = {
    "len_3": "3 years",
    "len_6": "6 years",
    "len_9": "9 years",
    "len_12": "12 years",
}


def format_top_suffix(top_k_configs=None):
    """Helper to suffix filenames when averaging over top-k configs."""
    if not top_k_configs:
        return ""
    return f"_topConfigs{int(top_k_configs)}"

def extract_outer_folds(outer_folds_error, pred_types):
    """Extract available outer fold identifiers from the nested error dictionary."""
    for subset in outer_folds_error.values():
        for pred in pred_types:
            if pred in subset and subset[pred]:
                return sorted(subset[pred].keys())
    return []

def generate_spatial_plots(
    lengths,
    pred_types,
    error_type,
    sparse_years=None,
    top_k_configs=None,
    sparse=None,
    save_dir=None,
    base_path= "results/result_sum"
):

    errors, size = errors_data_outer_fold_generic(
        mode="spatial",
        lengths=lengths,
        base_path=base_path,
        sparse=sparse,
        sparse_year=sparse_years,
    )


    if not errors:
        raise RuntimeError("No spatial CSVs found. Check folder structure and filenames.")


    outer_folds = extract_outer_folds(errors, pred_types)
    suffix = format_top_suffix(top_k_configs)

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {pred: base_colors[idx % len(base_colors)] for idx, pred in enumerate(pred_types)}
    size_values = np.asarray(list(size.values())) 
    D_plot = np.linspace(size_values.min(), size_values.max(), 3500) 
    error_label= errotype_to_string(error_type)
    #%%%plots seperated by stations but all models together
    if outer_folds:
        nrows, ncols = 2, 2
        max_axes = nrows * ncols
        folds_to_plot = outer_folds[:max_axes]
        if len(outer_folds) > max_axes:
            print(
                f"[warn] More than {max_axes} outer folds detected; plotting only the first {max_axes}: "
                f"{', '.join(map(str, folds_to_plot))}"
            )

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6 * ncols, 6 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            "Generalization Error vs. Training Size | Spatial NCV",
            fontsize=16,
        )

        legend_handles = []
        legend_labels = []
        axes_flat = axes.ravel()
        axes_with_data = []
       
        
        for ax, outer_fold in zip(axes_flat, folds_to_plot):
            results=[]
            for pred in pred_types:
                popt, pcov, errors_mean, errors_std = fit_power_law(
                    errors,
                    size_values,
                    pred_type=pred,
                    error_type=error_type,
                    outer_fold=outer_fold,
                )
                perr = np.sqrt(np.diag(pcov))
                
                results.append({
                "pred_type": pred,
                "A": f"{np.round(popt[0],2)} $\pm$ {np.round(perr[0],2)}",
                "alpha": f"{np.round(popt[1],2)} $\pm$ {np.round(perr[1],2)}",
                "B": f"{np.round(popt[2],2)} $\pm$ {np.round(perr[2],2)}"
                })

                color = color_map[pred]
                ax.errorbar(
                    size_values,
                    errors_mean,
                    yerr=errors_std,
                    fmt="o",
                    capsize=2,
                    color=color,
                    alpha=0.6,
                )
                label = None
                if pred not in legend_labels:
                    label = pred
                line, = ax.plot(
                    D_plot,
                    power_law(D_plot, *popt),
                    color=color,
                    alpha=0.8,
                    label=label,
                )
                ax.grid(True)
                if label:
                    legend_handles.append(line)
                    legend_labels.append(pred)
          
            ax.set_title(STATION_NAME_MAP[outer_fold])
            ax.set_xlabel("Training Set Size [Days]")
            ax.set_ylabel(f"{error_label}")
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.grid(True)
            axes_with_data.append(ax)
            pd_results = pd.DataFrame(results)
            # if save_dir is not None:
            #     pd_results.to_csv(save_dir / f"powerlaw_fit_params_{outer_fold}_{error_type}_spatial{suffix}.csv", index=False)
       
        plt.tight_layout(pad=2.0)
        if legend_labels:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(len(legend_labels), 7),
                fontsize=8,
                frameon=False,
            )
            fig.subplots_adjust(top=0.85, bottom=0.12, hspace=0.4, wspace=0.3)
        else:
            fig.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)

        filename = f"new_generalization_error_outerfold_grid_{error_type}_spatial{suffix}.png"
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,filename), dpi=300, bbox_inches="tight")
        plt.close(fig)

        #%%%plots for each model type and each outer fold seperately
        ncols = 2
        n_folds = len(outer_folds)
        nrows = max(1, int(np.ceil(n_folds / ncols)))

        for pred in pred_types:
            has_pred_data = any(pred in subset and subset[pred] for subset in errors.values())
            if not has_pred_data:
                print(f"[warn] No outer fold data for model {pred}.")
                continue

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 4 * nrows),
                squeeze=False,
            )
            fig.suptitle(
                f"Generalization Error vs. Training Size\n{pred} | {error_type} | Spatial NCV",
                fontsize=16,
            )

            for idx, outer_fold in enumerate(outer_folds):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]

                popt, pcov, errors_mean, errors_std = fit_power_law(
                    errors,
                    size_values,
                    pred_type=pred,
                    error_type=error_type,
                    outer_fold=outer_fold,
                )
              
                ax.errorbar(size_values, errors_mean, yerr=errors_std, fmt="o", capsize=2, label="Observed")
                ax.plot(D_plot, power_law(D_plot, *popt), label="Power-law fit")

                ax.set_title(outer_fold)
                ax.set_xlabel("Training Set Size [Days]")
                ax.set_ylabel(f"{error_label}")

                
                ax.grid(True)

            total_axes = nrows * ncols
            if n_folds < total_axes:
                for idx in range(n_folds, total_axes):
                    row, col = divmod(idx, ncols)
                    axes[row][col].axis("off")

            plt.tight_layout(pad=3.0)
            fig.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
            filename = f"generalization_error_{pred}_{error_type}_spatial{suffix}.png"
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir,filename), dpi=300, bbox_inches="tight")
            plt.close(fig)
    else:
        print("No outer folds found. Skipping outer-fold plot grid.")

    #%%%plots over all outer folds
    results= []
    fig, ax = plt.subplots(figsize=(11, 11))
    for i,pred in enumerate(pred_types):
        popt, pcov, errors_mean, errors_std = fit_power_law(
            errors,
            size_values,
            pred_type=pred,
            error_type=error_type,
        )
        perr = np.sqrt(np.diag(pcov))
        #compute uncertainity
        uncertainity = compute_uncertainty(popt,pcov,D_plot)
        Func_out =power_law(D_plot, *popt)
       
        D_upper= Func_out+uncertainity
        D_lower = Func_out-uncertainity
        color = color_map[pred]
        ax.scatter(
            size_values,
            errors_mean,
            color=color,
            s=100,
            alpha=0.7,
        )
        ax.plot(
                D_plot,
                Func_out,
                color=color,
                label=pred,
                alpha=0.6,
                linewidth=2,
                ms=20,
            )
        if pred!='process':
            ax.fill_between(
                D_plot,
                D_lower,
                D_upper,
                color=color,
                alpha=0.2,)
        results.append({
        "pred_type": pred,
        "A": f"{np.round(popt[0],3)} $\pm$ {np.round(perr[0],3)}",
        "alpha": f"{np.round(popt[1],3)} $\pm$ {np.round(perr[1],3)}",
        "B": f"{np.round(popt[2],3)} $\pm$ {np.round(perr[2],3)}"
        })   
    ax.set_xlabel("Training Set Size [Days] in log scale", fontsize=24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1,1.1)
    ax.set_ylabel(f"{error_label} in log scale",  fontsize=24)
    ax.tick_params(axis='both', labelsize=23)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    lines, labels = ax.get_legend_handles_labels()
    labels = [label.lower() for label in labels]
    ax.legend(
        loc="upper right",
        ncol=min(len(pred_types), 3),
        frameon=False,
        fontsize=18,
    )

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"generalization_all_models_{error_type}_spatial{suffix}_new.pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)

    if save_dir is not None:
        df_results= pd.DataFrame(results)
        df_results.to_csv(os.path.join(save_dir, f'powerlaw_fit_parameters_spatial_all_models_{error_type}{suffix}.csv'), index=False)

    #%%%generalization plots per model type
    for pred in pred_types:
        popt, pcov, errors_mean, errors_std = fit_power_law(
            errors,
            size_values,
            pred_type=pred,
            error_type=error_type,
        )
        
        perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([np.nan, np.nan, np.nan])

        fig, ax = plt.subplots(figsize=(15, 6))
        color = color_map[pred]
        ax.errorbar(
            size_values,
            errors_mean,
            yerr=errors_std,
            fmt="o",
            capsize=10,
            color=color,
            label=(
                f"A: {popt[0]:.3f}±{perr[0]:.3f}  "
                f"alpha: {popt[1]:.3f}±{perr[1]:.3f}  "
                f"B: {popt[2]:.3f}±{perr[2]:.3f}"
            ),
            alpha=0.7,
        )
        ax.plot(
            D_plot,
            power_law(D_plot, *popt),
            color=color,
            alpha=0.6,
            linewidth=4
        )
        ax.set_xlabel("Training Set Size [Days]")
        ax.set_ylabel("Generalization Error")
        ax.set_title(f"Model: {pred} | {error_type} | Spatial NCV")
        ax.grid(True)
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.tight_layout()
        if save_dir is not None:
            plot_path = f"generalization_{pred}_{error_type}_spatial{suffix}.png"
            plt.savefig(os.path.join(save_dir, plot_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Spatial NCV generalization plots saved.")
    plot_coefficent_of_variation(errors, pred_types, error_type, color_map, save_dir=save_dir, top_k_configs=top_k_configs)
    return

    
def plot_coefficent_of_variation(errors, pred_types, error_type, color_map, save_dir=None, top_k_configs=None):
    #plots coefficent of variation std/mean
    
    coef_var=list()
    pred_labels=list()
    for pred_type in pred_types:
        for _, error_dict in errors.items():
            mean_over_outer= []
            std_over_outer= []
            for outer_fold in error_dict[pred_type].keys():
                mean_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_mean'])
                std_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_std'])

            errors_mean= np.asarray(mean_over_outer).mean()
            errors_std = np.asarray(std_over_outer).mean()
        
            if errors_mean.shape != errors_std.shape:
                print(f'something wrong for model {pred_types}')
            
            coef_var_= errors_std/errors_mean
            coef_var_=coef_var_.flatten()
            coef_var.append(coef_var_)
            pred_labels.append(pred_type.lower())



    df = pd.DataFrame({
    "Model": np.repeat(pred_labels, [len(x) for x in coef_var]),
    "RSD": np.concatenate(coef_var)
    })
    plt.figure(figsize=(15, 15))
    ax=sns.violinplot(data=df, x="Model", y="RSD", inner="quart", 
                      palette={k.lower():v for k,v in color_map.items()}, 
                   alpha=0.7, density_norm="count", linewidth=2, bw_adjust=0.8, cut=0)
     #add star to process model
    for i, model in enumerate(df["Model"].unique()):
        model_data = df.loc[df["Model"] == model, "RSD"]

        if np.allclose(model_data, 0):
            ax.scatter(
                i,
                0,      # slightly above x-axis so it's visible
                marker="*",
                s=350,
                color="black",
                zorder=10
            )
    plt.ylabel('CV of Mean Squared Error', fontsize=28)
    plt.xlabel('Modelling strategies', fontsize=28)
    plt.xticks(fontsize=26, rotation=45)
    custom_ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    plt.ylim(-0.01,0.3)
    plt.yticks(custom_ticks, fontsize=26)
    plt.grid()
    plt.tight_layout()
    if save_dir is not None:    
        plt.savefig(os.path.join(save_dir, f'CV_violin_station_spatial.pdf'),  bbox_inches='tight')
    plt.show()
    plt.close()

    return
#%%%%%%

if __name__ == "__main__":
    lengths = (3, 6, 9, 12)
    sparse = [50, 30,7]
    pred_types = ['naive', 'regularized', 'residual','bias', 'parallel', 'Mixture', 'process']
    error_types = ["test_mse"]
    base_dir = os.getcwd()
    base_dir= os.path.join(base_dir, "results/result_sum")  
    top_k_configs = 5  # Set to integer (e.g., 5) to average the top-k configs per prediction type.
    sparse_years = 3
    for err in error_types:
        generate_spatial_plots(
            lengths=lengths,
            pred_types=pred_types,
            error_type=err,
            base_path=base_dir,
            top_k_configs=top_k_configs,
            sparse=sparse,
            sparse_years=sparse_years,
            save_dir='test_save',
        )
