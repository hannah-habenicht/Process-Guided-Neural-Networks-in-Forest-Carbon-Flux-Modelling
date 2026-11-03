import numpy as np
import seaborn as sns
import pandas as pd
#matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import pandas as pd
from pathlib import Path

from utilis_shorts import errotype_to_string
from powerlaw_utils import (
    power_law,
    compute_uncertainty,
    fit_power_law, 
    errors_data_outer_fold_generic,
)

STATION_NAME_MAP = {
    "outer_station_5": "Collelongo (Italy)",
    "outer_station_12": "Hyytiala (Finland)",
    "outer_station_14": "Le Bray (France)",
    "outer_station_21": "Soro (Denmark)",
}

def format_station_label(station: int) -> str:
    """Return a descriptive station label including mapped name when available."""
    key = f"outer_station_{station}"
    mapped_name = STATION_NAME_MAP.get(key)
    if mapped_name:
        return f"{mapped_name}"
    return f"Station {station}"


def format_top_suffix(top_k_configs=None):
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

def generate_station_plots(
    station,
    training_years,
    pred_types,
    error_type,
    sparse,
    sparse_year,
    color_map,
    top_k_configs=None,
    resolution=1000, 
    base_path="results/result_sum",
    save_dir=None
):
    """Create all per-station plots and return data needed for comparisons."""
    errors, size = errors_data_outer_fold_generic(
        mode="temporal",
        lengths=training_years,
        base_path=base_path,
        station=station,
        sparse=sparse,
        sparse_year=sparse_year,
    )
    outer_folds_error, size = errors_data_outer_fold_generic(
        mode="temporal",
        lengths=training_years,
        base_path=base_path,
        station=station,
        sparse=sparse,
        sparse_year=sparse_year,
    )
    size_values = np.asarray(list(size.values())) 
    suffix = format_top_suffix(top_k_configs)
    outer_folds = extract_outer_folds(outer_folds_error, pred_types)
    station_label = format_station_label(station)
    D_plot = np.linspace(min(size_values), max(size_values), resolution)
    #%% Plot per outer fold
    if outer_folds:
        ncols = 2
        n_folds = len(outer_folds)
        nrows = int(np.ceil(n_folds / ncols))
        results_per_outer_fold = {}

        for pred in pred_types:
            has_pred_data = any(pred in subset for subset in outer_folds_error.values())
            if not has_pred_data:
                print(f"No outer fold data for model {pred} at station {station}.")
                continue

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 4 * nrows),
                squeeze=False
            )
            fig.suptitle(
                f'Generalization Error vs. Training Size\n{pred} | {error_type} | {station_label}',
                fontsize=16
            )
            
            
            for idx, out_l in enumerate(outer_folds):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]

                popt, pcov, errors_mean, errors_std = fit_power_law(
                    outer_folds_error,
                    size_values,
                    pred_type=pred,
                    error_type=error_type,
                    outer_fold=out_l
                )

                
                ax.errorbar(size.values(), errors_mean, yerr=errors_std, fmt='o', label='Observed')
                ax.plot(D_plot, power_law(D_plot, *popt), label='Power-law fit')
                ax.set_title(f'Outer Fold: {out_l}')
                ax.set_xlabel('Training Set Size [Days]')
                ax.set_ylabel('Generalization Error')
                ax.set_xscale("log")
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=range(1, 10)))  # label every decade
                ax.xaxis.set_major_formatter(LogFormatterSciNotation())
                ax.grid(True)

            total_axes = nrows * ncols
            if n_folds < total_axes:
                for idx in range(n_folds, total_axes):
                    row, col = divmod(idx, ncols)
                    axes[row][col].axis('off')

            plt.tight_layout(pad=3.0)
            fig.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
            filename = f'generalization_error_{pred}_{error_type}_station{station}{suffix}.png'
            plt.close(fig)
    
            results_per_outer_fold_df = {
        pred: pd.DataFrame(data) for pred, data in results_per_outer_fold.items()
    }
    else:
        print(f"No outer folds found for station {station}. Skipping outer-fold plot.")

    #%% Combined plot for all models over all outer folds
    fig, ax = plt.subplots(figsize=(11, 11))
    errorlabel=errotype_to_string(error_type)
    for pred in pred_types:
        popt, pcov, errors_mean, errors_std = fit_power_law(
            errors, size_values, pred_type=pred, error_type=error_type
        )
        #get uncertainty in fit
        uncertainity= compute_uncertainty(popt,pcov,D_plot)
        color = color_map[pred]
        func_out= power_law(D_plot, *popt)
        ax.plot(
            D_plot,
            func_out,
            color=color,
            label=pred.lower(),
            alpha=0.6,
            linewidth=2
        )

        ax.scatter(
            list(size.values()),
            errors_mean,
            color=color,
            s=100,
            alpha=0.7,
        )
        D_upper = func_out + uncertainity
        D_lower = func_out - uncertainity
        if pred!='process':
            ax.fill_between(
                D_plot,
                D_lower,
                D_upper,
                color=color,
                alpha=0.2,)

    ax.set_xlabel('Training Set Size [Days] (log scale)', fontsize=24)
    ax.set_ylabel(f'{errorlabel} (log scale)', fontsize=24)
    #ax.set_title(f'{station_label}')
    lines, labels = ax.get_legend_handles_labels()
    labels = [label.lower() for label in labels]
    ax.legend(
        loc="upper right",
        #bbox_to_anchor=(0.5, 1.14),
        ncol=min(len(pred_types), 3),
        frameon=False,
        fontsize=18,
    )
   
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.1,1.1)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=23)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.tight_layout()
    
    suffix = format_top_suffix(top_k_configs)
    #plt.show()
    if save_dir is not None:
        comparison_filename = f'new_generalization_all_models_{error_type}_station{station}{suffix}.png'        
        save_path = Path(save_dir) / comparison_filename
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    results =[]
    #%% Individual model plots with fit parameters
    for pred in pred_types:
        fig, ax = plt.subplots(figsize=(15, 6))
        popt, pcov, errors_mean, errors_std = fit_power_law(
            errors, size_values, pred_type=pred, error_type=error_type
        )
        perr = np.sqrt(np.diag(pcov))
        color = color_map[pred]

        results.append({
                "pred_type": pred,
                "A": f"{np.round(popt[0],2)} $\pm$ {np.round(perr[0],2)}",
                "alpha": f"{np.round(popt[1],2)} $\pm$ {np.round(perr[1],2)}",
                "B": f"{np.round(popt[2],2)} $\pm$ {np.round(perr[2],2)}",
                })
        
        ax.errorbar(
            size_values,
            errors_mean.squeeze(),
            yerr=errors_std.squeeze(),
            fmt='o',

            capsize=50,
            capthick=2,
            color=color,
            label=(
                f"A: {popt[0]:.3f}$\\pm${perr[0]:.3f}  "
                f"alpha: {popt[1]:.3f}$\\pm${perr[1]:.3f}  "
                f"B: {popt[2]:.3f}$\\pm${perr[2]:.3f}"
            ),
            alpha=0.5
        )
        
        ax.plot(
            D_plot,
            power_law(D_plot, *popt),
            color=color,
            alpha=0.6
        )
        ax.set_xlabel('Training Set Size [Days]')
        ax.set_ylabel('Generalization Error')
        ax.set_title(f'Model: {pred} | {error_type} | {station_label}')
        ax.grid(True)
        ax.legend()
        ax.set_xscale("log")
        # ax.set_yscale("log")
        plt.tight_layout()

        if save_dir is not None:
            comparison_filename = f'generalization_all_models_{error_type}_station{station}{suffix}.pdf'
            save_path = Path(save_dir) / comparison_filename
            plt.savefig(save_path, bbox_inches='tight')
            df = pd.DataFrame(results)
            csv_filename = f"fitted_parameters_{station_label}.csv"
            df.to_csv(csv_filename, index=False)
        plt.close(fig)
   
    print(f'Completed {station_label}')
    return {
        "station": station,
        "errors": errors,
        "size": size,
        "station_label": station_label,
        "top_k_configs": top_k_configs,
    }



def plot_coefficent_of_variation(errors, pred_types, error_type, color_map,station, save_dir=None):
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
                print(f'something wrong for model {pred_type}')
            color = color_map[pred_type]
            
            coef_var_= errors_std/errors_mean
            coef_var_=coef_var_.flatten()
            coef_var.append(coef_var_)
            pred_labels.append(pred_type.lower())


    df = pd.DataFrame({
    "Model": np.repeat(pred_labels, [len(x) for x in coef_var]),
    "RSD": np.concatenate(coef_var)
    })
    print('CV of MSE of station', station)
    plt.figure(figsize=(15, 12))
    ax= sns.violinplot(data=df, 
                       x="Model", 
                       y="RSD", 
                       inner="quart",palette={k.lower():v for k,v in color_map.items()}, 
                   alpha=0.7, density_norm="count", linewidth=3, bw_adjust=0.8, cut=0)

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
    plt.ylabel('CV of Mean Squared Error', fontsize=25)
    plt.xlabel('Modelling strategies', fontsize=25)
    custom_ticks = [0,0.05, 0.1,0.15, 0.2,0.25,0.3,]
    plt.yticks(custom_ticks, fontsize=25)
    plt.ylim(-0.01,0.3)
    plt.xticks(fontsize=25, rotation=45)
    plt.grid()
   
    plt.tight_layout()
    plt.show()
    if save_dir is not None:
        save_path = Path(save_dir) / f'RSD_violin_station_{station}.pdf'
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return


if __name__ == "__main__":
    stations = [12,5] #12
    station_training_years = {
        5: [1, 2, 3, 5],
        12: [1, 2, 3],
    }
    pred_types = ['naive', 'regularized', 'residual','bias', 'parallel', 'Mixture'] # 
    error_type = 'test_mse'
    sparse = [50, 30, 7]
    sparse_year = 1
    
    result_dir= "results/result_sum" 

    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {pred: base_colors[idx % len(base_colors)] for idx, pred in enumerate(pred_types)}

    top_k_configs = 5  # Set to an integer (e.g., 5) to average the top-k configs per prediction type.

    station_results = []
    for station in stations:
        training_years = station_training_years.get(station, [1, 2, 3])
        
        errors, size = errors_data_outer_fold_generic(
        base_path=result_dir,
        mode="temporal",
        lengths=training_years,
        station=station,
        sparse=sparse,
        sparse_year=sparse_year
        )
        # plot_coefficent_of_variation(errors,
        #                              pred_types, 
        #                              error_type, 
        #                              color_map, 
        #                              station=station,
        #                              save_dir='test_save')

        result = generate_station_plots(
            station=station,
            training_years=training_years,
            pred_types=pred_types,
            error_type=error_type,
            sparse=sparse,
            sparse_year=sparse_year,
            color_map=color_map,
            base_path=result_dir,
            top_k_configs=top_k_configs,
            save_dir='test_save'
        )
        station_results.append(result)

    print('end')
