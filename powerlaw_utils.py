import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import os
import pandas as pd

#------------------------------
#Uncertainty computation functions
#------------------------------
def jacobian_power_law(D, A, alpha, B):  
    dA= D**(-alpha)
    dalpha=-A*np.log(D)*D**(-alpha)
    dB= np.ones_like(D)
    return   np.stack([dA, dalpha, dB], axis=-1)

def compute_uncertainty(poppt,pcov,Dplot):
    # Compute the Jacobian matrix at Dplot, derivatives w.r.t. parameters A, alpha, B
    jacobian = jacobian_power_law(Dplot, *poppt)
    # Compute the uncertainty in the fit, by including parameter covariance
    uncertainity = np.einsum('ij,jk,ik->i', jacobian, pcov, jacobian)
    uncertainity = np.sqrt(uncertainity)
    return uncertainity
#------------------------------
#Power law fitting functions
#------------------------------
def create_fills_between(sample_size, D_plot,  y_std, popt):
    #create standard deviation fill between area
    # Model evaluated on dense axis
    y_model_dense = power_law(D_plot, *popt)
    
    # Interpolate std from sparse sample points to dense grid
    y_std_dense = np.interp(D_plot, sample_size, y_std)
    
    # Upper and lower bounds
    D_std_upper = y_model_dense + y_std_dense
    D_std_lower = y_model_dense - y_std_dense
    return D_std_upper, D_std_lower

def power_law(D, A, alpha, B):
    """Canonical three-parameter power-law used for generalisation error fits."""
    return A * np.asarray(D, dtype=float) ** (-alpha) + B

def prepare_curve_fit_arrays(D_values, errors_mean, errors_std):
    """
    Regularise observations prior to curve fitting.

    Checks for non-finite entries, removes non-positive means (log-space fitting),
    raises errors and stops the anlyses process 
    Returns arrays suitable for scipy.curve_fit alongside a flag indicating whether
    sigma weighting can be used.
    """
    D_values = np.asarray(D_values, dtype=float)
    errors_mean = np.asarray(errors_mean, dtype=float)
    errors_std = np.asarray(errors_std, dtype=float)

    for arr in [D_values, errors_mean, errors_std]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Array contains NaN or infinite values.")

        if (arr<0).any():
            raise ValueError('Array smaller zero')

    if errors_mean.shape != errors_std.shape:
        raise ValueError('not the same size of std and mean') 
    
    use_sigma = np.any(errors_std > 0) #why do decide if you want to use uncertainity ?
    return D_values, errors_mean, errors_std, use_sigma

def fallback_params(errors_mean):
    """
    Provide a safe parameter triplet when insufficient data is available for fitting.
    """
    errors_mean = np.asarray(errors_mean, dtype=float)
    finite_vals = errors_mean[np.isfinite(errors_mean) & (errors_mean > 0)]
    if finite_vals.size == 0:
        finite_vals = np.array([1.0])
    baseline = finite_vals[0]
    tail = finite_vals[-1]
    return np.array([baseline, 0.0, tail], dtype=float)

def initial_guess(D_fit, err_mean_fit, floor=1e-4):
    """
    Estimate initial curve-fit parameters and clamp them to positive values.
    """
    # A0 = err_mean_fit[0]
    try:
        alpha0 = -np.polyfit(np.log(D_fit), np.log(err_mean_fit), 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        alpha0 = 0.5
    B0 = err_mean_fit.min()
    n_min = np.min(D_fit)
    A0 = (np.max(err_mean_fit) - B0) * (n_min ** alpha0)
    p0 = np.array([float(A0), float(alpha0), float(B0)], dtype=float)
    return np.where(p0 <= floor, floor, p0)

def fit_power_law(errors, sizes, 
                  pred_type,error_type='test_mae', 
                  outer_fold=None, use_uncer=True, 
                  absolute_sigma=True):

    D_values = sizes
    errors_std= []
    errors_mean= []

    if pred_type =='process':
        use_uncer= False
        absolute_sigma= False

    if outer_fold is not None:
        mean_over_outer= []
        std_over_outer= []
        for _, error_dict in errors.items():
            mean_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_mean'])
            std_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_std'])
        errors_mean = np.array(mean_over_outer)
        errors_std = np.array(std_over_outer)   
        
    else: 
        for _, error_dict in errors.items():
            mean_over_outer= []
            std_over_outer= []
            for outer_fold in error_dict[pred_type].keys():
                mean_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_mean'])
                std_over_outer.append(error_dict[pred_type][outer_fold][error_type+'_std'])
            errors_mean.append(np.mean(mean_over_outer))
            errors_std.append(np.mean(std_over_outer))


        errors_mean = np.array(errors_mean)
        errors_std = np.array(errors_std)

    
    D_fit, err_mean_fit, err_std_fit,_= prepare_curve_fit_arrays(D_values, errors_mean, errors_std)
    
    # Initial guess for A, alpha, B
    p0 = initial_guess(D_fit, err_mean_fit)

    curve_fit_kwargs = dict(
        p0=p0,
        bounds=([1e-3, 0, 0], [1e5, 10, 10]),
        maxfev=20000,
        absolute_sigma=absolute_sigma #change
    )
    if use_uncer:
        curve_fit_kwargs["sigma"] = err_std_fit
    try:
        popt, pcov = curve_fit(
            power_law,
            D_fit,
            err_mean_fit,
            **curve_fit_kwargs
        )
    except (RuntimeError, ValueError):
        popt = p0
        pcov = np.zeros((3, 3))
    return popt,pcov,errors_mean, errors_std

#-------
#data export and preparation 
#----------------------------------------------
def reduce_metric(values_mean, values_std, values_median=None, top_k=None):
    """
    Reduce metrics by selecting the top-k smallest mean values 
    (and corresponding std and median if provided).

    Parameters
    ----------
    values_mean : array-like
        Mean values for each metric.
    values_std : array-like
        Standard deviations corresponding to `values_mean`.
    values_median : array-like, optional
        Median values corresponding to `values_mean`. Can be None.
    top_k : int, optional
        Number of top entries (smallest mean values) to keep.

    Returns
    -------
    tuple
        A tuple containing:
        - reduced_means : np.ndarray
        - reduced_stds : np.ndarray
        - reduced_medians : np.ndarray or None
    """

    values_mean = np.asarray(values_mean, dtype=float)
    values_std = np.asarray(values_std, dtype=float)
    values_median = np.asarray(values_median, dtype=float) if values_median is not None else None

    if values_mean.size == 0:
        raise ValueError("Array has no values, please check")
        #return np.nan # not a good idea
    index_sort= np.argsort(values_mean)
    values_mean = values_mean[index_sort]
    values_std = values_std[index_sort]
    if values_median.all()!=None:
       values_median= values_median[index_sort]
    if top_k and top_k > 0 and top_k<= len(values_mean):
        values_mean=values_mean[:top_k]
        values_std= values_std[:top_k]

    return values_mean, values_std, values_median

#data extraction
def errors_data_outer_fold_generic(
    mode,                       # "temporal" or "spatial"
    lengths,
    base_path,
    station=None,
    sparse=None,
    sparse_year=1
):
    """
    Generic outer-fold error computation for temporal & spatial experiments.
    """

    size_dict = {}
    dict_error_outer_all = {}
    config_mins_model = []

    # ---- experiment-specific naming ----
    if mode == "temporal":
        name_dense = "temporal_len"
        name_sparse = "temporal_len1_sparse"
        parse_metric = lambda x: float(str(x).replace("tensor(", "").rstrip(")").split(",")[0])
        make_size = lambda t: t * 365
        make_sparse_size = lambda t: float(np.round(365 / t))
    elif mode == "spatial":
        name_dense = "spatial_len"
        name_sparse = "spatial_len" + str(sparse_year) + "_sparse"
        parse_metric = lambda x: float(str(x).replace("tensor(", "").rstrip(")").split(",")[0])
        make_size = lambda t: t * 365
        make_sparse_size = lambda t: float(np.round((365 * sparse_year) / t))
    else:
        raise ValueError("mode must be 'temporal' or 'spatial'")

    # ---- combine sparse & dense configs ----
    if sparse is not None:
        sparse_dic = {f"sparse{i}": sp for i, sp in enumerate(sparse)}
        dense_dic = {f"len{i}": t for i, t in enumerate(lengths)}
        data_size = {**sparse_dic, **dense_dic}
        name_ = len(sparse) * [name_sparse] + len(lengths) * [name_dense]
    else:
        data_size = {f"len{i}": t for i, t in enumerate(lengths)}
        name_ = len(lengths) * [name_dense]

    # ---- loop over datasets ----
    for idx, (dat_type, t) in enumerate(data_size.items()):
        csv_name = name_[idx] + str(t) + "/evaluation_all_prediction_types.csv"

        if mode == "temporal":
            csv_path = os.path.join(
                base_path,
                f"station_{station}",
                csv_name
            )
        else:
            csv_path = os.path.join(base_path, csv_name)

        if not os.path.exists(csv_path):
            print(f"[warn] Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        model_types = df["prediction_type"].unique()
        outer_folds = df["outer_fold"].unique()
        dict_outer_folds_error = {}

        for model in model_types:
            dict_error_outer = {}

            for outer_fold in outer_folds:
                df_fold = df[
                    (df["outer_fold"] == outer_fold)
                    & (df["prediction_type"] == model)
                ]

                test_metric_vals = np.array(
                    [parse_metric(v) for v in df_fold["test_metric"]]
                )

                # ---- compute metrics ----
                test_mse_mean = df_fold["test_loss"].mean()
                test_mse_std = df_fold["test_loss"].std()
                test_mse_std = 0 if np.isnan(test_mse_std) else test_mse_std

                test_mae_mean = test_metric_vals.mean()
                test_mae_std = test_metric_vals.std()
                test_mae_std = 0 if np.isnan(test_mae_std) else test_mae_std

                # ---- process model: enforce zero variance ----
                if model == "process":
                    test_mse_std = 0
                    test_mae_std = 0
                else:
                    config_mins_model.append(df_fold["config_id"].max())

                dict_error_outer[outer_fold] = {
                    "test_mse_mean": test_mse_mean,
                    "test_mse_std": test_mse_std,
                    "test_mae_mean": test_mae_mean,
                    "test_mae_std": test_mae_std,
                }

            dict_outer_folds_error[model] = dict_error_outer

        # ---- store results ----
        if "sparse" in dat_type:
            dict_error_outer_all[f"sparse{t}"] = dict_outer_folds_error
            size_dict[f"sparse{t}"] = make_sparse_size(t)
        else:
            dict_error_outer_all[f"year_{t}"] = dict_outer_folds_error
            size_dict[f"year_{t}"] = make_size(t)

    print("minimum config ids per model type:", np.min(config_mins_model))
    return dict_error_outer_all, size_dict

__all__ = [
    "power_law",
    "prepare_curve_fit_arrays",
    "fallback_params",
    "initial_guess",
]
