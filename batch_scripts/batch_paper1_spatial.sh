#!/bin/bash
#SBATCH -p alldlc2_gpu-l40s                       # partition (queue)
#SBATCH --mem 16G                                 # memory pool for the whole job (shared by all in-job workers)
#SBATCH -t 0-8:00                                 # time (D-HH:MM)
#SBATCH --cpus-per-task=16                        # cores for the job (shared by all in-job workers)
#SBATCH --gres=gpu:1                              # one GPU per fold, shared by all in-job workers
#SBATCH -o log/%x.%N.%A.%a.out                    # STDOUT (log folder should exist)
#SBATCH -e log/%x.%N.%A.%a.err                    # STDERR (log folder should exist)
#SBATCH -J p1_spatial                             # default job name; metaschedule_paper1.sh overrides via --job-name
#SBATCH -a 0-0%2                                  # default: one task per outer fold; submit_paper1.sh sizes this properly
#SBATCH --exclude=dlc2gpu07                       # faulty GPU memory (uncorrectable ECC); submit_paper1.sh's --exclude overrides this
#
# batch_paper1_spatial.sh — one array task = one outer fold = one GPU, running the
# spatial NCV entrypoint for the paper-1 PRELES rerun (the OOD CV_station_len* files).
#
# Differs from batch_spatial_new.sh in exactly two ways:
#   * SPARSITY selects the frozen config (${prediction_loss}_sparse${SPARSITY}.json)
#   * PROCESS_MODEL defaults to preles rather than basfor
#
# Normally launched via batch_scripts/submit_paper1.sh (which sizes --array from the
# CV config), which in turn is driven by batch_scripts/metaschedule_paper1.sh.

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_GPUS_PER_NODE GPUs per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "This is job number $SLURM_ARRAY_TASK_ID";

[ -f /etc/cuda_env ] && source /etc/cuda_env   # cluster-specific CUDA setup
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"   # submit from the repository root
source "${VENV_PATH:-$PROJECT_ROOT/.venv}/bin/activate"


# Defaults so a plain `sbatch batch_scripts/batch_paper1_spatial.sh` still does something.
DEFAULT_CV_CONFIG_PATH="configs/CV_splits/Paper_1_preles_ncvs/CV_station_len3_config.json"
DEFAULT_WORKERS_PER_FOLD=8
DEFAULT_TOP_TRIALS=5

# === EXPERIMENT SETTINGS ===
: "${PROCESS_MODEL:=preles}"
: "${prediction_loss:=naive}"                  # naive | bias | parallel | regularized | residual
: "${SPARSITY:=}"                              # empty = full resolution; otherwise 50 | 30 | 7
: "${calibration_flag:=require_existing}"
: "${CALIBRATION_ARTIFACT_NAMESPACE:=}"
: "${EXPERIMENT_ARTIFACT_NAMESPACE:=$CALIBRATION_ARTIFACT_NAMESPACE}"
export PROCESS_MODEL
export prediction_loss
export SPARSITY
export calibration_flag
export CALIBRATION_ARTIFACT_NAMESPACE
export EXPERIMENT_ARTIFACT_NAMESPACE

# Sparsity lives in the frozen config (key "data:train:sparse"), not on the CLI, so it
# is selected purely by filename. See Config_Files/create_frozen_config.py.
SPARSE_PART=""
if [ -n "$SPARSITY" ]; then
    SPARSE_PART="_sparse${SPARSITY}"
fi
EXPERIMENT_NAME="${prediction_loss}${SPARSE_PART}"

CONFIG_FILE="${PROJECT_ROOT}/configs/config_spatial/${EXPERIMENT_NAME}.json"
CV_CONFIG_PATH="${CV_CONFIG_PATH:-$DEFAULT_CV_CONFIG_PATH}"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "[ERROR] Frozen config not found: $CONFIG_FILE"
  echo "        Generate it with Config_Files/create_frozen_config.py"
  exit 1
fi

# Fail here rather than in the fold-listing python below, whose error would otherwise
# be repeated once per worker.
if [ ! -f "$CV_CONFIG_PATH" ]; then
  echo "[ERROR] CV config not found: $CV_CONFIG_PATH (relative to $PWD)"
  exit 1
fi

# spatial_training_years_len does NOT select the frozen config -- that is done by
# prediction_loss + SPARSITY above. It only feeds the *fallback* CV path in
# Optuna_spatial_NCV.py:_resolve_cv_config_path, which is inert because we always pass
# --CV_config_path explicitly. Derive it from the CV filename anyway so it can never
# disagree with the file actually being run; hardcoding a default would silently
# mislabel a len9 run as len3.
if [ -z "${spatial_training_years_len:-}" ]; then
    spatial_training_years_len=$(basename "$CV_CONFIG_PATH" | grep -oE 'len[0-9]+' | head -1)
    : "${spatial_training_years_len:=unknown}"
fi
export spatial_training_years_len

: "${SELECT_OUTER_FOLD:=}"
: "${SELECT_INNER_LOOP:=}"
: "${WORKERS_PER_FOLD:=$DEFAULT_WORKERS_PER_FOLD}"
: "${TOP_TRIALS:=$DEFAULT_TOP_TRIALS}"

# NOTE: this fold predicate must stay identical to the spatial one in
# submit_paper1.sh. The array size computed there indexes this FOLDS array; if the
# two disagree, tasks map to the wrong folds and the tail is silently never run.
mapfile -t FOLDS < <(
    python - "$CV_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cv_config = json.load(f)

for key in sorted(
    k for k in cv_config
    if (k.startswith("outer_fold_") or k.startswith("outer_station_"))
    and "_otrain" not in k
):
    print(key)
PY
)

if [ -n "$SELECT_OUTER_FOLD" ]; then
    FOLDS=("$SELECT_OUTER_FOLD")
fi

if (( ${#FOLDS[@]} == 0 )); then
    echo "[ERROR] No spatial outer folds found in $CV_CONFIG_PATH"
    exit 1
fi

FOLD_INDEX=$SLURM_ARRAY_TASK_ID
: "${N_TRIALS:=9}"                              # per worker; 9 * 8 workers = 72 HPO trials per fold

# The array range must cover every fold, or the tail of FOLDS never runs and nothing
# says so. Complain from the first task only, so the message appears once.
if (( SLURM_ARRAY_TASK_MAX < ${#FOLDS[@]} - 1 )); then
    if (( FOLD_INDEX == SLURM_ARRAY_TASK_MIN )); then
        echo "[ERROR] Array range 0-${SLURM_ARRAY_TASK_MAX} covers only $((SLURM_ARRAY_TASK_MAX + 1)) of ${#FOLDS[@]} folds in $CV_CONFIG_PATH."
        echo "        Unreachable folds: ${FOLDS[*]:SLURM_ARRAY_TASK_MAX+1}"
        echo "        Submit via batch_scripts/submit_paper1.sh, which sizes --array from the CV config."
    fi
fi

if (( FOLD_INDEX >= ${#FOLDS[@]} )); then
    echo "[INFO] FOLD_INDEX $FOLD_INDEX >= ${#FOLDS[@]} folds — task has no work. Exiting."
    exit 0
fi

FOLD=${FOLDS[$FOLD_INDEX]}

echo "LANE:             spatial"
echo "FOLD_INDEX:       $FOLD_INDEX"
echo "FOLD:             $FOLD"
echo "FOLDS_LOADED:     ${#FOLDS[@]}"
echo "WORKERS_PER_FOLD: $WORKERS_PER_FOLD (spawned in-job, sharing this task's 1 GPU)"
echo "INNER_LOOP:       ${SELECT_INNER_LOOP:-all}"
echo "CONFIG:           $CONFIG_FILE"
echo "CV_CONFIG:        $CV_CONFIG_PATH"
echo "TRAIN_YEARS_LEN:  $spatial_training_years_len"
echo "SPARSITY:         ${SPARSITY:-full}"
echo "TOP_TRIALS:       $TOP_TRIALS"
echo "N_TRIALS/worker:  $N_TRIALS"
echo "PROCESS_MODEL:    $PROCESS_MODEL"
echo "PREDICTION_LOSS:  $prediction_loss"
echo "CALIBRATION_MODE: $calibration_flag"
echo "EXPERIMENT_ARTIFACT_NAMESPACE: ${EXPERIMENT_ARTIFACT_NAMESPACE:-<unset>}"

WORKER_LOG_DIR="log/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}_workers"
mkdir -p "$WORKER_LOG_DIR"
echo "WORKER_LOGS:      $WORKER_LOG_DIR"

pids=()
for (( WORKER_ID=0; WORKER_ID<WORKERS_PER_FOLD; WORKER_ID++ )); do
    PY_ARGS=(
        --outer_fold "$FOLD"
        --worker_id "$WORKER_ID"
        --workers_per_fold "$WORKERS_PER_FOLD"
        --n_trials "$N_TRIALS"
        --top_trials "$TOP_TRIALS"
        --process-model "$PROCESS_MODEL"
        --config "$CONFIG_FILE"
        --CV_config_path "$CV_CONFIG_PATH"
    )

    if [ -n "$SELECT_INNER_LOOP" ]; then
        PY_ARGS+=(--inner_loop "$SELECT_INNER_LOOP")
    fi

    python Optuna_spatial_NCV.py "${PY_ARGS[@]}" \
        > "${WORKER_LOG_DIR}/worker${WORKER_ID}.out" \
        2> "${WORKER_LOG_DIR}/worker${WORKER_ID}.err" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if (( status != 0 )); then
    echo "[ERROR] One or more workers failed for $FOLD — see $WORKER_LOG_DIR"
    exit 1
fi

echo "DONE";
echo "Finished at $(date)";
