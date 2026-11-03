#!/bin/bash
#SBATCH -p alldlc2_gpu-l40s                       # partition (queue)
#SBATCH --mem 16G                                 # memory pool for the whole job (shared by all in-job workers)
#SBATCH -t 0-8:00                                 # time (D-HH:MM)
#SBATCH --cpus-per-task=16                        # cores for the job (shared by all in-job workers)
#SBATCH --gres=gpu:1                              # one GPU per fold, shared by all in-job workers
#SBATCH -o log/%x.%N.%A.%a.out                    # STDOUT (log folder should exist)
#SBATCH -e log/%x.%N.%A.%a.err                    # STDERR (log folder should exist)
#SBATCH -J p1_temporal                            # default job name; metaschedule_paper1.sh overrides via --job-name
#SBATCH -a 0-0%2                                  # default: one task per outer fold; submit_paper1.sh sizes this properly
#SBATCH --exclude=dlc2gpu07                       # faulty GPU memory (uncorrectable ECC); submit_paper1.sh's --exclude overrides this
#
# batch_paper1_temporal.sh — one array task = one outer fold = one GPU, running the
# temporal NCV entrypoint for the paper-1 PRELES rerun (the single-station files for
# station 5 and station 12).
#
# This is the temporal twin of batch_paper1_spatial.sh and differs only in:
#   * fold keys are "out_l_*" rather than "outer_fold_*" / "outer_station_*"
#   * entrypoint is Optuna_temporal_NCV.py
#   * exports temporal_training_years_len rather than spatial_training_years_len
#
# It deliberately does NOT reuse batch_temporal_new.sh, which still uses the old
# one-GPU-per-worker array mapping (FOLD_INDEX = ID / WORKERS_PER_FOLD), hardcodes
# its fold list, and activates conda venv2 instead of the project .venv.
#
# The temporal CV schema needs no conversion: Optuna_temporal_NCV.py reads
# "inner_loops" natively, and the outer split is resolved by
# process_model_calibration_manager._resolve_from_cv, which maps train_hold_out ->
# train and hold_out_years -> test.

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_GPUS_PER_NODE GPUs per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "This is job number $SLURM_ARRAY_TASK_ID";

[ -f /etc/cuda_env ] && source /etc/cuda_env   # cluster-specific CUDA setup
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"   # submit from the repository root
source "${VENV_PATH:-$PROJECT_ROOT/.venv}/bin/activate"


DEFAULT_CV_CONFIG_PATH="configs/CV_splits/Paper_1_preles_ncvs/temporal_cv_train_len1_station_5.json"
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
# is selected purely by filename. Both lanes share configs/config_spatial/.
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

if [ ! -f "$CV_CONFIG_PATH" ]; then
  echo "[ERROR] CV config not found: $CV_CONFIG_PATH (relative to $PWD)"
  exit 1
fi

# temporal_training_years_len does NOT select the frozen config -- that is done by
# prediction_loss + SPARSITY above. It feeds the *fallback* CV path (inert, since we
# always pass --CV_config_path) AND, unlike the spatial lane, the W&B run group at
# Optuna_temporal_NCV.py:910 (group=f"{ctx.training_years_len}"). Derive it from the CV
# filename so the group can never disagree with the file actually being run: a
# hardcoded default would file a len3 run under the len1 group.
if [ -z "${temporal_training_years_len:-}" ]; then
    temporal_training_years_len=$(basename "$CV_CONFIG_PATH" | grep -oE 'len[0-9]+' | head -1)
    : "${temporal_training_years_len:=unknown}"
fi
export temporal_training_years_len

: "${SELECT_OUTER_FOLD:=}"
: "${SELECT_INNER_LOOP:=}"
: "${WORKERS_PER_FOLD:=$DEFAULT_WORKERS_PER_FOLD}"
: "${TOP_TRIALS:=$DEFAULT_TOP_TRIALS}"

# NOTE: this fold predicate must stay identical to the temporal one in
# submit_paper1.sh. The array size computed there indexes this FOLDS array; if the
# two disagree, tasks map to the wrong folds and the tail is silently never run.
# "station" and "all_years" are the reserved top-level keys of the temporal schema.
mapfile -t FOLDS < <(
    python - "$CV_CONFIG_PATH" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cv_config = json.load(f)

for key in sorted(
    k for k in cv_config
    if k.startswith("out_l_") and "_otrain" not in k
):
    print(key)
PY
)

if [ -n "$SELECT_OUTER_FOLD" ]; then
    FOLDS=("$SELECT_OUTER_FOLD")
fi

if (( ${#FOLDS[@]} == 0 )); then
    echo "[ERROR] No temporal outer folds (out_l_*) found in $CV_CONFIG_PATH"
    exit 1
fi

FOLD_INDEX=$SLURM_ARRAY_TASK_ID
: "${N_TRIALS:=9}"                              # per worker; 9 * 8 workers = 72 HPO trials per fold

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

echo "LANE:             temporal"
echo "FOLD_INDEX:       $FOLD_INDEX"
echo "FOLD:             $FOLD"
echo "FOLDS_LOADED:     ${#FOLDS[@]}"
echo "WORKERS_PER_FOLD: $WORKERS_PER_FOLD (spawned in-job, sharing this task's 1 GPU)"
echo "INNER_LOOP:       ${SELECT_INNER_LOOP:-all}"
echo "CONFIG:           $CONFIG_FILE"
echo "CV_CONFIG:        $CV_CONFIG_PATH"
echo "TRAIN_YEARS_LEN:  $temporal_training_years_len (also the W&B run group)"
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

    python Optuna_temporal_NCV.py "${PY_ARGS[@]}" \
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
