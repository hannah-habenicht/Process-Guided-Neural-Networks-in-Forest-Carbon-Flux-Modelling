#!/bin/bash
#SBATCH -p alldlc2_cpu-epyc9655 #alldlc_gpu-rtx2080 #testdlc2_gpu-l40s #alldlc_gpu-rtx2080                       # partition (queue)
#SBATCH --mem 8G                                  # memory pool for each core (16GB)
#SBATCH -t 0-08:00                                # time (D-HH:MM)
#SBATCH --cpus-per-task=8                         # number of cores per job
##SBATCH --gres=gpu:1                             # number of GPUs per job
#SBATCH -o log/%x.%N.%A.%a.out                    # STDOUT (log folder should exist)
#SBATCH -e log/%x.%N.%A.%a.err                    # STDERR (log folder should exist)
#SBATCH -J p1_spatial_cal                        # sets the job name
##SBATCH --mail-type=END,FAIL                     # receive emails on job end or failure
#SBATCH -a 0-15%16                                # paper-1 CV_station_len*: 4 outer folds × (3 inner + 1 outer-only) = 16 tasks
## Task count = n_outer_folds × (n_inner_loops + 1 if ADD_OUTER_ONLY). Override with --array for other CV files.


# --- Print job info ---
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "This is job number $SLURM_ARRAY_TASK_ID"; # This is the job array index

# --- Activate environment ---
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"   # submit from the repository root
source "${VENV_PATH:-$PROJECT_ROOT/.venv}/bin/activate"


# ----------- User Configuration -------------
: "${PREDICTION_TYPE:=naive}"
: "${TRAIN_LEN:=len3}"
: "${SPARSITY:=}"                     # empty = full data; 50 | 30 | 7 → <PREDICTION_TYPE>_sparse<N>.json
: "${PROCESS_MODEL:=preles}"
: "${CALIBRATE_MODE:=force_calibrate}"
: "${CALIBRATION_ARTIFACT_NAMESPACE:=}"
: "${SELECT_OUTER_FOLD:=}"
: "${INNER_LOOPS:=inner_loop_1,inner_loop_2,inner_loop_3}"
: "${ADD_OUTER_ONLY:=true}"            # also calibrate each outer fold's outer train split
: "${ONLY_OUTER:=false}"
: "${CANONICAL_ONLY:=True}"
: "${OTRAIN_FOLD_LIST:=}"

export CALIBRATION_ARTIFACT_NAMESPACE

if [ -n "$SPARSITY" ]; then
    EXPERIMENT_NAME="${PREDICTION_TYPE}_sparse${SPARSITY}"
else
    EXPERIMENT_NAME="${PREDICTION_TYPE}"
fi
CONFIG_FILE="${PROJECT_ROOT}/configs/config_spatial/${EXPERIMENT_NAME}.json"

: "${CV_CONFIG:=${PROJECT_ROOT}/configs/CV_splits/Paper_1_preles_ncvs/CV_station_${TRAIN_LEN}_config.json}"
SCRIPT_PATH="${PROJECT_ROOT}/process_model_calibration_manager.py"
LOG_DIR="${PROJECT_ROOT}/log"
mkdir -p "$LOG_DIR"

mapfile -t OUTER_FOLDS < <(
    python - "$CV_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cv_config = json.load(f)

outer_folds = sorted(
    key for key in cv_config
    if key.startswith("outer_fold_") or key.startswith("outer_station_")
)

for key in outer_folds:
    print(key)
PY
)

if [ -n "$OTRAIN_FOLD_LIST" ]; then
    read -r -a OUTER_FOLDS <<< "$OTRAIN_FOLD_LIST"
fi

if [[ "$CANONICAL_ONLY" == "true" || "$CANONICAL_ONLY" == "1" ]]; then
    filtered=()
    for outer in "${OUTER_FOLDS[@]}"; do
        if [[ "$outer" != *"_otrain"* ]]; then
            filtered+=("$outer")
        fi
    done
    OUTER_FOLDS=("${filtered[@]}")
fi

if [ -n "$SELECT_OUTER_FOLD" ]; then
    OUTER_FOLDS=("$SELECT_OUTER_FOLD")
fi

if (( ${#OUTER_FOLDS[@]} == 0 )); then
    echo "[ERROR] No outer folds found in $CV_CONFIG"
    exit 1
fi

IFS=',' read -r -a INNER_LOOP_KEYS <<< "$INNER_LOOPS"

# Generate combinations based on selected modes and loops.
COMBINATIONS=()

if [[ "$ONLY_OUTER" == "true" || "$ONLY_OUTER" == "1" ]]; then
    for outer in "${OUTER_FOLDS[@]}"; do
        COMBINATIONS+=("${outer},None")
    done
else
    for outer in "${OUTER_FOLDS[@]}"; do
        for inner in "${INNER_LOOP_KEYS[@]}"; do
            COMBINATIONS+=("${outer},${inner}")
        done
        if [[ "$ADD_OUTER_ONLY" == "true" || "$ADD_OUTER_ONLY" == "1" ]]; then
            COMBINATIONS+=("${outer},None")
        fi
    done
fi

echo "[INFO] Loaded ${#OUTER_FOLDS[@]} outer folds from $CV_CONFIG"
echo "[INFO] Generated ${#COMBINATIONS[@]} calibration combinations"

if (( SLURM_ARRAY_TASK_ID >= ${#COMBINATIONS[@]} )); then
    echo "[ERROR] SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of range for ${#COMBINATIONS[@]} combinations"
    exit 1
fi

# --- Map task ID to config ---
IFS=',' read -r OUTER_FOLD INNER_LOOP <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
echo "[INFO] SLURM Task $SLURM_ARRAY_TASK_ID → OUTER_FOLD=$OUTER_FOLD, INNER_LOOP=$INNER_LOOP"
echo "[INFO] PROCESS_MODEL=$PROCESS_MODEL | CALIBRATE_MODE=$CALIBRATE_MODE"
echo "[INFO] CALIBRATION_ARTIFACT_NAMESPACE=${CALIBRATION_ARTIFACT_NAMESPACE:-<unset>}"

# --- Run ---
if [ "$INNER_LOOP" = "None" ]; then
    echo "[INFO] Running outer-only calibration for: $OUTER_FOLD"
    python "$SCRIPT_PATH" \
        --config "$CONFIG_FILE" \
        --cv-config "$CV_CONFIG" \
        --outer-fold "$OUTER_FOLD" \
        --process-model "$PROCESS_MODEL" \
        --calibrate "$CALIBRATE_MODE" \
        --no-plot
else
    echo "[INFO] Running inner calibration for: $OUTER_FOLD / $INNER_LOOP"
    python "$SCRIPT_PATH" \
        --config "$CONFIG_FILE" \
        --cv-config "$CV_CONFIG" \
        --outer-fold "$OUTER_FOLD" \
        --inner-loop "$INNER_LOOP" \
        --process-model "$PROCESS_MODEL" \
        --calibrate "$CALIBRATE_MODE" \
        --no-plot
fi
status=$?

if (( status != 0 )); then
    echo "[ERROR] Calibration failed for $OUTER_FOLD / ${INNER_LOOP:-None} (exit $status)"
    exit "$status"
fi

# Print some information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
