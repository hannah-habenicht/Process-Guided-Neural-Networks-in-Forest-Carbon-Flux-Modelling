#!/bin/bash
#SBATCH -p alldlc2_cpu-epyc9655 #alldlc_gpu-rtx2080 #testdlc2_gpu-l40s #alldlc_gpu-rtx2080                       # partition (queue)
#SBATCH --mem 8G                                  # memory pool for each core (16GB)
#SBATCH -t 0-08:00                                # time (D-HH:MM)
#SBATCH --cpus-per-task=8                         # number of cores per job
##SBATCH --gres=gpu:1                             # number of GPUs per job
#SBATCH -o log/%x.%N.%A.%a.out                    # STDOUT (log folder should exist)
#SBATCH -e log/%x.%N.%A.%a.err                    # STDERR (log folder should exist)
#SBATCH -J P7_temporal_cal                        # sets the job name
##SBATCH --mail-type=END,FAIL                     # receive emails on job end or failure
#SBATCH -a 0-4                                    # station 12 len1: 3 outer × (3 inner + 1 outer-only) = 12 combos
                                                  #   → 5 unique training-year sets (DEDUPE=true)
## Task count depends on the CV file and DEDUPE. Check before submitting and override with --array:
##   LIST_ONLY=1 bash batch_scripts/batch_calibrations_temporal.sh
## station 5 (6 outer × (5 inner + 1 outer-only) = 36 combos before dedupe)
##
## Examples:
##   sbatch batch_scripts/batch_calibrations_temporal.sh                                  # full, st12 len1
##   SPARSITY=50 sbatch batch_scripts/batch_calibrations_temporal.sh                      # sparse50 config
##   CV_CONFIG=.../temporal_cv_train_len1_station_5.json sbatch --array=0-N batch_scripts/batch_calibrations_temporal.sh


# --- Print job info ---
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME with JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "This is job number $SLURM_ARRAY_TASK_ID"; # This is the job array index

# --- Activate environment ---
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"   # submit from the repository root
source "${VENV_PATH:-$PROJECT_ROOT/.venv}/bin/activate"


# ----------- User Configuration -------------
: "${PREDICTION_TYPE:=naive}"         # bundles are shared across architectures; only picks the frozen config
: "${SPARSITY:=}"                     # empty = full data; 50 | 30 | 7 → <PREDICTION_TYPE>_sparse<N>.json
: "${PROCESS_MODEL:=preles}"
: "${CALIBRATE_MODE:=auto}"           # auto = calibrate only missing/incompatible bundles; force_calibrate redoes all
: "${CALIBRATION_ARTIFACT_NAMESPACE:=}"
: "${SELECT_OUTER_FOLD:=}"            # e.g. out_l_2 — restrict to one outer fold
: "${INNER_LOOPS:=}"                  # comma list, e.g. inner_loop1,inner_loop3; empty = all loops in the CV file
: "${ADD_OUTER_ONLY:=true}"           # also calibrate each outer fold's train_hold_out years
: "${ONLY_OUTER:=false}"              # skip inner loops entirely
: "${DEDUPE:=true}"                   # one task per unique training-year set (bundle path keys on train years only)
: "${LIST_ONLY:=false}"               # print the task table and exit (no calibration)

export CALIBRATION_ARTIFACT_NAMESPACE

if [ -n "$SPARSITY" ]; then
    EXPERIMENT_NAME="${PREDICTION_TYPE}_sparse${SPARSITY}"
else
    EXPERIMENT_NAME="${PREDICTION_TYPE}"
fi
CONFIG_FILE="${PROJECT_ROOT}/configs/config_spatial/${EXPERIMENT_NAME}.json"

: "${CV_CONFIG:=${PROJECT_ROOT}/configs/CV_splits/Paper_1_preles_ncvs/new_temporal_cv_train_len1_station_12.json}"
SCRIPT_PATH="${PROJECT_ROOT}/process_model_calibration_manager.py"
LOG_DIR="${PROJECT_ROOT}/log"
mkdir -p "$LOG_DIR"

for f in "$CONFIG_FILE" "$CV_CONFIG"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] File not found: $f"
        exit 1
    fi
done

# Temporal CV schema: {"station": N, "out_l_<k>": {"train_hold_out": [...], "hold_out_years": [...],
#                      "inner_loops": {"inner_loop<k>": {"train": [...], "test": [...], "val": [...]}}}}
# Emits one "outer,inner,train_years" line per calibration (inner = None for outer-only).
mapfile -t COMBINATIONS < <(
    python - "$CV_CONFIG" "$SELECT_OUTER_FOLD" "$INNER_LOOPS" "$ADD_OUTER_ONLY" "$ONLY_OUTER" "$DEDUPE" <<'PY'
import json
import sys

cv_path, select_outer, inner_arg, add_outer, only_outer, dedupe = sys.argv[1:7]
truthy = lambda s: s.lower() in {"true", "1"}

with open(cv_path, "r", encoding="utf-8") as f:
    cv_config = json.load(f)

if "station" not in cv_config:
    sys.exit(f"[ERROR] {cv_path} is not a temporal CV config (no 'station' key)")

outer_folds = sorted(
    (k for k in cv_config if k.startswith("out_l_")),
    key=lambda k: int(k.rsplit("_", 1)[1]),
)
if select_outer:
    if select_outer not in cv_config:
        sys.exit(f"[ERROR] Outer fold '{select_outer}' not in {cv_path}. Available: {outer_folds}")
    outer_folds = [select_outer]

requested_inner = [k for k in inner_arg.split(",") if k]
seen = set()

def emit(outer, inner, years):
    sig = "_".join(str(y) for y in sorted(years))
    if truthy(dedupe):
        if sig in seen:
            return
        seen.add(sig)
    print(f"{outer},{inner},{sig}")

for outer in outer_folds:
    fold = cv_config[outer]
    if truthy(add_outer) or truthy(only_outer):
        emit(outer, "None", fold["train_hold_out"])
    if truthy(only_outer):
        continue
    inner_loops = fold["inner_loops"]
    for inner in (requested_inner or sorted(inner_loops)):
        if inner not in inner_loops:
            sys.exit(f"[ERROR] '{inner}' not in {outer}. Available: {sorted(inner_loops)}")
        emit(outer, inner, inner_loops[inner]["train"])
PY
)

if (( ${#COMBINATIONS[@]} == 0 )); then
    echo "[ERROR] No calibration combinations generated from $CV_CONFIG"
    exit 1
fi

echo "[INFO] CV_CONFIG=$CV_CONFIG"
echo "[INFO] CONFIG_FILE=$CONFIG_FILE"
echo "[INFO] Generated ${#COMBINATIONS[@]} calibration tasks (DEDUPE=$DEDUPE)"

if [[ "$LIST_ONLY" == "true" || "$LIST_ONLY" == "1" ]]; then
    for i in "${!COMBINATIONS[@]}"; do
        IFS=',' read -r o n y <<< "${COMBINATIONS[$i]}"
        printf "  %3d  %-8s %-12s train_years=%s\n" "$i" "$o" "$n" "$y"
    done
    echo "[INFO] Submit with --array=0-$(( ${#COMBINATIONS[@]} - 1 ))"
    exit 0
fi

if (( SLURM_ARRAY_TASK_ID >= ${#COMBINATIONS[@]} )); then
    echo "[ERROR] SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID is out of range for ${#COMBINATIONS[@]} combinations"
    exit 1
fi

# --- Map task ID to config ---
IFS=',' read -r OUTER_FOLD INNER_LOOP TRAIN_YEARS <<< "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
echo "[INFO] SLURM Task $SLURM_ARRAY_TASK_ID → OUTER_FOLD=$OUTER_FOLD, INNER_LOOP=$INNER_LOOP, TRAIN_YEARS=$TRAIN_YEARS"
echo "[INFO] PROCESS_MODEL=$PROCESS_MODEL | CALIBRATE_MODE=$CALIBRATE_MODE | SPARSITY=${SPARSITY:-full}"
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
