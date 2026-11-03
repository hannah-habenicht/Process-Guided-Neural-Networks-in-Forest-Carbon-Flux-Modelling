#!/bin/bash
# submit_paper1.sh — submit ONE paper-1 PRELES experiment (one CV file x one sparsity
# x one architecture) as a single array job, one task per outer fold.
#
# Both lanes are handled here:
#   LANE=spatial   -> batch_paper1_spatial.sh   -> Optuna_spatial_NCV.py   (OOD CV_station_len*)
#   LANE=temporal  -> batch_paper1_temporal.sh  -> Optuna_temporal_NCV.py  (station 5 / 12)
#
# Usage (normally driven by metaschedule_paper1.sh, but standalone works):
#   LANE=temporal \
#   CV_CONFIG_PATH=configs/CV_splits/paper_1_preles_ncvs/temporal_cv_train_len1_station_5.json \
#   prediction_loss=naive SPARSITY=30 \
#     bash batch_scripts/submit_paper1.sh
#
# Chaining: pass DEPEND_ON=<jobid> to hold this submission until that job finishes
# (afterany, so a failed predecessor does not deadlock the rest of the chain), and
# JID_OUT=<file> to have the new job id written somewhere the caller can read it.
#
# Unlike submit_spatial.sh there is no second wave: none of the paper-1 CV files
# contain _otrain draw folds. If one ever does, this script says so rather than
# silently ignoring them.

set -euo pipefail

LANE="${LANE:-spatial}"
CV_CONFIG_PATH="${CV_CONFIG_PATH:-}"
SPARSITY="${SPARSITY:-}"
prediction_loss="${prediction_loss:-naive}"
PROCESS_MODEL="${PROCESS_MODEL:-preles}"
WORKERS_PER_FOLD="${WORKERS_PER_FOLD:-8}"
THROTTLE="${THROTTLE:-2}"            # max simultaneous tasks (%N in --array) = max concurrent GPUs
DEPEND_ON="${DEPEND_ON:-}"
JOB_NAME="${JOB_NAME:-}"
JID_OUT="${JID_OUT:-}"
EXPERIMENT_ARTIFACT_NAMESPACE="${EXPERIMENT_ARTIFACT_NAMESPACE:-}"
DRY_RUN="${DRY_RUN:-0}"
# dlc2gpu07 has faulty GPU memory: jobs die at the first CUDA allocation with
# "uncorrectable ECC error". Slurm has never drained it, so it keeps getting
# scheduled. Set EXCLUDE_NODES="" to submit to it anyway.
EXCLUDE_NODES="${EXCLUDE_NODES-dlc2gpu07}"

case "$LANE" in
    spatial)
        BATCH_SCRIPT="batch_scripts/batch_paper1_spatial.sh"
        # Must stay identical to the predicate in batch_paper1_spatial.sh.
        FOLD_PREDICATE='k.startswith("outer_fold_") or k.startswith("outer_station_")'
        ;;
    temporal)
        BATCH_SCRIPT="batch_scripts/batch_paper1_temporal.sh"
        # Must stay identical to the predicate in batch_paper1_temporal.sh.
        FOLD_PREDICATE='k.startswith("out_l_")'
        ;;
    *)
        echo "[ERROR] LANE must be 'spatial' or 'temporal', got '$LANE'"
        exit 1
        ;;
esac

if [ -z "$CV_CONFIG_PATH" ]; then
    echo "[ERROR] CV_CONFIG_PATH is required."
    exit 1
fi
if [ ! -f "$CV_CONFIG_PATH" ]; then
    echo "[ERROR] CV config not found: $CV_CONFIG_PATH"
    exit 1
fi
if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "[ERROR] Batch script not found: $BATCH_SCRIPT"
    exit 1
fi

# Sparsity is baked into the frozen config filename, not passed on the CLI.
SPARSE_PART=""
if [ -n "$SPARSITY" ]; then
    SPARSE_PART="_sparse${SPARSITY}"
fi
CONFIG_FILE="configs/config_spatial/${prediction_loss}${SPARSE_PART}.json"
if [ ! -f "$CONFIG_FILE" ]; then
    # Warn rather than abort under DRY_RUN, so the full plan stays inspectable while
    # the frozen configs are still being regenerated.
    if [ "$DRY_RUN" = "1" ]; then
        echo "[WARNING] Frozen config not found: $CONFIG_FILE (dry run — continuing)"
    else
        echo "[ERROR] Frozen config not found: $CONFIG_FILE"
        echo "        Generate it with Config_Files/create_frozen_config.py"
        exit 1
    fi
fi

# Count canonical folds, and report _otrain folds rather than dropping them silently.
# NOTE: the predicate below must match the one in the chosen batch script, or the array
# size computed here indexes a different fold list and the tail is never run.
read -r -d '' PY_COUNT <<PY || true
import json, sys
cv = json.load(open(sys.argv[1]))
outer = [k for k in cv if ${FOLD_PREDICATE}]
canonical = sorted(k for k in outer if "_otrain" not in k)
otrain = sorted(k for k in outer if "_otrain" in k)
print("N_CANONICAL=%d" % len(canonical))
print("N_OTRAIN=%d" % len(otrain))
print("FOLD_LIST='%s'" % " ".join(canonical))
PY

eval "$(python - "$CV_CONFIG_PATH" <<< "$PY_COUNT")"

if (( N_CANONICAL == 0 )); then
    echo "[ERROR] No $LANE outer folds found in $CV_CONFIG_PATH"
    exit 1
fi
if (( N_OTRAIN > 0 )); then
    echo "[WARNING] $CV_CONFIG_PATH contains $N_OTRAIN _otrain fold(s), which this"
    echo "          script does not submit. Use submit_spatial.sh's two-wave flow for those."
fi

ARRAY_MAX=$(( N_CANONICAL - 1 ))

if [ -z "$JOB_NAME" ]; then
    JOB_NAME="p1_${prediction_loss}_$(basename "$CV_CONFIG_PATH" .json)_${SPARSITY:+sparse}${SPARSITY:-full}"
fi

# Kept as an array so empty values expand to no argument at all rather than an empty
# string, which sbatch would reject.
SBATCH_EXTRA=()
if [ -n "$EXCLUDE_NODES" ]; then
    SBATCH_EXTRA+=(--exclude="$EXCLUDE_NODES")
fi
if [ -n "$DEPEND_ON" ]; then
    # afterany, not afterok: one failed experiment must not deadlock the rest of its chain.
    SBATCH_EXTRA+=(--dependency="afterany:${DEPEND_ON}")
fi

echo "Lane:            $LANE"
echo "CV config:       $CV_CONFIG_PATH"
echo "Frozen config:   $CONFIG_FILE"
echo "Architecture:    $prediction_loss"
echo "Sparsity:        ${SPARSITY:-full}"
echo "Process model:   $PROCESS_MODEL"
echo "Folds:           $N_CANONICAL  ($FOLD_LIST)"
echo "Workers/fold:    $WORKERS_PER_FOLD"
echo "Throttle:        $THROTTLE concurrent GPUs"
echo "Job name:        $JOB_NAME"
[ -n "$DEPEND_ON" ]     && echo "Depends on:      afterany:$DEPEND_ON"
[ -n "$EXCLUDE_NODES" ] && echo "Excluded nodes:  $EXCLUDE_NODES"

SBATCH_ARGS=(
    --parsable
    --job-name "$JOB_NAME"
    --array="0-${ARRAY_MAX}%${THROTTLE}"
    "${SBATCH_EXTRA[@]}"
    --export=ALL,CV_CONFIG_PATH="$CV_CONFIG_PATH",SPARSITY="$SPARSITY",prediction_loss="$prediction_loss",PROCESS_MODEL="$PROCESS_MODEL",WORKERS_PER_FOLD="$WORKERS_PER_FOLD",EXPERIMENT_ARTIFACT_NAMESPACE="$EXPERIMENT_ARTIFACT_NAMESPACE"
    "$BATCH_SCRIPT"
)

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] sbatch ${SBATCH_ARGS[*]}"
    # Emit a placeholder so a dry-run chain still exercises the dependency wiring.
    if [ -n "$JID_OUT" ]; then
        echo "DRYRUN" > "$JID_OUT"
    fi
    echo ""
    exit 0
fi

JID=$(sbatch "${SBATCH_ARGS[@]}")

echo "Submitted job:   $JID  ($((ARRAY_MAX + 1)) tasks)"
echo "Logs:            log/${JOB_NAME}.*.${JID}.*.{out,err}"
echo "Worker logs:     log/${JOB_NAME}.${JID}.<task>_workers/worker<N>.{out,err}"
echo ""

if [ -n "$JID_OUT" ]; then
    echo "$JID" > "$JID_OUT"
fi
