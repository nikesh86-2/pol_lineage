#!/bin/bash
# =============================================================================
# hbvpol — structural ensemble on a GPU node
# -----------------------------------------------------------------------------
#   cd /mnt/scratch/fbsnpat/pol_lineage
#   sbatch workflow/slurm/run_structure_gpu.slurm
#
# Folds the genotype/state/mutant ensemble (ColabFold/AlphaFold3/ESMFold — they
# must be installed and on PATH), then computes metrics and hinge calls.  The
# stage degrades to a manifest + warnings when no predictor is available, which
# makes this script safe to submit while you are still installing predictors.
#
# Tip: tune the ensemble size in config/config.yaml under `structure:` (states,
# lineages, strategies, seeds) before launching — that is what drives cost.
# =============================================================================

#SBATCH --job-name=hbvpol-struct
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=2-00:00:00
#SBATCH --output=workflow/slurm/logs/%x_%j.out
#SBATCH --error=workflow/slurm/logs/%x_%j.err

set -euo pipefail

resolve_repo_root() {
    local candidate
    for candidate in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        if [ -n "${candidate}" ] && [ -f "${candidate}/pyproject.toml" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}
REPO_ROOT="$(resolve_repo_root)" || {
    echo "ERROR: cannot find the repository root (a directory containing pyproject.toml)." >&2
    echo "       Submit from the repo root: cd /mnt/scratch/fbsnpat/pol_lineage && sbatch $0" >&2
    exit 2
}
cd "${REPO_ROOT}"

echo "== hbvpol structural ensemble =="
echo "job/node   : ${SLURM_JOB_ID:-local} / $(hostname)"
echo "gpu        : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "started    : $(date -Is)"

set +u
source /etc/profile.d/modules.sh
module load miniforge
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-/mnt/scratch/fbsnpat/envs/pol}"
set -u

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG="${CONFIG:-config/config.yaml}"
mkdir -p output

nvidia-smi || echo "WARNING: nvidia-smi unavailable; predictors may fall back to CPU"

python -m hbvpol.cli structure -c "${CONFIG}" --root .

# Molecular-dynamics pocket-persistence pass (GROMACS/OpenMM), if configured.
if [ "${RUN_MD:-0}" = "1" ]; then
    echo "== MD pocket persistence =="
    python -m hbvpol.cli structure -c "${CONFIG}" --root . structure.states='[apo]'
fi

echo "finished   : $(date -Is)"
