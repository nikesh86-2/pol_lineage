#!/bin/bash
# =============================================================================
# hbvpol — smoke test on SLURM (no network, no external tools)
# -----------------------------------------------------------------------------
#   cd /mnt/scratch/fbsnpat/pol_lineage
#   sbatch workflow/slurm/run_smoke_test.slurm
#
# Fabricates a tiny synthetic dataset and runs every stage in a few seconds.
# Submit this once after any environment change to prove the job exactly as
# SLURM will execute it (module loading, conda activation, PYTHONPATH, cwd)
# before committing a multi-day pipeline run.
# =============================================================================

#SBATCH --job-name=hbvpol-smoke
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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

echo "== hbvpol smoke test =="
echo "job/node   : ${SLURM_JOB_ID:-local} / $(hostname)"
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

python -c "import hbvpol, pandas, numpy; print('hbvpol', hbvpol.__version__, '| pandas', pandas.__version__)"

python scripts/run_synthetic.py --outdir output-smoke --genomes 8

echo "smoke test OK: $(date -Is)"
