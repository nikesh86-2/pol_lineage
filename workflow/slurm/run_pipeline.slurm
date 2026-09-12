#!/bin/bash
# =============================================================================
# hbvpol — run the full pipeline on one compute node
# -----------------------------------------------------------------------------
# Submit from the repository root (SBATCH log paths are relative to the cwd):
#
#   cd /mnt/scratch/fbsnpat/pol_lineage
#   sbatch workflow/slurm/run_pipeline.slurm
#
# Run a single stage instead of the whole pipeline:
#
#   sbatch --export=ALL,STAGE=select workflow/slurm/run_pipeline.slurm
#
# Run a specific Snakemake target (see workflow/Snakefile):
#
#   sbatch --export=ALL,SNAKEMAKE_TARGET=workflow/slurm/../... \
#          workflow/slurm/run_pipeline.slurm
#
# Notes
# -----
# * Snakemake 9 in this environment has no SLURM executor plugin, so Snakemake
#   drives the DAG *inside* this single allocation using --cores.  For very
#   large runs, install snakemake-executor-plugin-slurm and use
#   `--executor slurm` instead (see workflow/slurm/README.md).
# * `hbvpol` is used from `src/` via PYTHONPATH, so no editable install is
#   required.  External tools (mafft/iqtree/hyphy/...) are picked up if present
#   and skipped with a warning otherwise.
# =============================================================================

#SBATCH --job-name=hbvpol
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=workflow/slurm/logs/%x_%j.out
#SBATCH --error=workflow/slurm/logs/%x_%j.err
## Uncomment if your association requires an explicit account/QOS:
##SBATCH --account=default
##SBATCH --qos=normal

set -euo pipefail

# --- locate the repository ---------------------------------------------------
# SLURM executes a *copy* of this script from a spool directory, so paths
# derived from BASH_SOURCE are unreliable inside a job.  Prefer SLURM_SUBMIT_DIR
# and validate by looking for pyproject.toml.
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

echo "== hbvpol pipeline =="
echo "job        : ${SLURM_JOB_ID:-local}"
echo "node       : $(hostname)"
echo "repo       : ${REPO_ROOT}"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-1}"
echo "started    : $(date -Is)"

# --- environment -------------------------------------------------------------
# A non-login batch shell does not get the module function for free.
# `set +u`: the cluster's module/conda activation scripts reference variables
# (e.g. PS1) that are unset in a batch shell and would abort under `set -u`.
set +u
source /etc/profile.d/modules.sh
module load miniforge
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# The env is activated by absolute path: `conda activate pol` fails in a clean
# shell because the env lives under /mnt/scratch (see workflow/slurm/README.md).
CONDA_ENV="${CONDA_ENV:-/mnt/scratch/fbsnpat/envs/pol}"
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HBVPOL="${HBVPOL:-python -m hbvpol.cli}"

# OpenRDP pins numpy<2 and lives in its own environment; expose its bin so
# `recombination.openrdp.executable: openrdp` resolves without activating it.
OPENRDP_ENV="${OPENRDP_ENV:-/mnt/scratch/fbsnpat/envs/openrdp}"
if [ -x "${OPENRDP_ENV}/bin/openrdp" ]; then
    export PATH="${OPENRDP_ENV}/bin:${PATH}"
fi

echo "python     : $(command -v python)"
python -c "import hbvpol; print('hbvpol', hbvpol.__version__)"

# --- configuration -----------------------------------------------------------
CONFIG="${CONFIG:-config/config.yaml}"
STAGE="${STAGE:-all}"
CORES="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p output runs

# Keep every tool's thread count within the allocation: IQ-TREE 3 hard-errors
# when given more threads than the job has cores.  Stages also clamp internally
# (hbvpol.pipeline.effective_threads), but this keeps the config consistent.
export HBVPOL_OVERRIDES="${HBVPOL_OVERRIDES:-project.threads=${CORES}}"

# Forward an NCBI contact e-mail supplied at submit time, so it never has to be
# written into the repository:  sbatch --export=ALL,NCBI_EMAIL=you@example.org ...
if [ -n "${NCBI_EMAIL:-}" ]; then
    export HBVPOL_OVERRIDES="${HBVPOL_OVERRIDES} datasets.hbv.genbank.email=${NCBI_EMAIL}"
fi

# The human-HBV fetch needs a real contact e-mail for NCBI.
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 2
fi
if [ -z "${NCBI_EMAIL:-}" ] && grep -qE '^\s*email:\s*(null|~|)\s*$' "${CONFIG}"; then
    echo "WARNING: no NCBI e-mail (NCBI_EMAIL env or datasets.hbv.genbank.email);"
    echo "         the fetch stage will be skipped and QC will have no input."
fi

record_tools() {
    echo "== external tools =="
    for tool in mafft iqtree2 iqtree hyphy trimal cd-hit RDP5 3seq gmx; do
        if command -v "${tool}" >/dev/null 2>&1; then
            echo "  found   ${tool}"
        else
            echo "  missing ${tool}"
        fi
    done
}
record_tools

# --- run ---------------------------------------------------------------------
if [ "${STAGE}" = "all" ]; then
    TARGET="${SNAKEMAKE_TARGET:-}"
    echo "== snakemake (cores=${CORES}) target='${TARGET:-all}' =="
    # Clear any stale lock left by a killed/preempted run.  Safe here: the job
    # owns this working directory, and --unlock is a no-op when not locked.
    snakemake --snakefile workflow/Snakefile --unlock --quiet || true
    # By default Snakemake re-runs a rule when its code or config changed
    # (provenance).  Set SNAKEMAKE_RERUN_TRIGGERS=mtime to reuse existing outputs
    # (e.g. keep output/datasets and skip re-fetching) on a resubmission.
    RERUN_TRIGGERS="${SNAKEMAKE_RERUN_TRIGGERS:-}"
    snakemake \
        --snakefile workflow/Snakefile \
        --cores "${CORES}" \
        ${RERUN_TRIGGERS:+--rerun-triggers "${RERUN_TRIGGERS}"} \
        --rerun-incomplete \
        --keep-going \
        --printshellcmds \
        --latency-wait 60 \
        ${TARGET}
else
    echo "== single stage: ${STAGE} =="
    python -m hbvpol.cli "${STAGE}" -c "${CONFIG}" --root . ${HBVPOL_OVERRIDES}
fi

echo "finished   : $(date -Is)"
echo "outputs    : ${REPO_ROOT}/output"
