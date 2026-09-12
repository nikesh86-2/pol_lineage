# Running hbvpol on SLURM

Three batch scripts, all submitting from the repository root:

| Script | Partition | Purpose |
|---|---|---|
| `run_smoke_test.slurm` | `nodes` (2 CPU, 30 min) | Offline synthetic end-to-end run — prove the job environment before a real run |
| `run_pipeline.slurm` | `nodes` (16 CPU, 2 days) | The full pipeline, or a single stage, on one node |
| `run_structure_gpu.slurm` | `gpu` (1× l40s, 8 CPU) | Structural ensemble on a GPU node |

```bash
cd /mnt/scratch/fbsnpat/pol_lineage

sbatch workflow/slurm/run_smoke_test.slurm          # 1. sanity check
sbatch workflow/slurm/run_pipeline.slurm            # 2. real run

squeue -u "$USER"
tail -f workflow/slurm/logs/hbvpol_<jobid>.out
```

## Submitting a single stage

```bash
sbatch --export=ALL,STAGE=select   workflow/slurm/run_pipeline.slurm
sbatch --export=ALL,STAGE=atlas    workflow/slurm/run_pipeline.slurm
```

`STAGE` accepts any `hbvpol` subcommand (`fetch`, `deephep`, `reference`, `qc`,
`recombine`, `tree`, `select`, `epsilon`, `fitness`, `structure`, `atlas`) or
`all`.

## Overrides

| Variable | Default | Meaning |
|---|---|---|
| `CONDA_ENV` | `/mnt/scratch/fbsnpat/envs/pol` | Conda environment to activate |
| `CONFIG` | `config/config.yaml` | Configuration file |
| `STAGE` | `all` | Stage to run, or `all` |
| `SNAKEMAKE_TARGET` | (empty → full DAG) | Target rule/file passed to Snakemake |
| `HBVPOL` | `python -m hbvpol.cli` | How stages are invoked |
| `NCBI_EMAIL` | (unset) | NCBI contact e-mail, forwarded as `datasets.hbv.genbank.email` so it is never committed |
| `OPENRDP_ENV` | `/mnt/scratch/fbsnpat/envs/openrdp` | OpenRDP's own env; its `bin` is prepended to `PATH` so `executable: openrdp` resolves |
| `SNAKEMAKE_RERUN_TRIGGERS` | (unset → provenance) | Trigger type passed to Snakemake's `--rerun-triggers`; set to `mtime` to reuse existing outputs (e.g. skip re-fetching `output/datasets`) on a resubmission |
| `RUN_MD` | `0` | In the GPU script, also run the MD persistence pass |

Example — set the NCBI contact and launch for real:

```bash
sbatch --export=ALL,NCBI_EMAIL=you@example.org workflow/slurm/run_pipeline.slurm
```

Example — full run with an alternate config and a larger ensemble:

```bash
sbatch --export=ALL,CONFIG=config/config.gtA-H.yaml,SNAKEMAKE_TARGET=atlas \
       workflow/slurm/run_pipeline.slurm
```

Example — resubmit after code changes, keeping the already-fetched genomes:

```bash
sbatch --export=ALL,NCBI_EMAIL=you@example.org,SNAKEMAKE_RERUN_TRIGGERS=mtime \
       workflow/slurm/run_pipeline.slurm
```

By default Snakemake uses provenance (rule code + config hash) to decide what to
re-run, so editing the Snakefile or config re-triggers every stage, including the
network fetch. `SNAKEMAKE_RERUN_TRIGGERS=mtime` falls back to modification times,
which is what you want when the upstream outputs on disk are still valid.

## Why the environment setup looks unusual

* **Module function in batch jobs.** SLURM does not run a login shell, so the
  scripts `source /etc/profile.d/modules.sh` before `module load miniforge`.
* **Activation by path.** `conda activate pol` fails in a clean non-login shell
  because the environment lives under `/mnt/scratch/fbsnpat/envs`, outside the
  default `envs_dirs` searched by name. The scripts therefore activate
  `/mnt/scratch/fbsnpat/envs/pol` directly. On this cluster the miniforge
  activation hook is at
  `/opt/apps/pkg/interpreters/miniforge/*/bin/etc/profile.d/conda.sh`; the
  scripts derive it with `conda info --base` so they survive version bumps.
* **No editable install needed.** `PYTHONPATH=src` makes `python -m hbvpol.cli`
  work; the Snakemake rule uses the same CLI through `HBVPOL`.

## Scaling out with the SLURM executor plugin

Snakemake here runs the whole DAG **inside one allocation** (`--cores`), which is
fine for ~10k genomes on 16 cores. To submit one job per rule instead, install
the plugin and let Snakemake be the submitter:

```bash
mamba install -n pol -c conda-forge snakemake-executor-plugin-slurm
sbatch --partition=nodes --time=2-00:00:00 --mem=4G --cpus-per-task=2 \
  --wrap "cd /mnt/scratch/fbsnpat/pol_lineage && \
          source /etc/profile.d/modules.sh && module load miniforge && \
          source \$(conda info --base)/etc/profile.d/conda.sh && \
          conda activate /mnt/scratch/fbsnpat/envs/pol && \
          PYTHONPATH=src snakemake -s workflow/Snakefile --executor slurm \
            --jobs 50 --default-resources mem_mb=16000 runtime=240"
```

## Notes

* The smoke test writes `output-smoke/`; the real runs write `output/`
  (`project.output_root`). Both are git-ignored.
* `run_pipeline.slurm` warns if `datasets.hbv.genbank.email` is still unset —
  the fetch stage is skipped without it.
* Jobs above are sized for the observed partitions (`nodes` 2-day limit;
  `gpu` nodes expose `gpu:l40s:3`). Adjust `--mem`, `--cpus-per-task` and
  `--time` to your allocation.
