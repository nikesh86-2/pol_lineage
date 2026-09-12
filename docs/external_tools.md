# External tools

`hbvpol` is designed so that **every stage runs offline**. External programs are
used when present and are skipped — with an explicit fallback and a warning —
when absent, so the pipeline is always testable. For a real analysis you want
the real tools.

## Conda (recommended)

```bash
mamba env update -n pol -f environment-tools.yml
```

This installs MAFFT, HMMER, MUSCLE, trimAl, CD-HIT, IQ-TREE, RAxML-NG, HyPhy,
OpenMM, PDBFixer, Biotite, FreeSASA and DSSP.

## Programs distributed outside conda

### RDP5

RDP5 is distributed as a Java application (and a Windows build) rather than
through bioconda. Download it from the RDP project, then point the config at the
launcher:

```yaml
recombination:
  rdp5:
    executable: /path/to/RDP5/bin/RDP5_cli   # or RDP5.jar with a wrapper
```

The wrapper writes an RDP5 project file and parses its CSV output
(`hbvpol/recombination/rdp5.py`). If the executable is missing the tool is
skipped. Because RDP5 is interactive by design, `domain_wise: true` runs it once
per Pol domain and reconciles the breakpoints.

### OpenRDP (recommended RDP-family implementation)

[OpenRDP](https://github.com/PoonLab/OpenRDP) is an open-source Python
re-implementation of RDP4/RDP5 (PoonLab) with `rdp`, `geneconv`, `bootscan`,
`maxchi`, `siscan`, `chimaera` and `threeseq` methods; it bundles the third-party
3Seq and GENECONV binaries. It is used in place of RDP5 when installed.

It is **not** on conda or PyPI, and its `setup.py` pins `numpy<2` and
`h5py<3.11`, which conflicts with this project's NumPy 2 stack. Install it in
its own environment:

```bash
mamba create -p /path/to/envs/openrdp -c conda-forge \
      python=3.11 "numpy<2" scipy "h5py>=3.8,<3.11" pip
/path/to/envs/openrdp/bin/pip install "git+https://github.com/PoonLab/OpenRDP.git"
```

```yaml
recombination:
  tools: [openrdp, gard, threeseq, bootscan]
  openrdp:
    executable: /path/to/envs/openrdp/bin/openrdp
    methods: [rdp, threeseq]   # maxchi/siscan/chimaera abort on some inputs
    seed: 3
    fail: 100
```

The wrapper (`hbvpol/recombination/openrdp.py`) runs the binary and parses its
CSV (`Method,Start,End,Recombinant,Parent1,Parent2,Pvalue`). Because it lives in
another env, the SLURM script prepends `${OPENRDP_ENV}/bin` to `PATH`
(`OPENRDP_ENV` defaults to `/mnt/scratch/fbsnpat/envs/openrdp`), so
`executable: openrdp` resolves.

> OpenRDP's own README warns it is still under development and not a drop-in
> replacement for RDP5; some methods raise on some alignments. The wrapper fails
> soft (empty frame + warning) so the stage continues.

### 3SEQ

Bioconda ships 3SEQ: `mamba install -c bioconda 3seq`. Note its option style
differs from RDP's 3SEQ interface — the full run is
`3seq -full <alignment>`, options are **attached** (`-t0.05`, not `-t 0.05`),
`-id <name>` prefixes the output files (`<name>.3s.rec.csv`), and `-f`/`-l` mean
*first/last nucleotide* rather than input/output files. The wrapper
(`hbvpol/recombination/threeseq.py`) handles this and parses the report's fixed
column schema.

### GROMACS

```bash
mamba install -n pol -c conda-forge gromacs
```

`hbvpol.structure.md` emits GROMACS command plans and can fall back to OpenMM
when GROMACS is unavailable. Its purpose is the pocket-persistence criterion
before any large-scale docking: a predicted cavity must survive dynamics, not
just score well in a static structure.

### Structural predictors

ColabFold, AlphaFold3 and ESMFold are orchestrated, not bundled. Install them
in their own environments and expose the entry points on `PATH`:

| Strategy | Typical command |
|---|---|
| `colabfold` | `colabfold_batch` |
| `alphafold3` | `run_alphafold.py` |
| `esmfold` | a `fair-esm`/`esm` wrapper script |

`hbvpol.structure.models.run_predictor` only invokes a strategy when its
executable is found; otherwise the model manifest records where the model *would*
live and the stage continues. GPUs and model weights are your responsibility —
they are large and are not redistributed here.

## DCA backend (plmc)

`pydca`/`plmDCA` — the module the covariation stage originally imported — is
unmaintained.  **`plmc`** is the maintained C++ implementation of plmDCA and is
packaged on bioconda (already listed in `environment-tools.yml`):

```bash
mamba install -p /path/to/envs/pol -c bioconda plmc
```

`plmc` is a binary, not an importable module, so it cannot satisfy
`selection.covariation.dca_impl` on its own.  The pipeline ships a thin adapter,
`hbvpol.selection.plmc_backend`, which is the default backend:

```yaml
selection:
  covariation:
    methods: [mutual_information, dca]
    dca_impl: hbvpol.selection.plmc_backend   # '' disables DCA
    plmc_executable: plmc
    plmc_fast: false        # plmc --fast (stochastic gradient) for large sets
    require_backend: true   # error instead of silently using the MI proxy
```

The adapter writes the *selected variable columns* (not the whole genome — plmc
estimates an O(L²) parameter set) to FASTA, runs `plmc -c couplings.txt`,
parses the `i - j - 0 score` lines into a symmetric L×L matrix, and uses
a `-ACGT` alphabet for nucleotide input.  `dca_impl` can point at any importable
module exposing `dca_scores(alignment[, config])`.

## Requiring tools (publication runs)

By default a missing tool is skipped and a pure-Python fallback is used, which
is what lets the pipeline and its tests run anywhere. A publication run must not
do that silently, so list the tools you depend on:

```yaml
project:
  # Absence becomes a hard, up-front error instead of a fallback.
  required_tools: [mafft, iqtree2, hyphy]
```

The same principle applies to optional backends: set
`selection.covariation.require_backend: true` so a missing DCA implementation is
an error rather than a silent APC-corrected-MI proxy.

Note that the suite names differ between systems: IQ-TREE 2 ships `iqtree2`,
IQ-TREE 3 ships `iqtree` (both are detected). On many clusters MAFFT, IQ-TREE,
HyPhy and CD-HIT are present on compute nodes but not on login nodes — run the
smoke test (below) to confirm what a job actually sees.

## Verification

```bash
for t in mafft iqtree2 iqtree hyphy trimal cd-hit hmmsearch gmx RDP5 3seq; do
  printf '%-12s ' "$t"; command -v "$t" || echo missing
done
```

Stage summaries (`output/<stage>/*_summary.json`) record which tools actually
ran, so a methods section can be written directly from the run.
