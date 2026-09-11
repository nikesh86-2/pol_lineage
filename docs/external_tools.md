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

### 3SEQ

3SEQ is available from its author's site. Place the binary on `PATH` (or set the
path in `recombination.threeseq`). Output parsing lives in
`hbvpol/recombination/threeseq.py`.

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

## Verification

```bash
for t in mafft iqtree2 hyphy trimal cd-hit hmmsearch gmx RDP5 3seq; do
  printf '%-12s ' "$t"; command -v "$t" || echo missing
done
```

Stage summaries (`output/<stage>/*_summary.json`) record which tools actually
ran, so a methods section can be written directly from the run.
