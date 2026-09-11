# hbvpol — recombination-aware evolution of hepatitis B virus polymerase

A reproducible pipeline for asking what is *intrinsically* constrained in HBV
polymerase and what is a recent, lineage-specific accident. It builds two
related datasets (human HBV and deep hepadnaviruses), treats recombination
before inferring trees, classifies every nucleotide change in **both** reading
frames it occupies, folds genotype-specific structural ensembles, couples
polymerase to its cognate ε RNA, and anchors the whole analysis to an
experimental deep-mutational-scanning fitness map.

The guiding idea: a residue is only a strong mechanistic candidate when
**deep conservation**, **experimental intolerance**, **structural position** and
**covariation** all agree — and when the signal cannot be explained by the
overlapping surface-antigen frame.

---

## What it produces

| Output | Where |
|---|---|
| Recombination-aware Pol phylogeny (per-block + per-domain trees) | `output/phylogeny/` |
| Domain-specific ancestral sequences | `output/phylogeny/ancestral/` |
| Dual-frame evolutionary analysis | `output/selection/dual_frame.tsv` |
| Entropy / covariation / epistasis / genotype specificity | `output/selection/` |
| ε-RNA folds, Pol–ε coevolution, cross-genotype compatibility | `output/epsilon/` |
| DMS-integrated candidate residues | `output/fitness/` |
| Genotype-specific structural ensembles + hinges | `output/structure/` |
| **Interactive residue atlas** and ranked testable targets | `output/atlas/` |

---

## Quickstart

### 1. Environment

```bash
module load miniforge
mamba env create -f environment.yml        # or: mamba env update -n pol -f environment.yml
conda activate pol
pip install -e .

mamba env update -n pol -f environment-tools.yml   # mafft, iqtree, hyphy, trimal, ...
```

The Python stack runs everything offline; external tools (MAFFT, IQ-TREE, HyPhy,
RDP5, 3SEQ, GROMACS) are optional and degrade gracefully when absent. See
[`docs/external_tools.md`](docs/external_tools.md).

### 2. Prove the installation with synthetic data (no network, no tools)

```bash
python scripts/run_synthetic.py --outdir output-synthetic --genomes 8
```

This fabricates a tiny, structurally faithful dataset and runs every stage,
writing a joined residue atlas and an HTML report. It is the fastest way to see
the expected output tree.

### 3. Run on real data

```bash
# edit config/config.yaml — at minimum set datasets.hbv.genbank.email
hbvpol fetch     -c config/config.yaml      # human-HBV genomes + metadata
hbvpol deephep   -c config/config.yaml      # deep hepadnavirus / nackednavirus Pol
hbvpol qc        -c config/config.yaml      # circularise + ORF integrity
hbvpol recombine -c config/config.yaml      # RDP5 / GARD / 3SEQ / bootscan -> blocks
hbvpol tree      -c config/config.yaml      # per-block and per-domain trees
hbvpol select    -c config/config.yaml      # dual-frame + selection + covariation
hbvpol epsilon   -c config/config.yaml      # ε folding + Pol:ε coevolution
hbvpol fitness   -c config/config.yaml      # integrate the 2024 DMS map
hbvpol structure -c config/config.yaml      # structural ensemble (needs predictors)
hbvpol atlas     -c config/config.yaml      # join everything into the atlas

# or all at once / via the workflow
hbvpol all       -c config/config.yaml
snakemake -s workflow/Snakefile -j 8
```

Any config value can be overridden on the command line with dotted keys, e.g.
`hbvpol select selection.covariation.max_positions=800`.

---

## How the study maps onto the code

| Proposal step | Package | Notes |
|---|---|---|
| 1. Two evolutionary datasets | `hbvpol.datasets` | NCBI GenBank is the sequence-of-record; HBVdb is an independent cross-check; HBV-GLUE supplies maintained alignments. `deephep.py` builds the hepadnavirus/nackednavirus Pol alignment. |
| 2. Recombination before trees | `hbvpol.qc`, `hbvpol.recombination` | Circularise at the reference origin, verify ORFs, detect breakpoints with four methods, partition into non-recombinant blocks, infer per-block trees. |
| 3. Constraints at every level | `hbvpol.selection` | Entropy, the **dual-frame codon model**, covariation/DCA, epistasis, genotype Fst, resistance annotation. |
| 4. A structural ensemble | `hbvpol.structure` | Multiple states × lineages × predictors × seeds; pLDDT, PAE, domain orientation, catalytic geometry, hinges, MD pocket persistence. |
| 5. ε RNA + Pol coevolution | `hbvpol.epsilun` | Extract and fold ε, TP-residue × ε-feature covariation, cross-genotype Pol:ε compatibility. |
| 6. Experimental anchor | `hbvpol.fitness`, `hbvpol.atlas` | Join the 2024 single-nucleotide DMS map, score the six mechanistic criteria, rank candidate targets. |
| Outputs | `hbvpol.atlas` | Interactive residue atlas, conserved/lineage-specific interaction networks, predicted conformational switches, ranked target list. |

---

## Repository layout

```
config/config.yaml        master configuration (every stage reads from here)
environment.yml           Python analysis stack
environment-tools.yml     optional bioconda CLI tools
resources/                resistance catalogue, mutants, DMS schema
src/hbvpol/
  config.py domain.py io/ pipeline.py synthetic.py cli.py
  datasets/ qc/ recombination/ phylogeny/ selection/
  structure/ epsilun/ fitness/ atlas/
scripts/                  run_synthetic.py, profile_stages.py
workflow/Snakefile        Snakemake orchestration
tests/                    121 offline tests incl. full end-to-end
docs/                     methods, datasets, external tools, output contract
```

---

## Configuration

`config/config.yaml` is the single source of truth. Highlights:

* `reference.*` — genotype A2 coordinates (Pol 2307→1623 wrapping the EcoRI
  origin at nt 1), domain spans and the surface-frame offset.
* `qc.*` — circularisation, ORF integrity, dereplication thresholds.
* `recombination.*` — tool selection, consensus fraction, B/C hotspot handling.
* `selection.*` — domain partition, dual-frame model, covariation/epistasis
  performance guards.
* `structure.*` — states, lineages, predictors, seeds, MD and hinge thresholds.
* `epsilun.*` — ε span/folding, covarying features, compatibility predictor.
* `fitness.*` — DMS map path and the six candidacy criteria.
* `atlas.*` — join key, ranking weights, top-N targets.

---

## Output contract

Every stage writes under `output/<stage>/` in a fixed layout, so stages can be
re-run independently and the atlas can be rebuilt from partial results. The
full schema — including an atlas column dictionary — is in
[`docs/outputs.md`](docs/outputs.md).

---

## Testing

```bash
pytest                 # 121 tests, fully offline
pytest tests/test_end_to_end.py -q   # full stage chain on synthetic data
```

The end-to-end test fabricates a dataset and runs QC → recombination →
phylogeny → selection → ε → fitness → structure → atlas, asserting the
inter-stage contract and that the atlas is one row per (lineage, position).

---

## What is scaffold vs production

This repository is a working framework, not a finished analysis. Deliberate,
documented simplifications you should replace for a publication run:

* **Circularisation** assumes reference numbering unless `qc.detect_origin` is
  enabled; enable origin detection (or provide the reference) for arbitrarily
  rotated records.
* **Domain boundaries** are the approximate genotype-A2 spans in
  `DEFAULT_DOMAIN_SPANS`; refine them per genotype by lifting the reference
  annotation onto the curated alignment.
* **ε coordinates** (`epsilun.genome_span`) are approximate and origin-aware.
* **Offline fallbacks** (Neighbor-Joining, Fitch parsimony, Nussinov folding,
  pure-Python bootscan, APC-corrected MI for DCA) exist so the pipeline is
  testable with no external tools. Install the real tools for analysis; use the
  fallbacks only for wiring tests.
* **Covariation** scans only variable columns and caps them
  (`selection.covariation.max_positions`); for tens of thousands of sequences
  use a vectorised or external DCA backend.
* **The DMS map is not redistributed.** Supply
  `resources/hbv_pol_dms_2024.tsv` (see `resources/README.md`).

These are called out again, with the reasoning, in
[`docs/methods.md`](docs/methods.md).

---

## License

MIT (see `pyproject.toml`). Sequence data remain subject to their source
databases' terms.
