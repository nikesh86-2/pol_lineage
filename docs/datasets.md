# Datasets

The study needs two related datasets: recent human HBV diversity, and the deep
hepadnavirus record that tells us which features are ancient.

## 1. Human-HBV dataset

### Source choice and trade-offs

| Source | Role | Why |
|---|---|---|
| **NCBI GenBank** (`hbvpol.datasets.ncbi`) | Sequence-of-record | Broadest coverage of complete genomes; per-record provenance in the metadata qualifiers. Requires a valid `datasets.hbv.genbank.email`. |
| **HBVdb** (`hbvpol.datasets.hbvdb`) | Independent cross-check | Independently curated annotations, genotype assignments, protein annotations and resistance profiling. **Check `check_release_freshness`** before treating it as the principal sequence source — a stale release silently biases genotypes and annotations. |
| **HBV-GLUE** (`hbvpol.datasets.glue`) | Maintained alignments & references | Ships maintained alignments, a genotype reference set and an offline SAM/BAM implementation for read-level analyses. Located via config, `HBV_GLUE_HOME` or `PATH`. |

The pipeline prefers GenBank, falls back to HBVdb if GenBank is unavailable, and
always keeps HBV-GLUE artefacts separate rather than merging them.

### Metadata

`datasets.hbv.metadata.require` names the fields a genome must carry
(`genotype`, `country`, `year`); `desirable` fields (`subgenotype`,
`isolation_source`, `treatment_status`, `patient_id`) improve the analyses when
present. Missing values are normalised to `""`, never dropped silently — the
metadata stage reports how many records lacked each field.

`datasets.hbv.longitudinal` links records sharing a `patient_id` across sampling
dates, which is the basis for within-host variability.

### What "complete or near-complete" means here

Selection by title is deliberately permissive and then filtered hard by QC
(see below): near-complete genomes with small terminal truncations are retained,
heavily ambiguous or frameshifted records are not.

## 2. Deep-hepadnavirus / nackednavirus dataset

`hbvpol.datasets.deephep` builds a Pol-homologue alignment spanning the
taxonomic breadth requested: primate, rodent, bat, avian, reptile, amphibian and
fish hepadnaviruses, with fish **nackednaviruses** included only when a
defensible orthology can be established, plus carefully selected reverse
transcriptase outgroups (e.g. *Caulimoviridae*, Ty3/Gypsy LTR
retrotransposons).

Alignment strategy:

1. retrieve per-group sequences (`fetch_group_sequences`, pluggable);
2. filter by length (`deephep.min_pol_length`…`max_pol_length`);
3. align with MAFFT (L-INS-i) when available;
4. optionally anchor to the Pfam reverse-transcriptase profile (PF00078) via
   `hmmsearch`, which is what keeps nackednavirus and outgroup sequences in
   register where pairwise alignment is unreliable;
5. trim with trimAl.

Everything is optional: with no network the stage produces a documented,
logged placeholder rather than failing.

**Why this resolves a different question:** the deep alignment separates
features that belong to modern human HBV from those that have survived hundreds
of millions of years. Published structural prediction places TP around RT and
suggests the protein-priming fold is conserved from mammals to fish and in fish
nackednaviruses — a claim the deep alignment exists to test residue by residue.

## Numbering and circularisation

HBV is circular and databases emit arbitrary rotations. All genomes are recut at
a common origin (`reference.origin_nt`, the EcoRI site = nt 1 in standard
convention) before any coordinate is compared. Origin detection is on by default
(`qc.detect_origin: true`) and anchored to the reference sequence written by the
`reference` stage (`reference.sequence`); when the seed cannot be located it
falls back to the configured constant. Detection is exact for arbitrarily
rotated records; the fallback is exact only for standard-numbered ones. This is
documented in `hbvpol/qc/circularise.py`.

Reference-derived coordinates (see `hbvpol/reference.py` and
[`methods.md`](methods.md)) supply the Pol/S/C spans and the surface-frame offset
that QC, selection, epsilon and the atlas all consume.

## QC

`hbvpol.qc` filters on:

* `qc.min_length` (default 2800 nt) and `qc.max_length` (default 3250 nt, which
  rejects concatemers and multi-genome constructs),
* `qc.max_ambiguous_frac` (N/IUPAC fraction),
* ORF integrity for the frames in `qc.require_complete_orf` (Pol, S, C by
  default) via internal-stop counting,
* `qc.exclude_stop_in_pol`,
* dereplication at `qc.dereplicate_identity` (exact-hash, plus `cd-hit-est` for
  the approximate pass when it is installed; the pure-Python pass is bounded and
  skipped above a same-length bucket cap).

`hbv_qc_pass.tsv` records every check per genome; `hbv_qc_fail.tsv` keeps the
rejects with machine-readable `fail_reason` codes (`length`, `too_long`,
`ambiguous`, `orf_pol`/`orf_s`/`orf_c`, `stop_in_pol`) so the filter is
auditable rather than a silent black box. Only passing records are written to
`hbv_oriented.fasta` and therefore flow downstream.

## Reproducibility

Record, for each run: the GenBank query and download date, the HBVdb release
(check `check_release_freshness`), the HBV-GLUE version, all random seeds
(`project.seed`), and the tool versions reported in each stage's
`*_summary.json`.
