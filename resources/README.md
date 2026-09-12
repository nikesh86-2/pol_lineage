# hbvpol resources

These files are read by configuration paths in `config/config.yaml`.  They are
small, human-editable inputs that the manuscript supplements can point at.

| File | Config path | Purpose |
|---|---|---|
| `hbv_pol_resistance.tsv` | `selection.drug_resistance.catalogue` | Curated RT drug-resistance substitutions mapped to Pol positions |
| `pol_mutants.tsv` | `structure.mutants_file` | Mutants to fold as part of the structural ensemble |
| `epsilon_spans.tsv` | `epsilon.spans_file` | Per-genotype ε coordinates (`genotype, start_nt, end_nt`); calibrated for A–F with `scripts/locate_epsilon.py`; a `default` row applies when a genotype is absent |
| `pol_domain_spans.tsv` | `reference.domain_spans_file` | Per-genotype Pol domain boundaries in amino acids (`genotype, domain, start, end`); calibrated for A–F with `scripts/locate_pol_domains.py`; a `default` row applies when a genotype is absent |
| `genotype_references/` | (calibration input) | Pinned A–F genotype reference genomes (`genotype_<X>.fasta`) used to calibrate the two span tables; the NCBI accession is recorded in the `source` column |
| `hbv_pol_dms_2024.tsv` | `fitness.dms.map` | The 2024 single-nucleotide-resolution Pol deep-mutational-scanning fitness map |

## Calibrated spans (genotypes A–F)

The single-genotype fallbacks were replaced by alignment-based calibration for
genotypes A–F.  Both scripts normalise each reference to the reference frame
(origin + strand) before reporting coordinates, because public records carry
arbitrary rotation and sometimes the opposite strand.  In brief:

* **Pol domains** — a global BLOSUM62 alignment of the reference Pol transfers
  the four boundaries into the genotype's own Pol, producing a gap-free
  partition.  Genotype A adds 13 aa and B/C/F add 11 aa to the Pol length
  relative to D (845/843 vs 832 aa), while the RT and RNase H regions keep
  their canonical lengths.
* **ε** — a local nucleotide alignment of the 60-nt ε query, anchored on the
  invariant YMDD catalytic motif, places ε at **1846–1905** for every genotype
  A–F (identity 0.97–1.00; genotype D reproduces the default exactly).  The
  genotype length differences are outside the origin-to-ε interval, so the ε
  coordinate is unchanged; the risk is a *rotated or reversed record*, which the
  normalisation step removes.

Genotypes G–J have no calibrated rows: their suggested accessions are less
firmly attested (verify against HBVdb first), so they fall back to the `default`
row.  Re-run the scripts to add them.

## Numbering conventions

* `rt_mutation` uses clinical reverse-transcriptase numbering (`rtM204V`).
* `pol_position` uses Pol (genotype A2) amino-acid numbering.  The default
  offset is `rt + 346` (see `selection.drug_resistance.rt_pol_offset`), which
  places the YMDD catalytic motif at Pol 549–552.
