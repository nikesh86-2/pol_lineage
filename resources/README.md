# hbvpol resources

These files are read by configuration paths in `config/config.yaml`.  They are
small, human-editable inputs that the manuscript supplements can point at.

| File | Config path | Purpose |
|---|---|---|
| `hbv_pol_resistance.tsv` | `selection.drug_resistance.catalogue` | Curated RT drug-resistance substitutions mapped to Pol positions |
| `pol_mutants.tsv` | `structure.mutants_file` | Mutants to fold as part of the structural ensemble |
| `epsilon_spans.tsv` | `epsilon.spans_file` | Per-genotype ε coordinates (`genotype, start_nt, end_nt`); calibrated for A–J with `scripts/locate_epsilon.py`; a `default` row applies when a genotype is absent |
| `pol_domain_spans.tsv` | `reference.domain_spans_file` | Per-genotype Pol domain boundaries in amino acids (`genotype, domain, start, end`); calibrated for A–J with `scripts/locate_pol_domains.py`; a `default` row applies when a genotype is absent |
| `genotype_references/` | (calibration input) | Pinned A–J genotype reference genomes (`genotype_<X>.fasta`) used to calibrate the two span tables; the NCBI accession is recorded in the `source` column |
| `hbv_pol_dms_2024.tsv` | `fitness.dms.map` | The 2024 single-nucleotide-resolution Pol deep-mutational-scanning fitness map |

## Calibrated spans (genotypes A–J)

The single-genotype fallbacks were replaced by alignment-based calibration for
genotypes A–J, using the pinned reference genomes in `genotype_references/`.
Both scripts normalise each reference to the reference frame (origin + strand)
before reporting coordinates, because public records carry arbitrary rotation and
sometimes the opposite strand.  In brief:

* **Pol domains** — a global BLOSUM62 alignment of the reference Pol transfers
the four boundaries into the genotype's own Pol as a gap-free partition.  Pol
lengths are A 845, B/C/F/H/I 843, E/G 842, and D/J 832, and every one was
corroborated by the GenBank CDS translation length.  Genotype G is 3248 nt (not
in the usual 3182–3221 range) because of a 36-nt insertion in the *core* gene;
that insertion lies outside the Pol ORF, so G's Pol is 842 aa and the partition
needed no special handling.
* **ε** — a local nucleotide alignment of the 60-nt ε query, anchored on the
invariant YMDD catalytic motif, places ε at **1846–1905** for every genotype
A–J (identity 96.7–100%; genotype D reproduces the default exactly).  Even G's
core-gene insertion lies outside the origin-to-ε interval, so ε is unchanged;
the real hazard is a *rotated or reversed record*, which the normalisation step
removes.

### Provenance caveats

* **I** — genotype I is provisional/contested: proposed by Tran et al. 2008 (a
single isolate) and rejected by Kurbanov et al. 2008 (only ~7% divergence from
C, and a weak recombination signal).  It is understood to be an A/C/G
recombinant lineage, not a clean clade.  Row uses `EU833891` for consistency
with the industry reference panel, and its `source` records the caveat; treat
I coordinates as lower-confidence than A–H.
* **J** — the sole known J isolate (Tatematsu et al. 2009, JRB34); `AB486012`
is 3182 nt with Pol 2307→1623 (832 aa), coincidentally the same Pol length as D
despite ~10–15% sequence divergence.
* **G/H** — type-strain accessions from the original descriptions (Stuyver
et al. 2000; Arauz-Ruiz et al. 2002).

## Numbering conventions

* `rt_mutation` uses clinical reverse-transcriptase numbering (`rtM204V`).
* `pol_position` uses Pol (genotype A2) amino-acid numbering.  The default
  offset is `rt + 346` (see `selection.drug_resistance.rt_pol_offset`), which
  places the YMDD catalytic motif at Pol 549–552.
