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

### Cross-validation hierarchy

GenBank is the **primary retrieval layer** (accession versions, annotations,
translations, checksums). Specialist resources are validation layers, never
dependencies:

1. primary retrieval — NCBI GenBank / NCBI Virus;
2. independent annotation — HBV-GLUE (local, commit-pinned);
3. resistance validation — Stanford HBVseq / HBV RT database (RT region only);
4. legacy comparison — a checksummed HBVdb snapshot, if reachable;
5. local checks — translation, motifs, genotype placement, feature coordinates.

`hbvpol.datasets.crosscheck` never blocks the pipeline: it writes
`datasets/crosscheck_status.json` carrying `hbvdb_status`, `hbv_glue_status`,
`stanford_hbvseq_status`, `crosscheck_status`, `crosscheck_date` and
`crosscheck_details`. States are `matched | conflict | not_found |
not_attempted | service_unavailable | not_applicable`, so an unreachable HBVdb
becomes `hbvdb_status = service_unavailable` and `crosscheck_status = partial` —
never "record invalid". Phase-1 success is: GenBank annotation agrees with the
local translation **and** the sequence places in the expected HBV-GLUE clade
**and** the major Pol motifs are coherent; HBVdb is supporting evidence, not a
gate.

Pin the GLUE repositories rather than tracking a moving master:

```bash
git clone https://github.com/giffordlabcvr/HBV-GLUE.git external/HBV-GLUE
git -C external/HBV-GLUE rev-parse HEAD > results/metadata/hbv_glue.commit.txt
```

If HBVdb becomes reachable, snapshot it once under `external/hbvdb_snapshot/`
with a `provenance.tsv` (retrieval date, source URL) and `checksums.sha256`;
never require live access during a run, and check licensing before
redistribution.

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

### Group queries

Retrieval is driven by per-group Entrez queries. Two pitfalls shape the
shipped defaults (`DEFAULT_GROUP_QUERIES`):

* **`[Host]` is not indexed in the NCBI protein database.**
  `Hepadnaviridae[Organism] AND Rodentia[Host]` returns **zero** even though
  rodent hepadnaviruses exist. The default queries therefore target the actual
  **virus taxa**: `Woodchuck hepatitis virus`, `Ground squirrel hepatitis
  virus`, `Bat hepatitis B virus`, `Avihepadnavirus`, and so on.
* **Nackednaviruses are not indexed as an organism.**
  `Nackednaviridae[Organism]` and `Nackednavirus[Organism]` both return zero;
  the sequences are reachable by protein title (`nackednavirus[Title]`), where
  the Pol is annotated as `P`, `ORF2` or “reverse transcriptase”. That query
  embeds its own Pol restriction so the generic suffix is not applied.

Override a group with `deephep.group_queries` (a string or a list of
alternative terms) and adjust the Pol-restriction filter with
`deephep.group_query_suffixes`. `probe_group_queries(config)` prints the hit
count per group and should be the first thing you run when a group comes back
empty — a zero is usually a wrong taxon name, not missing data.

**Amphibian, reptile and fish coverage comes from a curated manifest, not a broader taxon query.** The original gap was a *query-scope* problem: `txid10407[Organism:exp]` is human HBV and returns none of the Tibetan frog, fish meta/parahepadna-, avihepadna- or most bat viruses. Two changes fix it:

* the deep branch queries the **Hepadnaviridae family** (and the per-group virus taxa above), never the human HBV taxon;
* a version-controlled manifest, `config/deep_hepadnavirus_references.tsv`, pins the publication set (RefSeq representatives plus curated publication accessions). Database queries *discover* candidates; the manifest *controls* the dataset.

Manifest columns: `accession, display_name, genus, host_class, host_species,
sequence_origin, expected_complete, include_pol_tree, include_codon_analysis,
include_structure_analysis, source_publication, notes`. Pol is extracted per
record: the deposited CDS `translation` when present, otherwise a translation of
the CDS location (see `_pol_from_genbank`). A retrieval table is written to
`datasets/deephep_provenance.tsv`.

### Provenance categories

`sequence_origin` is one of `exogenous_complete`, `exogenous_partial`,
`assembly_derived_complete`, `assembly_derived_partial`,
`endogenous_complete_or_near_complete`, `endogenous_fragment`,
`metagenomic_unverified`. Do **not** pool these silently:

* **extant exogenous Pol** — mechanism, domains, structural conservation;
* **assembly-derived Pol** (WGS/TSA/SRA) — expanded lineage discovery, moderate confidence;
* **endogenous fragments** — deep-history evidence and motif/domain analyses only.

Endogenous elements may carry frameshifts, stop codons, host insertions,
fragmented Pol and ancient substitutions accumulated without viral replication,
so `include_codon_analysis` should be `no`/`conditional` for them. They are
integrated, not extant infectious genomes.

### Honest coverage states

Report coverage rather than forcing every vertebrate class to be equal:

| Clade | Coverage |
|---|---|
| Human and mammalian HBV | dense |
| Bat and non-human primate orthohepadnaviruses | moderate |
| Avihepadnaviruses | reference-level |
| Fish hepadnaviruses | sparse, reference + assembly-derived |
| Amphibian herpetohepadnaviruses | sparse, reference-level |
| Reptile herpetohepadnaviruses | limited / assembly-derived / unresolved |

Absence of reptile exogenous genomes is a dataset limitation, not something to
fill with simulated data. `Hepadnaviridae-GLUE`
(https://github.com/giffordlabcvr/Hepadnaviridae-GLUE) is the recommended
family-wide companion: pin it to a commit and use its reference set, alignments
and feature coordinates as the deep-dataset backbone, then add WGS/TSA/SRA
candidates from the literature separately.

**Why this resolves a different question:** the deep alignment separates
features that belong to modern human HBV from those that have survived hundreds
of millions of years. Published structural prediction places TP around RT and
suggests the protein-priming fold is conserved from mammals to fish and in fish
nackednaviruses — a claim the deep alignment exists to test residue by residue.

## Numbering and circularisation

HBV is circular and databases emit arbitrary rotations. All genomes are recut at
a common origin (`reference.origin_nt`, the EcoRI site = nt 1 in standard
convention) before any coordinate is compared. With `qc.origin_method: auto`
(default) the exact reference-origin seed is tried first (origin-anchored; the
bulk of GenBank records are rotated), then the invariant **YMDD** motif
(`hbvpol.calibrate.orient_to_reference`) for records whose origin k-mer is not
conserved or which are reverse-oriented, then the configured constant. A small
YMDD offset is arbitrated by ORF integrity, because one anchor cannot tell a
rotation from an indel upstream of the motif. Per-record `origin_method` is
recorded in the QC metadata, and `_lift` is the identity because the recut
sequence *is* the reference frame. See `hbvpol/qc/circularise.py`.

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
