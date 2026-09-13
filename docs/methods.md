# Methods and design

This document explains what each stage computes, the assumptions it makes, and
where the implementation is a scaffold. It is written to double as the skeleton
of a methods section.

---

## Stage 1 — Two datasets, two questions

* **Human HBV** answers: what is constrained within modern human diversity, and
  which genotypes/subgenotypes contributed which parts of a genome?
* **Deep hepadnaviruses** answers: which features predate the human virus by
  hundreds of millions of years?

Both are needed because a residue that is invariant across genotypes but also
invariant from fish to human is a fundamentally stronger candidate than one that
is merely conserved among human isolates. See [`datasets.md`](datasets.md).

---

## Stage 2 — Recombination before trees

A single whole-genome tree is not sufficient: recombinant genomes contain
regions with different histories, and B/C recombinants in particular are common
with non-randomly distributed breakpoints.

1. **Circularise** every genome at a common origin (`hbvpol.qc.circularise`).
2. **Verify ORFs** and exclude frameshifted or heavily ambiguous records
   (`hbvpol.qc.orfcheck`).
3. **Detect breakpoints with several methods** — RDP5, GARD, 3SEQ and
   bootscanning (`hbvpol.recombination.*`). A breakpoint is accepted when a
   configurable fraction of independent tools agree (`consensus_frac`).
4. **Partition** the alignment into non-recombinant blocks
   (`hbvpol.recombination.partition`).
5. **Infer a tree per block**, and separately for the TP, spacer, RT and RNase H
   domains (`hbvpol.phylogeny`).
6. **Annotate Pol changes with their surface-frame consequence** so that a
   lineage switch in Pol is read alongside what it does to surface antigen.

> **Offline bootscan specificity.** The pure-Python scan flags a window
> whenever the best-matching partner changes, which on thousands of
> near-identical genomes fires on ties (all pairs ~99% identical) and produced
> thousands of spurious calls. `recombination.bootscan.min_score_margin`
> requires the swapped partner to *beat* the primary reference by a margin, and
> `min_region_len`/`min_support` drop short or weak regions. The real detectors
> are **OpenRDP** (a Python re-implementation of RDP4, used in place of the RDP5
> binary) and **3SEQ**; both are wired as tools and should be installed for a
> publication run.

This is what makes genuine **domain-level lineage switching** visible: an RT
region may descend from one genotype while the spacer or the overlapping surface
region has a different origin, and only a partitioned analysis can show it.

> **Scaffold note.** The pure-Python bootscan is a documented offline fallback;
> the tree falls back to Neighbor-Joining and ancestral reconstruction to Fitch
> parsimony when IQ-TREE is absent. Use the real tools for inference.

---

## Stage 3 — Constraints at nucleotide, codon, residue and structural levels

For every polymerase position the pipeline computes: amino-acid entropy,
nucleotide entropy, genotype specificity (Fst), within-host variability,
pervasive and episodic selection, covariation, epistasis, drug-resistance
association, the consequence in the overlapping surface frame, and — joined from
Stage 4 — structural confidence and solvent accessibility.

### The dual-frame codon model

The novel core of the analysis. HBV Pol and the surface antigen are translated
from **overlapping open reading frames in different reading frames**, so a single
nucleotide substitution must be interpreted twice. Each change is classified as:

| Class | Meaning |
|---|---|
| `synonymous_both` | Silent in Pol **and** surface |
| `nonsynonymous_pol_only` | Changes Pol only |
| `nonsynonymous_surface_only` | Changes surface antigen only |
| `nonsynonymous_both` | Changes both proteins |
| `disruptive_rna_element` | Hits a known/predicted regulatory RNA element |

This separates genuine protein conservation from constraints imposed by the
overlapping gene — the single most common way HBV selection scans are
misinterpreted. Implementation: `hbvpol.selection.dualframe`, with the frame
arithmetic documented in the module docstring and the reference coordinates
(configurable) in `reference.*`.

> **The spacer is not junk.** It is retained as a first-class domain. It carries
> a disproportionate number of genotype-informative sites, tolerates variation
> unevenly, and may provide conformational plasticity while carrying lineage,
> immune-escape and functional signals.

### Performance guard

Pairwise metrics are O(L²). The covariation/epistasis scan is restricted to
variable columns and capped (`selection.covariation.max_positions`, repeat for
epistasis). By default it runs on the **translated Pol protein** alignment
(`selection.protein_covariation: true`), so positions are Pol residues that match
entropy, genotype specificity and the atlas; DCA is conventionally a protein
method. Set it to `false` to scan nucleotide columns instead. For tens of
thousands of sequences, replace the APC-corrected-MI proxy with a vectorised or
external DCA backend (`dca_impl`).

All position columns in this stage are **reference-frame** coordinates, derived
from the column map (see *Reference-frame coordinates* below), so a genotype's
inserted residues are retained in domain sub-alignments but do not shift the
numbering of anything downstream.

### Per-site selection (HyPhy)

`fel`, `meme`, `fubar` and `busted` are delegated to HyPhy when it is present.
Because the genotype tree is inferred from a capped subsample, the alignment is
restricted to the tree's taxa (matched after IQ-TREE's leaf-name sanitisation)
and the tree pruned to match, since HyPhy requires identical taxa; if fewer than
two taxa overlap, the method returns a schema-correct empty table with a
warning. Reported site indices are remapped through the column map to
reference-frame Pol residues, and sites outside the Pol ORF (or on insertion
columns) are dropped. Global tests such as `busted` legitimately report no
per-site rows.

---

## Stage 4 — A structural ensemble, not one static model

There is a credible predicted full-length architecture in which TP wraps around
RT placing the priming tyrosine near the catalytic centre, with a continuous
nucleic-acid-binding groove linking the RT and RNase H sites. But it is a
prediction and parts of Pol — especially the spacer and flexible interfaces —
are likely state-dependent.

The stage therefore enumerates several **classes of model**:

* apo Pol;
* Pol bound to ε RNA;
* priming-state Pol;
* Pol bound to an RNA:DNA hybrid;
* elongation-state Pol;
* lineage-specific models for the major genotypes;
* drug-resistant and compensatory mutants.

Each class is modelled with **multiple predictors and seeds**
(`structure.strategies`, `structure.seeds`), and compared on local confidence,
predicted aligned error, domain orientation, catalytic geometry, nucleic-acid
compatibility and conserved-residue packing.

The structure is treated as an **ensemble of hypotheses**: low-confidence regions
below `structure.hinge_plddt` are retained and reclassified as candidate
**hinges** rather than discarded as prediction failures.

> **Pocket caution.** Docking scores alone are not sufficient. A recent
> computational screen found a predicted pocket to be conformationally unstable
> and its selected compounds non-inhibitory. `hbvpol.structure.md` therefore
> requires **pocket persistence** across an MD trajectory
> (`pocket_persistence_frac`) before any large-scale screening. The stage plans
> MD and analyses it with MDAnalysis, so **OpenMM (default) and GROMACS are
> interchangeable**: run the plan with `scripts/run_md_openmm.py`, which writes
> `topology.pdb` + `trajectory.dcd` under `structure/md/trajectories/` for the
> persistence pass to read.

---

## Stage 5 — ε RNA evolution coupled to polymerase evolution

Analysing Pol alone would miss a central part of its mechanism: ε is both the
encapsidation recognition element and the template for protein priming, and
recent work supports a polymerase-responsive alternative ε fold with a cryptic
stem-loop.

For each genome the stage pairs the polymerase sequence, the cognate ε
sequence, an ε structural ensemble, and a predicted Pol–ε compatibility.
Covariation between **TP residues and ε base pairs** may identify a molecular
recognition code; cross-genotype compatibility tests whether genotype-specific
Pol proteins prefer their own ε and whether apparent Pol conservation masks
compensatory evolution in RNA.

> **Scaffold note.** The ε span is resolved per genotype through
> `epsilon.spans_file` (table row → `default` row → `epsilon.genome_span` →
> built-in); the shipped table carries a documented reference span, so curate
> genotype rows to make it exact. Folding falls back to a Nussinov fold when
> RNAfold is absent, and the compatibility predictor is an uncalibrated
> contact-map-energy heuristic intended for ranking, not for quantitative
> affinities.

---

## Stage 6 — Experimental fitness as a mechanistic anchor

A 2024 deep-mutational-scanning study produced a single-nucleotide-resolution
fitness map of HBV polymerase and found evidence that ribosome pausing helps
tether nascent Pol to its own RNA, promoting cis-preferential reverse
transcription. Conservation can now be compared with **measured tolerance**
rather than interpreted alone.

The pipeline integrates that map (`fitness.dms`) and scores the criteria for a
strong mechanistic candidate: conserved across deep hepadnavirus evolution,
intolerant in DMS, positioned at a predicted interface or hinge, supported by
covariation, not explained solely by surface-frame constraint, and distinct from
the already well-characterised catalytic motifs.

> **The DMS map is not redistributed.** See [`../resources/README.md`](../resources/README.md).

---

## Mechanisms the atlas is built to surface

| Hypothesis | How the code looks for it |
|---|---|
| **Conserved TP–RT priming clamp** | Deep conservation ∧ structural TP–RT contact ∧ DMS intolerance ∧ covariation at the interface; more selective than the conserved active site. |
| **Spacer-controlled conformational switch** | Lineage-specific spacer substitutions near domain interfaces/hinges (`conformational_switches.tsv`) that alter relative TP–RT motion rather than forming a stable domain. |
| **RT–RNase H coevolution** | Covarying residue pairs along the predicted nucleic-acid channel, tuning the distance/timing between synthesis and degradation. |
| **Hidden allosteric pockets** | Conserved transient cavities at TP–RT, RT–RNase H or Pol–RNA interfaces, **only** after MD demonstrates persistence. |
| **Resistance as a trajectory** | Resistance substitutions reconstructed as paths (primary → compensatory → surface consequence) rather than independent events. |

---

## Stage 7 — The residue atlas

Everything joins on `[lineage, pol_position]` into one table
(`output/atlas/residue_atlas.parquet`), from which the ranked target list, the
conserved and lineage-specific interaction networks, the conformational-switch
candidates and a self-contained HTML report are built. The column dictionary is
in [`outputs.md`](outputs.md).

---

## Assumptions to revisit before publication

Status after wiring the reference into the pipeline:

1. **Domain boundaries** — resolved per genotype through
   `reference.domain_spans_file` (genotype row → explicit `reference.domain_spans`
   list → `default` row → built-in), clamped to `reference.pol_length_aa` unless
   the row is genotype-calibrated. Calibrated for all of A–J (Pol lengths: A 845,
   B/C/F/H/I 843, E/G 842, D/J 832; all corroborated by the GenBank CDS
   translation). Pol boundaries are protein-level, so rows are calibrated by
   alignment with `scripts/locate_pol_domains.py` (see below), not by copying
   genotype D's numbers.
2. **The surface-frame offset** — *resolved automatically.* The `reference`
   stage computes it as `(s_start − pol_start) mod 3` from the CDS annotation,
   so it is no longer assumed.
3. **Circularisation** — origin detection is on (`qc.detect_origin: true`) and
   anchored to the derived reference sequence, falling back to the configured
   constant when the seed is absent.
4. **Offline fallbacks** — set `project.required_tools: [mafft, iqtree2, hyphy]`
   to make a missing tool a hard, up-front error instead of a silent
   pure-Python substitute.
5. **Covariation capping** — the quadratic scan is still capped for
   tractability, but a real DCA backend is now wired. `pydca`/`plmDCA` is
   unmaintained, so the shipped backend is `hbvpol.selection.plmc_backend`,
   which shells out to `plmc` (bioconda). Set `require_backend: true` to make a
   missing backend an error and `max_positions: 0` to scan every variable
   column.
6. **ε coordinates** — resolved per genotype through `epsilon.spans_file`
   (table row → `default` row → `genome_span` → built-in). Calibrated for A–J,
   which all place ε at 1846–1905 in the reference frame (genotype D reproduces
   the default exactly; identity 96.7–100%), including genotype G despite its
   36-nt core-gene insertion. Rows are calibrated by alignment
   (`scripts/locate_epsilon.py`) after normalising each reference's rotation and
   strand. The Pol–ε compatibility heuristic remains uncalibrated; genotype I's
   row is lower-confidence (provisional/contested genotype).
7. **The DMS map** — supplied (`resources/hbv_pol_dms_2024.tsv`).

### Calibrating per-genotype spans (`hbvpol.calibrate`)

ε coordinates and Pol domain boundaries are single-genotype fallbacks, and
reference genomes are not interchangeable: a 6-nt indel upstream of preS1 makes
genotype A ~3221 nt versus genotype D's 3182 nt. Both elements are
sequence-conserved, so the reliable calibration is alignment, not arithmetic:
locate the element by aligning a conserved query to the genotype's own reference,
then read the coordinates off *that* sequence.  Public records carry arbitrary
rotation and sometimes the opposite strand, so before reading coordinates both
scripts normalise the reference to the reference frame using an invariant anchor
(the YMDD catalytic motif, whose codons vary synonymously).

* **Pol domains** (`scripts/locate_pol_domains.py`) — align a query Pol protein
  (default: the derived reference Pol) to the genotype's Pol with a global
  BLOSUM62 alignment and transfer the four boundaries as a gap-free partition;
  the target may be a Pol protein or a genome (coordinates supplied, or detected
  by six-frame translation of the doubled genome, which handles origin-wrapping
  ORFs). Emits/append rows to `resources/pol_domain_spans.tsv`.
* **ε** (`scripts/locate_epsilon.py`) — normalise rotation/strand with
  `hbvpol.calibrate.orient_to_reference`, then local-align the conserved ε query
  (`calibrate_epsilon_span`) and report 1-based coordinates in the reference
  frame. Emits/append rows to `resources/epsilon_spans.tsv`.

Both report alignment identity and coverage and warn below the caller's
threshold, so a wrong accession or orientation is visible rather than silently
mis-calibrated.  Genotype references are pinned in
`resources/genotype_references/` so the calibration is reproducible offline.

### Reference-frame coordinates (`hbvpol.coordinates`)

Every position the pipeline reports is in the **reference genome's numbering**
(like clinical rt numbering). That is only well-defined when alignment columns
can be mapped to reference positions, and neither raw records nor a gapped MSA
satisfy "column *i* == reference position *i + 1*".

`hbvpol.coordinates.reference_positions` aligns the configured reference genome
to a representative row of the alignment once and returns a per-column
reference position (`None` for insertions). `extract_domain_subalignments` and
the selection stage (entropy, dual-frame, genotype specificity, translated Pol
for DCA) select codons and label positions through that map, so:

* gapped alignments no longer shift downstream slices;
* a genotype insertion internal to a domain is retained in that domain's
  sub-alignment (via its nearest anchored neighbour) but has **no numbered
  position**, which is the standard reference-frame convention (e.g. rt
  numbering), rather than being excluded or mis-numbered;
* calibrated genotype rows (A 845 aa, B/C/F 843, E/G 842, …) are genotype-frame
  and are therefore *not* applied to reference-frame tables — the map handles
  genotype indels, and the reference `default` spans are the correct labels.

Selection reads the aligned MSA (`recombination/hbv_aligned.fasta`) so the map
applies; the raw oriented FASTA remains the documented fallback when no
alignment exists (e.g. the offline tests).

### Wiring the reference (`hbvpol.reference`)

Hardcoded genotype constants were the source of several of the above. They are
now derived from the annotated reference and threaded through one merge point:

```
hbvpol reference -c config/config.yaml        # writes output/reference/*
    -> reference_features.json                #   {reference: {...}, features: {...}}
    -> load_config() merges it into `reference`   (config file < derived < overrides)
    -> qc / dualframe / partition / epsilon / atlas all read the same values
```

On NC_003977.2 this corrected several shipped constants: the accession is
**genotype D (ayw), 3182 bp, 832 aa Pol** — not the "genotype A2, 3215 bp,
843 aa" the config previously claimed — with Pol 2309..1625, surface
2850..837 and frame offset 1.
