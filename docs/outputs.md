# Output contract

Every stage writes a fixed layout under `output/<stage>/` and returns a
`{name: path}` mapping. Stages can therefore be re-run independently, and the
atlas can be rebuilt from whatever subset of upstream files exists.

```
output/
  datasets/     hbv_genomes.fasta  hbv_metadata.tsv  deephep_pol.fasta  deephep_alignment.fasta
  qc/           hbv_oriented.fasta  hbv_qc_pass.tsv  hbv_qc_fail.tsv
  recombination/ breakpoints.tsv  blocks.tsv  block_alns/block_<id>.fasta
                domain_subalns/{TP,spacer,RT,RNaseH}.fasta  recombination_summary.json
  phylogeny/    genotype_tree.treefile  block_trees/  domain_trees/
                ancestral/ancestral_<domain>.fasta  ancestral/ancestral_states.tsv  tree_summary.json
  selection/    entropy.tsv  dual_frame.tsv  selection_sites.tsv  covariation.tsv
                epistasis.tsv  genotype_specificity.tsv  resistance.tsv  selection_summary.json
  epsilon/      epsilon_sequences.fasta  epsilon_folds/<id>.dotbracket  epsilon_folds/<id>.json
                pol_epsilon_coevolution.tsv  compatibility.tsv  epsilon_summary.json
  fitness/      dms_annotated.tsv  dms_candidates.tsv  fitness_summary.json
  structure/    models/  model_manifest.tsv  metrics.tsv  hinges.tsv
                md/md_manifest.tsv  md/pocket_persistence.tsv
  atlas/        residue_atlas.parquet  residue_atlas.tsv  ranked_targets.tsv
                conserved_interactions.tsv  conformational_switches.tsv  report.html  atlas_summary.json
```

## Key table schemas

| Table | Columns |
|---|---|
| `qc/hbv_qc_pass.tsv` | `accession, length, ambiguous_frac, pol_start, pol_end, pol_stops, orf_pol, orf_s, orf_c, qc_pass, fail_reason` |
| `recombination/breakpoints.tsv` | `recombinant_id, partner, tool, bp_start, bp_end, support, region` |
| `recombination/blocks.tsv` | `block_id, start_nt, end_nt, n_seqs, n_tools_supporting` |
| `selection/entropy.tsv` | `lineage, pol_position, domain, aa_entropy, nt_entropy, gap_frac, n_seqs` |
| `selection/dual_frame.tsv` | `lineage, nucleotide_position, ref_nt, pol_codon_position, pol_ref_aa, pol_alt_aa, surface_position, surface_ref_aa, surface_alt_aa, consequence_class, disrupts_rna` |
| `selection/selection_sites.tsv` | `lineage, pol_position, method, statistic, pvalue, significant` |
| `selection/covariation.tsv` | `lineage, position_i, position_j, method, score, pvalue` |
| `selection/epistasis.tsv` | `lineage, position_i, position_j, method, score, support` |
| `selection/genotype_specificity.tsv` | `pol_position, domain, fst, genotype_informative` |
| `selection/resistance.tsv` | `lineage, pol_position, rt_mutation, drug_class, associated, canonical_motif` |
| `epsilon/pol_epsilon_coevolution.tsv` | `pol_position, pol_residue, epsilon_feature, epsilon_position, mutual_information, n_sequences` |
| `epsilon/compatibility.tsv` | `pol_genotype, epsilon_genotype, energy, compatibility, rank` |
| `structure/model_manifest.tsv` | `model_id, kind, state, lineage, strategy, seed, variant, rel_path, sequence_source` |
| `structure/metrics.tsv` | `model, path, status, n_residues, plddt_mean, plddt_min, plddt_max, pae_mean, sasa_mean, orientation_angle_deg, n_domains, catalytic_distance, nucleic_has_chain, nucleic_contact_fraction, nucleic_min_distance, packing_correlation` |
| `structure/hinges.tsv` | `model_id, hinge_start, hinge_end, length, plddt_mean, plddt_min` |

## The residue atlas

`output/atlas/residue_atlas.parquet` (also `.tsv`) is the joined per-position
table. The join key is `[lineage, pol_position]`; tables that are finer-grained
(dual-frame has up to three nucleotide rows per codon; resistance can list two
substitutions at one residue) are collapsed to the key first, and pair-indexed
tables (covariation, epistasis) are reduced to a per-position support score.
Rows outside the annotated Pol domains are dropped unless
`atlas.include_unannotated` is set.

`lineage` takes the genotype labels found in metadata plus `all` for
pan-genotype statistics (notably Fst), which by construction are not
lineage-specific.

### Column dictionary

| Column | Meaning | Source stage |
|---|---|---|
| `lineage` | Genotype label, or `all` for pan-genotype statistics | — |
| `pol_position` | 1-based Pol amino-acid position (join key) | — |
| `domain` | `TP` / `spacer` / `RT` / `RNaseH` | domain model |
| `aa_entropy`, `nt_entropy` | Shannon entropy (bits) at codon and nucleotide level | selection |
| `gap_frac`, `n_seqs` | Alignment coverage at the position | selection |
| `nucleotide_position` | Reference nucleotide coordinate of a variable site | selection |
| `ref_nt`, `pol_ref_aa`, `pol_alt_aa` | Reference base and Pol substitution | selection |
| `surface_position`, `surface_ref_aa`, `surface_alt_aa` | The same site in the overlapping surface frame | selection |
| `consequence_class` | `synonymous_both` / `nonsynonymous_pol_only` / `nonsynonymous_surface_only` / `nonsynonymous_both` / `disruptive_rna_element` | selection |
| `disrupts_rna` | Whether the site hits a configured RNA element | selection |
| `fst`, `genotype_informative` | Between-genotype differentiation and flag | selection |
| `rt_mutation`, `drug_class`, `associated`, `canonical_motif` | Resistance annotation | selection |
| `covariation_support`, `epistasis_support` | Max pair score over the position's partners | selection |
| `conservation` | `1 − minmax(aa_entropy)` | atlas (derived) |
| `is_hinge`, `is_interface` | Low-pLDDT hinge / predicted interface flags | structure |
| `conserved_deep_hepadna` | Conserved across the deep hepadnavirus alignment | fitness criteria |
| `intolerant_dms` | Intolerant in the 2024 DMS map | fitness criteria |
| `interface_or_hinge` | Structural criterion | fitness criteria |
| `supported_covariation` | Covariation criterion | fitness criteria |
| `not_explained_by_surface_frame` | Not solely a surface-frame constraint | fitness criteria |
| `distinct_from_canonical_motifs` | Outside YMDD / motifs A–F / NJG / RNase H motif C | fitness criteria |
| `n_criteria_met`, `passes_criteria` | Count and conjunction of the enabled criteria | fitness criteria |

### Other atlas outputs

| File | Contents |
|---|---|
| `ranked_targets.tsv` | Weighted shortlist (`atlas.ranked_targets.weights`, `top_n`) with normalised component scores |
| `conserved_interactions.tsv` | Conserved and lineage-specific residue contacts (covariation + structure) |
| `conformational_switches.tsv` | Lineage-specific spacer substitutions near interfaces/hinges |
| `report.html` | Self-contained HTML report; embeds py3Dmol when `atlas.interactive` |
| `atlas_summary.json` | Row/lineage counts, criteria met, which upstream tables were present |
