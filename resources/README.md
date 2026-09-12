# hbvpol resources

These files are read by configuration paths in `config/config.yaml`.  They are
small, human-editable inputs that the manuscript supplements can point at.

| File | Config path | Purpose |
|---|---|---|
| `hbv_pol_resistance.tsv` | `selection.drug_resistance.catalogue` | Curated RT drug-resistance substitutions mapped to Pol positions |
| `pol_mutants.tsv` | `structure.mutants_file` | Mutants to fold as part of the structural ensemble |
| `hbv_pol_dms_2024.tsv` | `fitness.dms.map` | The 2024 single-nucleotide-resolution Pol deep-mutational-scanning fitness map

## Numbering conventions

* `rt_mutation` uses clinical reverse-transcriptase numbering (`rtM204V`).
* `pol_position` uses Pol (genotype A2) amino-acid numbering.  The default
  offset is `rt + 346` (see `selection.drug_resistance.rt_pol_offset`), which
  places the YMDD catalytic motif at Pol 549–552.
