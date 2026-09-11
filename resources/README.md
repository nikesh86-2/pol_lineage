# hbvpol resources

These files are read by configuration paths in `config/config.yaml`.  They are
small, human-editable inputs that the manuscript supplements can point at.

| File | Config path | Purpose |
|---|---|---|
| `hbv_pol_resistance.tsv` | `selection.drug_resistance.catalogue` | Curated RT drug-resistance substitutions mapped to Pol positions |
| `pol_mutants.tsv` | `structure.mutants_file` | Mutants to fold as part of the structural ensemble |
| `hbv_pol_dms_2024.tsv` | `fitness.dms.map` | The 2024 single-nucleotide-resolution Pol deep-mutational-scanning fitness map — **you must supply this** |

## Supplying the DMS map

The 2024 deep-mutational-scanning fitness map is not redistributed here.  Place
it at `resources/hbv_pol_dms_2024.tsv` as a tidy TSV with the columns:

```
wt_nt    position    mut_nt    fitness
A        1           C         0.98
```

`position` is the 1-based reference nucleotide coordinate; `fitness` is the
measured relative fitness/intolerance.  A schema-only example is provided as
`hbv_pol_dms_2024.example.tsv`.  Until the real file exists, the fitness stage
logs a loud warning and emits empty-but-correctly-schemed outputs.

## Numbering conventions

* `rt_mutation` uses clinical reverse-transcriptase numbering (`rtM204V`).
* `pol_position` uses Pol (genotype A2) amino-acid numbering.  The default
  offset is `rt + 346` (see `selection.drug_resistance.rt_pol_offset`), which
  places the YMDD catalytic motif at Pol 549–552.
