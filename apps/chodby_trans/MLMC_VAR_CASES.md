# Initial variance diagnostics of the coarse-fine model

Setup the model and its configuration which has a good chance to be effective in MLMC.
Runs MC sampling with the fine-homogenization-coarse transport model, collects the QoI and run times
and evaluates the fine-coarse variance ratio, reduction ratio (time costs), correlation.

## Variance diagnostics
### Observed quantities
We compute from the QoI of the collected MC samples:
- `Var(fine)` - 
- `Var(coarse)`
- `Var(fine - coarse)`

and evaluate the time dependent value of:

`r_V = Var(coarse) / Var(fine - coarse)`

and the cost reduction ratio:

`r_C = Cost(fine)/Cost(coarse)`

Note: for significant `Cost(coarse)` we should consider:

`r_C = 1 + Cost(fine)/Cost(coarse)`

### Desired observations
For MLMC, it is desired:
- that fine and coarse outputs differ with low variance (in comparison to coarse and fine output itself), therefore:
`r_V > 1`
- coarse evaluation is cheap:
`r_C > 1`
- additionaly, the correlation between fine and coarse output is high (close to 1)


Ratios comparison:
1. `r_V>r_C`

The correction variance decreases more strongly than its cost increases.
The correction level is relatively efficient. Under optimal sample allocation, the coarse level may consume more total work than the correction level.

2. `r_V ~ r_C`

The coarse and correction levels make roughly similar contributions to the total optimal computational work.

3. `r_V < r_C`

The correction level is relatively expensive compared with the variance reduction it provides. The fine correction will tend to dominate the total MLMC cost.

**Summary:**
`r_V` should ideally be at least comparable to, and preferably larger than, `r_C`. If `r_V >> r_C`, the coarse model is doing very well: the variance reduction obtained by the coarse/fine coupling is greater than the computational penalty of going to the fine level.


# CASES

### `workdir_mlmc_var_01`
- bug in source term sigma homogenization => significantly higher source concentration
- `r_V < 1`  between years 1k and 100ky

### `workdir_mlmc_var_02`, `workdir_mlmc_var_03`
- fixed source term sigma homogenization
- `r_V > 2`  except some interval around 10ky
- `r_C ~ 1.5`
- conclusions:
  - we need much higher `r_V` since we also want to increase `r_C`
  - still suspecting the large diffusion of the source concentration in coarse term (the diff mean `Mean(fine - coarse)` is a bit lower with the fixed source sigma, but still significant, the zero diff is also shifted in time from 60ky to 20ky)