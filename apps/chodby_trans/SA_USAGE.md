# Sobol Sensitivity Analysis Usage

This note is intended as context for integrating MLMC Sobol-index support into
an external sensitivity-analysis project. It describes the current API and the
data shapes expected by the implementation.

## Main Modules

- `mlmc.sim.saltelli_simulation.SaltelliSchemaSimulation`
  wraps a forward `Simulation` so one MLMC sample evaluates a complete
  Saltelli row.
- `mlmc.quantity.sobol.SaltelliSchema`
  defines the Saltelli term order and constructs mixed `A`/`B` parameter rows.
- `mlmc.quantity.sobol.estimate_sobol_indices`
  builds lazy `Quantity` expressions and estimates first-order, total-order,
  and second-order Sobol indices from stored MLMC samples.

## Saltelli Row Layout

For `D` uncertain parameters, each MLMC sample contains
`2 * (D + 1)` forward-model evaluations in this order:

```text
A, AB_0, AB_1, ..., AB_{D-1}, BA_0, BA_1, ..., BA_{D-1}, B
```

Use the schema rather than hard-coded indices:

```python
from mlmc.quantity.sobol import SaltelliSchema

schema = SaltelliSchema.make(n_parameters=D)

schema.a      # index of A
schema.ab     # ndarray of AB_i indices
schema.ba     # ndarray of BA_i indices
schema.b      # index of B
schema.n_terms
```

`schema.terms(a_row, b_row)` takes two parameter vectors with shape `(D,)` and
returns a Saltelli input matrix with shape `(schema.n_terms, D)`.

## Forward Simulation Contract

The wrapped forward simulation must be an ordinary MLMC `Simulation`:

```python
class ForwardSimulation(Simulation):
    def level_instance(self, fine_level_params, coarse_level_params):
        ...

    def result_format(self):
        return [
            QuantitySpec(
                name="value",
                unit="1",
                shape=(output_size,),
                times=[0],
                locations=["0"],
            )
        ]

    def calculate(self, config_dict, input_vector):
        # input_vector has shape (D,)
        # return fine_result, coarse_result
        ...
```

`SaltelliSchemaSimulation` calls `forward_simulation.calculate(...)` once for
each Saltelli term. It flattens each forward output and stores the result with
a leading Saltelli term axis.

For a wrapped output spec with `shape=s`, the Saltelli wrapper exposes
`shape=(schema.n_terms, *s)`.

## Matrix Generator Contract

`SaltelliSchemaSimulation` needs a callable:

```python
def matrix_generator(n_rows: int, n_parameters: int) -> np.ndarray:
    ...
```

It is called twice for each scheduled batch: once for matrix `A` and once for
matrix `B`. It must return a numeric array with shape
`(n_rows, n_parameters)` and values in `[0, 1]`.

This generator is the integration point for the SA project parameter sampling.
If physical parameter transforms are needed, either:

- generate already-transformed values and make the forward model interpret
  them directly; or
- generate unit-cube values here and transform inside the forward simulation.

Use one convention consistently. The Sobol implementation itself only assumes
independent columns and compatible `A`/`B` rows.

## Running Samples

Typical local/HDF workflow:

```python
import numpy as np

from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.sampler import Sampler
from mlmc.sampling_pool import OneProcessPool
from mlmc.sim.saltelli_simulation import SaltelliSchemaSimulation

D = 3
level_parameters = [[0.1], [0.05], [0.025]]

def matrix_generator(n_rows, n_parameters):
    return np.random.random((n_rows, n_parameters))

simulation = SaltelliSchemaSimulation(
    forward_simulation=ForwardSimulation(...),
    matrix_generator=matrix_generator,
    n_parameters=D,
)

storage = SampleStorageHDF(file_path="mlmc_sa.hdf5")
sampler = Sampler(
    sample_storage=storage,
    sampling_pool=OneProcessPool(),
    sim_factory=simulation,
    level_parameters=level_parameters,
)

sampler.set_initial_n_samples([1000, 100])
sampler.schedule_samples()
sampler.ask_sampling_pool_for_samples()
```

For production integrations, replace `OneProcessPool` with the appropriate
MLMC sampling pool. The scheduled sample input is the full Saltelli row matrix,
so failed-sample renewal reuses the original `A`/`B` rows instead of generating
new rows.

## Building The Quantity For Sobol Estimation

After sampling, load the stored result format and build a root `Quantity`:

```python
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.sobol import estimate_sobol_indices

result_format = storage.load_result_format()
root_quantity = make_root_quantity(storage, result_format)
```

Select one scalar/vector output from the quantity tree. For a result spec:

```python
QuantitySpec(name="value", unit="1", shape=(output_size,), times=[0], locations=["0"])
```

the first time/location quantity is:

```python
saltelli_quantity = root_quantity["value"][0]["0"]
```

Pass that Saltelli-axis quantity and the schema from the simulation:

```python
estimate = estimate_sobol_indices(saltelli_quantity, simulation.schema)
```

`estimate_sobol_indices(...)` validates that the selected quantity has a
leading Saltelli axis with `schema.n_terms` entries. It currently constructs
`AB_i` and `BA_i` groups with `Quantity.QArray`; avoid replacing that with
advanced indexing unless the Quantity shape behavior is fixed and tested.

## Reading Results

The returned `SobolIndexEstimate` exposes cached MLMC estimates:

```python
estimate.mean_mlmc
estimate.denominator_mlmc
estimate.first_order_numerator_mlmc
estimate.total_order_numerator_mlmc
estimate.second_order_numerator_mlmc
```

Final Sobol indices:

```python
estimate.first_order      # ndarray, shape (D, ...)
estimate.total_order      # ndarray, shape (D, ...)
estimate.second_order     # dict: (i, j) -> value for i < j
```

Standard-deviation diagnostics:

```python
estimate.denominator_std
estimate.first_order_numerator_std
estimate.total_order_numerator_std
estimate.second_order_numerator_std
estimate.first_order_std
estimate.total_order_std
estimate.second_order_std
```

Level variance diagnostics are available on the cached MLMC mean objects,
for example:

```python
estimate.first_order_numerator_mlmc.l_vars
estimate.total_order_numerator_mlmc.l_vars
estimate.second_order_numerator_mlmc.l_vars
estimate.denominator_mlmc.l_vars
```

## Formula Conventions

The implementation estimates:

- first-order numerators with Saltelli/SALib-compatible
  `E[B * (AB_i - A)]`;
- total-order numerators with Jansen
  `0.5 * E[(AB_i - A) ** 2]`;
- second-order numerators with the SALib-compatible pair formula for `i < j`;
- the denominator as the variance estimate based on `A` and `B`.

Indices are formed as ratios of MLMC-estimated numerator means and denominator
mean. The per-sample `Quantity` expressions are not divided by the denominator.

## Minimal In-Memory Check

For unit tests or prototyping without the sampler:

```python
from mlmc.quantity.quantity import make_root_quantity
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.quantity.sobol import SaltelliSchema, estimate_sobol_indices
from mlmc.sample_storage import Memory

schema = SaltelliSchema.make(n_parameters=D)
term_values = ...  # shape (n_samples, schema.n_terms)

storage = Memory()
result_format = [
    QuantitySpec(name="value", unit="1", shape=(schema.n_terms,), times=[0], locations=["0"])
]
storage.save_global_data(result_format=result_format, level_parameters=[[1.0]])
storage.save_samples({
    0: [
        ("L00_S{:07d}".format(i_sample), (values, np.zeros_like(values)))
        for i_sample, values in enumerate(term_values)
    ]
}, {})

root_quantity = make_root_quantity(storage, result_format)
saltelli_quantity = root_quantity["value"][0]["0"]
estimate = estimate_sobol_indices(saltelli_quantity, schema)
```

See `test/test_sobol_quantity.py` for a complete analytic two-parameter
interaction-model check against SALib formulas.

## Integration Notes For Another Codex Instance

- Start from `SaltelliSchemaSimulation` if the SA project can provide an MLMC
  `Simulation` wrapper around its forward model.
- Keep one scheduled MLMC sample equal to one complete Saltelli row.
- Do not generate new random parameter rows inside worker-side
  `calculate(...)`; rows are planned on the master and passed as `sample_input`.
- Preserve the leading Saltelli result axis through storage and quantity
  selection.
- Use `estimate_sobol_indices(...)` only after selecting the output quantity
  whose first axis is the Saltelli term axis.
- For multi-output models, estimate each selected output quantity separately.
- Targeted regression tests for this feature are:

```bash
.tox/py312/bin/python -m pytest -c test/pytest.ini \
  test/test_saltelli_simulation.py test/test_sobol_quantity.py -vv
```
