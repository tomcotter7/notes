# Spark (pySpark)

## MLLib

### Pipelines

A pipeline is a specified set of stages, running both Transformers (df --> df) and Estimators (provide a `.fit()` method).

During training, you use the `.fit()` of a Pipeline, which produces a `PipelineModel`, which is actually a Transformer, so during test time, you can use the ` .transform()` method.

It is possible to create non-linear pipelines, provided that they are in the form of a DAG (the stages must be represented in topological order).

## Data Manipulation

### Joins

#### `left-anti`

The left-anti (one possible argument to the parameter `how`) says return all items in the left table that don't have a corresponding match in the right table. You should use this instead of any not in filtering as it's more performant on larger datasets.
