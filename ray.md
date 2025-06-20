# ray

`ray` is a useful package for performing batch inference of LLMs at scale (among other things). [This](https://docs.vllm.ai/en/v0.8.1/getting_started/examples/distributed.html) is there example for doing offline batch inference at scale.

## ray Data

Ray Data has two main concepts: `Datasets` and `Blocks`. Dataset is the main user facing API, whereas a BLock is a set of rows representing a single partition of the Dataset. Processing of a ray dataset is typically parallelized / distributed at the block level.

Ray acts like spark in the sense that it is completely lazy until it actually needs to be executed. When it does, it creates two sets of plans, a logical one and a physical one. The logical plan is optimized, and then translated into a physical plan.
