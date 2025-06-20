# Reinforcement Learning in LLMs

## The Suprising Effect of Negative Reinforecement in LLM Reasoning

This [paper](https://arxiv.org/abs/2506.01347) details an interesting phenomenom. They found that when training a model using Reinforcement Learning w/ Verifiable Rewards it's quite efficient / powerful to only train the model on negative samples. Exclusively training on negative samples improves performance across the entire `pass@k` spectrum (sampling answers `k` times, and seeing if the right answer is obtained). It also improves on GRPO/DPO based reinforcement learning slightly.

It is essentially keeping the entropy of the model high, whilst still pushing it away from poor reasoning steps. The model is not forced down the specific steps that are detailed in the positive samples.

