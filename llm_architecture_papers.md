# GenAI Architectures & Papers

## Generative AI Papers on Models/Architectures

### Scalable MatMul-free Language Modelling

[This paper](https://arxiv.org/pdf/2406.02528) details how MatMul operations (the expensive ops that require GPUs) can be eliminated from language models entirely.

However, the title is slightly misleading. Essentially, all the weights in the model have been constrained to be ternary - i.e in the set { -1, 0, 1 }. By doing this, it simplifies any equation that looks like $ y = xW_{i} $ (i.e a vector-matrix multiplication) into a element wise product (hadamard product). They also remove the attention layer, and replace it with something similar to a RNN that can be parallelized.

### Mamba2

[Tri Dao's Blog on the Mamba2 Release](https://tridao.me/blog/2024/mamba2-part1-model/)
The Mamba2 paper tries to combine the efficiency of Attention with the original State Space Model (SSM) - Mamba1. The SSM defines a map from $x \in R^{T} -> y \in R^{T}$. 


### DeepSeek R1

[Report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

This report details how people at DeepSeek built their reasoning model, which outperforms OpenAIs o1 at time of release.

Esssentially, how they did it was to use Group Relative Policy Optimization (GRPO), a RL technique that allows models to be 'RL'd' without accessing supervised data, it all depends on the reward model.

Their reward model was set up such that the R1 model was guided towards producing long 'chain-of-thought' like responses (as well as correct results). Since the two go hand-in-hand, the model now uses significantly more test-time compute and answers a significantly higher number of question correctly.

The rewards focused on:
- Accuracy Rewards (correctness)
- Format Rewards (forcing the model to use <think></think> tags)
- Language consistency reward (keeping the thinking output legible)

They initially created R1-Zero, DeepSeekv3 trained exclusively via GRPO. Whilst it performed well, they found that it struggled to respond in a readable way, so combined this process with some additional Supervised Finetuning (both before and after the RL) to create DeepSeek-R1

### DeepSeek Math

[Paper](https://arxiv.org/pdf/2402.03300)

DeepSeek's paper introduces GRPO (Grouped-Policy Optimization), a PPO variant that achieves remarkable efficiency in reinforcement learning for mathematical reasoning. Their 7B parameter model demonstrates superior performance compared to 72B parameter alternatives, primarily through two key innovations: a novel training approach and meticulously curated training data.

The team developed the DeepSeek Math Corpus, comprising 120B high-quality mathematical tokens. Their data collection strategy involved:
- Training a fastText model on OpenWebMath to identify similar content
- Implementing URL-based filtering, particularly targeting known mathematical domains (e.g., mathoverflow.net)
- Manual annotation of mathematical subdomains for precision

The Supervised Fine-Tuning (SFT) phase focused on mathematical reasoning, incorporating:
- Chain-of-thought demonstrations
- Program-of-thought examples
- Tool usage patterns

GRPO differentiates itself from traditional PPO by eliminating the value function typically used for reward calculation, reducing model complexity while maintaining effectiveness. It does so because it samples a group (hence the G) of outputs from the model in order to produce a reward.

Their training pipeline implements three major techniques:
- Outcome supervision (output-based rewards)
- Process supervision (step-wise reasoning rewards)
- Iterative RL (continuous reward model refinement)

Two significant discoveries emerged from the paper:

- Code token pre-training significantly outperforms general token pre-training for mathematical reasoning
- ArXiv papers proved surprisingly ineffective for improving mathematical reasoning capabilities

### DeepSeek MoE

[Paper](https://arxiv.org/pdf/2401.06066)

In this paper, DeepSeek introduce their variant on MoE (Mixture-of-Expert) models. They found that previous implementations, which activate the top-K out of N experts, face challenges in ensuring that each expert acquires non-overlapping and focused knowledge.

They mainly introduce two strategies: "finely segmenting the experts into $mN$ ones and activating $mK$ from them, allowing for a more flexible combination of activated experts" and "isolating $K$ experts as shared ones, aiming at capturing common knowledge".

Conventional MoE architectures swap the FFNs in a Tranformer with MoE layers, each of which contains multiple experts, each structurally identical to a standard FFN. DeepSeek improve on this by (a) segmenting the experts into a finer grain, the FFN intermediate hidden dimension. They also (b) isolate certain experts to serve as shared experts that are always activated. By compressing common knowledge into these shared experts, redunancy amount other routed experts will be mitigated.

(a) is achieved by segmenting each expert FFN into $m$ smaller experts by reducing the FFN intermediate dimension to $\frac{1}{m}$ its original size. Since each expert is now smaller, they also increase to the number of activated experts $m$ to keep the same computation cost. In short, they just use smaller experts.

(b) is achieved by always assigning the token to certain experts, regardless of the router module. To achieve a constant computational cost, the nubmer of activated experts among other experts will be reduced by the number of 'shared experts'.

They add a number of losses to prevent 'routing collapse' (aka when the model always selects the same experts). These are:
- Expert Level Balance Loss (minimizing the correlation between how often an expert is selected (fi) and its average routing probability (Pi))
- Device Level Balance Loss (Ensuring balanced computation across the devices upon which the expert 'group' is stored).


### Hermes 3

[Technical Report](https://nousresearch.com/wp-content/uploads/2024/08/Hermes-3-Technical-Report.pdf)

This is essentially a finetune on top of Llama3.1

The main point of this paper was the quality of dataset that they used. They collected & cleaned the data for 5 months before finetuning the model.

The model is finetuned on Llama3.1, and outpeforms the instruction-tuned version of Llama on some tasks. The authors also note the importance of adhereing to the system prompt, and not training on additional "guardrails" within the data. They suggest that the applications built around the model should be responsible for this.

### NemoTron-4 340B

[Paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_4_340B_8T_0.pdf).

This is the Nvidia model with 340B parameters. Notably, over 98% of the data used in the model alignment process is synthetically generated. Nvidia have released Nemotron-4-340B-Reward for helping label synthetically generated data (i.e. curate it to be high quality).

To generate the synthetic data, they had to create a prompt. They had 3 types of prompts:

- Single-Turn Prompts - they basically took a bunch of topics (i.e. ML, Geography, etc..) and generated data given a either a topic or a topic and some data (i.e summarize this paper).
- Instruction Following Prompts - e.g "Write an essay, your response should have three paragraphs". Specifically, the for each prompt ("Write an essay"), they randomly selected an instruction ("your response should have three paragraphs") from the verifiable prompt templates.
- Two-Turn Prompts - These are prompts that require the model to generate a response to a question, and then generate a response to a follow-up question. The follow-up question is generated based on the response to the first question. This is of the form "User: XXX, Assistant: XXX, User: XXX", where the XXX would be synthetic data.

They also used LLM as a Judge to compare the responses to the prompt. One cool trick they used to avoid positiional bias was to ask the LLM twice, swapping the order of the responses. If the LLM gave the same response twice, it was considered to be a good response.

They managed to show a "self-improving" flywheel of data generation, where each model would generate data, train a new model, ggenerate data, train a new model, etc...

### Gorilla 

[Gorilla](https://arxiv.org/pdf/2305.15334) is a model specifically designed to interact with APIs - i.e a model designed to be an agent. Gorilla is a finetuned 7B Llama model, specifically over APIs for TorchHub, TensorHub & HuggingFace. They generated 16,450 (instruction, API) pairs, these instructions were generated by GPT-4.

They used **Retrieve-Aware training**, which means finetuning the model with the same text that is appended to the input during inference (w/ RAG).

