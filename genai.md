# Generative AI

## Prompt Engineering

### Prompting Techniques

A [paper](https://arxiv.org/pdf/2312.16171v1.pdf) with 26 different prompting techniques has been released. They focus on strategies that help LLMs produce better outputs.

### DSPy

[DSPy](https://huggingface.co/papers/2310.03714)

The authors argue that prompt engineering is brittle, and akin to manually finetuning weights on a classifier. This abstracts LM pipelines as *text transformation graphs*. These can be automatically optimised for better results. From the paper "DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.". Prompts go from 'You are an AI assistant ....' to 'question -> answer'. Teleprompters are powerful optimizers (included in DSPy) that can learn to bootstrap and select effective prompts for the modules of any program. (The "tele-" in the name means "at a distance", i.e., automatic prompting at a distance.)

### F*ck you, show me the prompt

Working with LLM frameworks is annoying & the added complexity of not knowing the prompt or how many API calls are being made is annoying. [This blog post](https://hamel.dev/blog/posts/prompt/) details a library `mitmproxy` that can be used to intercept the API clalls, so you can determine exactly what is going on under the hood. The blog post also describes how current frameworks implement the "magic of LLMs". These include: `guardrails`, `guidance`, `langchain`, `instructor` & `dspy`.


## LLM Concepts

### Attention

#### Attention & Transformers - Explained

[Attention - Explained](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
Seq-2-Seq models were the first to use Attention. In a typical Seq-2-Seq, RNNs are used to encode the original input sentence into a context vector. However, since this context vector is a 'pipeline' in a sense, it struggles with long contexts.
In the decoder stage, the RNNs used Attention. The Encoder would pass in all N hidden states (depending on the number of tokens). The Decoder would give each hidden state a softmaxed score, multiply it with the hidden state, essentially computing attention on the Encoder hidden states. This context vector can be combined with the hidden state of the decoder, which is passed into a feed-forward NN to output a word.
[Transformers - Explained](https://jalammar.github.io/illustrated-transformer/)
Language Models use a Decoder-Only Transformer architecture. These contain a feed-forward neural (FFN) network to determine the the next word. However, FFN can't look at words in context. So we need a self-attention layer before this.
[Self Attention - Explained](https://www.youtube.com/@SebastianRaschka)

#### Basic Self Attention

Taken from Sebastian Raschka's lecture series. Self-attention is the attention used in transformers. A very basic version can be described as follows:
You have an input sequence -> a sequence of word embedding vectors. For each input (i), we compute the dot product to every other input (j). This produces a scalar value. We then apply the softmax function to these scalars, so that they sum up to one. This gives us $a_{ij}$.
We then multiply $a_{ij}$ with each $x_j$ (the input) and sum these vectors to produce $A_i$, which is a word embedding for the input $x_i$, however is context-aware.
This is done for each $i$, to produce a matrix $A_{ij}$. The equation for $A_i$ is: $A_i = \Sigma_{j=1}^{T} \space a_{ij}x_{j}$.

#### Scaled Dot-Product Attention

Sebastian also goes through the self-attention introduced in *Attention Is All You Need*. The basic self-attention has no learnable parameters.
We introduce three trainable weight matrices that are multiplied with the input sequence embeddings $x_i$'s. These are query, key, value ($q,k,v$).
The attention values are still computed as a dot product, however we do not use $x_i$ and $x_j$, we instead use $q_i$ and $k_j$.
The equation for the context vector of the second word ($x_2$) is: $A(q_2, K, V) = \Sigma_{i=1}^{T}[\frac{exp(q_2 \cdot k_i^T)}{\Sigma_j \space exp(q_2 \cdot k_j^T)} \cdot v_i]$
This is just a weighted sum, the values are weighted by the attention weight (which has been softmaxed).
This is done for each word, which can obviously be performed in paralell. After this you obtain an attention score matrix.

#### Multi-Head Attention

This scaled dot product attetention previously mentioned is 1-head attention. Multi-head attention is just this with different $q, k, v$ matrices. This means we can attend to different parts of the sequence different. Again this can be done in parallel.

#### Multi Query Attention

In this case, we still have N $q$ heads, but unlike MHA, we have only 1 $k$ and $v$ head. This means we still do the same number of operations, but have a lower memory footprint (less stored weights).

There is some quality loss in this case.

#### Grouped Query Attention

Assume, we have 8 $q$ heads, with 4 sub-groups. This means we have 4 $k$ and $v$ heads. This is a way of "parameterizing" the amount of quality loss in MHA.

[yt video explaining GQA & MQA](https://www.youtube.com/watch?v=pVP0bu8QA2w)

#### Flash Attention

Notes on [this yt video](https://youtube.com/watch?v=zy8ChVd_oTM).

### Positional Embeddings

Positional embeddings are needed because Transformer models are position invrariant by defaulf. For example, "the dog chased the pig" and "the pig chased the dog" have the same representation in the model.

If you want to preserve order, we need to add positional embeddings.

#### Absolute Positional Embeddings

Each position in the word is assigned a positional embedding as well as the word embedding. These are then added together to form the input to the model.

These can be learned or sinusoidal. These have similar performance. Learned embeddings have a max length (creating the MAX context length).

#### Relative Positional Embeddings

Learn a representation for every pair of tokens in a sentence. To do this, we have to modify the attention mechanism to handle these embeddings.

In practice, these are slow. These require an extra step in the attention layer.

#### RoPE

Rotary Positional Embeddings.

Rather than add a positional vector to the input, they propose adding a rotation to word embedding. The further in the sentence the word is, the more it is rotated.

The embed the position M, we rotate the word embedding by $M \cdot \theta$.

The relative position of words are preserved as well. 

[Video](https://www.youtube.com/watch?v=o29P0Kpobz0)
[Paper](https://arxiv.org/pdf/2104.09864)

## LLM Architectures

### Scalable MatMul-free Language Modelling

[This paper](https://arxiv.org/pdf/2406.02528) details how MatMul operations (the expensive ops that require GPUs) can be eliminated from language models entirely.

However, the title is slightly misleading. Essentially, all the weights in the model have been constrained to be ternary - i.e in the set { -1, 0, 1 }. By doing this, it simplifies any equation that looks like $ y = xW_{i} $ (i.e a vector-matrix multiplication) into a element wise product (hadamard product). They also remove the attention layer, and replace it with something similar to a RNN that can be parallelized.

### Mamba2

[Tri Dao's Blog on the Mamba2 Release](https://tridao.me/blog/2024/mamba2-part1-model/)
The Mamba2 paper tries to combine the efficiency of Attention with the original State Space Model (SSM) - Mamba1. The SSM defines a map from $x \in R^{T} -> y \in R^{T}$. 

## LLM Papers & Models

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

### MoA

[MoA](https://www.together.ai/blog/together-moa) is **M**ixture **o**f **A**gents. Their hypothesis is that "LLMs tend to generate better responses when presented with outputs from other models, even if these other models are less capable on their own.

They split up the models into *Proposers* & *Aggregators*:
    - Proposers generate initial reference responses.
    - Aggregators synthesize the different response from the proposers into a single, high-quality response.

It is essentially multiple models working in tandem.

## LLM Inference

### Math
[Maths of storing, inference and training of LLMs](https://blog.eleuther.ai/transformer-math/)
Model weights are stored in mixed precision - either fp16 + fp32 or fp16 + bf16. fpN is N-bit floating point. fp16 + fp32 means using lower precision for the majority of the model and higher precision for the parts where numerical stability is important. bf16 is bfloat16. This offers a larger dynamic range than fp16, whilst still providing the reduced memory usage and increased training speed.
        - Difference precisions require different memory:
In fp16/bf16, $memory_{model} = (2 bytes/param) \cdot (No.params)$.
In fp32 we require more memory: $memory_{model} = (4 bytes/param) \cdot (No.params)$.
There is also memory required for inference: $TotalMemory_{Inference} \approx (1.2) \cdot ModelMemory$.
So for example, Llama2-70b at fp16/bf16 requires around 168GB of RAM (most likely slightly more - so 200GB of RAM).
[LLM Inference Math](https://kipp.ly/transformer-inference-arithmetic/)
kv cache: we store the previously calculate key, value attention matrices for tokens that aren't changing. This happens when the model samples it's output token by token.

### Inference Optimization

[Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
Most interesting thing from here was *Inference Quantization*. The essentially means setting the weights to use int 8-bit precision and keeping activation at fp32 or bf16. This cuts down on the memory required to store the model, as we are using 50% of the memory per parameter.

### RoRF: Routing on Random Forests

See [this article](https://www.notdiamond.ai/blog/rorf) in which they detail how they trained a random forest on embeddings and used it to route queries between two LLMs. They outperformed the LLMs on their own (with two strong LLMs) and resulted in a cheaper cost with compartiable performance when looking at (weak vs strong LLMs).

### Transformer Inference Toolkit

[Article](https://astralord.github.io/posts/transformer-inference-optimization-toolset/).

#### Details on GPUs

This article starts off with an in-depth look at GPUs. One interesting point is that we can classify computations (layers) into one of three categories:

- Compute-bound: The time spent on arithmetic operations exceeds the time spent on memory accesses. Typical examples are linear layesr with a large inner dimensio or a convolutional layer with a large number of channels.
- Memory-bound: The time spent on memory accesses exceeds the time spent on computational operations. Most operations are classified as this, such as elementwise operations (activation functions, dropouts) or reductions (sum, softmax, normalization).
- Overhead-bound: Communication-bound, interpreter-bound, etc.

The balance between the first two is measured in arithmetic intensity, which is the number of arithmetic operations per byte of memory accessed required to run the computation. For example, apply a ReLU operation to an input tensor $x$ requires:

- read 2 bytes
- make 1 comparison
- write 2 bytes

Therefore, the aritmetic intensity of a ReLU is $\frac{1}{4}$. For each operation, we make 4 memory accesses. We can also look at an `ops:byte` ratio. Let's do an example of an input batch $x \in \mathcal{B}^{B \times d}$ and weight matrix $W \in \mathcal{R}^{d \times d}$. We want to compute the `ops:byte` ratio for a linear layer (i.e $xW$).

The linear layer compuation requries $2Bd^2$ flops and $2d^2$ (given that $B << d$). If we are doing this on a A100 gpu, we can calculate $T_{compute}$ and $T_{memory}$:

- $T_{compute} = \frac{2Bd^2}{312 \cdot 10^{12}}s$
- $T_{memory} = \frac{2d^2}{1.55 \cdot 10^{12}}s$

To find the bottleneck for our model, we can look at the ratio between these two terms, which is:

- $\frac{T_{compute}}{T_{memory}} = \frac{B}{200}$

Therefore, until our batch size is smaller than 200, our system performance is memory-bound. Enlarging the batch size to be greater than 200 increases the compuation time, while keeping the memory access time constant. This is the compute-bound scenario.

#### KV Cache

In GPTs, text generation occurs in two stages:

- Prefill - the model ingests a large chunk of our prompt tokens in parallel, computing all hidden states and outputs in one pass.
- Autoregressive Decoding - happens after prefill, where the model generates tokens one by one.

In this second pahse, we don't need to send the entire query vector into the mechanism (as we have already calculated the attention values). We can also cache the K and V matrices so we don't have to recalculate them.

We can also do things like *Multi-Query Attention*, which uses the same K and V matrices for each Q vector, which doesn't change the computational complexity, but does drastically reduce the memory footprint (depending on the number of Q heads).

*Grouped Query Attention* is a mixed between MQA and MHA. We split h query heads into g groups, each with its own keys and values. When g = 1 GQA is equivalent to MQA and when g = h, GQA is equivalent to MHA.

## Scaling Test Time Compute

[This blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) by hugging face goes into detail on some of the potential test time compute implementations. In this case, they use *search against a verifier*.

They focus on 3 topics:
- **Best of N**: generate multiple responses per problem, and assigns scores to each candidate answer, typically using a reward model. Then select the answer with the highest reward (or a weightd variant).
- **Beam search**: A systematic search method that explores the solution space, often combined with a process reward mdoel to optimse the sampling and evaluation of the intermediate steps. PRMs provide a sequence of scores, one for each step of the reasoning process.
- **Diverse verifier tree search (DVTS)**: AN extension of beam search that splits the initial beams into independent subtress, which are then expanded gredilly using a PRM. The method improves diversity and overall performance (with larger test-time compute budgets).

**Weighted Best of N**: Aggreate the scores across all identical responses and select the answer with the highest total rewrard. $a_weighted$ = $argmax_{a}\Sigma_{i=1}^{N} \mathcal{I}(a_i = a) \cdot RM(p, s_i)$.

where $RM(p, s_i)$ is the reward model score of the i-th solution to problem p. $\mathcal{I}(a_i = a)$ is an indicator function that is 1 if the i-th solution is equal to the answer a, and 0 otherwise.

**Beam search** is like doing this but at every step (usually indicated by a `\n\n`), and then looking at M (the beam width) different paths.

**DVTS** is like beam search, but each initial 'branch' becomes an independent subtree, generating more potential solutions

## Training Models

### Data Collection

[Here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is a good resource on a how a high quality training dataset for LLMs can be created.

## Resources

### ML Papers of the Week
- [ML Papers of Week](https://github.com/dair-ai/ML-Papers-of-the-Week)

### Deep Dive into LLMs

- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
This video is by Andrej Karpathy, talking about how LLMs are built.

(Byte Pair Encoding) BPE - transform all the words into bytes, and then look for common byte pairs, and coming them into single tokens. Take a look at [tiktokenizer.vercel.app](tiktokenizer.vercel.app) to do some tokenization demos.

- Pretraining - Background Knowledge.
- SFT - Imitating Being an Expert (A worked problem).
- Reinforcement Learning - Practicing Problems.

RLHF can produce *adversarial results* which means non-sensical results (an infinite number) which produce high rewards, but are not actually good. In verifiable domains you can run RL indefinitely without running into this (provided your reward function is set up well). However, RLHF is gameable so you cannot run it indefinitely.

## Productionizing LLMs

### LinkedIn - Musings on a Generative AI Product

[Article](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)

Most interesting part was getting structured data. In this case, they built a custom yaml parser that handled the specific mistakes the LLM could make. This reduced errors from 10% -> 0.01%.

They found they could easily reach "80% quality" but every subsequent 1% gain after that got harder and harder.

### Evaluation of LLM Applications

[Article](https://hamel.dev/blog/posts/evals/#motivation)

This article detailas the need for a good evaluation framework for LLMs.

Unit Tests -> Assertions the the model is working & producing the correct output.
Humna & Model Eval -> Keep logs of the interactions with the model, and validate that the correct thing is happening. Another LLM can periodically do this as well.
A/B Testing

### Google - AI Ideas

See [this article](https://blog.google/products/google-cloud/gen-ai-business-use-cases/) where Google detail a bunch of interesting AI usecases.


## HCI / UX

### Interacting with Chatbots

*Good tools make it clear how they should be used*. LLM's should not be chatbot interfaces with complex prompts, sliders for different settings should be used, i.e. competency with a topic, how verbose a response. Find an in depth article [here](https://wattenberger.com/thoughts/boo-chatbots).

## Sampling

Sampling refers to the process of generating text, how do we actually select the best next token?

### `softmax` is not enough (for sharp out-of-distribution)

[This paper](https://arxiv.org/pdf/2410.01104v1) details that the softmax function degrades in performance as sequence length grows, and propose *adaptive temperature* as an ad-hoc technique for improving the sharpnes of softmax at inference time.


*adaptive temperature* is not the solution but allows softmax to perform better even if the sequence length grows.

They suggest a Lemma that softamx must disperse: "Let $e(n) \in \mathcal{R}^{n}$ be a collection of n logits going into the `softmax`$_{\theta}$ function with temperature $\theta$ > 0, bounded above and below s.t. $m \leq e_{k}^{n} \leq M$ for some $m, M \in \mathcal{R}$. Then, as more items are added ($n \rightarrow +\infin$), it must hold that, for each item $1 \leq k \leq n$, `softmax`$_{\theta}(e^{(n)})k = \Theta(\frac{n}{1})$. That is, the computed attention coefficients disperse for all items." which they then prove in the paper.

They also note that setting $\theta$ to be 0 is problematic & decreases accuracy. Therefore, they suggest an adaptive temperature, which decreases as entropy increases (thereby decrease the entropy).

### Forking Paths in Neural Text Generation

[This paper](https://arxiv.org/pdf/2412.07961) works on the idea that LLMs have no intent behind what they will say, they are simply next-token predictors. They 'prove' this by looking at tokens that dramatically change the answer, for example:

'Who is the current British head of state? Well, since we know it's \[2024|2021\], we can infer that it is [King Charles|Queen Elizabeth]'. The token in the brackets drastically changes the final answer. They also found that seemingly arbitrary tokens affect this (like sampling a '(' instead of a word). This results in them saying:

"One interpretation might be that people typically holding intents and plan responses to some degree before they speak, whereas LLMs truly decide what to say next on the fly."

There is a lot of complex probability stuff in this paper, so it might be worth re-reading.
