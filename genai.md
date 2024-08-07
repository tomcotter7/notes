# Generative AI

## Prompt Engineering

### Prompting Techniques

A [paper](https://arxiv.org/pdf/2312.16171v1.pdf) with 26 different prompting techniques has been released. They focus on strategies that help LLMs produce better outputs.

### DSPy

[DSPy](https://huggingface.co/papers/2310.03714)

The authors argue that prompt engineering is brittle, and akin to manually finetuning weights on a classifier. This abstracts LM pipelines as *text transformation graphs*. These can be automatically optimised for better results. From the paper "DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.". Prompts go from 'You are an AI assistant ....' to 'question -> answer'. Teleprompters are powerful optimizers (included in DSPy) that can learn to bootstrap and select effective prompts for the modules of any program. (The "tele-" in the name means "at a distance", i.e., automatic prompting at a distance.)

### F*ck you, show me the prompt

Working with LLM frameworks is annoying & the added complexity of not knowing the prompt or how many API calls are being made is annoying. [This blog post](https://hamel.dev/blog/posts/prompt/) details a library `mitmproxy` that can be used to intercept the API clalls, so you can determine exactly what is going on under the hood. The blog post also describes how current frameworks implement the "magic of LLMs". These include: `guardrails`, `guidance`, `langchain`, `instructor` & `dspy`.

## Fine-Tuning

### Few-Shot PEFT is Better and Cheaper than In-Context Learning

[This paper](https://arxiv.org/pdf/2205.05638) discuss and shows that ICL is a) too expensive when adding lots of examples and b) does not provide enough benefit.

Given some general task (like doing CoT prompting, or Q&A over a document set) is it more efficient to finetune a model to act this way, rather than provide examples in the prompt.

### RLAIF

[RLAIF - Paper](https://arxiv.org/pdf/2309.00267.pdf)
This paper shows the comparision between RLHF and RLAIF. They seem to comparative results statistically. RLHF is slighlty better, but is more expensive & slower.

They used chain of thought (CoT) reasoning to compare the two summaries. First, they ask the model: 
    - "Given these two summaries - explain which is better".
Then, using the response from this they ask:
    - "Which is better?"

They don't generate a response here, they just look at the probability of the tokens for "1" and "2" for the answer to the first question.

### LoRA - Low-Rank Adaptation of Large Language Models {#lora}

[LoRA - Paper](https://arxiv.org/pdf/2309.00267.pdf)

This paper details `LoRA` a more parameter efficient way of fine-tuning LLMs. When fine-tuning a set of weights, we can represent the new weights $W'$ as $W' = W + \delta W$. Typically, in fine-tuning tasks, we would modify $W$ directly, in order to obtain $W'$. However, the paper proposes that we represent $\delta W$ as a low-rank matrix, i.e. $\delta W = BA$, where $B$ and $A$ are low-rank matrices. 

$A$ and $B$ are also smaller matrices, and their product $AB$ represents a `low-rank` approximation of the original matrix. A rank in matrix terms means "the number of linearly independent columns in the matrix".

By choosing a low-rank approximation, the number of parameters required to train th emodel is reduced. For example, if $W$ is a $d x d$ matrix, then updating $W$ would involve $d^2$ parameters. However, with two low-rank matrices, $A$ and $B$ of sizes $d x r$ and $r x d$ respectively, we only require $2d \cdot r$ parameters, where $r$ is the rank of the matrix.

There is also a good article [here](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6) that explains this in more detail.

### QLoRA

[QLoRA - Paper](https://arxiv.org/pdf/2305.14314.pdf)

QLoRA builds on top of [LoRA](#lora). The main innovation is to `quantize a pretrained model to 4-bit`. This means storing tensors and performing computations with a reduced precision, in this case 4-bit "NormalFloat" - which yields better results than 4-bit ints / floats. In this case, during evaluation, they found models fine-tuned with QLoRA still regain 16-bit performance.

"QLoRA improves over LoRA by quantizing the transformer model to 4-bit precision and using paged optimizers to handle memory spikes."

Their LoRA approache includes adapters at every network layer and "thereby avoids almost all of the accuracy tradeoffs seen in prior work".

"The memory footprint ... comes from activiation gradients and not from the learned LoRA parameters". This means the number of low-rank adapters will not affect the overall memory footprint

### GaLore

[GaLore - Paper](https://arxiv.org/pdf/2403.03507)

This is a fine-tuning / pre-training method similar to LoRa / QLoRa in the sense that it uses Low-Rank matrics to approximate certain states when fine-tuning. However, LoRA and QLora represents the "change" in the weights as a low-rank matrix, whereas GaLore represents the gradients for each Weight as a low-rank matrix. 

[This yt video](https://www.youtube.com/watch?v=VC9NbOir7q0) explains the concept well. However, GaLore is essentially more memory efficient that LoRA (and more accurate because this matrix is actually low rank, rather than being approximated by a low-rank matrix).

### Alternatives to Fine-Tuning (That Isn't In-Context Learning)

#### Prompt Tuning / Prefix Tuning

Prompt Tuning v2 paper [here](https://arxiv.org/pdf/2110.07602.pdf). This is the idea of *tuning only the continuous prompts*. Specificaly, adding trainable continous embeddings to the original sequence of input word embeddings. Only these *continuous prompts* are updated during training.

There are technically 3 types of prompt tuning:
- *Hard* Prompt Tuning
- *Soft* Prompt Tuning - This is PromptTuningV1 - where the continous trainable vector is only applied to the input embeddings.
    - I think this is also prompt tuning v2, however in v2, they actually apply multiple continous embeddings across multiple layers. This is because the prompts in deeper layers have more of an effect on the output.
- *Prefix-Tuning*

We do this to focuse on providing inputs like, "Amazing Movie, It is [MASK]". This focusing on the way LLMs were trained which is to generate the next word of the sentence.

The Prefix-Tuning can be found [here](https://arxiv.org/pdf/2101.00190.pdf), this is similar, in terms of the fact that it learns a continuous task-specific vector to prepend to the input. These vectors are added to *each* transformer block. Because they optimize over continous word embeddings, rather than discrete tokens (prompt engineering), you can get much more value, and automatic optimization.

They apply a MLP to each transformer block which is of the shape: $P_{idx} X dim(h_{i})$, where $P_{idx}$ is the length of the prefix, and $dim(h_{i})$ is the dimensionality of the activation layer in the transformer block.

I quite like this approach because you get a way of producing task specific vectors without having to keep copies of the LLM.

### Quiet-STaR

[Quiet-STaR - Paper](https://arxiv.org/pdf/2403.09629.pdf)

This is a process of fine-tuning to follow a CoT type approach, by producing "thoughts" after each token, which are combined with the previous tokens to predict the next token. They call it *Quiet* because they are training the model to think before it speaks, but not neccesarily outputting the "thoughts".

The start of thought and end of thought tokens are `---` as typically in text, this is used as a 'break' or 'pause'.


## LLM Core Concepts

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

## LLM Architectures

### Scalable MatMul-free Language Modelling

[This paper](https://arxiv.org/pdf/2406.02528) details how MatMul operations (the expensive ops that require GPUs) can be eliminated from language models entirely.

However, the title is slightly misleading. Essentially, all the weights in the model have been constrained to be ternary - i.e in the set { -1, 0, 1 }. By doing this, it simplifies any equation that looks like $ y = xW_{i} $ (i.e a vector-matrix multiplication) into a element wise product (hadamard product). They also remove the attention layer, and replace it with something similar to a RNN that can be parallelized.

### Mamba2

[Tri Dao's Blog on the Mamba2 Release](https://tridao.me/blog/2024/mamba2-part1-model/)
The Mamba2 paper tries to combine the efficiency of Attention with the original State Space Model (SSM) - Mamba1. The SSM defines a map from $x \in R^{T} -> y \in R^{T}$. 

## LLM Papers & Models

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

## Training Models

### Data Collection

[Here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is a good resource on a how a high quality training dataset for LLMs can be created.

## HCI / UX

*Good tools make it clear how they should be used*. LLM's should not be chatbot interfaces with complex prompts, sliders for different settings should be used, i.e. competency with a topic, how verbose a response. Find an in depth article [here](https://wattenberger.com/thoughts/boo-chatbots).

## Resources

- [ML Papers of Week](https://github.com/dair-ai/ML-Papers-of-the-Week)

## Productionizing LLMs

### LinkedIn - Musings on a Generative AI Product

[Article](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)

Most interesting part was getting structured data. In this case, they built a custom yaml parser that handled the specific mistakes the LLM could make. This reduced errors from 10% -> 0.01%.

They found they could easily reach "80% quality" but every subsequent 1% gain after that got harder and harder.
