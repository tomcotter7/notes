# LLMs Core

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

### Packages for LLM Inference

#### vLLM

[This article](https://croz.net/run-your-own-ai-at-scale-vol-1-tuning-vllm/) has a really good intro into improving vLLMs efficiency, specifically around `tensor_parallelism` and `pipeline_parallelism`.

##### Paged Attention

vLLM's core optimization is something called PagedAttention.

[This paper](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165) defines PagedAttention.

For each input token, an LLM will generate attention key and value tensors. Rather than recomputing these attention-related tensors from scratch for each step of the decoding process, the model will store these as the KV cache. The challenge is that this can quickly become a bottleneck (due to meemory constraints).

The core idea behind PagedAttention is to partition the KV cache of each sequence into smaller, more manageable "pages" or blocks. Each block contains key-value vectors for a fixed number of tokens. This means that the KV cache can be loaded and accessed more efficiently during the attention computation. Essentially, you have a lookup table to the KV attention vectors.

[Hopsworks](https://www.hopsworks.ai/dictionary/pagedattention) also has a nice blog post on this topic.

#### ray

`ray` is a useful package for performing batch inference at scale. [This](https://docs.vllm.ai/en/v0.8.1/getting_started/examples/distributed.html) is there example for doing offline batch inference at scale.

## LLM Sampling

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

