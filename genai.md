 Generative AI

## Prompt Engineering

### Prompting Techniques

A [paper](https://arxiv.org/pdf/2312.16171v1.pdf) with 26 different prompting techniques has been released. They focus on strategies that help LLMs produce better outputs.

### DSPy

[DSPy](https://huggingface.co/papers/2310.03714)

The authors argue that prompt engineering is brittle, and akin to manually finetuning weights on a classifier. This abstracts LM pipelines as *text transformation graphs*. These can be automatically optimised for better results. From the paper "DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.". Prompts go from 'You are an AI assistant ....' to 'question -> answer'. Teleprompters are powerful optimizers (included in DSPy) that can learn to bootstrap and select effective prompts for the modules of any program. (The "tele-" in the name means "at a distance", i.e., automatic prompting at a distance.)

### F*ck you, show me the prompt

Working with LLM frameworks is annoying & the added complexity of not knowing the prompt or how many API calls are being made is annoying. [This blog post](https://hamel.dev/blog/posts/prompt/) details a library `mitmproxy` that can be used to intercept the API clalls, so you can determine exactly what is going on under the hood. The blog post also describes how current frameworks implement the "magic of LLMs". These include: `guardrails`, `guidance`, `langchain`, `instructor` & `dspy`.

## Retrieval Augmented Generation

### Retrieval

#### Is Cosine Similarity of Embeddings Really About Similarity?

[Paper](https://arxiv.org/pdf/2403.05440.pdf). In this paper they derive analytically that cosine-similarity can yield arbitrary and therefore meaningless `similarities`, this is even true for deep learned embeddings (such as the ones used in RAG). Cosine similarity has become popular under the motiviation that the norm of the vecotrs is not as important as the directional alignment between the vectors.

The reason they found that cosine similarity can be arbitrary is "We find that the underlying reason is not cosine similarity itself, but the fact that the learned embeddings have a degree of freedom that can render arbitrary cosine-similarities even though their (unnormalized) dot-products are well-defined and unique.".

The paper defines matrix factorization weirdly, [here](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) is a better explanation.


#### Vector Databases

Multi-Tenancy or Role-Based Access Control (RBAC) can be implemented using metadata filtering. There is blog post by Q-drant on this [here](https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/)

#### Sentence Window Retrieval

Sentence Window Retrieval. When indexing, we provide the ids of the 'chunks' that are either side of the current chunk. We use the ids instead of the chunks themselves, to not increase storage costs, and fetching the chunks based on ids is very quick.

This is a benefit because we can keep embeddings small, i.e similar to the query, but sitll provide lot's of context to the LLM.

#### RAPTOR

[RAPTOR - Recurse Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/pdf/2401.18059.pdf)
RAPTOR is an approach of embedding document(s) in a `tree-like` structure, and summarizing the child nodes at each level. This allows for a both a `fine-grained` retrieval and a more abstract retrieval. So, this means we can retrieval answers to very specific queries, but also more general queries, like `Tell me about the 5 upgrades listed in the document`.

Process:
    - Chunk the document into N chunks.
    - Cluster the chunks using a clustering algorithm based on Gaussian Mixture Models (GMM).
        - They are using `soft clustering` here, which means that each chunk can belong to multiple clusters.
    - For each cluster, they summarize the chunks in the cluster.
    - Repeat the process for the summarized chunks, until we have a single summary.

For retrieval, they have two processes. Either, they query the tree level by level, taking the `top-k` chunks at each level. Or, they collapse the tree and do traditional retrieval methods obtaining the most similar chunks.

Since the embedding vectors have a high dimensionality, they use Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality of the vectors, this results in better performance. As of 14/03/2024, they are ranked 3rd on the QuALITY benchmark, found [here](https://paperswithcode.com/sota/question-answering-on-quality?p=raptor-recursive-abstractive-processing-for).

#### Fusion / Hybrid Retrieval

We can use both a dense vector search (produced by an embedding model) and a sparse vector search (produced by a model like BM-25, which uses TF-IDF). We can then combine the results of both searches to produce a better result. These can be combined using RRF.

#### Reciprocal Rank Fusion (RRF)

RRF simply sorts the documents according to a naive scoring formula. Given a set of $D$ documents , and a set of rankings $R$, we can compute: $RRFScore(d \in D) = \Sigma_{r \in R} \frac{1}{k + r(d)}$. This is then used to sort the documents.

We can use this to combine the results of multiple queries.

#### Query Transformation

**Sub Query Transformation**. This means using an LLM to break a query down into multiple sub-queries if required. For example, "Which has more Github stars, Langchain or LlamaIndex?", can be broken down into "How many Github stars does Langchain have?" and "How many Github stars does LlamaIndex have?", then the results can be combined.

#### Knowledge Graphs for RAG

Neo4j offers built in `vector indexes`, which can be used to store embeddings. Once we have the `top-n` documents, we can grab the surrounding documents in the knowledge graph and use them as context for the LLM. See [here](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)

How to build these KGs?
    - In the basic course, they define 2 relationship, "NEXT" & "PART-OF", where "NEXT" links chunks to chunks and "PART-OF" links chunks to documents.
    - They also create a relationship "FIRST" which links the first chunk of a section to the section.

The true power would come from using a more complex KG, with relationship that have more meaning. 

#### Self-RAG

Self-RAG is a technique to prevent unnessecary retrieval, or remove irrelevant passages from the retrieved documents if required. In the [paper](https://arxiv.org/pdf/2310.11511.pdf), they train their own LM to do this. The steps are as folows:

- For a given input $x$, they generate a response $y$.
- Then, they generate a single token given $x$ and $y$ to determine if retrieval is required, collecting the documents $d$.
- Given $x$, $y$ and $d$, they generate 3 `reflection` tokens
    - (1) IsRel - whether the document $d_i$ is relevant to the input $x$.
    - (2) IsSup - whether the document $d_i$ fully supports, partially supports or does not support the output $y$.
    - (3) IsUseful - how useful the response $y$ is to the input $x$.

(1) is generated first, then $y$ would be updated. Then (2) and (3) would be generated to determine what to do with the output $y$.

I believe we can use a classifier model / smaller LM to do the same thing.

Youtube Video from langchain [here](https://www.youtube.com/watch?v=pbAd8O1Lvm4). Notes:
- `Active RAG`: LLM decides when and what to retrieve.
- In this they use `langgraph`, which essentially creates a state-machine. The model can generate the `action` it could take.
    - How would I implement this without langchain?
    - What benefit does langgraph add? -> I think this can be done with function calling and a custom state machine.

## Fine-Tuning

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


## The Math of LLMs

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

## HCI / UX

*Good tools make it clear how they should be used*. LLM's should not be chatbot interfaces with complex prompts, sliders for different settings should be used, i.e. competency with a topic, how verbose a response. Find an in depth article [here](https://wattenberger.com/thoughts/boo-chatbots).

## Resources

- [ML Papers of Week](https://github.com/dair-ai/ML-Papers-of-the-Week)

## Training

### Data Collection

[Here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is a good resource on a how a high quality training dataset for LLMs can be created.
