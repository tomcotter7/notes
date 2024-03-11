# Generative AI

## Prompt Engineering

A [paper](https://arxiv.org/pdf/2312.16171v1.pdf) with 26 different prompting techniques has been released. They focus on strategies that help LLMs produce better outputs.

## Retrieval Augmented Generation

### Retrieval

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

#### Fusion / Hybrid Retrieval

We can use both a dense vector search (produced by an embedding model) and a sparse vector search (produced by a model like BM-25, which uses TF-IDF). We can then combine the results of both searches to produce a better result. These can be combined using RRF.

#### Reciprocal Rank Fusion (RRF)

RRF simply sorts the documents according to a naive scoring formula. Given a set of $D$ documents , and a set of rankings $R$, we can compute: $RRFScore(d \in D) = \Sigma_{r \in R} \frac{1}{k + r(d)}$. This is then used to sort the documents.

We can use this to combine the results of multiple queries.

#### Query Transformation

**Sub Query Transformation**. This means using an LLM to break a query down into multiple sub-queries if required. For example, "Which has more Github stars, Langchain or LlamaIndex?", can be broken down into "How many Github stars does Langchain have?" and "How many Github stars does LlamaIndex have?", then the results can be combined.

## Fine-Tuning

### RLAIF

[RLAIF - Paper](https://arxiv.org/pdf/2309.00267.pdf)
This paper shows the comparision between RLHF and RLAIF. They seem to comparative results statistically. RLHF is slighlty better, but is more expensive & slower.

They used chain of thought (CoT) reasoning to compare the two summaries. First, they ask the model: 
    - "Given these two summaries - explain which is better".
Then, using the response from this they ask:
    - "Which is better?"

They don't generate a response here, they just look at the probability of the tokens for "1" and "2" for the answer to the first question.
