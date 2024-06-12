# Retrieval Augmented Generation

## Evaluation

### C-RAG

[Paper](https://arxiv.org/pdf/2406.04744) describes a detailed dataset for evaluating RAG systems. It also contains 'mock' APIs for KGs & Web Search retrieval in order to test that.

### Evaluating IR

[ir-measures](https://ir-measur.es/en/latest/getting-started.html) - A Python package for evaluating Information Retrieval systems.

### Evaluating Embedding Models

[Article](https://ar5iv.labs.arxiv.org/html/2305.06300).

They found that BM25 outperforms embedding models for the re-ranking use case - which makes sense. The best performance came from combining the Cohere **embedding** model with BM25. Note that they are using bi-encoder embedding models for re-ranking, not cross-encoders.

The metrics they used were:
- [nDCG@k](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) - Normalized Discounted Cumulative Gain, k is the number of documents to consider - i.e. the top-k documents. The also looked at recall - which is useful when you have multiple "correct/relevant" answers.

## Retrieval

### Is Cosine Similarity of Embeddings Really About Similarity?

[Paper](https://arxiv.org/pdf/2403.05440.pdf). In this paper they derive analytically that cosine-similarity can yield arbitrary and therefore meaningless `similarities`, this is even true for deep learned embeddings (such as the ones used in RAG). Cosine similarity has become popular under the motiviation that the norm of the vecotrs is not as important as the directional alignment between the vectors.

The reason they found that cosine similarity can be arbitrary is "We find that the underlying reason is not cosine similarity itself, but the fact that the learned embeddings have a degree of freedom that can render arbitrary cosine-similarities even though their (unnormalized) dot-products are well-defined and unique.".

The paper defines matrix factorization weirdly, [here](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) is a better explanation.


### Vector Databases

Multi-Tenancy or Role-Based Access Control (RBAC) can be implemented using metadata filtering. There is blog post by Q-drant on this [here](https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/)

### Sentence Window Retrieval

Sentence Window Retrieval. When indexing, we provide the ids of the 'chunks' that are either side of the current chunk. We use the ids instead of the chunks themselves, to not increase storage costs, and fetching the chunks based on ids is very quick.

This is a benefit because we can keep embeddings small, i.e similar to the query, but sitll provide lot's of context to the LLM.

### RAPTOR

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

### Fusion / Hybrid Retrieval

We can use both a dense vector search (produced by an embedding model) and a sparse vector search (produced by a model like BM-25, which uses TF-IDF). We can then combine the results of both searches to produce a better result. These can be combined using RRF.

### Reciprocal Rank Fusion (RRF)

RRF simply sorts the documents according to a naive scoring formula. Given a set of $D$ documents , and a set of rankings $R$, we can compute: $RRFScore(d \in D) = \Sigma_{r \in R} \frac{1}{k + r(d)}$. This is then used to sort the documents.

We can use this to combine the results of multiple queries.

### Query Transformation

**Sub Query Transformation**. This means using an LLM to break a query down into multiple sub-queries if required. For example, "Which has more Github stars, Langchain or LlamaIndex?", can be broken down into "How many Github stars does Langchain have?" and "How many Github stars does LlamaIndex have?", then the results can be combined.

### Knowledge Graphs for RAG

Neo4j offers built in `vector indexes`, which can be used to store embeddings. Once we have the `top-n` documents, we can grab the surrounding documents in the knowledge graph and use them as context for the LLM. See [here](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)

How to build these KGs?
    - In the basic course, they define 2 relationship, "NEXT" & "PART-OF", where "NEXT" links chunks to chunks and "PART-OF" links chunks to documents.
    - They also create a relationship "FIRST" which links the first chunk of a section to the section.

The true power would come from using a more complex KG, with relationship that have more meaning. 

### Self-RAG

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

## Frameworks

## FlashRAG

[Paper](https://arxiv.org/pdf/2405.13576)
[GitHub](https://github.com/RUC-NLPIR/FlashRAG)

A lightweight framework for running RAG applications. Most interesting thing is that they have examples of advanced RAG techniques. I can use this to implement my own techniques.

The metrics they are using (as of 12/06/2024) are:

- 'em': Exact Match
- 'f1': F1 Score
- 'sub_em': Substring Exact Match
- 'precision': Precision
- 'recall': Recall
