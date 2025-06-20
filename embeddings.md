# Embedding Models

## Techniques

### Matroyshka Embeddings

This [article](https://huggingface.co/blog/matryoshka) details a method of using matroyshka embeddings to essentially "front-load" the important information in the embeddings at the front of the vector. This is useful because we can actually decrease the dimensionality of our output vector, and therefore have more efficient retrieval.

It works by summing the loss for each of the Matroyshka dimensions, and then trying to minimize that total loss, which will in-effect "front-load" the model. Each Matroyshka dimension is just a truncation of the output embeddings.

For more detail, [this is the paper](https://arxiv.org/pdf/2205.13147). The official loss function is: $min_{W^{m}_{m \in M}, \Theta_{f}}\frac{1}{N} \Sigma_{i \in N}\Sigma_{m \in M} c_{m} \cdot \mathcal{L} (W^{m} \cdot F(x_{i};\Theta_{F})_{1:m}; y_{i})$. In normal loss, $c_{m}$ is set to 1 and $\mathcal{L}$ is the normal loss for the embedding model with dimension $m$.

The inference cost is the same, but it's cheaper to store the embeddings once they are produced.

### Contextual Document Embeddings

See [this paper](https://arxiv.org/pdf/2410.02525).

It defines how to create contextualized document embeddings, i.e embeddings that are specific to a document corpus.

Statistical methods of retrieval (e.g BM25) can easily incorporate prior corpus statistics, such as TF-IDF into their representations. This  imparts corpus-context dependence into the model, since this can't be updated with information specific to retrieval in a given domain at test time.

They propose two techniques. The first of which is focused on training - 'contextual training', which aims to build a notion of the neighboring documents directly into the contrastive learning process. Essentially, each training batch is comprised completely of neighbors (collected via query-document clustering) such that embeddings can distinguish documents even in the most challenging contexts. By being forced to distinguish between similar documents, the model becomes more sensitive to subtle differences that are important in specific contexts.

The second is their new encoder architecture (CDE) augments the standard BERT-style encoder with additional conditioning that provides aggregated document-level information about neighboring documents. The paper proposes a method to add contextualization directly to the architecture of the embedding model. Given some context documents, they embed each and then concatenate embeddings into a sequence. Then, to compute the embeddings for a different document $d'$, they concatenate the sequence of context embeddings with the token embedding matrix of another model applied to each token in $d'$. This is then passed through a transformer to produce the final embedding.

In a more casual way, this can be described as "First, it calculates a shared embedding for the cluster to which the document belongs. Then, it combines this shared embedding with the document’s unique features to create a contextualized embedding."

[This article](https://venturebeat.com/ai/new-technique-makes-rag-systems-much-better-at-retrieving-the-right-documents/) provides a more casual explanation.

## Finetuning / Training Models

### augmented-SBERT

This [paper](https://arxiv.org/pdf/2010.08240) details a powerful method of finetuning bi-encoders. 

Initially, they compared the performance of cross-encoders & bi-encoders and showed that with little data, cross-encoders greatly outperform bi-encoders on retrieval metrics. Therefore, they present a data augmentation method that uses a cross-encoder to create a labelled dataset which can then be used to fine-tune a bi-encoder.

In order to create this labelled dataset, they try out a number of various sampling techniques (from some larger unlablled sentence pair dataset) attempting to "weakly label" a large amount to keep the distribution of negative pairs to positive pairs the same as their "gold-standard labelled training dataset". They use BM25 & a bi-encoder for this weak labelling - BM25 produced the best results.

In this paper, they also do something called "Seed Optimization" which basically tries out training with 20% of the total required training steps with different random seeds, as the seed can greatly affect the model performance, especially with smaller datasets.

They found the best results for domain adaption from having a base model trained on a generic domain, and then adapting it with the cross-encoder labelled data (along with the gold-standard data) to a specific domain.

### Semantic Re-Tuning with Constrastive Tension

[Paper](https://openreview.net/pdf?id=Ov_sMNau-PF)

This is an self-supervised method for finetuning, and when combining CT with supervised data they found that it outperformed supervised-only methods. 

Their method trains two separate language models on the task of maximizing the dot product between the two models’ representations for identical sentences, and minimizing the dot product between the models’ representations for different sentences. This training objective encourages the model to retain a semantically distinguishable sentence representation until the final layer.

This outperforms sBERT on supervised regression tasks.

"we find little reason to believe that the realignment enforced by CT to be beneficial for fine-tuning tasks where ample training data is available." -> i.e. it's useful when you don't have much data.

### GPL: Generative Pseudo Labeling for the Unsupervised Doman Adaptation of Dense Retrieval

[Paper](https://arxiv.org/pdf/2112.07577)

Dense retrieval models are extremely sensitive to domain shifts. The authors train a dense retriever with pairs with similarity scores (which are generated by a cross-encoder - aka the 'pseudo'-labels). The 'queries' in the pairs are created by a query generator and therefore requires less unlabeled data from the target domain.

The dataset format was $(Q, P_{pos}, P_{neg}, M)$, where $M$ is the 'margin' between the cross-encoder score of (Q, P_{pos}) and (Q, P_{neg}). MarginMSE loss was used to train the dense retriever. Interestingly, the margin is taken into account here, which means they aren't just trying to push these as far apart as possible.

Using the cross-encoder labels softens how much either a passage that is incorrectly marked as negative or a badly generated query that is not answerable by $P_{pos}$ will affect because the finetuned dense retriever will mimic the score margins of the cross-encoder.

Each $P_{pos}$ had 3 Q associated with it. However, they found that ~250K queries was enought for good performance on DistillBERT.

In their evaluation, they showed that without using the cross-encoder labels, aka just 0 & 1, using hard negatives actually hurt the performance of the fine-tuned model.

The synthetic queries were generated using a specific version of T5 designed for generating queries. I think modern LLMs like Claude are fine at doing this without finetuning.

### Text and Code Embeddings by Contrastive Pre-Training

[Paper](https://arxiv.org/pdf/2201.10005)

In the paper, they show that 'contrastive' pre-training (i.e training to get closer to positives and further away from negatives) leads to high quality embeddings. This can be done unsupervised, as long as you have $A$nchor and $P$ositive pairs, [$A_{1}, P_{1}$], [$A_{2}, P_{2}$], etc. as you can build the negative pairs by doing [$A_{1}, P_{2}$], [$A_{1}, P_{3}$], etc.

Typically, you'll want to do hard negative mining but pseudo-random works somewhat. I have found this to be applicable to cross-encoder models too.

They also training code embedding models, and used the docstring and function implementation as the $A$nchor and $P$ositive respectively. In this case, the goal was to retrieve the relevant piece of code given a search string. One interesting point that they found was no performance improvement when increasing the number of parameters from 300M $\rightarrow$ 1.2B.

A nice way of explaining embedding models: **The hidden state from the last layer corresponding to the special token [EOS] is considered as the embedding of the input sequence.** A quote from paper shows that what is chosen as these tokens is important too: *We found that using different delimiters leads to more stable training. For x, we use ‘[’ as [SOS]x and ‘]’ as [EOS]x , while we use ‘{’ and ‘}’ as [SOS]y and [EOS]y respectively for y.*

They also mention **Linear Probe Classifcation** here, which is essentially freezing the weights of the representation model, and training a linear classifier head on top. In theory, if the embeddings can be linearly separated into categories, they are high quality embeddings.

### Finetuning with Modal

[See this article](https://modal.com/blog/fine-tuning-embeddings) for details on using the modal platform to finetune embeddings

A couple important notes:
- They use a Grid Search for finding hyperparameters, in this case it is good because it is easy to parallelize.
- They ended up choosing `bge-base-en-v1.5`.

### GISTEmbed

This [paper](https://arxiv.org/pdf/2402.16829) details a method on collecting negatives for text embedding fine-tuning. It uses a guide model to enhance in-batch negative selection.

They use large & high performing embedding models to finetune smaller embedding models. So, given a (Q, P) pair, they sample the entire batch for negatives, then use the guide model to filter out any "negatives" with a higher similarity to Q than P. This means you can finetune a model with just (Q, P) pairs, and not have to worry about collecting negatives. This is their official explanation: *If any of the similarities in the similarity matrices $\sigma$, derived from vectors generated by $G$, is greater than the similarity $\sigma_{qp}^{i}$ of the query-positive pair, then we assume that these are examples that must not be considered as irrelevant.

I think this sits as an alternative to GPL, because that used MarginMSE loss with cross-encoder labels, whereas this filters out the potentially 'bad' negatives by using a guide model.

### A Simple Framework for Contrastive Learning of Visual Representations

In [this paper](https://arxiv.org/abs/2002.05709), they focus on on Contrastive Learning for Visual Representations, but potentially some of the knowledge could be useful for an type of embedding based contrastive learning.

They show a number of things:
    - (1) For image-based tasks, data augmentation is v. important.
    - (2) Introducing a non-linear transformation between the representation (the embedding) and the contrastive loss function substantially improves perfomance.
        - They hypothesized that this is because optimizing for the contrastive loss may cause a slight information loss in the embeddings (for the final classification task), so adding this non-linear transformation means that this is mitigated.  
    - (3) Contrastive Loss requires a much higher batch size compared to supervised learning.

## Evaluation

### Evaluating Embedding APIs for Inofmration Retrieval

[Article](https://ar5iv.labs.arxiv.org/html/2305.06300).

In this article, they compare a variety of Embedding APIs (including:   Aleph-Alpha, Cohere & OpenAI). They compared the models across a full-ranking (aka a dense retrieval) and also from a re-ranking task from the top100 retrieve by BM25. On the BEIR benchmark, OpenAIs ada2 performed the best, however across all languages Cohere+BM25 retrieval was the best.

The metrics they used were:
- [nDCG@k](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) - Normalized Discounted Cumulative Gain, k is the number of documents to consider - i.e. the top-k documents. The also looked at recall - which is useful when you have multiple "correct/relevant" answers.

## Loss Functions

### RankedListLoss for Deep Metric Learning

[Paper](https://arxiv.org/pdf/1903.03238)

*Existing pairwise or tripletwise loss functions are known to suffer from slow convergence due to a large proportion of trivial pairs or triplets as the model improves. To improve this, ranking-motivated structured losses are proposed recently to incorporate multiple examples and exploit the structured information among them.*

This paper proses a novel ranked list loss. For every mini-batch, the learning objective of RLL is to make the query closer to the positive set than to the negative set by a margin. They also propose some regularization to prevent positive pairs from being 'pulled' as close as possible in embedding space.

Specifically, given a query, they obtain a ranked list by sorting all other data points according to the similarities, ideally all the positive examples are supposed to be ranked before the negatives samples. To acheive this, they introduced RLL to organise the samples of each query. Given a query, the optimisation of RLL is to rank all positive points before negative points and forcing a margin between them. Ranking across the entire set contains richer information than just triplet based loss.

For their regularization, they have a threshold at which they only push the distance between positive and query to be under that threshold, to prevent the similarity from "collapsing" (in a sense).

They argue that existing pairwise or triplet loss functions will suffer from slow converagence to due to a large proproration of trivial pairs / triplets as the model improves.

## Reranking 

### ColBERT

[HuggingFace](https://huggingface.co/colbert-ir/colbertv2.0)
[Demo](https://www.youtube.com/watch?v=cN6S0Ehm7_8)

ColBERT is an alternative to an cross-encoder. Cross-Encoder is an "early interaction" model, it takes in a query & a document, combines them and then outputs a similarity score. This can get pretty slow due to the attention mechanism.

ColBERT is a "late interaction" model. For both the query and the document, you produce an embedding for each **token**. For each token vector in the query, you get the max similarity score across all token vectors in the document. The final score is the sum of all these max similarities.

This is different to a cross-encoder because the q & d are not combined. ColBERT is more efficient because the document embeddings only need to be calculate once. Therefore, if you are reranking over many documents it may be better to use ColBERT.

[JinaAI Article](https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/) - also useful.

### Any* Embedding Model Can Be a Late Interaction Model

[Article](https://qdrant.tech/articles/late-interaction-models/)

In embedding models, the final output is a single vector. However, this vector is produced by a pooling operation (typically mean pooling) of the final representation of the token vectors. 

These token vectors are pretty similar to the way the ColBERT works. This article showed that using these as multi-vector retrieval models actually outperformed ColBERT.

The only issue is that using these multi-vector models with all the output vectors uses up more memory than ColBERT embeddings. However, the authors of the article showed that in this case, quantization to int8 was fine. 

### Late Chunking

Late Chunking is slightly different to Late Interaction defined by ColBERT. Late Interaction compares every token in the query to every token in the document. This can get costly (imagine storing an embedding for each token).

Late Chunking uses Pooling similar to embedding models. Embedding models, would chunk into 512 token chunks, then embed those tokens, and then produce a vector via pooling to a produce a single representation for that entire chunk. But what if we want to span multiple chunks?

Late Chunking would first embed the entire chunk (therefore retaining the attention scores across all tokens). This obtains token embeddings. We split these into a determined sized (512) and then pool the token embeddings after that. This allows us to span multiple chunks.



