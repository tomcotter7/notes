# Encoder Models

## Finetuning / Training Models

### augmented-SBERT

This [paper](https://arxiv.org/pdf/2010.08240) details a powerful method of finetuning bi-encoders. Essentially, they create a labelled dataset using a cross-encoder, which then can be used to fine-tune a bi-encoder.

Cross-Encoders can be finetuned with less data as well, so this is the typical method for adapting the retrieval models to a specific domain.

### Semantic Re-Tuning with Constrastive Tension

[Paper](https://openreview.net/pdf?id=Ov_sMNau-PF)

This is an self-supervised method for finetuning, and when combining CT with supervised data they found that it outperformed supervised-only methods. 

Two independent models, with identical initialized weights, are set to maximise the dot product between their sentence representations for identical sentences, and minimize the dot product for their sentence representations of differing sentences.

"we find little reason to believe that the realignment enforced by CT to be beneficial for fine-tuning tasks where ample training data is available." -> i.e. it's useful when you don't have much data.

### GPL: Generative Pseudo Labeling for the Unsupervised Doman Adaptation of Dense Retrieval

[Paper](https://arxiv.org/pdf/2112.07577)

Dense retrieval models are extremely sensitive to domain shifts. They train a dense retriever with pairs with similarity scores (which are generated by a cross-encoder).

The dataset format was $(Q, P_{pos}, P_{neg}, M)$, where $M$ is the 'margin' between the cross-encoder score of (Q, P_{pos}) and (Q, P_{neg}). They then use this to train a dense retriever. They used MarginMSE loss to train the dense retriever.

They also have a finetuned generator model to generate the synthetic queries, although I'm not sure how much benefit this would have over something like Claude. 

They use MarginMSE loss with the cross-encoder labels to prevent errors from the query generator (i.e generating queries that are not answerable by the passage). This makes it more robust to badly generated queries.

### Text and Code Embeddings by Contrastive Pre-Training

[Paper](https://arxiv.org/pdf/2201.10005)

In the paper, they show that 'contrastive' pre-training (i.e training to get closer to positives and further away from negatives) leads to high quality embeddings. This can be done unsupervised, as long as you have $A$nchor and $P$ositive pairs, [$A_{1}, P_{1}$], [$A_{2}, P_{2}$], etc. as you can build the negative pairs by doing [$A_{1}, P_{2}$], [$A_{1}, P_{3}$], etc.

Typically, you'll want to do hard negative mining but pseudo-random works somewhat. I have found this to be applicable to cross-encoder models too.

### Finetuning with Modal

[See this article](https://modal.com/blog/fine-tuning-embeddings) for details on using the modal platform to finetune embeddings

### GISTEmbed

This [paper](https://arxiv.org/pdf/2402.16829) details a method on collecting negatives for text embedding fine-tuning. It uses a guide model to enhance in-batch negative selection.

They use large & high performing embedding models to finetune smaller embedding models. So, given a (Q, P) pair, they sample the entire batch for negatives, then use the guide model to filter out any "negatives" with a higher similarity to Q than P. This means you can finetune a model with just (Q, P) pairs, and not have to worry about collecting negatives.

## Evaluation

### Evaluating for IR

[Article](https://ar5iv.labs.arxiv.org/html/2305.06300).

They found that BM25 outperforms embedding models for the re-ranking use case - which makes sense. The best performance came from combining the Cohere **embedding** model with BM25. Note that they are using bi-encoder embedding models for re-ranking, not cross-encoders.

The metrics they used were:
- [nDCG@k](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) - Normalized Discounted Cumulative Gain, k is the number of documents to consider - i.e. the top-k documents. The also looked at recall - which is useful when you have multiple "correct/relevant" answers.

## Loss Functions

### RankedListLoss for Deep Metric Learning

[Paper](https://arxiv.org/pdf/1903.03238)

They argue that existing pairwise or triplet loss functions will suffer from slow converagence to due to a large proproration of trivial pairs / triplets as the model improves.


