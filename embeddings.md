# Embeddings

## Finetuning / Training Embedding Models

### Text and Code Embeddings by Contrastive Pre-Training

[Paper](https://arxiv.org/pdf/2201.10005)

In the paper, they show that 'contrastive' pre-training (i.e training to get closer to positives and further away from negatives) leads to high quality embeddings. This can be done unsupervised, as long as you have $A$nchor and $P$ositive pairs, [$A_{1}, P_{1}$], [$A_{2}, P_{2}$], etc. as you can build the negative pairs by doing [$A_{1}, P_{2}$], [$A_{1}, P_{3}$], etc.

Typically, you'll want to do hard negative mining but pseudo-random works somewhat. I have found this to be applicable to cross-encoder models too.

### Finetuning with Modal

[See this article](https://modal.com/blog/fine-tuning-embeddings) for details on using the modal platform to finetune embeddings
