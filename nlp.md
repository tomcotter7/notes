# NLP

## Topic Modelling
### Latent Dirichlet Allocation (LDA)
An unsupervised ML model that can be used to discover topics in a corpus of documents.
    - https://www.youtube.com/watch?v=T05t-SqKArY
    - https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
    - https://github.com/rwalk/gsdmm
Gibbs Sampling LDA is more suited to short form text (i.e Tweets)

## Preparing Data

### PDFs

#### Headers and Footers

In order to determine the approximate size of a header or footer across an entire document, we can use PyMuPDF and DBSCAN.

```python
import fitz
import numpy as np

from collections import Counter
from sklearn.cluster import DBSCAN

def get_hf_size(source_doc: str) -> int:
    
    doc = fitz.open(source_doc)
    hf_size = 0
    categorize_vectors.append([])
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            block_rect = block[:4]
            block_text = block[4]
            categorize_vectors.append((*block_rect, len(block_text))
    
    page_rect = doc[0].rect
    X = np.array(categorize_vectors)
    dbscan = DBSCAN()
    dbscan.fit(X)
    labels = dbscan.labels_
    label_counter = Counter(labels)
    most_common_label = label_counter.most_common(1)[0][0]

    labels = [0 if label == most_common_label else 1 for label in labels]
    for vector, label in zip(categorize_vectors, labels):
        if label == 1:
            y0 = abs(vector[1] - page_rect.y1)
            y1 = abs(vector[3] - page_rect.y0)
            hf_size = max(min(y0, y1), hf_size)

    return hf_size
```

Essentially, DBSCAN is used to cluster the vectors of the text blocks on the page. The most common cluster is assumed to be the main body of the document, and the distance from the top and bottom of the page is used to determine the size of the header and footer.



