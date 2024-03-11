# Generative AI

[Andrew Ng's Course](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)

## Opinion Blog Posts
[Chatbots / HCI](https://wattenberger.com/thoughts/boo-chatbots)
*Good tools make it clear how they should be used*. 
Also, people want to interact with tools in different ways. The only current way to do that with LLMs is to add context via text.
Perhaps, we could think about adding "sliders". This shows your competency with the topic, how verbose a response, etc...
There is a spectrum of how much human input is required for a task - we want to stay human input > 50% of the total input.
    - This keeps the human engaged.
## ML Papers
### Resources for accessing papers
[ML Papers of Week](https://github.com/dair-ai/ML-Papers-of-the-Week)
A list of ML papers worth reading - updated every week. I think I should dedicate some time every week to reading and understanding one of these.
[AI Foundational Basics](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
Another list of AI foundational basics - should take some time to read through these.
### Eco-Assistant
[Eco-Assistant - Paper](https://arxiv.org/pdf/2310.03046.pdf)
Uses multiple calls to smaller language models (which are orders of magnitude cheaper) to interact with APIs. Only if the small model can't do it does it move on to more expensive models. Overall, this is cheaper. Results in a 5x cost reduction.
### DSPy
[DSPy](https://huggingface.co/papers/2310.03714)
They argue that prompt engineering is brittle, and akin to manually finetuning weights on a classifier. This abstracts LM pipelines as *text transformation graphs*. These can be automatically optimised for better results. From the paper "DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.". Prompts go from 'You are an AI assistant ....' to 'question -> answer'. Teleprompters are powerful optimizers (included in DSPy) that can learn to bootstrap and select effective prompts for the modules of any program. (The "tele-" in the name means "at a distance", i.e., automatic prompting at a distance.)
### CoVe - Chain of Verification
[CoVe](https://arxiv.org/pdf/2309.11495.pdf)
TODO: Read through this and make notes on it.
### QMoE: Sub 1-bit compression
[QMoE](https://arxiv.org/pdf/2310.16795.pdf)
TODO: Read through this and make notes on it.
## Foundational LLM Concepts
### Embeddings
[What are embeddings?](https://vickiboykis.com/what_are_embeddings/)
[1 2 3] - this is a 3 dimensional vector.
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
## Prompt Engineering
[Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
TODO: Read through this and make notes on it.
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
## Serving LLMs
### FlexFlow
[FlexFlow](https://github.com/flexflow/FlexFlow/). Uses speculative inference to improve the inference speed.
### Medus
[Medus](https://github.com/FasterDecoding/Medusa). Improves the inference speed without using speculative inference.
### FastChat
[FastChat](https://github.com/lm-sys/FastChat). Python library to easily serve LLMs -> also useful for training and fine-tuning.
### TinyChat
[TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main). In this repo is Tiny Chat - which uses quantization to produce super fast LLMs.
### Petals
[Petals](https://petals.dev/). This is a library for running LLMs in the style of bit-torrent. This means other people run different parts of the model. Llama2 runs at 6 tokens/sec.
### LlamaIndex
[LlamaIndex](https://docs.llamaindex.ai/en/stable/) - this a library for performing RAG. TODO: Read through the docs & check this out.
## LLM Training
### ReST
[ReST for Language Modelling](https://arxiv.org/pdf/2308.08998.pdf)
### Enforcing Outputs
[Guardrails](https://shreyar.github.io/guardrails/)
[Magnetic](https://github.com/jackmpcollins/magentic#magentic)
## Other Useful LLM Stuff
### Medium Articles
[Basic Langchain Rag](https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed). This condenses down this [course](https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/1/introduction) - is a pretty cool implementation of Q&A with your own data.
### LLM Utilization
[LangChain](https://python.langchain.com/docs/get_started/introduction.html). LangChain is a library for interacting with LLMs. I have found this to be rather bloated - it's often better to just interact with the LLM yourself.


# Image Classification
## ViT
[ViT](https://arxiv.org/pdf/2010.11929.pdf)
Comparable to ResNet when trained on large amounts of data. Turns the 2-D image into a 1-D encoding to feed into the encoder of a transformer (similar to BERT).
Once you have this, a classification head is trained -> but you can remove this and retrain it when fine-tuning
## Swin
[Swin Transformer](https://arxiv.org/abs/2103.14030): This improves on the original ViT by using Shifted WINdows (SWIN) -> i.e convolutions, which mean it has a better potential to be suited towards more computer vision tasks, not just image classification.
## Other Methods
There are also other methods produced that further combine convolutions and transformer (found [here](https://arxiv.org/pdf/2201.03545.pdf)).
## Image Recognition / Labelling
AWS has a good off-the-shelf service, found [here](https://aws.amazon.com/rekognition/resources/?nc=sn&loc=6) and [here](https://docs.aws.amazon.com/rekognition/latest/dg/labels-detect-labels-image.html).
The second link especially is very interesting - picking out objects in an image and returning them as multiple bounding boxes.


# Lectures
## Stanford NLP
[Youtube](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=1)
## MIT - Efficient ML
[Youtube](https://www.youtube.com/watch?v=rCFvPEQTxKI&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB)
## AI Engineer Summit 2023
[YouTube](https://www.youtube.com/@aiDotEngineer)

# NLP
## Preprocessing
[python-docx](https://python-docx.readthedocs.io/en/latest/index.html). This python library is useful for handling .docx files. Used previously for handling poorly formatted tables inside .docx files.
[regex101](https://regex101.com/). Useful website for testing regex.
## Text Summarization
[sumy](https://github.com/miso-belica/sumy)
## Topic Modelling
### Latent Dirichlet Allocation (LDA)
An unsupervised ML model that can be used to discover topics in a corpus of documents.
    - https://www.youtube.com/watch?v=T05t-SqKArY
    - https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
    - https://github.com/rwalk/gsdmm
Gibbs Sampling LDA is more suited to short form text (i.e Tweets)


# Programming Languages
## Mojo
### Resources
[The mojo homepage](https://www.modular.com/)
### Syntax Highlighters
[Vim](https://github.com/czheo/mojo.vim)
### Examples & Cheatsheets
[Mojo Cheatsheet](https://github.com/czheo/mojo-cheatsheet/blob/main/README.md)
## Go (Golang)
### Creating a Go Module
`go mod init <module-name>`.
If you want to use that folder (module) in a different folder you can do `go mod edit -replace <module-name>=<path-to-module>`. Finally, run `go mod tidy` to clean up any imports.
## Python
### Pip
You can find the requirements for a package using the pypi json endpoint `https://pypi.org/pypi/<package>/<version>/json`.
## Bash
You can run a bash script without forking it by prefixing it with a dot: `. ./script.sh`

# Traditional ML
## Interpretability
[Interpretability Python Package](https://github.com/interpretml/interpret)
Allows you to train "glass-box" models and explainability for black-box models. **Glass-Box**: ML Models designed for interpretability.
Private data explainability here too - could definitely be useful. [link](https://arxiv.org/pdf/1602.04938.pdf)
LIME learns an interpretable model locally around the prediction of the black box model. This means it is model agnostic, although quite slow when performing this on 1000s of records.
The reasoning behind this is that AI/ML essentially requires a human in the loop for a lot of tasks. If you have an explaination of the prediction - the human is more likely to make the correct decision - i.e accept or reject the models prediction.
It could also be required that the model doesn't exploit certain features (such as clickbait titles on articles, etc as this could hurt user retention) to get higher cross validation accuracy scores.


# WebApps
## Backends
### FastAPI
A python framework for building APIs. Homepage is [here](https://fastapi.tiangolo.com/).
Find a tutorial [here](https://www.travisluong.com/how-to-build-a-full-stack-next-js-fastapi-postgresql-boilerplate-tutorial/) on using FastAPI as a backend for a NextJS app.
## Deploying
### Nginx
Find a tutorial [here](https://www.travisluong.com/how-to-deploy-next-js-fastapi-and-postgresql-with-shell-scripts/) explaining how to deploy a FastAPI / NextJS app using Nginx & PM2 with shell scripts.
### PM2 
PM2 a application manager that ensures the app is online 24/7, you can run it with `pm2 start %command%`. Pretty cool, here are the [docs](https://pm2.keymetrics.io/)
## Frontend
### Tailwind / CSS
Keeping a `fixed` position element to use the parents width (even though it is floating) - `style={{width: 'inherit'}}`
### Chrome Dev Tools
If you enable Chrome dev tools, you can change the size of viewport for testing purposes.
### Axios
A cool npm package which is basically `fetch` but with a wayyy better syntax. Find it [here](https://github.com/axios/axios)
### NextJS
#### URL Handling
Use `import { useSearchParams } from next/navigation` to get the params passed in with the url i.e code in `http://localhost:3000/auth?code=test`.
## OAuth
Tricky concept, with a fastapi / nextjs app. Here are some good resources, [Google OAuth docs](https://developers.google.com/identity/openid-connect/openid-connect#exchangecode), [SSO FastAPI](https://github.com/tomasvotava/fastapi-sso/tree/master), [A good example with Github auth](https://github.com/fuegoio/fastapi-frontend-auth-example/tree/main).
I found the best method to use all 3 in conjunction with the Github auth giving a good general idea of how to go about it.

