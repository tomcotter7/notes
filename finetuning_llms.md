# Fine-Tuning

## Full Finetunes

### Quiet-STaR

[Quiet-STaR - Paper](https://arxiv.org/pdf/2403.09629.pdf)

This is a process of fine-tuning to follow a CoT type approach, by producing "thoughts" after each token, which are combined with the previous tokens to predict the next token. They call it *Quiet* because they are training the model to think before it speaks, but not neccesarily outputting the "thoughts".

The start of thought and end of thought tokens are `---` as typically in text, this is used as a 'break' or 'pause'.

### Phi-4 Technical Report

[Paper](https://arxiv.org/pdf/2412.08905)

Phi-4 is a 14B parameter model that is fine-tuned on synthetic data from GPT-4.

They used an array of synthetic data techniques to generate this:
- Multi-Agent Prompting.
- Self-Revision Workflows.
- Instruction Reversal.

They carefully curated the data "seeds", to ensure that the generated synthetic data is correct. They actually propose that synthetic data is better than real data. You can prompt the models to generate the exact data you want, and in the format you need.

*For example, web forums are very different in style from LLM interactions. If a fact only appears
in web forum data, the pretrained model will think it is very unlikely to occur in the chats it produces.
Rewriting facts from the web forum into the language style of an LLM makes the facts more accessible
during the LLM chat context of inference.* - sample from the paper.

They also introduce something called Pivotal Token Search (PTS), which basically finds out when the model either generates a really good token for the sequence or a really bad one and tries to filter out the bad ones.

## PEFT

### Adapters

[Adapters](https://huggingface.co/docs/peft/en/conceptual_guides/adapter)

Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of frozen pretrained models to reduce memory usage and speed up training.

It could simply be an extra added layer, or it could be expressing the weight updates $\delta W$ as a low-rank decomposition of the weight matrix.

### Few-Shot PEFT is Better and Cheaper than In-Context Learning

[This paper](https://arxiv.org/pdf/2205.05638) discuss and shows that ICL is a) too expensive when adding lots of examples and b) does not provide enough benefit.

Given some general task (like doing CoT prompting, or Q&A over a document set) is it more efficient to finetune a model to act this way, rather than provide examples in the prompt.

This paper also introduces $IA^{3}$. To make fine-tuning more efficient, IA3 rescales inner activations with learned vectors. These learned vectors are injected in the attention and ff modules in a typical transformer-based architecture. Dealing with learned vectors (as opposed to learned low-rank matrices) keeps the number of trainable parameters smaller.

It seems like these learned vectors are added to the K, V matrices of the attention block. which respectively rescale (via element-wise multiplication) these matrices.

### LoRA - Low-Rank Adaptation of Large Language Models {#lora}

[LoRA - Paper](https://arxiv.org/pdf/2309.00267.pdf)

This paper details `LoRA` a more parameter efficient way of fine-tuning LLMs. When fine-tuning a set of weights, we can represent the new weights $W'$ as $W' = W + \delta W$. Typically, in fine-tuning tasks, we would modify $W$ directly, in order to obtain $W'$. However, the paper proposes that we represent $\delta W$ as a low-rank matrix, i.e. $\delta W = BA$, where $B$ and $A$ are low-rank matrices. 

$A$ and $B$ are also smaller matrices, and their product $AB$ represents a `low-rank` approximation of the original matrix. A rank in matrix terms means "the number of linearly independent columns in the matrix".

By choosing a low-rank approximation, the number of parameters required to train th emodel is reduced. For example, if $W$ is a $d x d$ matrix, then updating $W$ would involve $d^2$ parameters. However, with two low-rank matrices, $A$ and $B$ of sizes $d x r$ and $r x d$ respectively, we only require $2d \cdot r$ parameters, where $r$ is the rank of the matrix.

There is also a good article [here](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6) that explains this in more detail.

If we imagine the W (d x d) and B (d x r) and A (r x d), then the number of parameters goes from $d^{2}$ to $2dr$, which is smaller when ( r << d ).

It seems like we only finetune the low-rank matrices, as the out of the model is as we don't have to update the original during finetuning. During inference, you can merge these low-rank weights to the original weights to reduce latency.

The original pretrained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.

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

### LoRA vs Full Fine-Tuning: An Illusion of Equivalence

[This paper](https://arxiv.org/pdf/2410.21228v1) shows the LoRA finetuning (especially for low-rank) doesn't fully replace full finetuning. LoRA introduces *intruder dimensions* into the SVD of the weight matrices, which are singular vectors (with a high singular value, and therefore important) which are orthgonal to the original singular vectors in the original weight matrix.

However, they show that using higher rank (with rank-stablization) results in less of these intruder dimensions (and therefore better finetuning). They also suggest to scale the $\alpha$ parameter with the rank, using $\alpha$ = $2r$.

Models with these intruder dimensions (whilst performing well on the task they are finetuned for), lose the adaptability to handle different tasks.

## Reinforcement Learning

### RLAIF

[RLAIF - Paper](https://arxiv.org/pdf/2309.00267.pdf)
This paper shows the comparision between RLHF and RLAIF. They seem to comparative results statistically. RLHF is slighlty better, but is more expensive & slower.

They used chain of thought (CoT) reasoning to compare the two summaries. First, they ask the model: 
    - "Given these two summaries - explain which is better".
Then, using the response from this they ask:
    - "Which is better?"

They don't generate a response here, they just look at the probability of the tokens for "1" and "2" for the answer to the first question.

## Reinforcement Finetuning

Reinforcement Finetuning is a technique that uses reinforcement learning (GRPO, PPO, etc...) to teach models to perform specific tasks by giving them a reward for doing that.

### Teaching an AI to Write GPU Code: A Deep Dive into Reinforcement Fine-Tuning


[Predibase's article](https://predibase.com/blog/teaching-ai-to-write-gpu-code-a-deep-dive-into-reinforcement-fine-tuning) demonstrates using Reinforcement Fine-Tuning (RFT) to teach AI models to convert high-level PyTorch code into GPU-optimized Triton code.

Why RFT is Well-Suited for This Task
- No large labeled dataset required
- Output code is directly verifiable
- Large search space with no analytical solution

Reward Function Components
1. Formatting (Partial Credit)
    - Correct output tags
    - Appropriate imports
    - Adherence to Triton best practices
2. Compilation
     - Binary reward: 1 if code compiles and runs, 0 if not
3. Correctness
    - Binary reward: 1 if code passes test suite
    - Anti-reward-hacking measures: Monkey patching kernels with no-ops to verify actual computation
4. Performance
    - Not implemented but recommended
    - Would incentivize more efficient code generation strategies

## Alternatives to Fine-Tuning (That Isn't In-Context Learning)

### Prompt Tuning / Prefix Tuning

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


## Domain Adaptation

### Fine-Tuning LLMs For Domain Adaptation: Exploration of Training Strategies, Scaling, Model Merging and Synergistic Capabilities.

[Paper](https://arxiv.org/pdf/2409.03444)

The authors found that the best way to fine-tune a model for domain adaptation was always to merge the finetune model with some other model (in this case if it was a finetune of Llama3.1-8b\[-Instruct\] then they would merge it with Llama3.1-8b-Instruct.

To perform the merging they used a technique called SLERP, which is a non-linear merging technique. The S stands for Spherical, so by combining weights in this way, the combined models were better than the average of the individual models.
