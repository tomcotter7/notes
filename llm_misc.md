# Miscellanous LLM Topics (Paper / Blogs)

## Why Language Models Hallucinate (OpenAI)

A team at OpenAI put out [this paper](https://cdn.openai.com/pdf/d04913be-3f6f-4d2b-b283-ff432ef4aaa5/why-language-models-hallucinate.pdf), which argues that these stem from both pre-training & post-training.

In pre-training, they create this reduction that the generation of a valid output is harder that classifying output validity. This is an obvious statement, but by proving it mathematically, they have allowed us to a) have a lower bound for the 'hardness' of this problem & b) allows us to apply the lens of computational learning theory (which has a lot of information on supervised learning) to the idea of generative unsupervised learning.

For post-training, most evaluation benchmarks are typically in the same test space as humans - aka you are not penalized for getting a question wrong. Therefore, during post-training, typically models are rewarded for "confident guessing", because they *might* get the question right. They prove this point by proving that for any test with a binary grading system (where the model can get a score of 0 for an incorrect answer, and 1 for a correct one), the maximum utility of abstaining from answering is 0 (since it will never be right), but the maximum utility of answering is defined by the test taker's "belief distribution" (aka how likely the correct answer is in the models output probability distribution).

To solve this post-training they suggest to bring in a kind of threshold, but this is essentially a penalty for guessing when providing rewards.

Extras;
- They also mention this idea of a *reduction*:
    - "the reduction from supervised learning (binary classification) to unsupervised learning (density estimation or self-supervised learning)"
    - A reduction (in computational theory) from problem A to problem B is an algorithm that solves problem A by using a hypothetical subroutine (an *oracle*) that can solve any instance of problem B in a single step.
    - They show that the generative task of an LLM can be reduced to the supervised learning problem of "is this answer valid to the question" - it's a continual stream of these checks.
- "agnostically". If an algorithm is agnostically learning, then it is making no assumptions about the true nature of the data or the target function. Specifically, it does not assume that a perfect, zero-error model exists within the hypothesis class H.

