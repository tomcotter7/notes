# Applied LLMs / Generative AI 

## Prompt Engineering

### Prompting Techniques

A [paper](https://arxiv.org/pdf/2312.16171v1.pdf) with 26 different prompting techniques has been released. They focus on strategies that help LLMs produce better outputs.

### DSPy

[DSPy](https://huggingface.co/papers/2310.03714)

The authors argue that prompt engineering is brittle, and akin to manually finetuning weights on a classifier. This abstracts LM pipelines as *text transformation graphs*. These can be automatically optimised for better results. From the paper "DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.". Prompts go from 'You are an AI assistant ....' to 'question -> answer'. Teleprompters are powerful optimizers (included in DSPy) that can learn to bootstrap and select effective prompts for the modules of any program. (The "tele-" in the name means "at a distance", i.e., automatic prompting at a distance.)

### F*ck you, show me the prompt

Working with LLM frameworks is annoying & the added complexity of not knowing the prompt or how many API calls are being made is annoying. [This blog post](https://hamel.dev/blog/posts/prompt/) details a library `mitmproxy` that can be used to intercept the API clalls, so you can determine exactly what is going on under the hood. The blog post also describes how current frameworks implement the "magic of LLMs". These include: `guardrails`, `guidance`, `langchain`, `instructor` & `dspy`.

### Chain of Draft: Thinking Faster by Writing Less

[Paper](https://arxiv.org/pdf/2502.18600)

Chain of Draft is a variant of CoT which uses as little as 7.6% of the tokens, whilst still performing as well as CoT does on reasoning based tasks.

An example of a CoD prompt:
```
Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of ther response after a separator ####.
```

Due to the scarcity of CoD-style reasoning patterns in the training data, this technique does not work well without few-shot prompting (as of 03/03/2025).


### MoA

[MoA](https://www.together.ai/blog/together-moa) is **M**ixture **o**f **A**gents. Their hypothesis is that "LLMs tend to generate better responses when presented with outputs from other models, even if these other models are less capable on their own.

They split up the models into *Proposers* & *Aggregators*:
    - Proposers generate initial reference responses.
    - Aggregators synthesize the different response from the proposers into a single, high-quality response.

It is essentially multiple models working in tandem.


## Scaling Test Time Compute

[This blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) by hugging face goes into detail on some of the potential test time compute implementations. In this case, they use *search against a verifier*.

They focus on 3 topics:
- **Best of N**: generate multiple responses per problem, and assigns scores to each candidate answer, typically using a reward model. Then select the answer with the highest reward (or a weightd variant).
- **Beam search**: A systematic search method that explores the solution space, often combined with a process reward mdoel to optimse the sampling and evaluation of the intermediate steps. PRMs provide a sequence of scores, one for each step of the reasoning process.
- **Diverse verifier tree search (DVTS)**: AN extension of beam search that splits the initial beams into independent subtress, which are then expanded gredilly using a PRM. The method improves diversity and overall performance (with larger test-time compute budgets).

**Weighted Best of N**: Aggreate the scores across all identical responses and select the answer with the highest total rewrard. $a_weighted$ = $argmax_{a}\Sigma_{i=1}^{N} \mathcal{I}(a_i = a) \cdot RM(p, s_i)$.

where $RM(p, s_i)$ is the reward model score of the i-th solution to problem p. $\mathcal{I}(a_i = a)$ is an indicator function that is 1 if the i-th solution is equal to the answer a, and 0 otherwise.

**Beam search** is like doing this but at every step (usually indicated by a `\n\n`), and then looking at M (the beam width) different paths.

**DVTS** is like beam search, but each initial 'branch' becomes an independent subtree, generating more potential solutions

## Training Models

### Data Collection

[Here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) is a good resource on a how a high quality training dataset for LLMs can be created.

## Resources

### ML Papers of the Week
- [ML Papers of Week](https://github.com/dair-ai/ML-Papers-of-the-Week)

### Deep Dive into LLMs

- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
This video is by Andrej Karpathy, talking about how LLMs are built.

(Byte Pair Encoding) BPE - transform all the words into bytes, and then look for common byte pairs, and coming them into single tokens. Take a look at [tiktokenizer.vercel.app](tiktokenizer.vercel.app) to do some tokenization demos.

- Pretraining - Background Knowledge.
- SFT - Imitating Being an Expert (A worked problem).
- Reinforcement Learning - Practicing Problems.

RLHF can produce *adversarial results* which means non-sensical results (an infinite number) which produce high rewards, but are not actually good. In verifiable domains you can run RL indefinitely without running into this (provided your reward function is set up well). However, RLHF is gameable so you cannot run it indefinitely.

## Productionizing LLMs

### LinkedIn - Musings on a Generative AI Product

[Article](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)

Most interesting part was getting structured data. In this case, they built a custom yaml parser that handled the specific mistakes the LLM could make. This reduced errors from 10% -> 0.01%.

They found they could easily reach "80% quality" but every subsequent 1% gain after that got harder and harder.

### Evaluation of LLM Applications

[Article](https://hamel.dev/blog/posts/evals/#motivation)

This article detailas the need for a good evaluation framework for LLMs.

Unit Tests -> Assertions the the model is working & producing the correct output.
Humna & Model Eval -> Keep logs of the interactions with the model, and validate that the correct thing is happening. Another LLM can periodically do this as well.
A/B Testing

### Google - AI Ideas

See [this article](https://blog.google/products/google-cloud/gen-ai-business-use-cases/) where Google detail a bunch of interesting AI usecases.


## HCI / UX

### Interacting with Chatbots

*Good tools make it clear how they should be used*. LLM's should not be chatbot interfaces with complex prompts, sliders for different settings should be used, i.e. competency with a topic, how verbose a response. Find an in depth article [here](https://wattenberger.com/thoughts/boo-chatbots).


## GenAI Orchestration

### Agents

#### Harrison Chase on Building the Orchestration Layer for AI Agents

Agents are defined as "having a LLM decide the control flow of an appplication" according to Harrison Chase. He also defines cognitive architecture as "the system architecture of the LLM application". This essentially means the flow of information, which is usually specific to your application and domain.

**LangGraph** - looks very useful for AI Agents, we can create loops and build a visual cognitive architecture for the agents to follow.

Interesting ideas for HCI with LLMs:
- The agent can request help when it gets stuck. Rather than a 1 Human Message / 1 AI message, the AI can do a bunch of tasks, and then send a message to an "inbox" where it will wait for Human help.
- We could also have like a review system. The agent would do things, and then wait for a human to review it's output at a specific point, before correcting the mistakes it made (which were labelled by the human)

Testing / Observing LLMs:
- Pairwise Testing. We can compare two different approaches (different LLMs, different graphs, etc) side by side. This is a very good way of interacting with LLMs, especially because they are non-deterministic.

#### Andrew Ng - What's next for AI Agentic Workflows?

Reflection - Generate a response, reprompt the LLM to provide feedback on the response, and then generate a new response based on the feedback. This is a good way to get the LLM to improve over time.

You could also have a Generator Agent & a Critic Agent.

#### Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models

[This paper](https://arxiv.org/pdf/2310.04406)

In this paper, they expand [ReAct](https://www.promptingguide.ai/techniques/react) prompting into a search over a combinatorial space of possible reasoning and acting steps. The key insight underpinning this is Monte-Carlo Tree Search (MCTS), along with the observation that mayn LM tasks should allow reverting back to earlier steps.

**MCTS**: This algorithm builds a decision tree where every node in the tree is a state and edge is an action. MCTS run for k-episodes; for each episode it starts from the root and iteratively conducts two steps to expand the tree. (1) Expansion, where multiple children states $s$ are explored from the current state p by sampling n actions. (2) Selection, where the children twith the highest UCT (Upper confidence bounds applied to trees) value is selected for expansion in the next iteration.

In the paper, as they explore different routes, they actually backpropagate the rewards upwards through the tree, essentially steering the search algorithm towards the more promising areas of the tree. My understanding would be that the nodes with lot's of useful child nodes would build up a high overall score as you continue to backpropagate.

The full set of stages that they use are as follows:
- Selection
- Expansion
- Evaluation
- Simulation: Sample until a terminal state is reached.
- Backpropagation: Only used if the terminal state is partially successful. As mentioned, we want to guide the model to this area of the tree, but not nessecerilly this terminal state.
- Reflecting: If the terminal state is incorrect, they add some reflection about why to the context for the next branch to be expanded.
