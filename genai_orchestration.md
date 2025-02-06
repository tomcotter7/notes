# GenAI Orchestration

## Agents

### Harrison Chase on Building the Orchestration Layer for AI Agents

Agents are defined as "having a LLM decide the control flow of an appplication" according to Harrison Chase. He also defines cognitive architecture as "the system architecture of the LLM application". This essentially means the flow of information, which is usually specific to your application and domain.

**LangGraph** - looks very useful for AI Agents, we can create loops and build a visual cognitive architecture for the agents to follow.

Interesting ideas for HCI with LLMs:
- The agent can request help when it gets stuck. Rather than a 1 Human Message / 1 AI message, the AI can do a bunch of tasks, and then send a message to an "inbox" where it will wait for Human help.
- We could also have like a review system. The agent would do things, and then wait for a human to review it's output at a specific point, before correcting the mistakes it made (which were labelled by the human)

Testing / Observing LLMs:
- Pairwise Testing. We can compare two different approaches (different LLMs, different graphs, etc) side by side. This is a very good way of interacting with LLMs, especially because they are non-deterministic.

### Andrew Ng - What's next for AI Agentic Workflows?

Reflection - Generate a response, reprompt the LLM to provide feedback on the response, and then generate a new response based on the feedback. This is a good way to get the LLM to improve over time.

You could also have a Generator Agent & a Critic Agent.

### Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models

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


