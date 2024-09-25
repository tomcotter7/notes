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
