# Synthetic Data
## Techniques & Papers

### Scaling Synthetic Data Creation with 1B Personas

[Paper](https://arxiv.org/pdf/2406.20094)

Rather than have 1 LLM produce all the synthetic data, we can create personas and have an LLM act as if it were that persona.

This team have created the 'Persona Hub' which contains these personas. They have created personas via *Text-To-Persona* and *Persona-To-Persona*.

Text-To-Persona: Given some text, create a Persona.
Persona-To-Persona: Once we have a Persona (e.g Nurse), we can create more Personas by creating relations (e.g. Medical Supplier, Patient, Colleage).

Deduplication: They used MinHash and Embedding Based (filtering out similarity of >0.9 in both cases) to remove duplicates.

### Orca AgentInstruct: Towards Generative Teaching w/ Agentic Flows

[Paper](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/AgentInstruct.pdf)

This concise summary of AgentInstruct is:
 - Assemble a collection of raw seeds (e.g. textbook chapters, web articles, code snippets)
 - for each seed, do:
    - Transform the seed with the aid of one or more content transformation Agents.
    - Route it through a series of instruction creation Agents to create a diverse set of instructions.
    - Utilize another group of Refinement Agents to iteratively refine the complexity and quality of the seed instructions.
 - end for

 This is more focused on generating "generic" synthetic data for finetuning.

### Promptagator

[Paper](https://arxiv.org/pdf/2209.11755)

This paper is an implementation of using encoder-decoder (generative LLMs) to generate synthetic data and then finetune cross-encoders / bi-encoders on that data.

They suggest that adding examples to the prompt resulted in much better data quality, but this was also with FLAN (137B), whereas now we have GPT-4 (1760B). I don't think we neccessarily need few-shot prompting anymore.

### InPars

[Paper](https://arxiv.org/pdf/2301.01820)

This paper is a way of generating synthetic data for a domain specific adaption of for monoT5. The [Github](https://github.com/zetaalphavector/InPars/tree/master) is available.
