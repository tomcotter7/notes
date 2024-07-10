# Synthetic Data

## Techniques & Papers

### Scaling Synthetic Data Creation with 1B Personas

[Paper](https://arxiv.org/pdf/2406.20094)

Rather than have 1 LLM produce all the synthetic data, we can create personas and have an LLM act as if it were that persona.

This team have created the 'Persona Hub' which contains these personas. They have created personas via *Text-To-Persona* and *Persona-To-Persona*.

Text-To-Persona: Given some text, create a Persona.
Persona-To-Persona: Once we have a Persona (e.g Nurse), we can create more Personas by creating relations (e.g. Medical Supplier, Patient, Colleage).

Deduplication: They used MinHash and Embedding Based (filtering out similarity of >0.9 in both cases) to remove duplicates.
