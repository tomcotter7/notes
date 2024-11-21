# Tokenization

## Customizing Tokenizers

### Getting the most out of your tokenizer for pre-training and domain adaption.

[Paper](https://arxiv.org/pdf/2402.01035v2)

In this paper, they specialize Byte-Pair Encoding (BPE) tokenizers, and found that, when finetuning on more than 50b tokens, they could specialize the tokenizer of a pre-trained LLM to obtain large gains in generation speed and effective context size.

For example, CodeLlama uses the tokenizer from the original LLama model, whereas InCoder uses a custom code tokenizer, which as a result uses 25% less tokens than the LLama tokenizer on average when encoding source code. This is how they increase the effective context size.

*tokenizers should be trained on the data mix that they are expected to see during training/inference*

They note this *if Code Llama had changed its tokenizer before fine-tuning, it would have had a negligible impact on downstream performance, but a large positive impact on compression and inference speed.*
