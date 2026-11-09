# VLLMs (Vision Language Models)

### Qwen3.5

The architecture of Qwen3.5 is different to older versions of Qwen, since the vision + text parts of the model are effectively combined right at the start of the model architecture.

```mermaid
graph TD
  subgraph Vision["Vision Encoder (27 blocks, d=1152)"]
      direction TB
      PE["PatchEmbed<br/>Conv3d(3→1152, k=2×16×16)"]
      POS["Position Embed<br/>Embedding(2304, 1152)"]
      RoPE_V["Rotary Pos Emb"]
      VB["27× VisionBlock<br/>LayerNorm → Attention(1152→3456→1152)<br/>LayerNorm → MLP(1152→4304→1152, GELU)"]
      PM["PatchMerger (spatial_merge=2)<br/>LayerNorm → Linear(4608→4608)<br/>GELU → Linear(4608→4096)"]

      PE --> POS --> VB
      RoPE_V -.-> VB
      VB --> PM
  end

  subgraph LLM["Language Model (32 layers, d=4096)"]
      direction TB
      EMB["Embedding(248320, 4096)"]

      subgraph Pattern["Repeated 8× (layers 0-31)"]
          direction TB
          LIN["3× GatedDeltaNet Layer<br/>SiLU → Conv1d(8192, k=4) →
RMSNormGated<br/>in_proj_qkv(4096→8192)<br/>in_proj_z(4096→4096)<br/>in_proj_a/b(4096→32)<br/>out_proj(4096→4096)"]
          FA["1× Full Attention Layer<br/>GQA: q(4096→8192) k,v(4096→1024)<br/>32 Q heads, 4 KV heads,
head_dim=256<br/>QK-Norm(RMSNorm)"]
          MLP_L["MLP (both layer types)<br/>gate_proj(4096→12288) ⊙ up_proj(4096→12288)<br/>SiLU → down_proj(12288→4096)"]

          LIN --> FA
      end

      NORM["RMSNorm(4096)"]
      EMB --> Pattern --> NORM
  end

  HEAD["LM Head<br/>Linear(4096→248320)"]

  PM -->|"image tokens<br/>(400 per 640×640 img)"| EMB
  NORM --> HEAD

  style Vision fill:#1a1a2e,stroke:#e94560,color:#fff
  style LLM fill:#1a1a2e,stroke:#0f3460,color:#fff
  style Pattern fill:#16213e,stroke:#533483,color:#fff
```

We can dive into the layer types:

- Vision Encoder
    - This uses standard ViT blocks, which is Attention -> MLP over the image patches (produced by RoPE). These are repeated 27 times.
    - The "PatchMerger" does a "spatial merge" which concatenates groups of 2x2 spatially adjacent patches, and then are projected down to 4096 through a couple linear layers.
- The output vision embedding is combined with the text embedding, with the following pattern:
    - 3 layers of GatedDeltaNet - "LinearAttention", a novel architectuer which provides faster attention
    - 1 layer of full GQA, which is more precise.
- Finally, you have the LMHead which predicts the next token.

The GatedDeltaNet is interesting, essentiall you have the following:
    - The linear attention means keeping a fixed size matrix, and because it's fixed size, having to overwrite the data within it. This means you don't get a perfect lookup.
    - The Delta part means that rather than just adding the new value to the previous, the model adds the "corrections" e.g new value - old value, which means it performs clean overwrites, rather than adding noise.
    - Finally, the gated part is similar to an RNN, and controls how much of the old state should be forgotten, and what is the strength and which new states should be written.

The VisionEncoder itself  is a Conv3d PatchEmbed. The reason that we use 3d here instead of 2d is because this model supports video, so it actually produces a token per 16x16 pixel square for each 2 frames in the video.

## Videos / Talks

### Computer Vision in the Age of LLMs

[This is a talk](https://www.youtube.com/watch?v=kxO6ARgI_SU&t=974s) by Lucas Beyer.

>[!TODO]
> - [ ] Make notes on this talk.
