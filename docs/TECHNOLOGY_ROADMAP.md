# Technology roadmap

Learn one mechanism at a time, then use DeepSeek papers to understand how the pieces fit together. Everything below is planned; the starting point is the [single-head attention notebook](../attention_mechanism/transformer.ipynb).

## Six-step plan

| Step | Study | Why | Coding checkpoint |
|---|---|---|---|
| 1. Tiny enocder + decoder | Batching, masks, multi-head attention, residual blocks, embeddings, tokenization, next-token loss, sampling | Connect attention to a working language model | Train a tiny decoder; verify future tokens cannot affect earlier logits |
| 2. Modern blocks | RoPE, RMSNorm, SwiGLU | Bridge the original Transformer to modern designs | Implement each separately; explain shapes and verify RoPE's relative-position identity |
| 3. Efficient inference | KV caching → MQA/GQA → FlashAttention → paging → MLA | Understand distinct memory and computation bottlenecks | Match cached/full logits; check tiled attention and latent/reconstructed MLA against references |
| 4. DeepSeek-V2/V3 | MoE routing, shared experts, balancing, multi-token prediction, low precision; data and scaling | See how architecture and training choices interact | Build a tiny routed MLP; account for active computation versus total weight memory |
| 5. Post-training and R1 | SFT, LoRA, preference learning, DPO, policy gradients, GRPO, distillation | Understand instruction following and reasoning training | Implement small loss examples, then a toy reward task; evaluate held-out answers |
| 6. Explore | Speculative decoding, KV quantization, sparse attention, long context, recurrent hybrids, retrieval/tools, multimodality | Develop informed interests in emerging directions | Choose one branch and compare quality, memory, and runtime with a baseline |

Read [the corresponding papers](PAPERS.md) alongside each step. Start V2's MLA section after step 3, V3 after MoE/training foundations, and R1 after DeepSeekMath. BERT and translation are optional comparison branches.

## Keep the optimization targets separate

| Technique | Main target |
|---|---|
| KV cache | Avoid recomputing past keys and values |
| MQA/GQA | Store fewer KV heads |
| MLA | Cache a compressed latent representation |
| KV quantization | Store fewer bits per element |
| Paging and prefix sharing | Reduce allocation waste and duplicate storage |
| FlashAttention | Reduce temporary storage and memory traffic; dense arithmetic remains quadratic |
| Windowing/eviction | Retain less history, changing available context |
| Sparse attention | Visit fewer keys; persistent cache need not shrink |
| Speculative decoding | Produce more tokens per target-model verification |

For ordinary equal-width K/V heads:

`KV bytes = 2 × layers × batch × cached_tokens × KV_heads × head_width × bytes_per_element`

Account for weights, temporary memory, and metadata separately. Compare exact implementations numerically; evaluate quality changes for approximations. CPU exercises establish correctness, while GPU speed claims need suitable hardware and measurements.

## Working method

**Read → derive → code → test → summarize.**

For each mechanism, keep equations and tensor shapes, a small explicit implementation, an independent check, one failure case, and a short tradeoff note. Use the [note template](PAPER_NOTE_TEMPLATE.md). Build the library yourself in small pieces; consult authors' code afterward.

For frontier reading, ask: what bottleneck changed, what budget was held fixed, and what evidence supports the improvement? Promising study directions include memory efficiency, adaptive reasoning, sparse/hybrid models, and better training data. Treat these as research questions, not predictions of a winner.

**Next action:** annotate the current notebook's shapes, then add and test a causal mask. Its `num_att_heads` argument currently changes projection width; it does not create multiple heads.
