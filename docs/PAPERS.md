# Paper list

Follow the [six-step roadmap](TECHNOLOGY_ROADMAP.md). Read the named mechanism first; derive and code it before moving on. Links checked on 2026-10-05; record the version you study.

## 1. Tiny decoder

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — attention, masks, multiple heads, and blocks.
- [Subword Units / BPE](https://arxiv.org/abs/1508.07909) — tokenization foundations.

Keep *Understanding Deep Learning*, Chapter 12, as your familiar reference.

## 2. Modern blocks

- [LLaMA](https://arxiv.org/abs/2302.13971) — overview of a modern dense decoder.
- [RMSNorm](https://arxiv.org/abs/1910.07467) — normalization.
- [GLU Variants](https://arxiv.org/abs/2002.05202) — SwiGLU and gated feed-forward layers.
- [RoFormer](https://arxiv.org/abs/2104.09864) — derive rotary position embeddings.

## 3. Efficient inference

Read in this order:

1. [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) — incremental decoding and MQA.
2. [GQA](https://arxiv.org/abs/2305.13245) — shared KV heads and memory accounting.
3. [FlashAttention](https://arxiv.org/abs/2205.14135) — online softmax, tiling, and memory traffic.
4. [PagedAttention](https://arxiv.org/abs/2309.06180) — cache allocation and sharing.
5. [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — MLA, decoupled RoPE, and matrix absorption.

Optional: [FlashAttention-2](https://arxiv.org/abs/2307.08691) for GPU work partitioning.

## 4. DeepSeek architecture and training

- [Switch Transformers](https://arxiv.org/abs/2101.03961) → [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — routing, balancing, and shared experts.
- [Chinchilla](https://arxiv.org/abs/2203.15556) — model size, data, and compute budgets.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — integrate architecture, multi-token prediction, low precision, and training systems.
- Optional: [ZeRO](https://arxiv.org/abs/1910.02054) — distributed training memory.

## 5. Post-training and reasoning

- [InstructGPT](https://arxiv.org/abs/2203.02155) — SFT, rewards, and RLHF pipeline.
- [LoRA](https://arxiv.org/abs/2106.09685), optionally [QLoRA](https://arxiv.org/abs/2305.14314) — efficient adaptation.
- [DPO](https://arxiv.org/abs/2305.18290) — derive the preference loss.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) → [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO, reasoning training, and distillation.
- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) — compare reasoning strategies at equal budgets.

Learn policy-gradient basics before GRPO. Distinguish R1-Zero, R1, and distilled models.

## 6. Optional directions

| Interest | Sources and focus |
|---|---|
| Faster generation | [Speculative Decoding](https://arxiv.org/abs/2211.17192): acceptance and rejection correction |
| Smaller caches | [KIVI](https://arxiv.org/abs/2402.02750): KV quantization |
| Longer context | [YaRN](https://arxiv.org/abs/2309.00071): positional extension |
| Sparse attention | [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556): key selection |
| Retrieval and tools | [RAG](https://arxiv.org/abs/2005.11401), [ReAct](https://arxiv.org/abs/2210.03629) |
| Multimodality | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485): image-to-language interface |
| Recurrent/hybrid models | [Mamba](https://arxiv.org/abs/2312.00752) → [Mamba-2](https://arxiv.org/abs/2405.21060): state-space models |
| Integration studies | [Qwen3](https://arxiv.org/abs/2505.09388), [Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf): compare different design choices |

These are selected case studies, not an exhaustive frontier survey. For each paper, ask **what changed, what it costs, and what the experiments actually establish**. Capture the answer in a [short note](PAPER_NOTE_TEMPLATE.md).
