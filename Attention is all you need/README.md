# Attention Is All You Need — Minimal Implementation

A from-scratch PyTorch implementation of the Transformer architecture introduced in
[Vaswani et al., 2017 — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

The original paper trained on WMT 2014 EN→DE (~4.5M sentence pairs) and WMT 2014 EN→FR
(~36M pairs). This repo follows the same architecture but trains on a much smaller,
similar dataset — **Multi30K** (~29K EN→DE sentence pairs) — so the model can train on a
single GPU (or even CPU, slowly) in minutes-to-hours rather than days.

It is meant as a study implementation, not a production translator: the architecture matches
the paper, but the data is too small to reach paper-level BLEU.

## Layout

| File          | Contents                                                                  |
| ------------- | ------------------------------------------------------------------------- |
| `model.py`    | `LayerNorm`, sinusoidal `PositionEmbeddings`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer` |
| `data.py`     | Multi30K loading, BPE tokenizer training, `TranslationDataset`, padding/causal masks, `build_dataloaders` |
| `train.py`    | Adam + `WarmupScheduler` (the paper's `d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)` schedule), train/eval loop, checkpointing |
| `generate.py` | Greedy autoregressive decoding from a saved checkpoint                    |

## Setup (uv)

```bash
# from this directory
uv sync
```

This creates a `.venv/` and installs PyTorch, `datasets`, and `tokenizers`.

If you want CUDA-specific PyTorch wheels, install torch separately, e.g.:

```bash
uv pip install --torch-backend=cu121 torch
```

## Train

```bash
uv run python train.py --epochs 20 --batch-size 32
```

Default hyperparameters mirror the paper's *base* model (`d_model=512`, 6 layers,
8 heads, `d_ff=2048`, dropout 0.1, warmup 4000). On Multi30K you'll typically see
validation loss plateau within a few dozen epochs.

The script saves `transformer.pt` plus the trained BPE tokenizers
(`tokenizer_en-vocab.json` / `-merges.txt` and the German pair).

## Translate

```bash
uv run python generate.py --text "Two young men are outside near bushes."
```

Greedy decoding is used; swap in beam search if you want closer-to-paper inference.

## Dataset

- [`bentrevett/multi30k`](https://huggingface.co/datasets/bentrevett/multi30k) —
  29K train / 1014 val / 1000 test EN↔DE image-caption sentence pairs.
- Tokenization: byte-level BPE, 8K vocab per language, with `<pad>`, `<sos>`, `<eos>`,
  `<unk>` special tokens.

## Notes vs. the paper

- Architecture is faithful: scaled dot-product multi-head attention, sinusoidal positional
  encodings, residual + LayerNorm, encoder–decoder stacks, Adam (β1=0.9, β2=0.98, ε=1e-9)
  with the noam-style warmup schedule.
- This implementation uses **post-LN** (LayerNorm after residual), like the paper.
- Trained on Multi30K, not WMT — expect lower BLEU but a fully working pipeline.
- Greedy decoding only; no label smoothing, beam search, or checkpoint averaging.
