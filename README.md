# BERT from Scratch

> A complete BERT implementation in PyTorch — embeddings, multi-head self-attention, transformer blocks, and six task heads — in a single file, with no `transformers` dependency.

[`bert.py`](bert.py) builds the whole architecture from `nn.Module` up. Running
it directly constructs a BERT-Base-sized model and prints its structure.

## Run

```bash
pip install torch
python bert.py
```

## Architecture

| Class | Role |
|---|---|
| `BertEmbedding` | Sums token + position + segment embeddings, then dropout |
| `BertMultiHeadAttention` | Fused QKV projection, scaled dot-product attention, optional mask |
| `BertTransformerBlock` | Attention + feed-forward, each wrapped in a residual and LayerNorm |
| `BertModel` | Embedding → `num_layers` stacked blocks |

### Task heads

| Head | Output | Reads |
|---|---|---|
| `BertForMaskedLM` | `(batch, seq, vocab)` | All token positions |
| `BertForNextSentencePrediction` | `(batch, 2)` | `[CLS]` token only |
| `BertForSequenceClassification` | `(batch, num_classes)` | `[CLS]` token only |
| `BertForTokenClassification` | `(batch, seq, num_classes)` | All token positions |
| `BertForQuestionAnswering` | `(batch, seq, 2)` | All positions — start/end logits |

The split is the interesting part: sentence-level tasks read `outputs[:, 0, :]`
— the `[CLS]` position, which attention has aggregated into a whole-sequence
summary — while token-level tasks read every position. Same encoder, different
slice.

## Default configuration

`__main__` instantiates BERT-Base:

| Parameter | Value |
|---|---|
| `vocab_size` | 30522 (WordPiece) |
| `embed_size` | 768 |
| `max_len` | 512 |
| `num_heads` | 12 |
| `num_layers` | 12 |
| `num_segments` | 2 |
| `dropout` | 0.1 |

## Design notes

**Three embeddings, summed.** Token embeddings carry meaning, position
embeddings carry order (learned, not sinusoidal — BERT's choice), and segment
embeddings distinguish sentence A from sentence B for next-sentence prediction.
They're added rather than concatenated, so the model dimension stays at
`embed_size`.

**Fused QKV.** A single `nn.Linear(embed_size, embed_size * 3)` produces
queries, keys, and values in one matmul, then splits them — fewer kernel
launches than three separate projections.

**Bidirectional by default.** Unlike a GPT decoder, there's no causal mask.
Every token attends to every other token in both directions, which is what the
"B" in BERT stands for and why it's trained with masked-language-modelling
rather than next-token prediction.

**GELU and a 4× feed-forward expansion**, matching the paper.

**Post-norm residuals** — `norm(x + sublayer(x))`, as in the original BERT.
Modern implementations usually prefer pre-norm for training stability at depth.

## ⚠️ Known bug — the forward pass crashes

`BertMultiHeadAttention.forward` raises a `RuntimeError` on any real input:

```
permute(sparse_coo): number of dimensions in the tensor input does not match
the length of the desired ordering of dimensions i.e. input.dim() = 5 is not
equal to len(dims) = 4
```

`__main__` only constructs the model and prints it — it never runs a forward
pass — so this isn't visible when you run the file.

**Cause.** After `permute(0, 2, 3, 1, 4)` the tensor is
`(batch, 3, heads, seq, head_size)`. Calling `chunk(3, dim=1)` splits it into
three tensors that each *keep* a singleton dim — shape
`(batch, 1, heads, seq, head_size)`, still 5-D. The later
`out.permute(0, 2, 1, 3)` expects 4-D.

**Fix.** Use `unbind`, which removes the dimension it splits along:

```python
q, k, v = qkv.unbind(1)        # instead of qkv.chunk(3, dim=1)
```

Verified — with that one-line change the forward pass returns `(2, 16, 64)` for
a 2×16 batch, and the `BertForMaskedLM` and `BertForSequenceClassification`
heads return `(2, 16, vocab)` and `(2, num_classes)` as expected.

## Scope

Architecture only. There's no tokenizer, no training loop, no pretraining
objective implementation, and no pretrained weights — this is the model
definition, for understanding how BERT is assembled.
