# HMER — Handwritten Mathematical Expression Recognition

A deep learning model that converts images of handwritten math expressions into LaTeX.

## Architecture

```
Image (128×256)
  → ResNet-34 (truncated at layer3, stride 16)
  → 128 spatial tokens (8×16 feature map, 256 channels)
  → Transformer Encoder (2 layers, 8 heads)
  → Transformer Decoder (4 layers, 8 heads)
  → LaTeX token sequence
```

**Why this architecture:**
- ResNet-34 truncated at `layer3` gives stride 16 → 128 spatial tokens at 128×256 input. More tokens than stride-32 models, enough for the transformer encoder to reason about spatial relationships.
- 256 output channels from `layer3` matches `EMBED_DIM` directly — no projection layer needed.
- Transformer encoder adds global self-attention over spatial tokens, important for math's non-local structure (fractions, radicals, superscripts).
- Transformer decoder autoregressively generates LaTeX tokens with cross-attention to encoder output.

## Dataset

Trained on [MathWriting](https://huggingface.co/datasets/deepcopy/MathWriting-human) — a large-scale handwritten math dataset.

## Results

| Metric | Score |
|--------|-------|
| Token Accuracy (test) | ~96% |
| Expression Recognition Rate (test) | ~45.5% |

Expression Recognition Rate (ExpRate) measures exact full-sequence matches — the standard metric for HMER.

## Installation

```bash
pip install -r requirements.txt
```

## Inference

```python
from inference import load_model, greedy_decode, beam_search_decode
from PIL import Image

model = load_model("hmer.pt")
img   = Image.open("your_image.png")

# greedy (faster)
print(greedy_decode(model, img))

# beam search (more accurate, beam_size=5)
print(beam_search_decode(model, img, beam_size=5))
```

## Training

```bash
python model.py
```

Trains from scratch on MathWriting. Checkpoint saved to `hmer.pt`.

To resume from a checkpoint, add before the training loop:

```python
ckpt = torch.load("hmer.pt", weights_only=False)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])
best_val_loss = ckpt["val_loss"]
```

## Fine-tuning the backbone

After initial training, unfreeze the last CNN layers for fine-tuning:

```python
model.cnn.unfreeze(num_layers=2)  # unfreeze layer2 + layer3
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5, weight_decay=1e-4
)
```
