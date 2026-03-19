import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
from datasets import load_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H      = 128
IMG_W      = 256
MAX_LEN    = 150
EMBED_DIM  = 256
NUM_HEADS  = 8
FF_DIM     = 1024
ENC_LAYERS = 2
DEC_LAYERS = 4
DROPOUT    = 0.1
BATCH_SIZE = 32
LR         = 3e-4
EPOCHS     = 50
PATIENCE   = 10
CLIP_NORM  = 1.0
CHECKPOINT = "hmer.pt"

VOCAB = [
    "<pad>", "<sos>", "<eos>", "<unk>",
    "_", "^", "{", "}", "&", "\\\\", " ",
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "0","1","2","3","4","5","6","7","8","9",
    "\\mathbb{A}","\\mathbb{B}","\\mathbb{C}","\\mathbb{D}","\\mathbb{E}",
    "\\mathbb{F}","\\mathbb{G}","\\mathbb{H}","\\mathbb{I}","\\mathbb{J}",
    "\\mathbb{K}","\\mathbb{L}","\\mathbb{M}","\\mathbb{N}","\\mathbb{O}",
    "\\mathbb{P}","\\mathbb{Q}","\\mathbb{R}","\\mathbb{S}","\\mathbb{T}",
    "\\mathbb{U}","\\mathbb{V}","\\mathbb{W}","\\mathbb{X}","\\mathbb{Y}",
    "\\mathbb{Z}",
    ",",";",":","!","?",".","(",")","]","[","\\{","\\}",
    "*","/","+","-","\\_","\\&","\\#","\\%","|","\\backslash",
    "\\alpha","\\beta","\\delta","\\Delta","\\epsilon","\\eta",
    "\\chi","\\gamma","\\Gamma","\\iota","\\kappa","\\lambda",
    "\\Lambda","\\nu","\\mu","\\omega","\\Omega","\\phi","\\Phi",
    "\\pi","\\Pi","\\psi","\\Psi","\\rho","\\sigma","\\Sigma",
    "\\tau","\\theta","\\Theta","\\upsilon","\\Upsilon","\\varphi",
    "\\varpi","\\varsigma","\\vartheta","\\xi","\\Xi","\\zeta",
    "\\frac","\\sqrt","\\prod","\\sum","\\iint","\\int","\\oint",
    "\\hat","\\tilde","\\vec","\\overline","\\underline","\\prime",
    "\\dot","\\not",
    "\\begin{matrix}","\\end{matrix}",
    "\\langle","\\rangle","\\lceil","\\rceil","\\lfloor","\\rfloor","\\|",
    "\\ge","\\gg","\\le","\\ll","<",">",
    "=","\\approx","\\cong","\\equiv","\\ne","\\propto","\\sim","\\simeq",
    "\\in","\\ni","\\notin","\\sqsubseteq","\\subset","\\subseteq",
    "\\subsetneq","\\supset","\\supseteq","\\emptyset",
    "\\times","\\bigcap","\\bigcirc","\\bigcup","\\bigoplus","\\bigvee",
    "\\bigwedge","\\cap","\\cup","\\div","\\mp","\\odot","\\ominus",
    "\\oplus","\\otimes","\\pm","\\vee","\\wedge",
    "\\hookrightarrow","\\leftarrow","\\leftrightarrow","\\Leftrightarrow",
    "\\longrightarrow","\\mapsto","\\rightarrow","\\Rightarrow",
    "\\rightleftharpoons","\\iff",
    "\\bullet","\\cdot","\\circ",
    "\\aleph","\\angle","\\dagger","\\exists","\\forall","\\hbar",
    "\\infty","\\models","\\nabla","\\neg","\\partial","\\perp",
    "\\top","\\triangle","\\triangleleft","\\triangleq","\\vdash",
    "\\Vdash","\\vdots",
]

VOCAB_SIZE    = len(VOCAB)
char_to_index = {c: i for i, c in enumerate(VOCAB)}
index_to_char = {i: c for c, i in char_to_index.items()}

PAD_ID = char_to_index["<pad>"]
SOS_ID = char_to_index["<sos>"]
EOS_ID = char_to_index["<eos>"]
UNK_ID = char_to_index["<unk>"]

_CMD = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")

def tokenize(s: str) -> list:
    tokens = []
    while s:
        if s[0] == "\\":
            tokens.append(_CMD.match(s).group(0))
        else:
            tokens.append(s[0])
        s = s[len(tokens[-1]):]
    return tokens

def encode(label: str) -> list:
    tokens = tokenize(label)
    ids    = [SOS_ID]
    ids   += [char_to_index.get(t, UNK_ID) for t in tokens]
    ids   += [EOS_ID]
    ids    = ids[:MAX_LEN]
    ids   += [PAD_ID] * (MAX_LEN - len(ids))
    return ids

def process_image(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("L")
    img.thumbnail((IMG_W, IMG_H), Image.BILINEAR)
    padded = Image.new("L", (IMG_W, IMG_H), color=255)
    offset_x = (IMG_W - img.width)  // 2
    offset_y = (IMG_H - img.height) // 2
    padded.paste(img, (offset_x, offset_y))
    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    return torch.tensor(arr, dtype=torch.float32)


class MathWritingDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img    = process_image(sample["image"])
        tokens = torch.tensor(encode(sample["label"]), dtype=torch.long)
        return img, tokens[:-1], tokens[1:]

def build_dataloaders():
    ds = load_dataset("deepcopy/MathWriting-human")
    return (
        DataLoader(MathWritingDataset(ds["train"]),      batch_size=BATCH_SIZE,
                   shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(MathWritingDataset(ds["validation"]), batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(MathWritingDataset(ds["test"]),       batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=2, pin_memory=True),
    )


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class CNNBackbone(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        for p in self.features.parameters():
            p.requires_grad = False
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.features(x))

    def unfreeze(self, num_layers=None):
        children = list(self.features.children())
        target   = children if num_layers is None else children[-num_layers:]
        for layer in target:
            for p in layer.parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CNN trainable params: {trainable:,}")


class HMERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn    = CNNBackbone(DROPOUT)
        self.enc_pe = PositionalEncoding(EMBED_DIM, max_len=IMG_H * IMG_W, dropout=DROPOUT)
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS,
                                             enable_nested_tensor=False)
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.dec_pe  = PositionalEncoding(EMBED_DIM, max_len=MAX_LEN, dropout=DROPOUT)
        dec_layer    = nn.TransformerDecoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS)
        self.fc_out  = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def encode(self, images):
        feats = self.cnn(images)
        B, C, H, W = feats.shape
        feats = feats.flatten(2).permute(0, 2, 1)
        feats = self.enc_pe(feats)
        return self.encoder(feats)

    def decode(self, enc_out, dec_inp):
        T        = dec_inp.size(1)
        tgt      = self.dec_pe(self.tok_emb(dec_inp))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=dec_inp.device, dtype=torch.bool)
        pad_mask = dec_inp == PAD_ID
        out      = self.decoder(tgt, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
        return self.fc_out(out)

    def forward(self, images, dec_inp):
        return self.decode(self.encode(images), dec_inp)


def build_scheduler(optimizer, total_steps, warmup_steps=1000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for imgs, dec_inp, dec_tgt in loader:
        imgs    = imgs.to(DEVICE)
        dec_inp = dec_inp.to(DEVICE)
        dec_tgt = dec_tgt.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(imgs, dec_inp)
            loss   = criterion(logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        mask    = dec_tgt != PAD_ID
        correct = (logits.detach().argmax(-1) == dec_tgt) & mask
        total_loss    += loss.item()
        total_correct += correct.sum().item()
        total_tokens  += mask.sum().item()
    return total_loss / len(loader), total_correct / total_tokens


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_tokens = 0, 0, 0
    for imgs, dec_inp, dec_tgt in loader:
        imgs    = imgs.to(DEVICE)
        dec_inp = dec_inp.to(DEVICE)
        dec_tgt = dec_tgt.to(DEVICE)
        logits  = model(imgs, dec_inp)
        loss    = criterion(logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1))
        mask    = dec_tgt != PAD_ID
        correct = (logits.argmax(-1) == dec_tgt) & mask
        total_loss    += loss.item()
        total_correct += correct.sum().item()
        total_tokens  += mask.sum().item()
    return total_loss / len(loader), total_correct / total_tokens


@torch.no_grad()
def greedy_decode(model, pil_img: Image.Image) -> str:
    model.eval()
    img     = process_image(pil_img).unsqueeze(0).to(DEVICE)
    enc_out = model.encode(img)
    dec_inp = torch.tensor([[SOS_ID]], device=DEVICE)
    result  = []
    for _ in range(MAX_LEN):
        logits  = model.decode(enc_out, dec_inp)
        next_id = logits[:, -1, :].argmax(-1).item()
        if next_id == EOS_ID:
            break
        result.append(next_id)
        dec_inp = torch.cat([dec_inp, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return " ".join(index_to_char.get(i, "<unk>") for i in result)


@torch.no_grad()
def beam_search_decode(model, pil_img: Image.Image, beam_size: int = 5) -> str:
    model.eval()
    img     = process_image(pil_img).unsqueeze(0).to(DEVICE)
    enc_out = model.encode(img)
    beams   = [(0.0, [SOS_ID])]
    completed = []
    for _ in range(MAX_LEN):
        candidates = []
        for score, tokens in beams:
            dec_inp  = torch.tensor([tokens], device=DEVICE)
            logits   = model.decode(enc_out, dec_inp)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_ids = log_probs.topk(beam_size)
            for prob, idx in zip(topk_probs[0], topk_ids[0]):
                new_score  = score + prob.item()
                new_tokens = tokens + [idx.item()]
                candidates.append((new_score, new_tokens))
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for score, tokens in candidates[:beam_size * 2]:
            if tokens[-1] == EOS_ID:
                completed.append((score / len(tokens), tokens))
            else:
                beams.append((score, tokens))
        beams = beams[:beam_size]
        if not beams:
            break
    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        best = completed[0][1]
    else:
        best = max(beams, key=lambda x: x[0])[1]
    result = [t for t in best if t not in (SOS_ID, EOS_ID, PAD_ID)]
    return " ".join(index_to_char.get(i, "<unk>") for i in result)


@torch.no_grad()
def expression_accuracy(model, val_split, max_samples=500):
    model.eval()
    correct, total = 0, 0
    for sample in list(val_split)[:max_samples]:
        pred  = greedy_decode(model, sample["image"])
        truth = " ".join(tokenize(sample["label"]))
        if pred.strip() == truth.strip():
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"  {total}/{max_samples} | running acc: {correct/total:.4f}")
    return correct / total


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders()

    model     = HMERModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device=DEVICE.type)

    total_steps = len(train_loader) * EPOCHS
    scheduler   = build_scheduler(optimizer, total_steps, warmup_steps=1000)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                scheduler, criterion, scaler)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss":  val_loss,
            }, CHECKPOINT)
            print("  ✓ saved")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    ckpt = torch.load(CHECKPOINT, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest | loss: {test_loss:.4f} acc: {test_acc:.4f}")
