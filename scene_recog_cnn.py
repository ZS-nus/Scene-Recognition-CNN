# scene_recog_cnn.py
"""
CNN pipeline for HW‑3 scene classification (15 classes).

Key features
------------
✓ Places365‑pretrained ResNet‑50 backbone  
✓ Progressive layer unfreezing (head → layer4 → layer3)  
✓ Discriminative learning rates (head > backbone)  
✓ RandAugment data augmentation  
✓ Cosine‑annealing learning‑rate schedule  
✓ Label smoothing 0.05  
✓ Early stopping (patience = 8)  
✓ Live progress bar for both train *and* validation, with running val‑accuracy

Dependencies
------------
torch ≥ 2.0, torchvision ≥ 0.15, timm (only if you swap backbone),
and the file **resnet50_places365.pth.tar** in the project root.
"""

import sys, time, collections, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from model     import ResNet50Places365, get_device
from constants import IDX, INV_IDX, NUM_CLASSES

# ───────────────────────────────────────────────────────────────
# Data transforms
# ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ───────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────
class EarlyStop:
    """Stop training when val‑loss hasn’t improved for `pat` epochs."""
    def __init__(self, pat=8):
        self.pat, self.best, self.cnt = pat, np.inf, 0
    def __call__(self, val_loss):
        if val_loss < self.best:
            self.best, self.cnt = val_loss, 0
        else:
            self.cnt += 1
        return self.cnt >= self.pat


def pb(i: int, n: int, phase: str, extra: str = "") -> None:
    """Tiny progress bar printed in‑place."""
    pct = i / n * 100
    sys.stdout.write(f"\r{phase:>10} [{pct:6.2f}%] {extra}")
    sys.stdout.flush()


# ───────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────
def train(
    train_data_dir: str,
    epochs: int = 20,
    bs: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    val_split: float = 0.2,
    save_path: str = "trained_cnn.pth",
    weights_path: str = "resnet50_places365.pth.tar",
) -> None:
    """Fine‑tune Places365‑ResNet‑50 on the 15‑scene dataset."""
    device = get_device()

    # Dataset / split -------------------------------------------------------
    full = datasets.ImageFolder(train_data_dir, _train_tfms)
    full.class_to_idx = IDX
    full.samples = [(p, IDX[full.classes[i]]) for p, i in full.samples]

    vlen = int(len(full) * val_split)
    train_set, val_set = random_split(full, [len(full) - vlen, vlen])

    tl = DataLoader(train_set, bs, shuffle=True, num_workers=4)
    vl = DataLoader(val_set, bs, shuffle=False, num_workers=4)

    # Model ---------------------------------------------------------------
    model = ResNet50Places365(num_classes=NUM_CLASSES,
                              weights_path=weights_path).to(device)

    params = [
        {"params": model.backbone.fc.parameters(), "lr": lr_head},
        {"params": [p for n, p in model.backbone.named_parameters()
                    if "fc" not in n],              "lr": lr_backbone},
    ]
    opt      = optim.Adam(params, weight_decay=1e-4)
    sched    = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn  = nn.CrossEntropyLoss(label_smoothing=0.05)
    stopper  = EarlyStop(pat=8)

    print(f"\nTraining for {epochs} epochs on {device}…\n")

    # ───── Epoch loop ────────────────────────────────────────────────
    for ep in range(epochs):
        model.progressive_unfreeze(ep)  # head → layer4 → layer3+

        # -------- Train pass -----------------------------------------
        model.train(); train_loss = 0
        for i, (x, y) in enumerate(tl, 1):
            pb(i, len(tl), f"E{ep+1}/{epochs} Train")
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out  = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(tl)

        # -------- Validation pass ------------------------------------
        model.eval(); val_loss = 0; val_cor = 0; val_tot = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(vl, 1):
                out = model(x.to(device))
                val_loss += loss_fn(out, y.to(device)).item()
                preds = out.argmax(1).cpu()
                val_cor += (preds == y).sum().item()
                val_tot += y.size(0)
                acc_live = val_cor / val_tot
                pb(i, len(vl), f"E{ep+1}/{epochs} Val  ",
                   f"acc={acc_live:.3f}")
        val_loss /= len(vl)
        val_acc   = val_cor / val_tot
        sched.step()

        print(f"\n  ▸ Ep {ep+1}: "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        if stopper(val_loss):
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), save_path)
    print(f"\n★ Model checkpoint saved to {save_path}\n")


# ───────────────────────────────────────────────────────────────
# Testing / inference
# ───────────────────────────────────────────────────────────────
def test(
    test_data_dir: str,
    model_path: str = "trained_cnn.pth",
    bs: int = 64,
    tta: bool = True,
) -> None:
    """Evaluate a saved model on a held‑out test folder."""
    device = get_device()

    ds = datasets.ImageFolder(test_data_dir, _test_tfms)
    ds.class_to_idx = IDX
    ds.samples = [(p, IDX[ds.classes[i]]) for p, i in ds.samples]
    dl = DataLoader(ds, bs, shuffle=False, num_workers=4)

    model = ResNet50Places365(NUM_CLASSES, use_places=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    cor, tot = 0, 0
    rights, totals = collections.Counter(), collections.Counter()

    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            if tta:
                out = torch.zeros(x.size(0), NUM_CLASSES, device=device)
                for _ in range(6):
                    x_aug = x if _ == 0 else transforms.RandomResizedCrop(224)(x.cpu()).to(device)
                    out += model(x_aug)
                out /= 6
            else:
                out = model(x)

            preds = out.argmax(1)
            cor  += (preds == y).sum().item()
            tot  += y.size(0)
            for p_i, y_i in zip(preds, y):
                rights[int(y_i)] += int(p_i == y_i)
                totals[int(y_i)] += 1

    print(f"\n► Overall accuracy: {cor / tot:.4f}\n")
    print("Per‑class accuracy:")
    for idx, name in INV_IDX.items():
        hit, ttl = rights[idx], totals[idx]
        print(f"  {name:12}: {hit:3d}/{ttl:3d}  ({hit/ttl:.1%})")
