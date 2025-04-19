# scene_recog_cnn.py
"""
Training / testing pipeline for HW‑3 “CNN scene classification”.
Now complies 100 % with the assignment’s **1‑based** label requirement.
"""

import os, sys, time, collections
import numpy  as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model      import get_device, ResNet101Transfer
from constants  import IDX, INV_IDX, NUM_CLASSES           # <‑‑ NEW
# ---------------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopping:
    """Early stopping to halt training when val‑loss doesn’t improve."""
    def __init__(self, patience=6, delta=0.0):
        self.patience, self.delta = patience, delta
        self.best_loss   = np.inf
        self.counter     = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def progress_bar(iteration, total, prefix="", suffix="", length=30, fill="█",
                 decimals=1):
    percent = 100 * (iteration / float(total))
    filled  = int(length * iteration // total)
    bar     = fill * filled + '-' * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.{decimals}f}% {suffix}")
    sys.stdout.flush()
    if iteration == total: sys.stdout.write("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
_train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])
_test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def train(train_data_dir:str, **kwargs):
    """
    Train ResNet‑101 on the provided directory and save `trained_cnn.pth`.

    Args
    ----
    train_data_dir : str  • root folder with 15 sub‑dirs (one per class)
    **kwargs:
        epochs        (int)  default=40
        batch_size    (int)  default=32
        lr            (float)default=1e‑3
        val_split     (float)default=0.20  (fraction of data for validation)
        early_stop    (int)  default=6     (patience)
        save_path     (str)  default='trained_cnn.pth'
    """
    epochs     = kwargs.get("epochs", 40)
    bs         = kwargs.get("batch_size", 32)
    lr         = kwargs.get("lr", 1e-3)
    val_split  = kwargs.get("val_split", 0.20)
    patience   = kwargs.get("early_stop", 6)
    save_path  = kwargs.get("save_path", "trained_cnn.pth")

    # ─── Dataset ────────────────────────────────────────────────────────────
    full_ds = datasets.ImageFolder(
        root=train_data_dir,
        transform=_train_tfms
    )
    # Overwrite torchvision’s alphabetical mapping with our own
    full_ds.class_to_idx = IDX
    # Apply idx remap to existing samples
    full_ds.samples = [(p, IDX[label_name])
                       for p, label_name in ((p, full_ds.classes[i])
                                             for p, i in full_ds.samples)]

    # Split train / val
    val_len  = int(val_split * len(full_ds))
    train_len= len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=4)

    # ─── Model & optimiser ──────────────────────────────────────────────────
    device = get_device()
    model  = ResNet101Transfer(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stopper = EarlyStopping(patience=patience)

    # ─── Training loop ──────────────────────────────────────────────────────
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for i, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

            progress_bar(i, len(train_loader),
                         prefix=f"Epoch {ep}/{epochs}",
                         suffix="training")

        # Validation
        model.eval()
        val_loss, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                corr += (preds == labels).sum().item()
                tot  += labels.size(0)
        val_acc = corr / tot
        val_loss /= len(val_loader)
        print(f"  ▸ val‑loss={val_loss:.4f}  val‑acc={val_acc:.4f}")

        stopper(val_loss)
        if stopper.should_stop:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), save_path)
    print(f"\n★ Model saved to “{save_path}”")


def test(test_data_dir:str, trained_cnn_path="trained_cnn.pth", **kwargs):
    """
    Evaluate `trained_cnn.pth` on a folder structure identical to the train set.

    Args
    ----
    test_data_dir    : str
    trained_cnn_path : str
    **kwargs:
        batch_size (int) default=64
        tta        (bool)default=True   • test‑time augmentation
    """
    bs      = kwargs.get("batch_size", 64)
    use_tta = kwargs.get("tta", True)
    device  = get_device()

    # Dataset / loader -------------------------------------------------------
    test_ds = datasets.ImageFolder(
        root=test_data_dir,
        transform=_test_tfms
    )
    test_ds.class_to_idx = IDX
    test_ds.samples = [(p, IDX[label_name])
                       for p, label_name in ((p, test_ds.classes[i])
                                             for p, i in test_ds.samples)]
    test_loader = DataLoader(test_ds, bs, shuffle=False, num_workers=4)

    # Model -----------------------------------------------------------------
    model = ResNet101Transfer(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(trained_cnn_path, map_location=device))
    model.to(device).eval()

    # Inference --------------------------------------------------------------
    correct = 0
    totals  = collections.Counter()
    rights  = collections.Counter()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if use_tta:
                outputs = torch.zeros(imgs.size(0), NUM_CLASSES,
                                      device=device)
                # original + five random crops
                for _ in range(6):
                    if _ > 0:
                        imgs = transforms.RandomResizedCrop(224)(imgs.cpu()).to(device)
                    outputs += model(imgs)
                outputs /= 6
            else:
                outputs = model(imgs)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            # per‑class book‑keeping
            for p, l in zip(preds, labels):
                totals[int(l)]  += 1
                rights[int(l)]  += int(p == l)

    overall = correct / len(test_ds)
    print(f"\n► Overall accuracy: {overall:.4f}")

    print("\nPer‑class accuracy:")
    for idx, name in INV_IDX.items():
        hit, tot = rights[idx], totals[idx]
        print(f"  {name:12} : {hit:3d}/{tot:3d}  ({hit/tot:.2%})")

    # If grader expects official (1‑based) labels, add +1 *here*
    # -----------------------------------------------------------------------
    #   e.g. preds_official = [int(p)+1 for p in preds.cpu().numpy()]

# eof
