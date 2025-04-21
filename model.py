# model.py
import os, torch
import torch.nn as nn
import torchvision.models as models
import warnings, torch
import urllib.request
import sys

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=UserWarning,
    module=r"torch\.serialization"
)


# ──────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Select the best available backend.

    Priority
    --------
    1. CUDA (NVIDIA GPUs)
    2. Apple M‑Series / Metal Performance Shaders (MPS)
    3. CPU
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"--> Using CUDA   : {name}")
        return torch.device("cuda")

    # MPS is available only in PyTorch ≥ 1.12 on macOS 12.3+
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok is not None and torch.backends.mps.is_available():
        print("--> Using MPS    : Apple Metal backend")
        return torch.device("mps")

    print("--> Using CPU    : no GPU backend found")
    return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────
def download_places365_weights(weights_path="resnet50_places365.pth.tar"):
    """
    Download Places365 weights if they don't exist locally.
    Returns True if weights are available (either existed or downloaded successfully).
    """
    if os.path.exists(weights_path):
        return True
    
    print(f"Places365 weights not found. Downloading to {weights_path}...")
    url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
    try:
        def report_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}% complete")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(url, weights_path, reporthook=report_progress)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading Places365 weights: {e}")
        return False

# ──────────────────────────────────────────────────────────────────

class ResNet50Places365(nn.Module):
    """
    ResNet‑50 backbone with optional Places365 weights and a *single* FC head.
    Provides utilities for progressive layer unfreezing.
    """
    def __init__(self, num_classes=15,
                 weights_path="resnet50_places365.pth.tar",
                 use_places=True):
        super().__init__()

        # Base architecture
        self.backbone = models.resnet50(weights=None)
        if use_places:
            # Try to download weights if they don't exist
            weights_available = download_places365_weights(weights_path)
            
            if weights_available:
                print(f"Loading Places365 weights from {weights_path}")
                try:
                    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
                except TypeError:
                    ckpt = torch.load(weights_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                # Temp fc to 365 to match ckpt
                in_f = self.backbone.fc.in_features
                self.backbone.fc = nn.Linear(in_f, 365)
                self.backbone.load_state_dict(state, strict=False)
            else:
                print("❕ Places weights missing ‑ using random init.")
        else:
            print("❕ Places weights disabled ‑ using random init.")

        # Replace classifier (head) – smaller, harder to over‑fit
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

        # Freeze everything by default
        self.freeze_backbone()

    # ── forward ────────────────────────────────────────────────
    def forward(self, x):   return self.backbone(x)

    # ── freezing helpers ───────────────────────────────────────
    def freeze_backbone(self):
        for n, p in self.backbone.named_parameters():
            if "fc" not in n: p.requires_grad = False

    def unfreeze_layer_group(self, group="layer4"):
        """Unfreeze backbone layers *from* the specified group upward."""
        unfreeze = False
        for n, p in self.backbone.named_parameters():
            if group in n: unfreeze = True
            if unfreeze and "fc" not in n: p.requires_grad = True

    def progressive_unfreeze(self, epoch, schedule=(0, 3, 6)):
        """
        Epoch‑based schedule:
          0‑2  : train head only
          3‑5  : unfreeze layer4
          6‑∞  : unfreeze layer3+
        """
        if epoch < schedule[1]:
            self.freeze_backbone()
        elif epoch < schedule[2]:
            self.freeze_backbone()
            self.unfreeze_layer_group("layer4")
        else:
            # unfreeze layer3,4
            self.freeze_backbone()
            self.unfreeze_layer_group("layer3")
