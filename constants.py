# constants.py
"""
Central place for all label mappings so we never make an “off‑by‑one” mistake again.
"""

OFFICIAL_LABELS = {
    "bedroom"      : 1,
    "Coast"        : 2,
    "Forest"       : 3,
    "Highway"      : 4,
    "industrial"   : 5,
    "Insidecity"   : 6,
    "kitchen"      : 7,
    "livingroom"   : 8,
    "Mountain"     : 9,
    "Office"       : 10,
    "OpenCountry"  : 11,
    "store"        : 12,
    "Street"       : 13,
    "Suburb"       : 14,
    "TallBuilding" : 15,
}

# Same keys, but 0‑based values for use inside PyTorch
IDX          = {cls: lbl - 1 for cls, lbl in OFFICIAL_LABELS.items()}
INV_IDX      = {v: k for k, v in IDX.items()}          # 0‑>name
INV_OFFICIAL = {v: k for k, v in OFFICIAL_LABELS.items()}  # 1‑>name
NUM_CLASSES  = len(OFFICIAL_LABELS)                    # = 15
