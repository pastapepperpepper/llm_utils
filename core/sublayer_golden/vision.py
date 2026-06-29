"""Entry point for vision-domain sublayer golden extraction.

Reads only the VL_* settings from config. Looks up the spec by VL_MODEL_ID
(get_spec), builds the inputs, and runs a single forward through the hook engine,
saving each sublayer's golden tensor.

Run:  uv run core/sublayer_golden/vision.py
"""

import sys
from pathlib import Path

# config.py lives at the project root (two levels above core/sublayer_golden/),
# so add the root to the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DEVICE,
    VL_IMAGE_PATH, VL_IMAGE_SIZE, VL_MODEL_ID, VL_SAVE_DIR, VL_SAVE_DTYPE,
    VL_SDK_DATA_DIR, VL_TAPS,
)
from core.sublayer_golden.engine import capture
from core.sublayer_golden.spec import get_spec
import core.sublayer_golden.specs  # noqa: F401  (populate the spec registry)


def main():
    spec = get_spec(VL_MODEL_ID)
    print("=" * 60)
    print(f"[sublayer_golden] model = {VL_MODEL_ID}")
    print(f"[sublayer_golden] spec  = {spec.name}  device = {DEVICE}")
    print("=" * 60)

    print(f"Loading {VL_MODEL_ID} ...")
    root = spec.load(VL_MODEL_ID)
    root.to(DEVICE).eval()

    forward_fn, n_layers = spec.prepare(
        root,
        {
            "model_id": VL_MODEL_ID,
            "sdk_data_dir": VL_SDK_DATA_DIR,
            "image_path": VL_IMAGE_PATH,
            "image_size": VL_IMAGE_SIZE,
            "device": DEVICE,
        },
    )

    taps = spec.make_taps(root, VL_TAPS, n_layers)
    mode = "ALL" if not VL_TAPS else f"{len(VL_TAPS)} requested"
    print(f"n_layers = {n_layers} | taps = {len(taps)} ({mode}) | save_dtype = {VL_SAVE_DTYPE}")
    print(f"Running forward and capturing -> {VL_SAVE_DIR}/")

    result = capture(forward_fn, taps, save_dir=VL_SAVE_DIR, save_dtype=VL_SAVE_DTYPE)

    print("-" * 60)
    for key in sorted(result):
        tensor = result[key]
        print(f"  {key:24s} {str(tuple(tensor.shape)):18s} {tensor.dtype}")
    print("-" * 60)
    print(f"[sublayer_golden] saved {len(result)} tensors to {VL_SAVE_DIR}/")


if __name__ == "__main__":
    main()
