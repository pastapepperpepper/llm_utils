"""Model-agnostic hook engine.

Registers forward hooks on modules addressed by name and runs the model forward
once to capture each point's intermediate output (the sublayer golden). This file
knows nothing about any specific model -- per-model knowledge lives in the
ModelSpec objects under specs/.
"""

import os

import torch


def capture(forward_fn, taps, *, save_dir=None, save_dtype="bfloat16"):
    """Capture the intermediate tensors at `taps` with a single forward pass.

    Args:
        forward_fn: No-arg thunk that runs the model forward exactly once.
        taps: List of (key, mode, module) tuples.
            - mode="out" -> capture the module's output (register_forward_hook)
            - mode="pre" -> capture the module's input (register_forward_pre_hook)
        save_dir: If given, each tensor is saved to {save_dir}/{key}.pt.
        save_dtype: dtype string used when saving ("bfloat16" / "float32" / ...).

    Returns:
        dict {key: cpu tensor}.
    """
    store = {}
    handles = []

    def out_hook(key):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                store[key] = tensor.detach()
        return hook

    def pre_hook(key):
        def hook(_module, inputs):
            tensor = inputs[0] if isinstance(inputs, tuple) else inputs
            if isinstance(tensor, torch.Tensor):
                store[key] = tensor.detach()
        return hook

    for key, mode, module in taps:
        if mode == "pre":
            handles.append(module.register_forward_pre_hook(pre_hook(key)))
        else:
            handles.append(module.register_forward_hook(out_hook(key)))

    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for handle in handles:
            handle.remove()

    dtype = getattr(torch, save_dtype)
    result = {key: tensor.to(dtype).cpu() for key, tensor in store.items()}

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for key, tensor in result.items():
            torch.save(tensor, os.path.join(save_dir, f"{key}.pt"))

    return result
