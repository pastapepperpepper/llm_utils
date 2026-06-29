"""ModelSpec (architecture descriptor) + registry + tap-resolution helper.

Supporting a new model = add one ModelSpec under specs/ (engine/driver unchanged).
The sublayer map is data, not code: a dict of "path templates" resolved via get_submodule.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

# Registry: name -> ModelSpec
REGISTRY: dict[str, "ModelSpec"] = {}


@dataclass
class ModelSpec:
    """Describes one architecture (mostly data)."""

    name: str                                   # registry key (e.g. "qwen3_vl")
    match: str                                  # substring-matched against model_id for dispatch (e.g. "qwen3-vl")
    load: Callable[[str], nn.Module]            # model_id -> root module to hook/forward
    prepare: Callable                           # (root, opts) -> (forward_fn, n_layers)
    make_taps: Callable                         # (root, requested, n_layers) -> [(key, mode, module)]
    note: str = field(default="")               # human-readable description (optional)


def register(spec: ModelSpec) -> None:
    REGISTRY[spec.name] = spec


def get_spec(model_id: str) -> ModelSpec:
    """Find the spec matching `model_id`."""
    for spec in REGISTRY.values():
        if spec.match.lower() in model_id.lower():
            return spec
    raise ValueError(
        f"No ModelSpec matches model_id={model_id!r}. Registered: {list(REGISTRY)}"
    )


def template_taps(root, requested, n_layers, block_map, global_map):
    """Build the (key, mode, module) list from path-template dicts.

    block_map: friendly name -> "blocks.{i}.path" template ({i} = layer index)
    global_map: friendly name -> "path" (layer-independent module)
    requested: if empty, capture everything (all layers of every block + globals);
        otherwise a list of (layer, sub). If `layer` is an int it is a block tap;
        if it is "global" the name is looked up in global_map.
    """
    taps = []
    if not requested:
        for i in range(n_layers):
            for friendly, template in block_map.items():
                taps.append((f"{friendly}_{i}", "out", root.get_submodule(template.format(i=i))))
        for friendly, path in global_map.items():
            taps.append((friendly, "out", root.get_submodule(path)))
        return taps

    for layer, sub in requested:
        if layer == "global":
            taps.append((sub, "out", root.get_submodule(global_map[sub])))
        else:
            taps.append((f"{sub}_{layer}", "out", root.get_submodule(block_map[sub].format(i=layer))))
    return taps
