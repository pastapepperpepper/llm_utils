"""Qwen3-VL vision tower spec.

A single forward of the vision tower (modeling_qwen3_vl.Qwen3VLVisionModel.forward)
runs patch_embed -> pos_emb -> rope/cu_seqlens construction -> the full blocks loop
(including deepstack) -> merger internally, so registering hooks captures every
sublayer of every layer in one pass.
"""

import os

from core.sublayer_golden.spec import ModelSpec, register, template_taps

# friendly name (= saved file name) -> HF module path template ({i} = block index)
BLOCK_MAP = {
    "norm1":          "blocks.{i}.norm1",           # pre-attn LayerNorm   (= kernel layernorm_0)
    "attn_qkv":       "blocks.{i}.attn.qkv",         # fused qkv (pre-rope)
    "attn_proj":      "blocks.{i}.attn.proj",        # out-projection       (= attn_out)
    "attn":           "blocks.{i}.attn",             # attention output (pre-residual)
    "norm2":          "blocks.{i}.norm2",            # post-attn LayerNorm  (= kernel layernorm_1)
    "mlp_linear_fc1": "blocks.{i}.mlp.linear_fc1",   # FFN up-proj          (= ffn_up)
    "mlp_act":        "blocks.{i}.mlp.act_fn",        # gelu(fc1)
    "mlp_linear_fc2": "blocks.{i}.mlp.linear_fc2",   # FFN down-proj        (= ffn_down)
    "block":          "blocks.{i}",                  # block output (after both residuals)
}

# Layer-independent global modules
GLOBAL_MAP = {
    "merger_norm": "merger.norm",
    "merger_fc1":  "merger.linear_fc1",
    "merger_act":  "merger.act_fn",
    "merger_out":  "merger",
}


def load(model_id):
    import torch
    import transformers

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16)
    return model.model.visual


def prepare(visual, opts):
    """Reuse the sim input (input_tensor.pt) and recompute grid_thw to build the forward thunk."""
    import torch
    import transformers
    from PIL import Image

    device = opts.get("device", "cpu")

    # pixel_values: identical to what the sim consumed (guarantees input parity with HW)
    pixel_values = torch.load(os.path.join(opts["sdk_data_dir"], "input_tensor.pt"))
    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    # grid_thw: recomputed deterministically from the same (512-resized) image
    processor = transformers.Qwen3VLProcessor.from_pretrained(opts["model_id"])
    image = Image.open(opts["image_path"]).resize((opts["image_size"], opts["image_size"]))
    grid_thw = processor.image_processor(image, return_tensors="pt").image_grid_thw.to(device)

    def forward_fn():
        visual(pixel_values, grid_thw=grid_thw)

    return forward_fn, len(visual.blocks)


def make_taps(visual, requested, n_layers):
    taps = template_taps(visual, requested, n_layers, BLOCK_MAP, GLOBAL_MAP)
    # Deepstack mergers exist only at deepstack_visual_indexes (e.g. 5/11/17).
    # Auto-included in full mode ([]). deepstack_merger_list[k] is the k-th deepstack layer.
    if not requested:
        for k, layer in enumerate(visual.deepstack_visual_indexes):
            taps.append((f"deepstack_{layer}", "out", visual.deepstack_merger_list[k]))
    return taps


SPEC = ModelSpec(
    name="qwen3_vl",
    match="qwen3-vl",
    load=load,
    prepare=prepare,
    make_taps=make_taps,
    note="Qwen3-VL vision tower (patch_embed -> blocks -> merger, deepstack at config indexes)",
)

register(SPEC)
