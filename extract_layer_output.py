import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
from config import (
    EXTRACT_MODEL_ID, DEVICE, TORCH_DTYPE,
    EXTRACT_IMAGE_PATH, EXTRACT_IMAGE_SIZE, EXTRACT_PROMPT,
    EXTRACT_LIST_MODULES, EXTRACT_TARGET, EXTRACT_BLOCK_IDX,
    EXTRACT_INCLUDE_MERGER, EXTRACT_VISION_ONLY, EXTRACT_ATTN_INTERNALS,
    EXTRACT_SAVE_PT, EXTRACT_SAVE_SEPARATE, EXTRACT_SAVE_DIR,
)


def make_forward_hook(store, name):
    def hook(module, inp, out):
        # self_attn/block 등은 tuple을 반환할 수 있으므로 첫 원소를 실제 출력으로 본다
        store[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


def make_pre_hook(store, name):
    def hook(module, inp):
        x = inp[0] if isinstance(inp, tuple) else inp
        store[name] = x.detach()
    return hook


def list_module_names(model):
    """모델의 모든 레이어(모듈) 이름과 타입을 출력한다."""
    print("\n" + "=" * 70)
    print("[All Module Names]  (이 중 하나를 config.EXTRACT_TARGET 에 입력)")
    print("-" * 70)
    for name, mod in model.named_modules():
        if name == "":
            continue
        print(f"  {name:52s} {type(mod).__name__}")
    print("=" * 70)


def register_block_internal_hooks(block, store, idx=None):
    """단일 vision block 내부 모듈들의 출력을 모두 캡처한다.
    idx 가 주어지면 store key 뒤에 "_<idx>" 를 붙여 layer 인덱스를 구분한다."""
    sfx = f"_{idx}" if idx is not None else ""
    return [
        block.register_forward_pre_hook(make_pre_hook(store, f"block_input{sfx}")),
        block.norm1.register_forward_hook(make_forward_hook(store, f"norm1{sfx}")),
        block.attn.qkv.register_forward_hook(make_forward_hook(store, f"attn.qkv{sfx}")),
        block.attn.proj.register_forward_hook(make_forward_hook(store, f"attn.proj{sfx}")),
        block.attn.register_forward_hook(make_forward_hook(store, f"attn{sfx}")),
        block.norm2.register_forward_hook(make_forward_hook(store, f"norm2{sfx}")),
        block.mlp.linear_fc1.register_forward_hook(make_forward_hook(store, f"mlp.linear_fc1{sfx}")),
        block.mlp.linear_fc2.register_forward_hook(make_forward_hook(store, f"mlp.linear_fc2{sfx}")),
        block.mlp.register_forward_hook(make_forward_hook(store, f"mlp{sfx}")),
        block.register_forward_hook(make_forward_hook(store, f"block_output{sfx}")),
    ]


def register_all_blocks_hooks(visual, store):
    """모든 vision block 의 내부 모듈 출력 전부를 layer 인덱스를 붙여 캡처한다."""
    handles = []
    for i, block in enumerate(visual.blocks):
        handles += register_block_internal_hooks(block, store, idx=i)
    return handles


def register_named_module_hooks(model, path, store):
    """임의 경로 모듈 + 직속 하위 모듈의 출력을 캡처한다."""
    module = model.get_submodule(path)  # 잘못된 경로면 AttributeError 발생
    handles = [module.register_forward_hook(make_forward_hook(store, path))]
    for child_name, child in module.named_children():
        handles.append(
            child.register_forward_hook(make_forward_hook(store, f"{path}.{child_name}"))
        )
    return handles


class _ForwardPatch:
    """monkeypatch 한 forward 를 복원하기 위한 핸들 (hook 핸들과 동일하게 .remove() 제공)."""
    def __init__(self, module, orig):
        self.module = module
        self.orig = orig

    def remove(self):
        self.module.forward = self.orig


def patch_attn_internals(block, store, idx=None):
    """block.attn.forward 를 래핑해 q/k/v(pre-rope), q/k(post-rope) 중간값을 캡처한다.
    실제 출력은 원본 forward 에 위임하므로 attention 결과는 변하지 않는다.
    캡처 레이아웃은 modeling 소스(L211-215)와 동일: 각 텐서 (seq, num_heads, head_dim)."""
    attn = block.attn
    orig_forward = attn.forward  # bound method
    sfx = f"_{idx}" if idx is not None else ""

    def wrapper(hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        # 소스와 동일한 reshape/permute/unbind 로 q,k,v 분리 (단순 slice 가 아님에 주의)
        q, k, v = (
            attn.qkv(hidden_states)
            .reshape(seq_length, 3, attn.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        store[f"attn.q_pre_rope{sfx}"] = q.detach()
        store[f"attn.k_pre_rope{sfx}"] = k.detach()
        store[f"attn.v{sfx}"] = v.detach()
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_post, k_post = apply_rotary_pos_emb_vision(q, k, cos, sin)
            store[f"attn.q_post_rope{sfx}"] = q_post.detach()
            store[f"attn.k_post_rope{sfx}"] = k_post.detach()
        # 실제 출력은 원본 forward 가 계산 → attention 결과 동일성 보장
        return orig_forward(hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)

    attn.forward = wrapper
    return _ForwardPatch(attn, orig_forward)


def main():
    dtype = getattr(torch, TORCH_DTYPE)
    target = EXTRACT_TARGET.strip()

    if EXTRACT_LIST_MODULES:
        mode_desc = "LIST MODULE NAMES"
    elif target == "all":
        mode_desc = "ALL vision blocks (internals per layer)"
    elif target == "":
        mode_desc = f"single vision block [{EXTRACT_BLOCK_IDX}] internals"
    else:
        mode_desc = f"named module '{target}'"

    print("\n" + "=" * 70)
    print("[Configuration]")
    print(f"Model ID      : {EXTRACT_MODEL_ID}")
    print(f"Device        : {DEVICE}")
    print(f"Torch Dtype   : {TORCH_DTYPE}")
    print(f"Image         : {EXTRACT_IMAGE_PATH}")
    print(f"Image size    : {EXTRACT_IMAGE_SIZE or 'processor smart_resize'}")
    print(f"Prompt        : {EXTRACT_PROMPT!r}")
    print(f"Mode          : {mode_desc}")
    print("=" * 70)

    print(f"\nLoading model and processor: {EXTRACT_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(EXTRACT_MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        EXTRACT_MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="eager",  # attention 내부값(q/k/v 등)을 실제 forward 경로에서 캡처하기 위함
    )
    model.to(DEVICE)
    model.eval()

    # 모듈 이름만 출력하고 종료
    if EXTRACT_LIST_MODULES:
        list_module_names(model)
        return

    visual = model.model.visual

    # 대상에 따라 hook 등록 (patches: attention 내부값 캡처용 forward 패치)
    store = {}
    patches = []
    if target == "all":
        handles = register_all_blocks_hooks(visual, store)
        if EXTRACT_ATTN_INTERNALS:
            patches = [patch_attn_internals(b, store, idx=i) for i, b in enumerate(visual.blocks)]
        if EXTRACT_INCLUDE_MERGER:
            handles.append(visual.merger.register_forward_hook(make_forward_hook(store, "merger")))
    elif target == "":
        num_blocks = len(visual.blocks)
        if not (0 <= EXTRACT_BLOCK_IDX < num_blocks):
            raise ValueError(
                f"EXTRACT_BLOCK_IDX={EXTRACT_BLOCK_IDX} 범위 초과. 0 ~ {num_blocks - 1} 사이여야 함."
            )
        handles = register_block_internal_hooks(
            visual.blocks[EXTRACT_BLOCK_IDX], store, idx=EXTRACT_BLOCK_IDX
        )
        if EXTRACT_ATTN_INTERNALS:
            patches = [patch_attn_internals(visual.blocks[EXTRACT_BLOCK_IDX], store, idx=EXTRACT_BLOCK_IDX)]
        if EXTRACT_INCLUDE_MERGER:
            handles.append(visual.merger.register_forward_hook(make_forward_hook(store, "merger")))
    else:
        handles = register_named_module_hooks(model, target, store)

    # 이미지 + 텍스트 입력 구성 (비전 forward가 돌려면 이미지가 반드시 필요)
    image = Image.open(EXTRACT_IMAGE_PATH).convert("RGB")
    processor_kwargs = {}
    if EXTRACT_IMAGE_SIZE is not None:
        image = image.resize(EXTRACT_IMAGE_SIZE, Image.Resampling.BILINEAR)
        processor_kwargs["do_resize"] = False
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": EXTRACT_PROMPT},
        ],
    }]
    apply_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if processor_kwargs:
        apply_kwargs["processor_kwargs"] = processor_kwargs
    inputs = processor.apply_chat_template(messages, **apply_kwargs).to(DEVICE)

    with torch.no_grad():
        if EXTRACT_VISION_ONLY:
            # vision 타워만 forward → LLM decoder 미실행
            if "pixel_values" not in inputs:
                raise ValueError("이미지 입력이 없어 vision-only forward 불가 (EXTRACT_IMAGE_PATH 확인)")
            pixel_values = inputs["pixel_values"].to(dtype=dtype)
            visual(pixel_values, grid_thw=inputs["image_grid_thw"])
        else:
            # 전체 forward (language_model 모듈 추출 시 필요)
            model(**inputs)

    for h in handles:
        h.remove()
    for p in patches:
        p.remove()

    # 결과 출력
    print("\n" + "=" * 70)
    print(f"[Outputs]  mode = {mode_desc}")
    print("-" * 70)
    print(f"{'name':30s} {'shape':22s} {'mean':>9s} {'std':>9s}")
    print("-" * 70)
    for name, t in store.items():
        tf = t.float()
        print(f"{name:30s} {str(tuple(t.shape)):22s} "
              f"{tf.mean().item():9.4f} {tf.std().item():9.4f}")
    print("=" * 70)

    # 텐서 저장 (옵션). store key 에 이미 layer 인덱스가 포함돼 있다 (예: norm1_1, block_output_23)
    if EXTRACT_SAVE_PT:
        os.makedirs(EXTRACT_SAVE_DIR, exist_ok=True)

        if EXTRACT_SAVE_SEPARATE:
            # 레이어(block)마다 새 파일 생성: "<이름>_<index>.pt"
            print(f"\nSaved {len(store)} files to {EXTRACT_SAVE_DIR}/ :")
            for name, t in store.items():
                fname = name.replace(".", "_")           # attn.qkv_1 -> attn_qkv_1
                save_path = os.path.join(EXTRACT_SAVE_DIR, f"{fname}.pt")
                torch.save(t.cpu(), save_path)
            print(f"  e.g. {os.path.join(EXTRACT_SAVE_DIR, list(store)[0].replace('.', '_'))}.pt ...")
        else:
            # 하나의 .pt 로 묶어서 저장
            tag = "all" if target == "all" else (target.replace(".", "_") if target else f"block{EXTRACT_BLOCK_IDX}")
            save_path = os.path.join(EXTRACT_SAVE_DIR, f"vision_{tag}_outputs.pt")
            torch.save({k: v.cpu() for k, v in store.items()}, save_path)
            print(f"\nSaved tensors to: {save_path}")


if __name__ == "__main__":
    main()
