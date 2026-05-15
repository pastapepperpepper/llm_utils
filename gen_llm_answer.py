import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    MODEL_ID, DEVICE, TORCH_DTYPE,
    GEN_INPUT_MODE, GEN_INPUT_TEXT, GEN_INPUT_IDS, MAX_NEW_TOKENS,
    DO_SAMPLE, TEMPERATURE, REPETITION_PENALTY, TOP_K, TOP_P,
)


def main():
    dtype = getattr(torch, TORCH_DTYPE)

    print(f"Loading LLM model and tokenizer: {MODEL_ID}...")
    print("\n" + "=" * 50)
    print("[Common Configuration]")
    print(f"Model ID      : {MODEL_ID}")
    print(f"Device        : {DEVICE}")
    print(f"Torch Dtype   : {TORCH_DTYPE}")
    print("=" * 50)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    # Pad 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    # 입력 모드에 따라 input_ids와 attention_mask 구성
    if GEN_INPUT_MODE == 0:  # text mode
        print(f"Input mode: TEXT")
        inputs = tokenizer(GEN_INPUT_TEXT, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        display_input = GEN_INPUT_TEXT
    elif GEN_INPUT_MODE == 1:  # ids mode
        print(f"Input mode: TOKEN IDs")
        input_ids = torch.tensor([GEN_INPUT_IDS], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        display_input = f"{GEN_INPUT_IDS}"
    else:
        raise ValueError(f"Unknown GEN_INPUT_MODE: {GEN_INPUT_MODE}. Use 0 (text) or 1 (ids).")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            top_k=TOP_K,
            top_p=TOP_P,
        )

    generated_ids = output_ids[0][len(input_ids[0]):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Input 텍스트 디코딩
    input_text_decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

    print("\n" + "=" * 50)
    print("[Generation Result]")
    mode_name = "TEXT" if GEN_INPUT_MODE == 0 else "IDs"
    print(f"Mode          : {mode_name}")
    print(f"Input ({mode_name})    : {display_input}")
    print("-" * 50)
    print(f"Input IDs     : {input_ids[0].tolist()}")
    print(f"Input Text    : {input_text_decoded}")
    print("-" * 50)
    print(f"Output IDs    : {generated_ids}")
    print(f"Output Text   : {generated_text}")
    print("=" * 50)

if __name__ == "__main__":
    main()
