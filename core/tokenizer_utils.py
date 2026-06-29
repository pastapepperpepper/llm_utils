import sys
from pathlib import Path

# config.py lives at the project root (parent of core/), so add the root to the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enum import IntEnum
from transformers import AutoTokenizer
from config import LLM_MODEL_ID, TOKENIZER_MODE, TOKENIZER_INPUT_IDS, TOKENIZER_INPUT_TOKENS


class TokenizerMode(IntEnum):
    IDS_TO_TOKENS = 0
    TOKENS_TO_IDS = 1


def ids_to_tokens(tokenizer, ids):
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    decoded = tokenizer.decode(ids, skip_special_tokens=False)

    print("\n" + "=" * 50)
    print(f"Input IDs : {ids}")
    print("-" * 50)
    print(f"Tokens    : {tokens}")
    print(f"Decoded   : '{decoded}'")
    print("=" * 50)


def tokens_to_ids(tokenizer, tokens):
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]

    print("\n" + "=" * 50)
    print(f"Input Tokens : {tokens}")
    print("-" * 50)
    print(f"Token IDs    : {ids}")
    print("=" * 50)


def main():
    print(f"Loading tokenizer: {LLM_MODEL_ID}...")
    print("\n" + "=" * 50)
    print("[Common Configuration]")
    print(f"Model ID      : {LLM_MODEL_ID}")
    print("=" * 50)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

    if TOKENIZER_MODE == TokenizerMode.IDS_TO_TOKENS:
        ids_to_tokens(tokenizer, TOKENIZER_INPUT_IDS)
    elif TOKENIZER_MODE == TokenizerMode.TOKENS_TO_IDS:
        tokens_to_ids(tokenizer, TOKENIZER_INPUT_TOKENS)
    else:
        raise ValueError(f"Unknown TOKENIZER_MODE: {TOKENIZER_MODE}. Use 0 (ids_to_tokens) or 1 (tokens_to_ids).")

if __name__ == "__main__":
    main()
