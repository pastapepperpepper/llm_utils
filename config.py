# =============================================================================
# 공통 설정 (모든 스크립트에서 사용)
# =============================================================================
MODEL_ID    = "NousResearch/Llama-2-7b-hf"  # HuggingFace 모델 ID
DEVICE      = "cuda"                          # "cuda" 또는 "cpu"
TORCH_DTYPE = "bfloat16"                       # "float16", "bfloat16", "float32"

# =============================================================================
# gen_llm_answer.py 설정
# =============================================================================
# GEN_INPUT_MODE: 0 = text, 1 = ids
#   - 0: GEN_INPUT_TEXT를 텍스트로 입력
#   - 1: GEN_INPUT_IDS를 토큰 ID로 직접 입력
GEN_INPUT_MODE     = 1       # 입력 모드 선택 (0=text, 1=ids)
GEN_INPUT_TEXT     = "Hello, my name"  # 입력 프롬프트 (GEN_INPUT_MODE=0일 때 사용)
GEN_INPUT_IDS      = [2]     # 토큰 ID 리스트 (GEN_INPUT_MODE=1일 때 사용)
MAX_NEW_TOKENS     = 200      # 생성할 최대 토큰 수

# Sampling 파라미터
# DO_SAMPLE=False 이면 Greedy Search (TEMPERATURE/TOP_K/TOP_P 무시됨)
# DO_SAMPLE=True  이면 아래 파라미터가 적용됨
DO_SAMPLE          = True   # False=Greedy Search, True=샘플링
TEMPERATURE        = 1.0    # 높을수록 다양한 출력 (기본값: 1.0)
REPETITION_PENALTY = 1.0    # 1.0=패널티 없음, >1.0이면 반복 억제
TOP_K              = 1      # 확률 상위 k개 토큰만 후보로 사용
TOP_P              = 0.9    # 누적 확률 p 이하 토큰만 후보로 사용

# =============================================================================
# tokenizer_utils.py 설정
# =============================================================================
# TOKENIZER_MODE: 0 = ids_to_tokens, 1 = tokens_to_ids
TOKENIZER_MODE       = 1

TOKENIZER_INPUT_IDS  = [1]                      # ids_to_tokens 모드에서 사용
TOKENIZER_INPUT_TOKENS = ["Hello"]  # tokens_to_ids 모드에서 사용
