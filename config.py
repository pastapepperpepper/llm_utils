# =============================================================================
# 공통 설정 (모든 스크립트에서 사용)
# =============================================================================
MODEL_ID    = "hyper-accel/ci-random-bfloat16-llama3-3b"  # HuggingFace 모델 ID
TORCH_DTYPE = "bfloat16"                       # "float16", "bfloat16", "float32"
DEVICE      = "cpu"                          # "cuda" 또는 "cpu"

# =============================================================================
# gen_llm_answer.py 설정
# =============================================================================
# GEN_INPUT_MODE: 0 = text, 1 = ids
#   - 0: GEN_INPUT_TEXT를 텍스트로 입력
#   - 1: GEN_INPUT_IDS를 토큰 ID로 직접 입력
GEN_INPUT_MODE     = 1       # 입력 모드 선택 (0=text, 1=ids)
GEN_INPUT_TEXT     = "Hello, my name"  # 입력 프롬프트 (GEN_INPUT_MODE=0일 때 사용)
GEN_INPUT_IDS      = [2]     # 토큰 ID 리스트 (GEN_INPUT_MODE=1일 때 사용)
MAX_NEW_TOKENS     = 1024      # 생성할 최대 토큰 수
IGNORE_EOS         = True      # True면 EOS 조기 종료 비활성화 (max_new_tokens까지 생성)

# Sampling 파라미터
# DO_SAMPLE=False 이면 Greedy Search (TEMPERATURE/TOP_K/TOP_P 무시됨)
# DO_SAMPLE=True  이면 아래 파라미터가 적용됨
DO_SAMPLE          = True   # False=Greedy Search, True=샘플링
TEMPERATURE        = 1.0    # 높을수록 다양한 출력 (기본값: 1.0)
REPETITION_PENALTY = 1.0    # 1.0=패널티 없음, >1.0이면 반복 억제
TOP_K              = 1      # 확률 상위 k개 토큰만 후보로 사용
TOP_P              = 0.9    # 누적 확률 p 이하 토큰만 후보로 사용

# =============================================================================
# extract_layer_output.py 설정 (Vision encoder 레이어 출력 추출)
# =============================================================================
EXTRACT_MODEL_ID       = "Qwen/Qwen3-VL-2B-Instruct"  # 비전-언어 모델 ID
EXTRACT_IMAGE_PATH     = "000000002149.jpg"  # 입력 이미지 경로
EXTRACT_IMAGE_SIZE     = (512, 512)          # (width, height) 고정 리사이즈. None이면 processor smart_resize 사용
EXTRACT_PROMPT         = ""  # 함께 넣을 텍스트 프롬프트

# EXTRACT_LIST_MODULES: True면 모델의 모든 레이어(모듈) 이름을 출력하고 종료
#   (어떤 이름을 EXTRACT_TARGET 에 넣을지 확인할 때 사용)
EXTRACT_LIST_MODULES   = False

# EXTRACT_TARGET: 출력을 추출할 대상 지정
#   ""               → EXTRACT_BLOCK_IDX 로 지정한 vision block 1개의 내부 모듈 전체
#   "all"            → 모든 vision block 의 최종 출력(block_output)
#   "<module.path>"  → 해당 경로 모듈 + 직속 하위 모듈 출력 (예: "model.visual.blocks.3")
EXTRACT_TARGET         = ""

EXTRACT_BLOCK_IDX      = 0      # EXTRACT_TARGET="" 일 때 추출할 vision block 인덱스 (0 ~ 23)
EXTRACT_INCLUDE_MERGER = False  # True면 merger(patch merger) 출력도 함께 추출 ("" / "all" 모드)
EXTRACT_VISION_ONLY    = True   # True면 vision 타워만 forward (LLM decoder 미실행). language_model 모듈을 볼 땐 False
EXTRACT_ATTN_INTERNALS = True   # True면 attn 내부 q/k/v(pre-rope), q/k(post-rope) 도 캡처 ("" / "all" 모드, eager attention)
EXTRACT_SAVE_PT        = True   # True면 추출한 텐서를 .pt 파일로 저장
EXTRACT_SAVE_SEPARATE  = True   # True면 레이어별로 "이름_index.pt" 개별 파일 저장 / False면 하나의 .pt로 묶어서 저장
EXTRACT_SAVE_DIR       = "layer_outputs"  # 텐서 저장 디렉토리

# =============================================================================
# tokenizer_utils.py 설정
# =============================================================================
# TOKENIZER_MODE: 0 = ids_to_tokens, 1 = tokens_to_ids
TOKENIZER_MODE       = 1

TOKENIZER_INPUT_IDS  = [1]                      # ids_to_tokens 모드에서 사용
TOKENIZER_INPUT_TOKENS = ["Hello"]  # tokens_to_ids 모드에서 사용
