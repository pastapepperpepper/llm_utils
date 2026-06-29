# =============================================================================
# 전체 공통 설정 (모든 스크립트에서 사용)
# =============================================================================
DEVICE      = "cpu"                          # "cuda" 또는 "cpu"

# =============================================================================
# LLM 공통 설정 (gen_llm_answer.py / tokenizer_utils.py 전용)
#   * sublayer_golden(vision)은 이 값들을 쓰지 않고 VL_* 를 사용한다.
# =============================================================================
LLM_MODEL_ID = "hyper-accel/ci-random-bfloat16-llama3-3b"  # HuggingFace 모델 ID (gen + tokenizer)
TORCH_DTYPE  = "bfloat16"                     # "float16", "bfloat16", "float32" (gen_llm_answer 전용)

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
# tokenizer_utils.py 설정
# =============================================================================
# TOKENIZER_MODE: 0 = ids_to_tokens, 1 = tokens_to_ids
TOKENIZER_MODE       = 1

TOKENIZER_INPUT_IDS  = [1]                      # ids_to_tokens 모드에서 사용
TOKENIZER_INPUT_TOKENS = ["Hello"]  # tokens_to_ids 모드에서 사용

# =============================================================================
# Vision model sublayer golden 추출 설정 (core/sublayer_golden/vision.py)
#   TODO(추후 통합 예정): 현재는 vision 전용 네임스페이스(VL_*)다. sublayer_golden 에
#   LLM(text) 도메인이 추가되면 도메인 공통 구조(LLM_* / VL_* 통합)로 재정리할 것.
# =============================================================================
VL_MODEL_ID     = "Qwen/Qwen3-VL-2B-Instruct"   # vision 모델 ID (spec 자동 매칭)
# sim이 소비한 입력을 그대로 재사용해 HW와 동일 입력 보장 (input_tensor.pt = pixel_values)
VL_SDK_DATA_DIR = "/root/hyperaccel-sdk/tests/rt_simulator_integration/data"
# grid_thw 재계산용 원본 이미지 (sim이 쓰는 것과 동일)
VL_IMAGE_PATH   = "/root/hyperaccel-sdk/tests/simulator_compiler_integration/utils/data/000000002149.jpg"
VL_IMAGE_SIZE   = 512                            # 512/patch16 → 32×32 = 1024 patch
VL_SAVE_DIR     = "golden/sublayer/qwen3_vl"     # tap별 .pt 저장 위치 (.gitignore 됨)
VL_SAVE_DTYPE   = "bfloat16"                      # HW와 동일.

# VL_TAPS: 추출할 (layer, sublayer) 목록.
#   - []  (빈 리스트) → 전체 레이어 × 전체 sublayer + 전역(merger/deepstack) 자동 덤프
#   - (정수 layer_idx, "sublayer 이름") → 해당 block sublayer (예: (0, "mlp_linear_fc1"))
#   - ("global", "sublayer 이름")        → 전역 모듈 (예: ("global", "merger_out"))
# sublayer 이름 카탈로그(block): norm1 / attn_qkv / attn_proj / attn / norm2 /
#   mlp_linear_fc1 / mlp_act / mlp_linear_fc2 / block
# sublayer 이름 카탈로그(global): merger_norm / merger_fc1 / merger_act / merger_out
VL_TAPS = []
