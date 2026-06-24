# LLM Utils

`rt_simulator_integration` 테스트의 정답지(golden output)를 생성·검증하는 스크립트 모음.

---

## 디렉토리 구조

```
llm_utils/
├── config.py                # 모든 스크립트의 공통/개별 설정 (여기만 수정하면 됨)
├── gen_llm_answer.py        # LLM 텍스트 생성 정답지 스크립트
├── tokenizer_utils.py       # 토큰 ↔ ID 변환 유틸
├── extract_layer_output.py  # 비전 인코더(Qwen3-VL) 레이어 출력(.pt) 추출
├── layer_outputs/           # 추출된 텐서(.pt) 저장 위치
├── pyproject.toml           # 프로젝트 메타 + 의존성 (uv source of truth)
├── uv.lock                  # 잠금된 의존성 버전 (재현성 보장, 커밋함)
├── .python-version          # Python 3.10 고정 (hyperaccel-sdk와 일치)
└── README.md
```

> **버전 정책**: 이 스크립트들은 `hyperaccel-sdk`의 `rt_simulator_integration`
> 정답지를 만든다. 정답지가 SDK 런타임과 어긋나지 않도록 Python·`torch`·
> `transformers` 등 주요 패키지를 **SDK와 동일한 버전으로 고정**한다
> (Python 3.10, torch 2.10.0, torchvision 0.25.0, transformers 4.57.1,
> accelerate 1.13.0, pillow 12.0.0, numpy <2.0.0).

---

## 환경 설정 및 실행 (uv)

이 프로젝트는 [uv](https://docs.astral.sh/uv/)로 가상환경(`.venv`)과 의존성을 관리한다.

### 0. uv 설치 (최초 1회, 시스템에 uv가 없을 때만)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 의존성 설치 + 가상환경 생성 (최초 1회 / 의존성 변경 시)

```bash
cd llm_utils
uv sync          # .venv 자동 생성 후 uv.lock 기준으로 패키지 설치
```

> `uv sync`는 `.venv`가 없으면 자동으로 만들고, `pyproject.toml` + `uv.lock`에
> 명시된 정확한 버전을 설치한다. 환경 구성은 `uv sync` 하나로 끝난다.

### 2. 스크립트 실행

가상환경을 직접 활성화하지 않고 `uv run`으로 실행하는 것을 권장한다(자동으로 `.venv` 사용).

```bash
uv run gen_llm_answer.py        # LLM 텍스트 생성
uv run tokenizer_utils.py       # 토큰 ↔ ID 변환
uv run extract_layer_output.py  # 비전 레이어 출력 추출
```

또는 가상환경을 활성화한 뒤 실행해도 된다.

```bash
source .venv/bin/activate
python gen_llm_answer.py
```

### 의존성 추가/변경

```bash
uv add <패키지명>        # pyproject.toml + uv.lock 갱신 후 설치
uv remove <패키지명>
```

### 패키지 인덱스

- 대부분의 의존성은 환경변수 `UV_DEFAULT_INDEX`(사내 `nexus.hyperaccel.ai` 미러)에서
  받는다. 이 변수는 셸/시스템 환경에 이미 설정돼 있어 별도 작업이 필요 없다.
- `torch` / `torchvision` 만은 `pyproject.toml`의 `[tool.uv.sources]` 설정에 따라
  **PyTorch 공식 인덱스(`download.pytorch.org/whl/cu126`)** 에서 받는다. 이는
  hyperaccel-sdk 와 동일한 `torch 2.10.0+cu126`(CUDA 12.6) 빌드를 쓰기 위함이며,
  관련 `nvidia-*-cu12` 라이브러리도 SDK 와 동일한 12.6.x 로 맞춰진다.

### GPU(CUDA) / CPU 선택

CPU/GPU 사용 여부는 설치된 torch가 아니라 **`config.py`의 `DEVICE`** 로 결정한다.

```python
DEVICE = "cuda"   # GPU 사용 (CUDA 빌드 torch + GPU 필요)
DEVICE = "cpu"    # GPU 없는 머신에서 실행
```

> 설치된 `torch`는 CUDA 빌드라 GPU 없는 머신에서도 import는 되지만,
> `DEVICE="cuda"` 로 실행하면 런타임 에러가 난다. GPU가 없으면 `DEVICE="cpu"` 로 둘 것.

---

## 옵션 변경 방법

모든 설정은 **`config.py`** 한 곳에서 관리한다. 스크립트 코드를 직접 고칠 필요 없다.

### 공통 설정 (`config.py` 상단)

```python
MODEL_ID    = "hyper-accel/ci-random-bfloat16-llama3-3b"  # HuggingFace 모델 ID
TORCH_DTYPE = "bfloat16"   # "float16", "bfloat16", "float32"
DEVICE      = "cuda"       # "cuda" 또는 "cpu"
```

> **정답지 대응 관계**
> - `DEVICE = "cpu"` 로 실행한 결과 → `rt_simulator_integration` 테스트의 **torch 모드** 정답지
> - `DEVICE = "cuda"` 로 실행한 결과 → `rt_simulator_integration` 테스트의 **rtl 모드** 정답지

### LLM 생성 설정 (`gen_llm_answer.py`)

```python
GEN_INPUT_MODE     = 1       # 0=텍스트(GEN_INPUT_TEXT), 1=토큰 ID(GEN_INPUT_IDS)
GEN_INPUT_TEXT     = "Hello, my name"
GEN_INPUT_IDS      = [2]
MAX_NEW_TOKENS     = 1024     # 생성할 최대 토큰 수
IGNORE_EOS         = True     # True면 EOS 조기 종료 비활성화(max_new_tokens까지 생성)

DO_SAMPLE          = True     # False=Greedy Search(재현성 보장), True=샘플링
TEMPERATURE        = 1.0
REPETITION_PENALTY = 1.0
TOP_K              = 1
TOP_P              = 0.9
```

> **정답지 생성 시 주의**: `DO_SAMPLE=False` (Greedy)로 실행해야 동일 입력에 항상 동일 결과가 나온다.

### 비전 레이어 추출 설정 (`extract_layer_output.py`)

`config.py`의 `EXTRACT_*` 변수로 모델/이미지/추출 대상을 지정한다. 자세한 내용은
`config.py`의 주석을 참고한다.
