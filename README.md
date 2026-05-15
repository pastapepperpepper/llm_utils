# LLM Answer Generator

`rt_simulator_integration` 테스트의 정답지(golden output)를 생성하는 스크립트 모음.

---

## 디렉토리 구조

```
llm_answer/
├── gen_llama2_answer.py   # 정답지 생성 스크립트
├── requirements.txt       # 필요 패키지 목록
└── README.md
```

---

## 환경 설정 및 실행

### 1. 가상환경 생성 (최초 1회)

```bash
# llm_answer/ 내에서 실행
cd llm_answer
python3 -m venv .venv
```

### 2. 가상환경 활성화

```bash
cd llm_answer
source .venv/bin/activate
```

### 3. 패키지 설치 (최초 1회)

```bash
pip install -r requirements.txt
```

### 4. 스크립트 실행

```bash
python3 gen_llama2_answer.py
```

---

## 옵션 변경 방법

`gen_llama2_answer.py` 상단의 변수를 수정한다.

### 모델 변경

```python
model_id = "NousResearch/Llama-2-7b-hf"   # 원하는 HuggingFace 모델 ID로 변경
```

### 입력 프롬프트 변경

```python
input_text_str = "Hello"   # 원하는 프롬프트로 변경
```

### 생성 토큰 수 변경

```python
max_new_tokens = 200   # 생성할 최대 토큰 수
```

### 디바이스 변경 (CPU / CUDA)

기본적으로 CUDA가 있으면 자동으로 GPU를 사용하고, 없으면 CPU를 사용한다.
강제로 지정하려면 아래 라인을 수정한다.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
# CPU 강제: device = "cpu"
# GPU 강제: device = "cuda"
```

> **정답지 대응 관계**
> - `device = "cpu"` 로 실행한 결과 → `rt_simulator_integration` 테스트의 **torch 모드** 정답지
> - `device = "cuda"` 로 실행한 결과 → `rt_simulator_integration` 테스트의 **rtl 모드** 정답지

### 샘플링 파라미터 변경

```python
do_sample          = False  # Greedy Search (기본값, 재현성 보장)
                            # True로 바꾸면 아래 파라미터가 적용됨

temperature        = 1.0    # 높을수록 다양한 출력 (기본값: 1.0)
repetition_penalty = 1.0    # 1.0=패널티 없음, >1.0이면 반복 억제
top_k              = 50     # 확률 상위 k개 토큰만 후보로 사용
top_p              = 1.0    # 누적 확률 p 이하 토큰만 후보로 사용
```

> **정답지 생성 시 주의**: `do_sample=False` (Greedy)로 실행해야 동일한 입력에 대해 항상 동일한 결과가 나옴.
