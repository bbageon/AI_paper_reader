# Paper Tutor — AI 논문 리더 / 라이브러리

PDF 논문을 업로드해 **PDF 뷰어 + AI 튜터 + 노트**를 한 화면에서 사용하는 로컬 웹앱입니다.
모델은 로컬 Ollama를 사용하므로 API 키가 필요 없습니다.

## 주요 기능

- **3-pane UI (리사이즈 가능)** — 왼쪽: 폴더/논문 익스플로러, 가운데: PDF.js 뷰어, 오른쪽: 채팅/노트. 패널 경계 드래그로 너비 조절, 너비 자동 저장
- **드래그하여 해석 (이중 팝업)**
  - **MINI 팝업** — 텍스트 드래그 시 선택 위치 근처에 액션 3개 (단어 뜻 / 문장 번역 / 채팅에 질문) 즉시 표시
  - **DETAIL 팝업** — 좌측 상단에 영구 표시. 단어/문장 설명 결과, 노트 저장, 단어장 추가 등. X 또는 Esc 전까지 유지
- **단어 설명 3섹션** — 📖 일반 뜻 / 📄 이 논문에서의 의미 / ✍️ 영어 예문 (영어 학습용)
- **단어장** — 모르는 단어를 누적 저장, 상단바 `📚 단어장`에서 검색·조회·삭제
- **폴더 단위 라이브러리** — VSCode Explorer처럼 폴더 생성/이동/삭제, 논문 이름 변경
- **논문별 대화 세션** — 각 논문마다 독립된 채팅 히스토리 (서버 재시작 후에도 유지)
- **AI 자동 개요** — 업로드 직후 작은 모델로 ~10초 안에 학습 로드맵 포함 요약 생성
- **AI 요약 → 노트 파일** — 버튼 하나로 마크다운 요약을 `notes.md`에 저장 (교체/추가)
- **수동 노트 편집** — 노트 탭에서 자유롭게 메모, 자동 저장
- **체크포인트/되돌리기** — 대화 임의 지점 저장 후 롤백
- **디버그 패널** — 채팅 툴바 `디버그` — 실제 전송 프롬프트, 토큰 사용량, Ollama 로드 모델 확인
- **수식 렌더링** — KaTeX (`$...$`, `$$...$$`)

## 디스크 구조

```
data/library/
  Default/
    <paper_id>/
      original.pdf      # 원본 PDF (PDF.js가 직접 서빙)
      content.md        # pymupdf4llm 변환 결과
      notes.md          # 사용자 노트 + AI 요약 저장 위치
      meta.json         # 제목, 업로드 시각, 토큰 수
      chat.json         # 대화 + 체크포인트
  Research/
    NLP/
      <paper_id>/...
```

폴더 = 실제 디렉토리. UI에서 만든 폴더는 그대로 디스크에 반영됩니다.

## 사전 준비 — 호스트(Mac)에 Ollama

도커 컨테이너에서 Ollama를 호출하므로 **호스트에 Ollama가 떠 있어야 합니다**.

```bash
# macOS: ollama.com에서 .dmg 설치 (앱 실행 시 자동으로 서버 기동)
# 또는 직접:
ollama serve

# 기본 모델 두 개 받기 (각각 채팅용 / explain용)
ollama pull gemma4:e4b      # ~9GB, 채팅·자동 개요
ollama pull gemma3:4b       # ~3GB, 드래그 단어/문장
```

확인:

```bash
curl http://localhost:11434/api/tags
```

## 실행 — Docker

```bash
docker compose up --build
```

처음 빌드 후엔:

```bash
docker compose up        # 시작
docker compose down      # 정지
docker compose logs -f   # 로그 보기
```

브라우저:

```
http://localhost:8181
```

라이브러리는 `./data` 디렉토리에 볼륨 마운트되어 컨테이너를 지워도 보존됩니다.

### 환경 변수 (모두 선택)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | 호스트 Ollama 주소 |
| `DEFAULT_MODEL` | `gemma4:e4b` | 채팅 / 수동 개요 / 노트 요약에 쓰이는 메인 모델 |
| `OVERVIEW_MODEL` | `gemma4:e4b` | 업로드 직후 자동 개요 전용 (빠르게 첫 응답) |
| `EXPLAIN_WORD_MODEL` | `gemma3:4b` | 드래그한 단어 설명 (instant) |
| `EXPLAIN_SENTENCE_MODEL` | `gemma3:4b` | 드래그한 문장 번역 |
| `NUM_CTX` | `32768` | Ollama 컨텍스트 토큰 수 |
| `THINK_MODE` | `false` | qwen3 / r1 / gpt-oss thinking 모드 (`true` / `low` / `medium` / `high`). thinking 켜면 첫 토큰 지연 큼 |
| `SECRET_KEY` | dev key | Flask 세션 키 |

예:

```bash
# 채팅 모델을 qwen3:14b로
DEFAULT_MODEL=qwen3:14b docker compose up

# 한국어 자연스러움 우선이면 EXAONE
DEFAULT_MODEL=exaone3.5:7.8b OVERVIEW_MODEL=exaone3.5:7.8b docker compose up
```

> 사용할 모델은 모두 **호스트의 Ollama에 미리 받아놔야** 합니다 (`ollama pull <model>`).

## 실행 — 도커 없이 (로컬 개발)

```bash
pip install -r requirements.txt
python3 app.py
```

이 경우 `OLLAMA_BASE_URL`은 기본 `http://localhost:11434`이 됩니다.

## 사용 흐름

1. 좌측 상단 `+ 폴더`로 폴더 생성, 또는 가운데 영역에 PDF 드래그
2. 업로드 → `pymupdf4llm`이 마크다운 변환 + 자동 개요 생성 (작은 모델, ~10초)
3. PDF에서 **모르는 단어/문장을 드래그**:
   - 선택 근처에 **MINI 팝업** (액션 3개)
   - 액션 클릭 시 좌측 상단 **DETAIL 팝업**에 결과 표시 (X / Esc 까지 유지)
   - DETAIL은 드래그로 이동 가능
4. 단어 설명에서 `📚 단어장 추가` → 상단바 `📚 단어장` 버튼으로 누적 확인 / 검색 / 삭제
5. 팝업 결과 `노트에 저장` → 노트 탭의 `notes.md`에 누적
6. `AI 요약 (교체/추가)` 버튼으로 노트 파일에 마크다운 요약 자동 작성
7. 우측 채팅에서 자유롭게 질문 — 튜터 프롬프트가 단계적 설명으로 가이드

## 키보드 / 마우스 단축키

**PDF 뷰어** (뷰어 영역에 포커스가 있을 때, 또는 입력창이 아닐 때)

| 키 | 동작 |
|---|---|
| `↑` `↓` | 80px 스크롤 |
| `Space` / `PageDown` | 한 화면 아래 |
| `PageUp` | 한 화면 위 |
| `←` `→` | 이전 / 다음 페이지로 점프 |
| `Home` / `End` | 맨 위 / 맨 아래 |
| `Cmd/Ctrl + 휠` | 줌 인/아웃 |
| `Cmd/Ctrl + 드래그` | 위로 = 확대, 아래로 = 축소 (드래그 줌) |
| 툴바 `◀ N/M ▶` | 페이지 인디케이터 + 이전/다음 |

**팝업**

| 트리거 | 동작 |
|---|---|
| 새 텍스트 드래그 | MINI 팝업 새로 띄움 |
| MINI 외부 클릭 | MINI 닫힘 (DETAIL은 유지) |
| `Esc` | MINI가 있으면 MINI, 없으면 DETAIL 닫힘 |
| DETAIL 헤더 드래그 | 원하는 위치로 이동 |

**패널 리사이즈**

| 인터랙션 | 동작 |
|---|---|
| 패널 경계 드래그 | 너비 조절 (좌측 180~500px, 우측 280~800px) |
| 경계 더블클릭 | 기본 너비로 리셋 |

## 권장 모델 (M3 Max 64GB 기준)

기본 셋업(모두 설치 필수)은 가벼운 조합 — 즉각적인 응답이 우선이라.

| 모델 | 크기 | 역할 |
|---|---|---|
| `gemma4:e4b` | ~9GB | 메인 채팅 + 자동 개요 (기본값) |
| `gemma3:4b` | ~3GB | 드래그 단어/문장 설명 (instant) |

더 깊이 있는 응답이 필요할 때 옵션:

| 모델 | 크기 | 비고 |
|---|---|---|
| `qwen3:14b` | ~9GB | 한국어/수학/추론 균형, gemma4보다 한 단계 위 |
| `qwen3:32b` | ~20GB | 최상위 품질. Cold start ~40초 + thinking 모드 비추 |
| `exaone3.5:7.8b` | ~5GB | 한국어 자연스러움 최상 (수식 약함) |

`DEFAULT_MODEL` env var로 채팅 모델만 바꾸면, explain은 그대로 작은 모델 유지 — 응답 속도 보존.

## 문제 해결

**컨테이너에서 Ollama에 연결 실패**

- 호스트에서 `curl http://localhost:11434/api/tags`로 먼저 확인
- Docker Desktop이 아닌 환경(예: 순수 Linux)에서는 `OLLAMA_BASE_URL=http://172.17.0.1:11434`처럼 호스트 IP를 직접 지정

**모델 드롭다운이 비어있음**

- 호스트에 모델이 깔려있는지: `ollama list`
- 기본 셋업: `ollama pull gemma4:e4b && ollama pull gemma3:4b`
- 환경변수로 모델 바꿨다면 그 모델이 깔려있는지 함께 확인

**PDF가 안 보임**

- 컨테이너 로그(`docker compose logs -f`)에서 `pymupdf4llm` 변환 에러 확인
- 50MB 이상 PDF는 `MAX_CONTENT_LENGTH` 조정 필요 (app.py)

**응답이 안 오거나 프롬프트를 무시하는 것 같음**

채팅 툴바 `디버그` 버튼 → 모달에서 다음 확인:

- `configured num_ctx` 와 `현재 Ollama 로드 모델 (ctx=...)` 의 값이 **일치**해야 함. 로드된 ctx가 작으면(예: 2048) Ollama가 prompt를 잘라먹는 중 — 호스트 Ollama를 `OLLAMA_CONTEXT_LENGTH=65536 ollama serve` 로 재시작
- `총합` 토큰이 `num_ctx`를 넘으면 시스템 프롬프트 일부가 잘림. 더 큰 컨텍스트 모델로 변경하거나 짧은 논문 사용
- `think 모드` 가 `true`/`high` 인데 응답이 늦으면 thinking이 길어진 것. `THINK_MODE=false` 로 재시작
