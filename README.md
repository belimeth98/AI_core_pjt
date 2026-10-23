# 취준공시

기업 공시를 재무 초보 취업 준비생에게 설명하고 출처·후속 질문·취업 준비 브리핑을 제공하는 Streamlit RAG 서비스입니다.

## 현재 상태

- 기업 100개, 2025.12 연간 보고서 100건의 전체 4,856개 섹션을 검색합니다.
- 기존 text-embedding-3-large(3,072차원) 21,374개 + 전문 확장 text-embedding-3-small(1,536차원) 98,634개 = 120,008개 청크입니다. 실제 검증에서 추가 분기 공시도 처리했습니다.
- 임베딩 모델별 컬렉션·캐시를 분리하고 각각 검색한 결과를 RRF 순위로 합칩니다. 벡터나 거리 점수를 서로 섞지 않습니다.
- 현재 채팅 모델은 gpt-5.4-nano입니다. 실제 혼합 파이프라인과 기존 5문항 비교는 변경 전 gpt-6.1-sol로 검증했습니다. 사용자 설정 변경 뒤 추가 채팅 호출은 하지 않았습니다. .env의 임베딩 large 설정을 유지하며 small은 config/retrieval.json의 full_text_expansion으로 지정합니다.
- 필수 F2001~F2012 구현/검증. 선택 심화 F2015 비교 자료 보관. F2013/F2014/F2016은 미선택입니다.
- 132개 회귀 테스트 통과. 의미 정확성 전체나 현재 채용·금융조건의 보장은 아닙니다.

## 기존 프로젝트 실행 - Git Bash

```bash
cd ~/Desktop/AI_core_pjt_2
source .venv/Scripts/activate
python -m streamlit run app.py --server.port 8502
```

기업 선택 → 질문 → 쉬운 설명/원문 확인 → 추천 질문을 사용합니다. 자연어 기업 조건 탐색에서 기업 화면으로 이동할 수 있습니다. 기록은 기업별로 분리해 현재 세션에서 유지합니다. 브리핑 저장본은 4개 기업이며 새 생성은 CLI입니다.

질문에서 분기·반기 보고서를 지정하면 P1/P2가 후보를 선별하고 필요한 추가 문서를 small으로 색인합니다. 기업 자체 대표 연간 보고서는 항상 포함합니다. 한 질문의 선택 보고서는 최대 6개이며 2026-10-22까지의 검색 스냅샷입니다. 현재 금리·채용 여부나 임의 모든 과거 연도를 조회하는 서비스는 아닙니다.

## 제출 ZIP을 다른 PC에서 재현

키·.venv·Chroma DB·임베딩 캐시는 제외합니다. 데이터는 공개 DART 원문/파싱/코퍼스와 실제 검색에 필요한 청크를 포함합니다. 다른 PC에서 본인의 키/URL을 설정하고 두 모델의 색인을 생성해야 합니다. 새 임베딩을 GMS에 요청하면 사용량이 발생합니다.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
# 본인의 DART_API_KEY, GMS_API_KEY, GMS_BASE_URL 입력
python scripts/validate_dataset.py
python scripts/validate_chunks.py
python scripts/prepare_chunks.py --full-corpus
python scripts/validate_chunks.py --full-corpus
python scripts/prepare_chunks.py --expansion
python scripts/validate_chunks.py --expansion
python scripts/index_data.py --all --allow-new-embeddings
python scripts/index_data.py --all --expansion --allow-new-embeddings
python scripts/validate_chunks.py --expansion --verify-store
python -m streamlit run app.py --server.port 8502
```

이미 정상인 원래 프로젝트는 다시 색인하지 않아도 됩니다. index_data.py는 기본적으로 캐시만 재사용하며 새 요청은 --allow-new-embeddings를 지정해야 합니다. UI 추가 문서의 허용 여부는 allow_document_embedding으로 제어합니다. 혼합 모드에서 기존 large 문서에 신규 임베딩을 요청하지 않습니다. Python 3.12에서 검증했습니다.

## 명령

```bash
python scripts/run_pipeline.py --corp-code 01133217 --question "카카오뱅크의 2026년 1분기보고서와 2025년 사업보고서의 주요 사업을 비교해 주세요."
python scripts/search_companies.py --query "모바일 금융 서비스를 제공하는 기업"
python scripts/generate_briefing.py --corp-code 00149646
python -B -m pytest tests -q -p no:cacheprovider
```

evaluate_rag.py는 설정이 바뀌면 5문항을 다시 생성하므로 추가 사용량이 생깁니다. 보관한 5문항 전후 비교는 혼합 확장 전 large 코퍼스 기준입니다. 이번에는 재실행하지 않았습니다.

## 산출물

- [요구사항 정의서·대조표](docs/submission/01_요구사항정의서.md)
- [파이프라인 설계](docs/submission/02_RAG파이프라인설계.md)
- [검증 결과·화면 증거](docs/submission/03_검증결과.md)
- [RAG 전후 비교](docs/submission/04_RAG전후품질비교.md)
- [공식 셀프 체크리스트](docs/submission/05_셀프체크리스트.md)
- [정리·복구·제출 안내](docs/submission/07_정리및제출안내.md)
- [남은 제한사항](docs/submission/08_미구현및제한사항.md)

지역·반·성명, 본인의 셀프 결과 확인과 실제 상호 점검은 남아 있습니다. 외부 업로드는 하지 않았습니다.

## 코드 책임

collection.py/disclosure_parser.py는 수집·파싱, chunking.py는 전문/확장/추가 보고서 청크, vector_store.py는 모델별 캐시·Chroma·동시 갱신 방지·색인을 담당합니다. report_selection.py/report_preparation.py는 P1/P2와 추가 문서 준비를 담당합니다. retrieval.py는 모델별 검색과 RRF 병합, answering.py/qa_graph.py/pipeline.py는 생성과 흐름, briefing.py/company_search.py/evaluation.py는 특화 기능을 담당합니다. app.py와 assets/app.css는 화면입니다. 공통 JSON·해시·모델 호출은 common.py/llm.py를 재사용합니다.

## 데이터·라이선스

[데이터 출처·이용 조건](docs/data_sources.md), [폰트 라이선스](docs/FONT_LICENSES.md), licenses/를 유지합니다. .env를 제출하거나 키를 코드에 쓰지 마세요. 제출 보안 검사는 docs/submission/evidence/package_validation.json에 있습니다.

## 분리 제출 문서

- [요구사항 정의서](output/제출문서/01_요구사항정의서.pdf)
- [RAG 파이프라인 설계문서](output/제출문서/02_RAG파이프라인설계문서.pdf)
- [검증 결과](output/제출문서/03_검증결과.pdf)
- [공식 셀프 체크리스트 엑셀](output/제출문서/04_셀프체크리스트.xlsx)
- [셀프 체크리스트 PDF](output/제출문서/04_셀프체크리스트.pdf)
- [분리 산출물 묶음](output/취준공시_분리산출물.zip)

증빙 PNG 13개를 output/제출문서/증빙_캡처에 각각 저장했다. 체크리스트의 증빙 파일명과 증빙파일_목록.csv를 대조했다. 셀프 결과는 통과 21, 미확인 3, 해당없음 3이다. 미확인 20/25/27번의 사유를 작성했고 실제 상호 점검은 수행하지 않았다. 성명은 미기입이다. 이전 통합 PDF는 이 분리 문서로 대체하며 제출 ZIP에는 분리 문서를 포함한다.

