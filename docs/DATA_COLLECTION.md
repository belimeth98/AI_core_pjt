이 문서는 초기 수집 단계 기록입니다. 현재 전문/혼합 검색은 [최신 설계](submission/02_RAG파이프라인설계.md)와 [검증](submission/03_검증결과.md)을 기준으로 합니다.

# 취준공시 데이터 수집 결과

수집·검증일: 2026-10-22 / 프로젝트: `C:\Users\SSAFY\Desktop\AI_core_pjt_2`

## 완료 결과

| 항목 | 결과 |
|---|---:|
| 고유 기업 (`corp_code` 기준) | 100곳 |
| 기업별 사업보고서 원문 ZIP | 100건 |
| 보고서 기준 기간 | 2025.12 (연간) |
| 공시 접수일 검색 범위 | 2026-01-01 ~ 2026-10-22 |
| 추출 본문 섹션 | 4,856개 |
| 추출 본문 글자 수 | 80,235,841자 |
| 취업 준비용으로 별도 선별한 섹션 | 1,815개 / 14,006,153자 |
| 검증 | 전체 파일·연결 검사 통과, 12개 기업의 원문 문단 60개 표본 대조 통과 |
| 수집·파싱 코드 테스트 | 7개 통과 |

보고서의 표를 텍스트로 펼치면서 병합된 셀 값을 반복하므로 글자 수는 원본 XML의 글자 수와 다르다. 본문 수집 과정에서 LLM 요약·임베딩·Vector DB 색인은 실행하지 않았다. 1,815개는 문서의 섹션 수이며 검색용 청크 수가 아니다.

## 기업 구성

| 선정 그룹 | 기업 수 |
|---|---:|
| 금융 | 15 |
| IT·통신·게임 | 23 |
| 제조·에너지·인프라 | 22 |
| 유통·소비재·콘텐츠 | 22 |
| 바이오·헬스케어 | 9 |
| 운송·지주·서비스 | 9 |

이 그룹은 수집 대상을 고르게 고르기 위한 편집상 분류다. 공식 업종분류는 별도의 `industry_code`에 DART 업종코드로 저장했다. 업종명 매핑은 아직 수행하지 않았다.

금융 기업: 신한금융지주, 하나금융지주, 우리금융지주, 기업은행, BNK금융지주, iM금융지주, 카카오뱅크, 카카오페이, DB손해보험, 현대해상, 삼성생명, NH투자증권, 삼성증권, 키움증권, 한국금융지주.

## 생성 파일과 사용 방법

아래 경로는 프로젝트 폴더 기준이다.

| 파일 | 내용·용도 |
|---|---|
| `data/companies.json` | 기업 목록 100개. 기업 탐색·메타데이터 색인의 입력 |
| `data/processed/report_candidates.json` | 기업별 대표 사업보고서 100건의 접수번호·출처·기간 |
| `data/processed/disclosure_manifest.json` | 원문 ZIP·추출 JSON/TXT 위치와 해시, 수집 상태 |
| `data/raw/disclosures/{corp_code}/{receipt_no}.zip` | 내려받은 원본. 수정하지 않음 |
| `data/processed/disclosures/{corp_code}/{receipt_no}.json` | 사업보고서 본문을 섹션별로 추출한 결과 |
| `data/processed/disclosures/{corp_code}/{receipt_no}.txt` | 섹션 제목과 표를 포함하는 본문 텍스트 |
| `data/processed/rag_sections.jsonl` | 취업 준비용 섹션을 한 줄에 하나씩 저장. 아직 청킹되지 않음 |
| `data/processed/collection_summary.json` | 수집 수, 제외 기업과 사유, 검색 기준일 |
| `data/processed/dataset_validation.json` | 검증 결과와 원본 표본 대조 기록 |
| `data/raw/company_details/` | 기업개황 API 응답 캐시 |
| `data/raw/report_lists/` | 기업별 공시검색 응답 캐시와 조회 조건 |
| `data/raw/snapshots/20261022T021744Z/` | 이번 확장 이전의 집계 JSON 3개 백업 |

`rag_sections.jsonl`에는 사업 내용, 경영진단, 회사 개요, 요약 재무정보·연결/별도 재무제표, 임원·직원 현황 관련 섹션을 선별했다. 긴 재무제표 주석 등 다른 섹션은 기업별 원문 추출 JSON/TXT에서 확인할 수 있다. 별첨 감사보고서를 사업보고서 본문과 중복 적재하지 않는다.

기업 메타데이터의 주요 필드:

- `corp_code`, `corp_name`, `stock_name`, `stock_code`: 기업 식별·표시.
- `industry_code`, `established_date`, `accounting_month`, `homepage_url`, `ir_url`: 기업개황.
- `business_overview_excerpt`: 사업 개요 원문 발췌, 최대 1,800자. 생성한 요약이 아니다.
- `business_overview_section_id`, `canonical_receipt_no`, `report_source_url`: 발췌의 근거 연결.
- `report_period`, `collected_at`: 보고기간·기업개황 수집시각.
- `index_text`: 기업명·종목명·업종코드·보고기간·사업 개요 발췌를 결합한 색인 입력 텍스트.

섹션 데이터에는 `corp_code`, `receipt_no`, `document_id`, `section_id`, `section_path`, `title`, `text`, `source_url`, `report_period`, `receipt_date` 등을 저장했다. XML에는 PDF 쪽수가 없으므로 인용할 때 보고서명·섹션명·원문 링크를 사용한다.

Python에서 읽는 예:

```python
import json
from pathlib import Path

root = Path.cwd()  # 프로젝트 루트에서 실행
companies = json.loads((root / "data/companies.json").read_text(encoding="utf-8"))
print(len(companies))  # 100
print(companies[0]["index_text"])

# JSONL은 전체를 한 번에 json.load하지 않고 줄별로 읽는다.
with (root / "data/processed/rag_sections.jsonl").open(encoding="utf-8") as file:
    first_section = json.loads(next(file))
print(first_section["title"], first_section["source_url"])
```

## 보고서 선택·파싱 기준

1. 기존 기업 후보 목록에서 12월 결산 상장사 후보를 선정했다. 보고서가 없는 기업을 서로 다른 기간으로 대체하지 않았다.
2. 공시검색 `A001`, `last_reprt_at=Y` 결과에서 해당 기업의 `(2025.12)` 사업보고서를 선택했다. 정정본이 있으면 수집 스냅샷 기준 최신 접수번호를 따른다.
3. ZIP 무결성을 확인하고, `접수번호.xml`을 우선 선택했다. 문서 코드 `11011` 및 XML 내부 회사 고유번호도 대조했다.
4. XML의 이스케이프 누락·표시용 속성 오류를 정규화했다. 회사·문서 식별자와 표의 행·열 병합 속성은 남겼다. 원본 ZIP은 변경하지 않았다.
5. 정규화 이후 엄격한 XML 파싱을 사용했다. 오류를 무시하는 복구 모드로 본문 일부를 조용히 버리지 않았다.
6. 표의 행·열 내용과 병합 셀 값을 텍스트로 보존했다. 병합 셀 값은 반복되며, 시각적 배치까지 재현한 표는 아니다. 복잡한 수치 계산은 원문·단위·연결/별도 기준을 별도로 확인해야 한다.
7. 사업 내용 존재 여부, 표지의 기준 연도, 본문 최소 길이, 해시·기업·접수번호·출처 연결을 검사했다.

## 제외 기업과 알려진 범위

후보 117개를 검토해 100개를 확보했다. 제외 17개는 `collection_summary.json`에 기록했다.

- 원문 API 상태 `014` 5개: KB금융, 제주항공, 삼성화재, 현대제철, 한화에어로스페이스.
- 엄격한 XML 파싱 실패 10개: SK하이닉스, 한미약품, JB금융지주, 크래프톤, 하이트진로, POSCO홀딩스, 한화생명, 미래에셋증권, CJ ENM, 대신증권.
- 기업개황 응답의 회사구분이 대상 상장구분 Y/K/N에 속하지 않은 후보 2개: 더존비즈온, 동양생명보험. 이는 API 응답 기준의 제외이며 별도 상장 이력을 조사한 결과는 아니다.

KB금융 최신 정정본 `20260619000667`은 수집 시점에 원문을 받지 못했다. 이전 정정본·원보고서를 최신본처럼 섞지 않고 이번 데이터셋에서 제외했다. 나중에 원문이 제공되면 다시 포함할 수 있다. 상태 `014`의 공식 설명은 [공시서류원본파일 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019003)에서 확인할 수 있다.

이 데이터는 2025년 연간 공시를 바탕으로 한다. 분기자료·현재 채용공고·실제 면접 기출·기업문화 자료는 수집하지 않았다. 기업개황 시점과 보고서 기준기간은 다르며 각 필드에 기록했다. 보고서에 전기 비교수치가 포함돼도 별도의 여러 연도 데이터셋을 수집한 것은 아니다.

## 재실행 (Git Bash)

프로젝트 루트에서 가상환경을 활성화한 상태로 실행한다. 현재 수집은 끝났으므로 확인만 하려면 두 번째 명령만 실행하면 된다.

```bash
python scripts/collect_dataset.py --target 100
python scripts/validate_dataset.py
```

첫 명령은 `.env`의 `DART_API_KEY`를 사용한다. 수집기는 기존 기업 후보 파일을 입력으로 받는다. 같은 조회 조건의 목록 캐시와 유효한 원문·파싱 캐시를 재사용하고, 일시적 통신 오류는 최대 3회 시도한다. API 호출 시작 간격은 최소 0.35초다. 인증·요청 한도 오류는 중단한다. 실패한 원문을 성공 캐시로 남기지 않는다.

기본 수집 기준일은 이번 스냅샷인 `20261022`로 고정돼 있다. 이후 정정본까지 조회하려면 `--as-of YYYYMMDD`를 명시해 새 검색 범위를 사용한다. 기업개황 캐시는 자동 갱신되지 않으므로 이는 모든 기업 속성의 최신화를 보장하는 명령은 아니다. 검증기는 현재 100개·2025.12 데이터셋을 검사하도록 작성했다.

수집 재실행으로 대표 문서가 달라졌다면 검증을 다시 실행해 JSONL도 갱신한다. 이전 raw/processed 파일은 캐시로 남으므로 폴더 안 파일 개수 대신 현재 manifest의 100개 문서를 기준으로 사용한다. `smoke_` 접두사가 있는 집계 파일은 사전 점검용이다.

사용자가 작성한 기존 단계별 스크립트는 변경하지 않았다. 기존 3개 기업용 스크립트가 같은 집계 경로를 다시 저장할 수 있으므로, 이번 100개 데이터는 위 통합 수집·검증 스크립트로 관리한다.

## 다음 구현 단계

`data/companies.json`의 `index_text`를 기업 메타데이터 컬렉션에, 선택 기업의 `rag_sections.jsonl` 데이터를 청킹한 결과를 공시 문서 컬렉션에 각각 색인한다. 문서 청킹 시 기업 ID·접수번호·섹션명·출처 링크를 유지한다. 100곳의 전체 본문을 한 번에 임베딩하기보다 우선 기업 1곳으로 검색과 인용을 검증한다.

질문 기반 검색어 추출(P1), LLM 후보 선별(P2), 근거 기반 답변(P3), 추천 질문, LangGraph와 UI는 다음 구현 범위다. 이번 규칙 기반 연간 보고서 선택이 P1/P2 구현을 대신하지는 않는다. 구조화된 재무계정 API 수집도 아직 수행하지 않았다.

출처와 이용 조건은 함께 제공한 `docs/data_sources.md`를 참고한다.
