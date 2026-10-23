"""화면과 명령행에서 공유하는 P1 → P2 → 원문 준비 → RAG 실행."""
import os
from dotenv import load_dotenv
from src.common import ROOT, read_json
from src.chunking import load_settings
from src.report_selection import (TARGET_PERIOD, START_DATE, END_DATE,
                                 extract_search_terms, find_candidates, select_reports)
from src.report_preparation import prepare_reports, ensure_prepared_index
from src.qa_graph import build_graph
from src.vector_store import available_reports
from src.ui_support import valid_result

def run_pipeline(question, corp_code):
    load_dotenv(ROOT / ".env")

    question = question.strip()

    if not 2 <= len(question) <= 1500:
        raise ValueError("질문은 2~1500자로 입력하세요.")

    companies = read_json(ROOT / "data/companies.json")
    company = next(
        (item for item in companies if item["corp_code"] == corp_code),
        None,
    )

    if company is None:
        raise ValueError("기업 목록에 없는 기업입니다.")

    if company["report_period"] != TARGET_PERIOD:
        raise ValueError("현재 검색 기준기간과 기업 자료의 기간이 다릅니다.")

    if not os.getenv("DART_API_KEY", "").strip():
        raise ValueError("DART_API_KEY 설정을 확인하세요.")

    # 색인이 없는 기업은 모델 호출 전에 중단합니다.
    settings = load_settings()
    ready = available_reports(
        ROOT,
        settings,
        os.getenv("GMS_BASE_URL", "").strip(),
        os.getenv("GMS_EMBEDDING_MODEL", "").strip(),
    )

    if (
        corp_code not in ready
        or ready[corp_code]["receipt_no"]
        != company["canonical_receipt_no"]
    ):
        raise ValueError(
            "이 기업의 현재 보고서 색인이 준비되지 않았습니다."
        )

    company_info = {
        key: company[key]
        for key in [
            "corp_code",
            "corp_name",
            "report_period",
            "canonical_receipt_no",
        ]
    }

    print("[1/6] P1: 검색어를 추출합니다.")
    plan = extract_search_terms(company_info, question)

    print(f"검색어: {', '.join(plan.keywords)}")

    print("\n[2/6] 보고서 후보를 조회합니다.")
    candidates, listing_requests = find_candidates(company, plan.model_dump())
    print(f"후보: {len(candidates)}건")

    context = {
        "company": company_info,
        "question": question,
        "search_plan": plan.model_dump(),
        "search_start": START_DATE,
        "search_end": END_DATE,
        "candidates": candidates,
    }

    print("\n[3/6] P2: 보고서를 선별합니다.")
    decisions = select_reports(context)
    selected = [item for item in decisions if item["selected"]]

    selected_ids = [item["receipt_no"] for item in selected]
    candidate_ids = {item["receipt_no"] for item in candidates}
    if company["canonical_receipt_no"] not in selected_ids or not set(selected_ids) <= candidate_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("선택 결과에 필수 보고서가 없거나 후보 밖·중복 보고서가 있습니다.")
    if len(selected_ids) > 6:
        raise ValueError("한 질문에서는 최대 6개 보고서를 선택할 수 있습니다. 기간이나 유형을 좁혀 주세요.")

    print(f"선택: {len(selected)}건")

    print("\n[4/6] 원문과 본문을 준비합니다.")
    prepared, document_requests = prepare_reports({
        **context,
        "selected_reports": selected,
    })

    print("\n[5/6] 선택한 모든 보고서를 청킹·색인합니다. 기존 완료 청크는 재사용합니다.")
    index_checks, embedding_usage = ensure_prepared_index({
        "company": company_info,
        "question": question,
        "reports": prepared,
    })

    print("\n[6/6] LangGraph로 검색과 답변을 실행합니다.")
    graph = build_graph()
    result = graph.invoke({
        "question": question,
        "corp_code": corp_code,
        "receipt_nos": selected_ids,
    })

    if not valid_result(result, company) or set(result.get("receipt_nos", [company["canonical_receipt_no"]])) != set(selected_ids):
        raise ValueError("최종 답변의 기업 또는 보고서가 일치하지 않습니다.")

    # 결과와 함께 P1/P2 처리 이력을 남깁니다.
    result["pipeline_trace"] = {
        "company": company_info,
        "search_plan": plan.model_dump(),
        "search_start": START_DATE,
        "search_end": END_DATE,
        "candidates": candidates,
        "selection_decisions": decisions,
        "prepared_reports": prepared,
        "index_checks": index_checks,
        "index_embedding_usage": embedding_usage,
        "dart_request_count": listing_requests + document_requests,
    }

    return result
