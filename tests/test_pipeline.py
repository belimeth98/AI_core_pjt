"""P1/P2와 전체 파이프라인의 연결·중단 조건을 모의 처리합니다."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pipeline, report_selection as selection


@pytest.fixture
def context():
    company = {"corp_code": "00000001", "corp_name": "테스트은행", "report_period": "2025.12", "canonical_receipt_no": "20260318000001"}
    candidate = {"corp_code": company["corp_code"], "report_period": "2025.12", "receipt_no": company["canonical_receipt_no"], "required": False}
    return {"company": company, "question": "주요 사업은 무엇인가요?", "search_plan": {"keywords": ["사업"], "report_types": ["사업보고서"]}, "candidates": [candidate]}


def test_required_report_is_forced_even_when_model_excludes(monkeypatch, context):
    monkeypatch.setattr(selection, "request_chat", lambda *args, **kwargs: json.dumps({"decisions": [{"receipt_no": context["company"]["canonical_receipt_no"], "selected": False, "reason": "관련성 낮음"}]}))
    result = selection.select_reports(context)
    assert result[0]["selected"] and result[0]["forced_include"]
    assert result[0]["model_selected"] is False


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate"])
def test_selection_rejects_incomplete_or_invented_decisions(monkeypatch, context, case):
    item = {"receipt_no": context["company"]["canonical_receipt_no"], "selected": True, "reason": "필수"}
    values = [] if case == "missing" else [{**item, "receipt_no": "20990101000001"}] if case == "unknown" else [item, item]
    monkeypatch.setattr(selection, "request_chat", lambda *args, **kwargs: json.dumps({"decisions": values}))
    with pytest.raises(ValueError):
        selection.select_reports(context)


def test_candidate_lookup_reuses_collection_client_and_closes_it(monkeypatch, context):
    calls = []
    class Client:
        def __init__(self, key):
            self.calls = 0
            self.session = type("Session", (), {"close": lambda self: calls.append("closed")})()
    monkeypatch.setenv("DART_API_KEY", "offline-test")
    monkeypatch.setattr(selection, "DartClient", Client)
    monkeypatch.setattr(selection, "report_list", lambda *args: {"reports": [], "collected_at": "test"})
    monkeypatch.setattr(selection, "select_target_reports", lambda *args: copy.deepcopy(context["candidates"]))
    candidates, requests = selection.find_candidates(context["company"])
    assert candidates[0]["required"]
    assert calls == ["closed"] and requests == 0


@pytest.fixture
def pipeline_setup(monkeypatch, context):
    events = []
    company = context["company"]
    monkeypatch.setattr(pipeline, "load_dotenv", lambda *args: None)
    for key in ("DART_API_KEY", "GMS_BASE_URL", "GMS_EMBEDDING_MODEL"):
        monkeypatch.setenv(key, "offline-test")
    monkeypatch.setattr(pipeline, "read_json", lambda *args: [company])
    monkeypatch.setattr(pipeline, "load_settings", lambda: {})
    monkeypatch.setattr(pipeline, "available_reports", lambda *args: {company["corp_code"]: {"receipt_no": company["canonical_receipt_no"]}})
    def record(name, result):
        def call(*args):
            events.append(name)
            return result
        return call
    monkeypatch.setattr(pipeline, "extract_search_terms", record("plan", selection.SearchPlan(**context["search_plan"])))
    monkeypatch.setattr(pipeline, "find_candidates", record("candidates", (context["candidates"], 0)))
    monkeypatch.setattr(pipeline, "select_reports", record("selection", [{**context["candidates"][0], "selected": True}]))
    monkeypatch.setattr(pipeline, "prepare_reports", record("prepare", ([{"receipt_no": company["canonical_receipt_no"]}], 0)))
    monkeypatch.setattr(pipeline, "ensure_prepared_index", record("check", ([{"status": "matched"}], {"api_requests_this_run": 0})))
    result = {"corp_code": company["corp_code"], "receipt_no": company["canonical_receipt_no"], "question": context["question"], "answer": "설명 [1]", "sources": []}
    graph = type("Graph", (), {"invoke": lambda self, state: record("graph", result)()})()
    monkeypatch.setattr(pipeline, "build_graph", lambda: graph)
    return events


def test_pipeline_keeps_order_and_trace(context, pipeline_setup):
    result = pipeline.run_pipeline(context["question"], context["company"]["corp_code"])
    assert pipeline_setup == ["plan", "candidates", "selection", "prepare", "check", "graph"]
    assert result["pipeline_trace"]["dart_request_count"] == 0


def test_unindexed_company_stops_before_models(monkeypatch, context, pipeline_setup):
    monkeypatch.setattr(pipeline, "available_reports", lambda *args: {})
    with pytest.raises(ValueError, match="색인이 준비되지"):
        pipeline.run_pipeline(context["question"], context["company"]["corp_code"])
    assert pipeline_setup == []


def test_unexpected_selected_report_stops_preparation(monkeypatch, context, pipeline_setup):
    monkeypatch.setattr(pipeline, "select_reports", lambda *args: [{"receipt_no": "20990101000001", "selected": True}])
    with pytest.raises(ValueError, match="필수 보고서"):
        pipeline.run_pipeline(context["question"], context["company"]["corp_code"])
    assert pipeline_setup == ["plan", "candidates"]


def test_pipeline_indexes_and_searches_multiple_selected_receipts(monkeypatch, context, pipeline_setup):
    company = context["company"]
    quarterly = {**context["candidates"][0], "receipt_no": "20260515000001", "report_period": "2026.03"}
    monkeypatch.setattr(pipeline, "extract_search_terms", lambda *args: selection.SearchPlan(keywords=["분기"], report_types=["분기보고서"]))
    monkeypatch.setattr(pipeline, "find_candidates", lambda *args: ([*context["candidates"], quarterly], 1))
    monkeypatch.setattr(pipeline, "select_reports", lambda *args: [{**r, "selected": True} for r in [*context["candidates"], quarterly]])
    monkeypatch.setattr(pipeline, "prepare_reports", lambda data: (data["selected_reports"], 1))
    monkeypatch.setattr(pipeline, "ensure_prepared_index", lambda data: ([{"receipt_no": r["receipt_no"]} for r in data["reports"]], {"api_requests_this_run": 2}))
    def invoke(state):
        assert state["receipt_nos"] == [company["canonical_receipt_no"], quarterly["receipt_no"]]
        return {**state, "receipt_no": company["canonical_receipt_no"], "answer": "분기 자료 설명 [2]", "sources": [{"corp_code": company["corp_code"], "receipt_no": quarterly["receipt_no"]}]}
    monkeypatch.setattr(pipeline, "build_graph", lambda: type("Graph", (), {"invoke": lambda self, state: invoke(state)})())
    result = pipeline.run_pipeline("분기보고서의 주요 사업은?", company["corp_code"])
    assert result["pipeline_trace"]["dart_request_count"] == 2
    assert len(result["pipeline_trace"]["index_checks"]) == 2
