"""실제 GMS 대신 가짜 그래프를 사용해 화면 동작과 비용 발생 경로를 검증합니다."""
import copy
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import read_json
from src.ui_support import used_sources


@pytest.fixture
def ui(monkeypatch):
    from src import pipeline as run_pipeline
    from src import vector_store
    from src import briefing

    def forbidden_generation(*args, **kwargs):
        pytest.fail("저장된 브리핑 화면에서 외부 생성을 실행했습니다.")
    monkeypatch.setattr(briefing, "generate_briefing", forbidden_generation)
    monkeypatch.setattr(briefing, "prepare_briefing_context", forbidden_generation)
    original_loader = briefing.load_saved_briefing
    kakao = read_json(ROOT / "data/companies.json")
    kakao = next(company for company in kakao if company["corp_code"] == "01133217")
    saved_briefing = original_loader(kakao)
    monkeypatch.setattr(briefing, "load_saved_briefing", lambda company: saved_briefing if company["corp_code"] == "01133217" else None)

    # 실제 .env·벡터 DB·외부 서비스에 의존하지 않는 UI 검사입니다.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    for name in ("GMS_API_KEY", "GMS_BASE_URL", "GMS_CHAT_MODEL", "GMS_EMBEDDING_MODEL", "DART_API_KEY"):
        monkeypatch.setenv(name, "offline-ui-test")
    monkeypatch.setattr(vector_store, "available_reports", lambda *args: {
        "01133217": {"receipt_no": "20260318000884", "fingerprint": "offline-fixture"},
    })

    st.cache_resource.clear()

    saved = read_json(
        ROOT / "data/processed/latest_pipeline_result.json"
    )
    by_code = {company["corp_code"]: company for company in read_json(ROOT / "data/companies.json")}

    class FakePipeline:
        def __init__(self):
            self.calls = []
            self.fail = False

        def run(self, question, corp_code):
            self.calls.append({
                "question": question,
                "corp_code": corp_code,
            })

            if self.fail:
                raise RuntimeError(
                    "PRIVATE_INTERNAL_DETAIL_MUST_NOT_LEAK"
                )

            result = copy.deepcopy(saved)
            result["question"] = question
            result["corp_code"] = corp_code
            result["receipt_no"] = by_code[corp_code]["canonical_receipt_no"]
            # 저장 예시가 다중 보고서여도 가짜 기업 응답의 범위는 자체 문서입니다.
            result["receipt_nos"] = [result["receipt_no"]]
            for source in result["sources"]:
                source["corp_code"] = corp_code
                source["receipt_no"] = result["receipt_no"]
                source["report_period"] = by_code[corp_code]["report_period"]
            result["answer"] = (
                "## 쉬운 설명\n화면 테스트용 응답입니다. [4]"
            )

            return result

    fake = FakePipeline()

    monkeypatch.setattr(
        run_pipeline,
        "run_pipeline",
        fake.run,
    )

    app = AppTest.from_file(
        str(ROOT / "app.py")
    ).run(timeout=30)

    assert not app.exception

    yield app, fake

    st.cache_resource.clear()


def submit(app, question):
    app.text_area[0].input(question)
    next(b for b in app.button if "ui_question_form_" in str(b.key)).click()
    app.run(timeout=30)
    assert not app.exception


def test_multiple_reports_show_actual_period_and_report_type(ui, monkeypatch):
    from src import pipeline
    app, _ = ui
    result = copy.deepcopy(read_json(ROOT / "data/processed/latest_pipeline_result.json"))
    receipt = "20260814000011"
    result["receipt_nos"] = [result["receipt_no"], receipt]
    result["question"] = "2026년 분기보고서의 사업은?"
    result["answer"] = "## 쉬운 설명\n분기 자료에서 확인한 내용입니다. [99]"
    result["cited_ids"] = [99]
    result["questions"] = []
    source = {**result["sources"][-1], "source_id": 99, "receipt_no": receipt,
              "report_period": "2026.03", "report_name": "분기보고서 (2026.03)",
              "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + receipt}
    result["sources"].append(source)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *args: result)
    submit(app, result["question"])
    captions = " ".join(c.value for c in app.caption)
    assert "여러 보고서 조회" in captions and "2026.03" in captions
    assert "분기보고서 본문" in captions


def test_initial_page_and_rerun_do_not_call_model(ui):
    app, fake = ui
    assert len(app.session_state["ui_turns"]["01133217"]) == 1
    app.run()
    assert not app.exception and fake.calls == []


def test_form_calls_once_and_same_question_reuses_session_answer(ui):
    app, fake = ui
    question = "UI 검증용 새로운 질문입니다."
    submit(app, question)
    assert len(fake.calls) == 1
    assert app.session_state["ui_turns"]["01133217"][-1]["question"] == question
    app.run()
    assert app.session_state["ui_active_turn"]["01133217"] == 1
    submit(app, question)
    app.run()
    assert not app.exception and len(fake.calls) == 1
    assert len(app.session_state["ui_turns"]["01133217"]) == 2
    assert app.session_state["ui_active_turn"]["01133217"] == 1


def test_unindexed_company_is_disabled_and_history_survives_switch(ui):
    app, fake = ui
    submit(app, "기업별 질문 기록을 확인합니다.")
    app.selectbox(key="ui_company").set_value("00382199").run()
    assert not app.exception
    assert next(b for b in app.button if "ui_question_form_" in str(b.key)).disabled
    assert app.session_state["ui_turns"]["00382199"] == []
    app.selectbox(key="ui_company").set_value("01133217").run()
    assert not app.exception and len(fake.calls) == 1
    assert len(app.session_state["ui_turns"]["01133217"]) == 2


def test_followup_button_runs_its_question(ui):
    app, fake = ui
    expected = app.session_state["ui_turns"]["01133217"][0]["questions"][0]["question"]
    app.button(key="ui_follow_01133217_0").click().run(timeout=30)
    assert not app.exception and len(fake.calls) == 1
    assert fake.calls[0]["question"] == expected


def test_failure_keeps_old_answer_and_hides_internal_exception(ui):
    app, fake = ui
    fake.fail = True
    submit(app, "실패 상황을 확인하는 질문입니다.")
    assert len(app.session_state["ui_turns"]["01133217"]) == 1
    assert app.error
    assert "PRIVATE_INTERNAL_DETAIL" not in " ".join(item.value for item in app.error)


def test_source_list_includes_only_references_used_by_answer_or_questions():
    result = {"cited_ids": [2], "questions": [{"source_ids": [3]}],
        "sources": [{"source_id": 1}, {"source_id": 2}, {"source_id": 3}]}
    assert [s["source_id"] for s in used_sources(result)] == [2, 3]
    assert [s["source_id"] for s in used_sources(result, include_questions=False)] == [2]


def test_uncited_answer_has_explicit_ui_notice(ui):
    app, fake = ui
    turn = app.session_state["ui_turns"]["01133217"][0]
    turn.update(answer="현재 자료로 확인할 수 없습니다.", cited_ids=[], questions=[])
    app.run()
    assert not app.exception and fake.calls == []
    assert any("답변에 인용된 근거가 없습니다" in item.value for item in app.info)


def test_saved_briefing_and_download_appear_without_generation(ui):
    app, fake = ui
    assert any("취업 준비 브리핑" in item.value for item in app.subheader)
    assert any("면접 연습 질문 — 실제 기출 아님" in item.value for item in app.markdown)
    downloads = app.get("download_button")
    assert any(item.proto.label == "브리핑을 노트로 저장 ↓" for item in downloads)
    app.run()
    assert not app.exception and fake.calls == []


def test_switching_company_hides_other_company_briefing(ui):
    app, fake = ui
    app.selectbox(key="ui_company").set_value("00382199").run()
    assert not app.exception and fake.calls == []
    assert any("브리핑이 아직 준비되지" in item.value for item in app.caption)
    assert not any(item.proto.label == "브리핑을 노트로 저장 ↓" for item in app.get("download_button"))
    assert not any("면접 연습 질문 — 실제 기출 아님" in item.value for item in app.markdown)


def test_invalid_briefing_shows_safe_warning_and_keeps_questions(ui, monkeypatch):
    from src import briefing
    app, fake = ui
    def invalid(*args):
        raise ValueError("PRIVATE_BRIEFING_DETAIL_MUST_NOT_LEAK")
    monkeypatch.setattr(briefing, "load_saved_briefing", invalid)
    app.run()
    assert not app.exception and fake.calls == []
    assert any("공시 근거를 확인하지 못했습니다" in item.value for item in app.warning)
    assert all("PRIVATE_BRIEFING_DETAIL" not in item.value for item in app.warning)
    assert not app.text_area[0].disabled
    assert len(app.session_state["ui_turns"]["01133217"]) == 1


def test_natural_company_search_runs_once_and_navigates(ui, monkeypatch):
    from src import company_search
    app, fake = ui
    company = next(c for c in read_json(ROOT / "data/companies.json") if c["corp_code"] == "00149646")
    calls = []
    def find(query):
        calls.append(query)
        return {"query": query, "company_count": 100, "candidate_count": 12, "matches": [{
            **company, "reason": "중소기업 지원 금융 업무를 합니다.",
            "supports": [{"condition": "중소기업 금융", "quote": company["business_overview_excerpt"][:100]}],
        }]}
    monkeypatch.setattr(company_search, "find_companies", find)
    app.text_input(key="ui_company_search_query").input("중소기업을 지원하는 금융 기업")
    next(b for b in app.button if "ui_company_search_form" in str(b.key)).click().run()
    assert not app.exception and len(calls) == 1
    app.run()
    assert len(calls) == 1 and fake.calls == []
    app.button(key="ui_search_select_00149646").click().run()
    assert not app.exception and app.selectbox(key="ui_company").value == "00149646"
    assert "01133217" in app.session_state["ui_turns"]


def test_two_ready_companies_keep_independent_question_histories(ui, monkeypatch):
    from src import vector_store
    app, fake = ui
    company = next(c for c in read_json(ROOT / "data/companies.json") if c["corp_code"] == "00149646")
    monkeypatch.setattr(vector_store, "available_reports", lambda *args: {
        "01133217": {"receipt_no": "20260318000884", "fingerprint": "offline-fixture"},
        "00149646": {"receipt_no": company["canonical_receipt_no"], "fingerprint": "offline-fixture-ibk"},
    })
    submit(app, "카카오뱅크 질문 기록입니다.")
    app.selectbox(key="ui_company").set_value("00149646").run()
    submit(app, "기업은행 질문 기록입니다.")
    assert app.session_state["ui_turns"]["00149646"][0]["question"] == "기업은행 질문 기록입니다."
    app.selectbox(key="ui_company").set_value("01133217").run()
    assert not app.exception and len(fake.calls) == 2
    assert app.session_state["ui_turns"]["01133217"][-1]["question"] == "카카오뱅크 질문 기록입니다."


def test_missing_api_key_shows_notice_without_pipeline(ui, monkeypatch):
    app, fake = ui
    monkeypatch.setenv("GMS_API_KEY", "")
    submit(app, "API 키 누락 상태의 질문입니다.")
    assert not fake.calls
    assert app.error
    assert any("환경변수" in item.value or "설정" in item.value for item in app.error)
