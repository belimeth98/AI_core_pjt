"""현재 화면의 답변·추천 질문·그래프를 외부 호출 없이 검사합니다."""
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import answering, llm
from src.qa_graph import build_graph


@pytest.fixture
def retrieval():
    meta = {"corp_code": "00000001", "receipt_no": "20260318000001", "report_period": "2025.12", "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20260318000001"}
    return {
        "question": "이 은행은 어떤 일을 하나요?", "corp_code": meta["corp_code"], "receipt_no": meta["receipt_no"],
        "company_metadata": {"text": "기업명: 테스트은행", "metadata": {**meta, "source_type": "company_metadata"}},
        "evidence": [{"id": "chunk1", "text": "예금과 대출 서비스를 제공합니다.", "metadata": {**meta, "source_type": "disclosure_xml", "title": "사업의 내용"}}],
    }


def test_prepare_sources_retains_company_period_and_numbering(retrieval):
    sources = answering.prepare_sources(retrieval)
    assert [s["source_id"] for s in sources] == [1, 2]
    assert sources[0]["source_type"] == "company_metadata"
    assert sources[1]["text"] == retrieval["evidence"][0]["text"]
    assert all(s["receipt_no"] == retrieval["receipt_no"] for s in sources)


@pytest.mark.parametrize("text", ["존재하지 않는 출처 [99]"])
def test_citations_reject_absent_or_unknown_ids(retrieval, text):
    with pytest.raises(ValueError):
        answering.check_citations(text, answering.prepare_sources(retrieval))


def test_citations_are_sorted_and_deduplicated(retrieval):
    assert answering.check_citations("설명 [2][1][2]", answering.prepare_sources(retrieval)) == [1, 2]


def test_answer_prompt_sends_only_given_evidence(monkeypatch, retrieval):
    calls = []
    def chat(prompt, payload, **kwargs):
        calls.append((prompt, payload, kwargs))
        return "## 쉬운 설명\n예금·대출 서비스를 제공합니다. [2]"
    monkeypatch.setattr(answering, "request_chat", chat)
    sources = answering.prepare_sources(retrieval)
    assert "[2]" in answering.generate_answer(retrieval["question"], sources)
    assert calls[0][1] == {"question": retrieval["question"], "sources": sources}


@pytest.mark.parametrize("value", [
    {"questions": [{"question": "예금이란 무엇인가요?", "source_ids": [99]}]},
    {"questions": [{"question": "예금이란 무엇인가요?", "source_ids": [True]}]},
    {"questions": [{"question": "예금이란 무엇인가요?", "source_ids": []}]},
    {"questions": [{"question": "예금이란 무엇인가요?", "source_ids": [2]}] * 4},
])
def test_invalid_recommendations_are_rejected(monkeypatch, retrieval, value):
    monkeypatch.setattr(answering, "request_chat", lambda *args, **kwargs: json.dumps(value))
    with pytest.raises((ValueError, ValidationError)):
        answering.generate_questions(retrieval["question"], "답변 [2]", answering.prepare_sources(retrieval))


def test_rag_result_preserves_ui_contract(monkeypatch, retrieval):
    monkeypatch.setattr(answering, "generate_answer", lambda *args: "답변 [2]")
    monkeypatch.setattr(answering, "generate_questions", lambda *args: answering.QuestionResult(
        questions=[answering.SuggestedQuestion(question="예금이란 무엇인가요?", source_ids=[2])]))
    result = answering.run_rag(retrieval)
    assert result["cited_ids"] == [2]
    assert result["corp_code"] == retrieval["corp_code"]
    assert result["questions"][0]["source_ids"] == [2]
    assert result["answer"] == "답변 [2]"


def test_invalid_answer_stops_before_recommendation(monkeypatch, retrieval):
    monkeypatch.setattr(answering, "generate_answer", lambda *args: "없는 근거 [99]")
    monkeypatch.setattr(answering, "generate_questions", lambda *args: pytest.fail("잘못된 답변으로 질문을 생성했습니다."))
    with pytest.raises(ValueError):
        answering.run_rag(retrieval)


def test_graph_runs_retrieve_before_answer(retrieval):
    events = []
    def search(question, code):
        events.append("retrieve")
        assert code == retrieval["corp_code"]
        return retrieval
    def answer(value):
        events.append("generate")
        return {"answer": "답변 [2]", "sources": answering.prepare_sources(value)}
    graph = build_graph(search, answer)
    result = graph.invoke({"question": retrieval["question"], "corp_code": retrieval["corp_code"]})
    assert events == ["retrieve", "generate"]
    assert result["answer"] == "답변 [2]"
    assert set(graph.get_graph().nodes) == {"__start__", "retrieve", "generate", "__end__"}


def test_graph_without_body_returns_unknown_without_model(monkeypatch, retrieval):
    retrieval["evidence"] = []
    monkeypatch.setattr(answering, "generate_answer", lambda *args: pytest.fail("본문 없이 모델을 호출했습니다."))
    monkeypatch.setattr(answering, "generate_questions", lambda *args: pytest.fail("근거 없이 추천 질문을 생성했습니다."))
    result = build_graph(lambda *args: retrieval).invoke({"question": retrieval["question"], "corp_code": retrieval["corp_code"]})
    assert "확인할 수 없습니다" in result["answer"]
    assert result["cited_ids"] == [] and result["questions"] == []
    assert result["sources"][0]["source_type"] == "company_metadata"


def test_uncited_answer_is_preserved_with_no_recommendations(monkeypatch, retrieval):
    monkeypatch.setattr(answering, "generate_answer", lambda *args: "현재 자료로 확인할 수 없습니다.")
    monkeypatch.setattr(answering, "generate_questions", lambda *args: pytest.fail("무인용 답변으로 추천을 생성했습니다."))
    result = answering.run_rag(retrieval)
    assert result["cited_ids"] == [] and result["questions"] == []


@pytest.mark.parametrize("name", ["GMS_API_KEY", "GMS_BASE_URL", "GMS_CHAT_MODEL"])
def test_shared_chat_requires_environment_before_client(monkeypatch, name):
    for key in ("GMS_API_KEY", "GMS_BASE_URL", "GMS_CHAT_MODEL"):
        monkeypatch.setenv(key, "offline-test")
    monkeypatch.delenv(name)
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: pytest.fail("설정 누락 상태에서 클라이언트를 만들었습니다."))
    with pytest.raises(ValueError):
        llm.chat_completion([])
