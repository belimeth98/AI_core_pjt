import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import briefing, llm


ORIGINAL = "한도 조정이나 여신취급규모 조정 등 적절한 조치를 취하도록 하고 있습니다."


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(briefing, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("GMS_API_KEY", "offline-test-key")
    monkeypatch.setenv("GMS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GMS_CHAT_MODEL", "offline-test-model")

    def forbidden(*args, **kwargs):
        pytest.fail("실제 API 클라이언트를 생성하면 안 됩니다.")

    monkeypatch.setattr(llm, "OpenAI", forbidden)


@pytest.fixture
def context():
    return {
        "company": {
            "corp_code": "00000001", "corp_name": "테스트은행",
            "canonical_receipt_no": "20260318000001", "report_period": "2025.12",
        },
        "sources": [{
            "source_id": 8, "corp_code": "00000001",
            "receipt_no": "20260318000001", "report_period": "2025.12",
            "title": "위험관리", "source_url": "https://example.invalid/report",
            "text": "기업: 테스트은행\n[본문]\n" + ORIGINAL,
        }],
    }


@pytest.fixture
def draft():
    return {
        "company_summary": [],
        "key_points": [{"text": "은행은 한도 조정 등으로 위험을 관리합니다.", "evidence_id": "S8Q1"}],
        "career_tips": [], "interview_questions": [],
        "limitations": ["2025.12 사업보고서에서 검색한 범위입니다."],
    }


def stub_client(monkeypatch, responses):
    calls = []

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        value = responses[len(calls) - 1]
        if isinstance(value, dict):
            value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False), refusal=None),
            )])
        return value

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: Client())
    return calls


def test_catalog_preserves_literal_text_and_length(context):
    source = context["sources"][0]
    source["text"] += "\n" + ORIGINAL + "\n" + "긴 원문 내용입니다 " * 70
    catalog = briefing.build_evidence_catalog(context["sources"])
    assert len(catalog) > 2
    assert len({entry["quote"] for entry in catalog.values()}) == len(catalog)
    assert all(8 <= len(entry["quote"]) <= 180 for entry in catalog.values())
    assert all(entry["quote"] in source["text"] for entry in catalog.values())
    assert catalog["S8Q1"]["quote"] == ORIGINAL


def test_restore_all_sections_keeps_final_schema(context, draft):
    point = copy.deepcopy(draft["key_points"][0])
    draft["company_summary"] = [point]
    draft["career_tips"] = [{**point, "text": "제안: 위험관리 방식을 정리해 보세요."}]
    draft["interview_questions"] = [{**point, "text": "한도 조정은 왜 필요한가요?", "reason": "위험관리 이해를 연습합니다."}]
    original_draft = copy.deepcopy(draft)
    result = briefing.restore_evidence(
        briefing.DraftBriefing.model_validate(draft),
        briefing.build_evidence_catalog(context["sources"]),
    )
    briefing.validate_briefing(result, context["sources"])
    assert all(item.source_id == 8 and item.quote == ORIGINAL for item in briefing.briefing_items(result))
    assert result.interview_questions[0].reason == draft["interview_questions"][0]["reason"]
    assert draft == original_draft
    assert "evidence_id" not in result.model_dump()["key_points"][0]
    assert "[8]" in briefing.render_briefing(context, result)


def test_original_quote_validator_still_rejects_missing_character(context):
    point = briefing.BriefingPoint(text="위험을 관리합니다.", source_id=8, quote=ORIGINAL.replace("취하도록", "하도록"))
    result = briefing.CareerBriefing(company_summary=[], key_points=[point], career_tips=[], interview_questions=[], limitations=["검색 범위에 한계가 있습니다."])
    with pytest.raises(ValueError, match="출처에 없는 원문 발췌"):
        briefing.validate_briefing(result, context["sources"])


def test_generation_attaches_authoritative_quote(monkeypatch, context, draft):
    calls = stub_client(monkeypatch, [draft])
    result = briefing.generate_briefing(context)
    assert len(calls) == 1
    assert result.key_points[0].quote == ORIGINAL
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["evidence_catalog"][0]["quote"] == ORIGINAL
    properties = payload["output_schema"]["$defs"]["DraftPoint"]["properties"]
    assert set(properties) == {"text", "evidence_id"}


@pytest.mark.parametrize("failure", ["unknown_id", "schema"])
def test_single_repair_uses_specific_feedback(monkeypatch, context, draft, failure):
    invalid = copy.deepcopy(draft)
    if failure == "unknown_id":
        invalid["key_points"][0]["evidence_id"] = "S8Q999"
        expected = "S8Q999"
    else:
        invalid["key_points"][0]["evidence_id"] = 8
        expected = "key_points.0.evidence_id"
    calls = stub_client(monkeypatch, [invalid, draft])
    result = briefing.generate_briefing(context)
    assert len(calls) == 2
    assert expected in calls[1]["messages"][-1]["content"]
    assert result.key_points[0].quote == ORIGINAL


def test_unknown_id_fails_after_one_repair(monkeypatch, context, draft):
    draft["key_points"][0]["evidence_id"] = "S8Q999"
    calls = stub_client(monkeypatch, [draft, draft])
    with pytest.raises(ValueError, match="자동 수정 후에도 검증에 실패"):
        briefing.generate_briefing(context)
    assert len(calls) == 2


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "company", "period", "receipt", "short_text"])
def test_invalid_context_never_calls_api(context, invalid):
    source = context["sources"][0]
    if invalid == "empty":
        context["sources"] = []
    elif invalid == "duplicate":
        context["sources"].append(copy.deepcopy(source))
    elif invalid == "short_text":
        source["text"] = "짧음"
    else:
        field = {"company": "corp_code", "period": "report_period", "receipt": "receipt_no"}[invalid]
        source[field] = "different"
    with pytest.raises(ValueError):
        briefing.generate_briefing(context)


@pytest.mark.parametrize("case", ["refusal", "truncated", "empty"])
def test_incomplete_response_never_returns_briefing(monkeypatch, context, case):
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length" if case == "truncated" else "stop",
        message=SimpleNamespace(content="{}" if case != "empty" else "", refusal="refused" if case == "refusal" else None),
    )])
    calls = stub_client(monkeypatch, [response])
    with pytest.raises(RuntimeError):
        briefing.generate_briefing(context)
    assert len(calls) == 1


@pytest.fixture
def saved_result(context, draft):
    value = briefing.restore_evidence(
        briefing.DraftBriefing.model_validate(draft),
        briefing.build_evidence_catalog(context["sources"]),
    )
    return {**copy.deepcopy(context), "briefing": value.model_dump(), "markdown": "UNVALIDATED_STORED_MARKDOWN"}


def save_result(tmp_path, result):
    path = tmp_path / "data/processed/practice_briefing_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return path


def test_saved_briefing_is_validated_and_markdown_rebuilt(tmp_path, context, saved_result):
    path = save_result(tmp_path, saved_result)
    before = path.read_bytes()
    result = briefing.load_saved_briefing(context["company"], tmp_path)
    assert result["briefing"].key_points[0].quote == ORIGINAL
    assert "UNVALIDATED_STORED_MARKDOWN" not in result["markdown"]
    assert "테스트은행 취업 준비 브리핑" in result["markdown"]
    assert "[8]" in result["markdown"]
    assert path.read_bytes() == before


def test_missing_saved_briefing_returns_none(tmp_path, context):
    assert briefing.load_saved_briefing(context["company"], tmp_path) is None


@pytest.mark.parametrize("field", ["corp_code", "canonical_receipt_no", "report_period"])
def test_saved_briefing_for_other_company_or_period_is_hidden(tmp_path, context, saved_result, field):
    saved_result["company"][field] = "different"
    save_result(tmp_path, saved_result)
    assert briefing.load_saved_briefing(context["company"], tmp_path) is None


@pytest.mark.parametrize("case", ["quote", "unknown_id", "schema", "duplicate", "empty", "malformed_source", "company", "receipt", "period"])
def test_invalid_saved_briefing_is_rejected(tmp_path, context, saved_result, case):
    if case == "quote":
        saved_result["briefing"]["key_points"][0]["quote"] = ORIGINAL.replace("취하도록", "하도록")
    elif case == "unknown_id":
        saved_result["briefing"]["key_points"][0]["source_id"] = 99
    elif case == "schema":
        saved_result["briefing"]["key_points"] = "wrong"
    elif case == "duplicate":
        saved_result["sources"].append(copy.deepcopy(saved_result["sources"][0]))
    elif case == "empty":
        saved_result["sources"] = []
    elif case == "malformed_source":
        saved_result["sources"][0]["text"] = None
    else:
        field = {"company": "corp_code", "receipt": "receipt_no", "period": "report_period"}[case]
        saved_result["sources"][0][field] = "different"
    save_result(tmp_path, saved_result)
    with pytest.raises(ValueError):
        briefing.load_saved_briefing(context["company"], tmp_path)


def test_two_companies_have_independent_briefing_files(tmp_path, context, saved_result):
    first = briefing.CareerBriefing.model_validate(saved_result["briefing"])
    path1 = briefing.save_briefing(context, first, tmp_path)
    second_context = copy.deepcopy(context)
    second_context["company"]["corp_code"] = "00000002"
    second_context["company"]["corp_name"] = "두번째은행"
    second_context["company"]["canonical_receipt_no"] = "20260318000002"
    second_context["sources"][0]["corp_code"] = "00000002"
    second_context["sources"][0]["receipt_no"] = "20260318000002"
    path2 = briefing.save_briefing(second_context, first, tmp_path)
    assert path1 != path2 and path1.exists() and path2.exists()
    assert briefing.load_saved_briefing(context["company"], tmp_path)["company"]["corp_code"] == "00000001"
    assert briefing.load_saved_briefing(second_context["company"], tmp_path)["company"]["corp_code"] == "00000002"


def test_company_specific_result_takes_priority_over_legacy(tmp_path, context, saved_result):
    save_result(tmp_path, {**saved_result, "briefing": {}})
    value = briefing.CareerBriefing.model_validate(saved_result["briefing"])
    briefing.save_briefing(context, value, tmp_path)
    assert briefing.load_saved_briefing(context["company"], tmp_path)["briefing"].key_points


def test_cached_briefing_creation_never_calls_models(tmp_path, context, saved_result, monkeypatch):
    save_result(tmp_path, saved_result)
    (tmp_path / "data/companies.json").write_text(json.dumps([context["company"]]), encoding="utf-8")
    monkeypatch.setattr(briefing, "prepare_briefing_context", lambda *args: pytest.fail("캐시가 있는데 검색했습니다."))
    monkeypatch.setattr(briefing, "generate_briefing", lambda *args: pytest.fail("캐시가 있는데 생성했습니다."))
    saved, path, reused = briefing.create_company_briefing("00000001", root=tmp_path)
    assert reused and path.exists() and saved["briefing"].key_points


@pytest.mark.parametrize("field", ["corp_code", "canonical_receipt_no"])
def test_briefing_storage_rejects_path_identifiers(context, tmp_path, field):
    context["company"][field] = "../outside"
    with pytest.raises(ValueError):
        briefing.briefing_path(context["company"], tmp_path)
