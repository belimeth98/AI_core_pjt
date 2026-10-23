import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import company_search as search


@pytest.fixture
def candidates():
    return [{
        "corp_code": f"0000000{i}", "corp_name": name, "stock_name": name,
        "stock_code": f"12345{i}", "canonical_receipt_no": f"2026031800000{i}",
        "report_period": "2025.12", "report_source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo=2026031800000{i}",
        "index_text": text,
    } for i, name, text in [
        (1, "알파은행", "모바일 앱에서 예금 및 대출 서비스를 제공하는 은행입니다."),
        (2, "베타제조", "산업용 기계를 제조하여 판매하는 회사입니다."),
    ]]


@pytest.fixture
def decision():
    return {"conditions": ["모바일 금융 서비스 제공"], "decisions": [
        {"corp_code": "00000001", "matched": True, "reason": "모바일 예금·대출을 제공합니다.", "supports": [{"condition_index": 0, "evidence_id": "S1Q1"}]},
        {"corp_code": "00000002", "matched": False, "reason": "해당 사업은 확인되지 않습니다.", "supports": []},
    ]}


def test_evidence_quotes_are_literal_and_bound_to_company(candidates):
    payload, evidence = search.prepare_candidate_evidence(candidates)
    by_code = {c["corp_code"]: c for c in candidates}
    assert len(payload) == 2
    assert all(item["quote"] in by_code[item["corp_code"]]["index_text"] for item in evidence.values())
    assert evidence["S1Q1"]["corp_code"] != evidence["S2Q1"]["corp_code"]


@pytest.mark.parametrize("case", ["unknown_company", "duplicate", "missing", "other_quote", "unknown_quote", "missing_condition", "duplicate_condition", "false_with_support", "boolean_id"])
def test_selection_rejects_invalid_companies_and_supports(candidates, decision, case):
    _, evidence = search.prepare_candidate_evidence(candidates)
    if case == "unknown_company":
        decision["decisions"][0]["corp_code"] = "99999999"
    elif case == "duplicate":
        decision["decisions"][1] = copy.deepcopy(decision["decisions"][0])
    elif case == "missing":
        decision["decisions"].pop()
    elif case == "other_quote":
        decision["decisions"][0]["supports"][0]["evidence_id"] = "S2Q1"
    elif case == "unknown_quote":
        decision["decisions"][0]["supports"][0]["evidence_id"] = "S9Q99"
    elif case == "missing_condition":
        decision["conditions"].append("현재 채용 중")
    elif case == "duplicate_condition":
        decision["decisions"][0]["supports"] *= 2
    elif case == "false_with_support":
        decision["decisions"][0]["matched"] = False
    else:
        decision["decisions"][0]["supports"][0]["condition_index"] = False
    with pytest.raises(ValueError):
        search.validate_company_selection(decision, candidates, evidence)


@pytest.fixture
def offline(monkeypatch, tmp_path, candidates, decision):
    monkeypatch.setattr(search, "load_dotenv", lambda *args: None)
    monkeypatch.setenv("GMS_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("GMS_CHAT_MODEL", "offline-test-model")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/companies.json").write_text(json.dumps(candidates), encoding="utf-8")
    monkeypatch.setattr(search, "load_settings", lambda *args: {})
    monkeypatch.setattr(search, "retrieve_company_candidates", lambda *args: (candidates, {"api_requests_this_run": 0}))
    calls = []
    def chat(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(decision)
    monkeypatch.setattr(search, "chat_completion", chat)
    return tmp_path, calls


def test_search_returns_authoritative_quotes_and_reuses_cache(offline, candidates):
    root, calls = offline
    first = search.find_companies("모바일 금융 서비스 기업", root=root)
    second = search.find_companies("모바일 금융 서비스 기업", root=root)
    assert len(calls) == 1
    assert first["matches"][0]["corp_code"] == "00000001"
    assert first["matches"][0]["supports"][0]["quote"] in candidates[0]["index_text"]
    assert first["chat_api_requests"] == 1
    assert second["chat_api_requests"] == 0 and second["cache_hit"]


def test_no_confirmed_condition_returns_empty_results(offline, decision):
    root, calls = offline
    for item in decision["decisions"]:
        item["matched"] = False
        item["supports"] = []
    result = search.find_companies("현재 신입 채용 기업", root=root)
    assert result["matches"] == []


def test_invalid_cache_is_revalidated(offline):
    root, calls = offline
    search.find_companies("모바일 금융 서비스 기업", root=root)
    cache = next((root / "data/private/company_search_cache").glob("*.json"))
    cached = json.loads(cache.read_text(encoding="utf-8"))
    cached["decisions"][0]["supports"][0]["evidence_id"] = "S2Q1"
    cache.write_text(json.dumps(cached), encoding="utf-8")
    with pytest.raises(ValueError):
        search.find_companies("모바일 금융 서비스 기업", root=root)
    assert len(calls) == 1


def test_invalid_input_stops_before_retrieval(monkeypatch):
    monkeypatch.setattr(search, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(search, "retrieve_company_candidates", lambda *args: pytest.fail("잘못된 입력으로 검색했습니다."))
    with pytest.raises(ValueError):
        search.find_companies(" ")
