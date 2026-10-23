"""최초 수집과 캐시 재사용을 실제 DART 요청 없이 검사합니다."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import collection
from src.common import read_json


COMPANIES_XML = b'''<result>
<list><corp_code>00000001</corp_code><corp_name>Test Bank</corp_name><stock_code>123456</stock_code></list>
<list><corp_code>00000002</corp_code><corp_name>Unlisted</corp_name><stock_code> </stock_code></list>
</result>'''


def test_missing_candidate_file_can_be_bootstrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(collection, "ROOT", tmp_path)
    monkeypatch.setattr(collection, "load_company_xml", lambda key: COMPANIES_XML)
    collection.collect_candidates("offline-test-key")
    result = read_json(tmp_path / "data/processed/company_candidates.json")
    assert result["count"] == 1
    assert result["companies"][0]["corp_code"] == "00000001"
    assert datetime.fromisoformat(result["downloaded_at"]).tzinfo is not None


def test_company_xml_cache_does_not_call_dart(monkeypatch, tmp_path):
    monkeypatch.setattr(collection, "ROOT", tmp_path)
    path = tmp_path / "data/raw/corp_codes.xml"
    path.parent.mkdir(parents=True)
    path.write_bytes(COMPANIES_XML)
    def forbidden(*args, **kwargs):
        raise AssertionError("기존 XML이 있는데 DART를 호출했습니다.")
    monkeypatch.setattr(collection.requests, "get", forbidden)
    assert collection.load_company_xml("offline-test-key") == COMPANIES_XML
