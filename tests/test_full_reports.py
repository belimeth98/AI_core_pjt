"""전문 누락, 문서 간 덮어쓰기, 다중 접수번호 검색 범위를 검증합니다."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import report_preparation as prep
from src import report_selection as selection
from src import chunking
from src.chunking import (build_chunks, chunk_paths, chunk_profile, prepare_report_chunks,
                          report_sections, prepare_expansion_chunks, expansion_paths)
from src.retrieval import merge_evidence
from src.common import digest, read_json, write_json
from src.ui_support import valid_result
from src.vector_store import (index_lock, open_store, query_evidence, state_path, sync_records)
from src.vector_store import available_reports, embedding_profile


@pytest.fixture
def settings():
    return {"chunk_size": 200, "chunk_overlap": 40, "top_k": 5, "candidate_k": 8,
            "embedding_batch_size": 2, "embedding_dimensions": 3, "max_input_utf8_bytes": 7500,
            "request_gap_seconds": 0, "database_path": "chroma", "embedding_cache_path": "cache",
            "corpus_scope": "full", "search_mode": "hybrid"}


def prepared(root, receipt, period="2025.12"):
    report = {"corp_code": "00000001", "corp_name": "테스트은행", "receipt_no": receipt,
              "report_period": period, "report_name": ("사업보고서" if period == "2025.12" else "분기보고서") + f" ({period})",
              "receipt_date": receipt[:8], "collected_at": "2026-10-22", "source": "DART",
              "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"}
    sections = [{"section_id": receipt + ":s001", "title": "II. 사업의 내용", "section_path": ["II. 사업의 내용"],
                 "text": "예금과 대출을 제공합니다. " * 30},
                {"section_id": receipt + ":s002", "title": "VI. 이사회 등 회사의 기관", "section_path": ["VI. 이사회 등 회사의 기관"],
                 "text": "이사회는 감사위원회를 두고 감독합니다. " * 30}]
    raw = root / "raw" / f"{receipt}.zip"
    raw.parent.mkdir(exist_ok=True)
    raw.write_bytes(b"offline-source-archive-" + receipt.encode())
    checksum = hashlib.sha256(raw.read_bytes()).hexdigest()
    parsed = {**report, "document_id": report["corp_code"] + ":" + receipt,
              "zip_sha256": checksum, "parser_version": "test-parser", "sections": sections}
    path = root / "parsed" / f"{receipt}.json"
    write_json(path, parsed)
    return {**report, "parsed_json_path": path.relative_to(root).as_posix(),
            "zip_path": raw.relative_to(root).as_posix(), "text_path": "unused.txt",
            "zip_sha256": checksum, "parser_version": "test-parser"}, parsed


def save_canonical(root, report, parsed, settings, *, omit_governance=False):
    records = [r for s in report_sections(report, parsed) for r in build_chunks(s, settings)]
    if omit_governance:
        records = [r for r in records if r["metadata"]["section_id"].endswith("s001")]
    folder, manifest = chunk_paths(settings, root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "00000001.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    info = {"filename": path.name, "receipt_no": report["receipt_no"], "chunk_count": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(manifest, {"companies": {"00000001": info}})
    return records, info


def test_full_report_includes_governance_and_reuses_identical_cache(tmp_path, settings):
    report, parsed = prepared(tmp_path, "20260318000001")
    records, first = prepare_report_chunks(report, settings, tmp_path)
    assert {r["metadata"]["section_id"] for r in records} == {s["section_id"] for s in parsed["sections"]}
    again, second = prepare_report_chunks(report, settings, tmp_path)
    assert records == again and first == second
    assert prep.report_index_records(report, settings, tmp_path)[2] is False


def test_full_report_rejects_missing_section_despite_valid_chunk_hash(tmp_path, settings):
    report, parsed = prepared(tmp_path, "20260318000001")
    save_canonical(tmp_path, report, parsed, settings, omit_governance=True)
    with pytest.raises(ValueError, match="색인되지 않은 섹션"):
        prep.report_index_records(report, settings, tmp_path)


def test_report_cache_rebuilds_when_source_changes(tmp_path, settings):
    report, parsed = prepared(tmp_path, "20260318000001")
    before, info = prepare_report_chunks(report, settings, tmp_path)
    parsed["sections"][1]["text"] += "추가 공시 사항"
    write_json(tmp_path / report["parsed_json_path"], parsed)
    after, new = prepare_report_chunks(report, settings, tmp_path)
    assert info["source_sha256"] != new["source_sha256"]
    assert info["sha256"] != new["sha256"] and before != after


def test_report_zip_tampering_is_rejected(tmp_path, settings):
    report, _ = prepared(tmp_path, "20260318000001")
    (tmp_path / report["zip_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="해시"):
        prep.report_index_records(report, settings, tmp_path)


def test_quarterly_prefix_and_ids_preserve_period_and_document(tmp_path, settings):
    report, parsed = prepared(tmp_path, "20260515000001", "2026.03")
    records, _ = prepare_report_chunks(report, settings, tmp_path)
    assert all("2026.03 (분기)" in r["text"] and "(연간)" not in r["text"] for r in records)
    assert all("20260515000001" in r["id"] for r in records)
    assert chunk_profile(settings) != chunk_profile({k: v for k, v in settings.items() if k != "corpus_scope"})


class FakeEmbedder:
    total_inputs = 0
    def __init__(self, settings):
        self.settings, self.profile, self.inputs = settings, "testprofile", 0
    def embed(self, texts):
        self.inputs += len(texts)
        FakeEmbedder.total_inputs += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]
    def stats(self):
        return {"new_embedding_inputs": self.inputs}
    def close(self):
        pass


def test_selected_reports_index_without_overwrite_and_repeat_without_api(tmp_path, settings, monkeypatch):
    annual, parsed = prepared(tmp_path, "20260318000001")
    quarter, _ = prepared(tmp_path, "20260515000001", "2026.03")
    _, info = save_canonical(tmp_path, annual, parsed, settings)
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    monkeypatch.setattr(prep, "GmsEmbeddings", FakeEmbedder)
    company = {"corp_code": "00000001", "canonical_receipt_no": annual["receipt_no"], "corp_name": "테스트은행",
               "index_text": "테스트은행 기업 메타데이터", "report_period": "2025.12"}
    _, meta, store = open_store(settings, "testprofile", tmp_path)
    embedder = FakeEmbedder(settings)
    sync_records(meta, [{"id": "00000001", "text": company["index_text"], "metadata": {
        "corp_code": "00000001", "content_sha256": digest(company["index_text"])}}], embedder)
    write_json(state_path(settings, "testprofile", tmp_path), {"company_metadata_complete": True})
    data = {"company": company, "reports": [annual, quarter]}
    results, usage = prep.ensure_prepared_index(data)
    assert len(results) == 2 and usage["new_embedding_inputs"] > 0
    before = set(store.get()["ids"])
    assert {m["receipt_no"] for m in store.get(include=["metadatas"])["metadatas"]} == {annual["receipt_no"], quarter["receipt_no"]}
    repeated, usage = prep.ensure_prepared_index(data)
    assert usage["new_embedding_inputs"] == 0 and set(store.get()["ids"]) == before
    assert all(r["sync"]["updated"] == 0 for r in repeated)
    # 대표 보고서를 다시 색인해도 추가 문서를 삭제하지 않습니다.
    prep.ensure_prepared_index({"company": company, "reports": [annual]})
    assert set(store.get()["ids"]) == before
    state = read_json(state_path(settings, "testprofile", tmp_path))
    extra = state["extra_documents"]["00000001:" + quarter["receipt_no"]]
    query = query_evidence("00000001", "감사위원회", settings, embedder, meta, store, state, company,
        info["sha256"], [annual["receipt_no"], quarter["receipt_no"]], {quarter["receipt_no"]: extra["chunk_file_sha256"]})
    assert {r["metadata"]["receipt_no"] for r in query["evidence"]} <= {annual["receipt_no"], quarter["receipt_no"]}
    with pytest.raises(ValueError, match="추가 보고서"):
        query_evidence("00000001", "감사위원회", settings, embedder, meta, store, state, company,
            info["sha256"], [annual["receipt_no"], "20260814000001"], {})


def test_candidate_types_periods_and_latest_correction():
    company = {"corp_code": "00000001", "corp_name": "테스트은행", "report_period": "2025.12"}
    rows = [
        {"corp_code": "00000001", "report_nm": "사업보고서 (2025.12)", "rcept_no": "20260318000001"},
        {"corp_code": "00000001", "report_nm": "분기보고서 (2026.03)", "rcept_no": "20260515000001"},
        {"corp_code": "00000001", "report_nm": "[기재정정]분기보고서 (2026.03)", "rcept_no": "20260516000001"},
        {"corp_code": "00000001", "report_nm": "반기보고서 (2026.06)", "rcept_no": "20260814000001"},
        {"corp_code": "00000002", "report_nm": "반기보고서 (2026.06)", "rcept_no": "20260814000002"}]
    results = selection.select_target_reports(company, {"reports": rows, "collected_at": "test"}, ["사업보고서", "반기보고서", "분기보고서"])
    assert {r["receipt_no"] for r in results} == {"20260318000001", "20260516000001", "20260814000001"}


def test_ui_only_accepts_selected_report_sources():
    company = {"corp_code": "00000001", "canonical_receipt_no": "20260318000001"}
    result = {"corp_code": "00000001", "receipt_no": "20260318000001", "answer": "설명", "sources": [
        {"corp_code": "00000001", "receipt_no": "20260515000001"}]}
    assert not valid_result(result, company)
    result["receipt_nos"] = ["20260318000001", "20260515000001"]
    assert valid_result(result, company)
    result["sources"][0]["corp_code"] = "00000002"
    assert not valid_result(result, company)


def test_index_writer_lock_blocks_overlap_and_releases(tmp_path):
    with index_lock(tmp_path):
        with pytest.raises(ValueError, match="다른 색인 작업"):
            with index_lock(tmp_path):
                pass
    with index_lock(tmp_path):
        pass


def test_full_scope_repairs_meaningful_gap_from_splitter(tmp_path, settings, monkeypatch):
    from types import SimpleNamespace
    report, parsed = prepared(tmp_path, "20260318000001")
    section = next(report_sections(report, parsed))
    section["text"] = "앞부분\n\n[표: 병합 셀 값 반복 표시]\n\n끝부분"
    text = section["text"]
    class OmittingSplitter:
        def __init__(self, **kwargs):
            pass
        def create_documents(self, texts):
            return [SimpleNamespace(page_content="앞부분", metadata={"start_index": 0}),
                    SimpleNamespace(page_content="끝부분", metadata={"start_index": text.index("끝부분")})]
    monkeypatch.setattr(chunking, "RecursiveCharacterTextSplitter", OmittingSplitter)
    covered = set()
    for record in build_chunks(section, settings):
        meta = record["metadata"]
        body = record["text"].split("\n[본문]\n", 1)[1]
        assert body == text[meta["start_index"]:meta["end_index"]]
        covered.update(range(meta["start_index"], meta["end_index"]))
    assert all(character.isspace() or index in covered for index, character in enumerate(text))


def test_expansion_excludes_existing_large_and_preserves_all_full_text(tmp_path, settings):
    report, parsed = prepared(tmp_path, "20260318000001")
    base_settings = {k: v for k, v in settings.items() if k != "corpus_scope"}
    large, _ = save_canonical(tmp_path, report, parsed, base_settings, omit_governance=True)
    full, _ = save_canonical(tmp_path, report, parsed, settings)
    _, path = chunk_paths(settings, tmp_path)
    manifest = read_json(path)
    manifest["source_sha256"] = "full-source-hash"
    write_json(path, manifest)
    result = prepare_expansion_chunks(base_settings, tmp_path)
    folder, _ = expansion_paths(base_settings, tmp_path)
    small = [json.loads(line) for line in (folder / "00000001.jsonl").read_text(encoding="utf-8").splitlines()]
    large_hashes = {r["metadata"]["content_sha256"] for r in large}
    small_hashes = {r["metadata"]["content_sha256"] for r in small}
    assert large_hashes.isdisjoint(small_hashes)
    assert large_hashes | small_hashes == {r["metadata"]["content_sha256"] for r in full}
    assert result["chunk_count"] == len(small) and result["excluded_existing_large"] == len(large)


def test_cross_model_fusion_uses_ranks_and_deduplicates_source_positions():
    def item(identifier, index, distance):
        return {"id": identifier, "distance": distance, "metadata": {"corp_code": "1", "receipt_no": "r",
            "section_id": "s", "start_index": index, "end_index": index+10, "content_sha256": str(index)}}
    rankings = [[item("large-shared", 0, 0.99), item("large-only", 20, 0.001)],
                [item("small-only", 40, 0.0001), item("small-shared", 0, 0.98)]]
    result = merge_evidence(rankings, 3)
    assert len(result) == 3
    assert result[0]["metadata"]["start_index"] == 0
    assert len({r["metadata"]["start_index"] for r in result}) == 3


def test_mixed_readiness_requires_completed_matching_small_index(tmp_path, settings):
    base = {k: v for k, v in settings.items() if k != "corpus_scope"}
    base["full_text_expansion"] = {"enabled": True}
    (tmp_path / base["database_path"]).mkdir()
    url, model = "https://example.test/v1", "text-embedding-3-large"
    large_profile = embedding_profile(url, model, base["embedding_dimensions"])
    _, manifest = chunk_paths(base, tmp_path)
    write_json(manifest, {"companies": {"00000001": {"receipt_no": "receipt", "sha256": "large-sha"}}})
    write_json(state_path(base, large_profile, tmp_path), {"company_metadata_complete": True, "documents": {
        "00000001": {"complete": True, "receipt_no": "receipt", "chunk_file_sha256": "large-sha"}}})
    assert available_reports(tmp_path, base, url, model) == {}
    from src.chunking import expansion_settings
    small = expansion_settings(base)
    profile = embedding_profile(url, small["embedding_model"], small["embedding_dimensions"])
    _, path = expansion_paths(base, tmp_path)
    write_json(path, {"companies": {"00000001": {"sha256": "small-sha", "base_sha256": "large-sha"}}})
    state = {"documents": {"00000001": {"complete": False, "receipt_no": "receipt", "chunk_file_sha256": "small-sha"}}}
    write_json(state_path(small, profile, tmp_path), state)
    assert available_reports(tmp_path, base, url, model) == {}
    state["documents"]["00000001"]["complete"] = True
    write_json(state_path(small, profile, tmp_path), state)
    ready = available_reports(tmp_path, base, url, model)
    assert set(ready) == {"00000001"} and profile in ready["00000001"]["fingerprint"]
