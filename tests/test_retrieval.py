import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import build_chunks, chunk_profile, digest
from src.vector_store import GmsEmbeddings, checked_vectors, lexical_ranking, open_store, query_evidence, sync_records


@pytest.fixture
def settings():
    return {"chunk_size": 200, "chunk_overlap": 40, "top_k": 3,
        "embedding_batch_size": 2, "embedding_dimensions": 3,
        "max_input_utf8_bytes": 7500, "request_gap_seconds": 0,
        "database_path": "chroma", "embedding_cache_path": "cache"}


def section(text):
    return {"corp_code": "00000001", "corp_name": "테스트은행", "document_id": "00000001:receipt",
        "receipt_no": "receipt", "report_name": "사업보고서 (2025.12)", "report_period": "2025.12",
        "receipt_date": "20260301", "source_type": "disclosure_xml", "collected_at": "2026-10-22",
        "source_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=receipt",
        "section_id": "receipt:s001", "title": "II. 사업의 내용 > 1. 사업의 개요",
        "version_status": "latest_available_in_search_snapshot", "text": text}


def test_chunks_cover_repeated_source_without_cross_company(settings):
    text = ("모바일 은행 서비스를 제공하고 있습니다.\n\n" * 80) + "마지막 원문"
    chunks = build_chunks(section(text), settings)
    covered = set()
    for item in chunks:
        meta = item["metadata"]
        body = item["text"].split("\n[본문]\n", 1)[1]
        assert text[meta["start_index"]:meta["end_index"]] == body
        assert len(body) <= settings["chunk_size"]
        assert meta["corp_code"] == "00000001"
        covered.update(range(meta["start_index"], meta["end_index"]))
    assert len(covered) == len(text)
    assert len({x["id"] for x in chunks}) == len(chunks)


def test_table_continuation_repeats_real_unit_and_heading(settings):
    text = "(단위: 백만원)\n\n[표: 병합 셀의 값은 해당 행·열에 반복 표시]\n계정 | 2025년 | 2024년\n" + "이자수익 | 12,345 | 10,000\n" * 40
    chunks = build_chunks(section(text), settings)
    continuations = [c for c in chunks if c["metadata"]["table_context_included"]]
    assert continuations
    for item in continuations:
        assert "(단위: 백만원)" in item["text"]
        assert "계정 | 2025년 | 2024년" in item["text"]


def test_long_table_rows_drop_no_non_whitespace_content(settings):
    text = "[표: 병합 셀의 값은 해당 행·열에 반복 표시]\n" + ("이자수익 | 2025 | " + "1234567890 " * 110 + "\n\n") * 5
    covered = set()
    for item in build_chunks(section(text), settings):
        meta = item["metadata"]
        covered.update(range(meta["start_index"], meta["end_index"]))
    assert all(character.isspace() or index in covered for index, character in enumerate(text))


def test_chunk_identity_stable_and_config_change_separate(settings):
    data = section("사업 내용 " * 200)
    first = build_chunks(data, settings)
    assert first == build_chunks(data, settings)
    other = {**settings, "chunk_overlap": 30}
    assert chunk_profile(settings) != chunk_profile(other)
    assert not ({x["id"] for x in first} & {x["id"] for x in build_chunks(data, other)})


def response(vectors, indices=None):
    indices = indices or list(range(len(vectors)))
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=v) for i, v in zip(indices, vectors)],
        model="test-model", usage=SimpleNamespace(total_tokens=9))


@pytest.mark.parametrize("vectors,indices", [([[1, 2]], [0]), ([[1, float('nan'), 3]], [0]),
    ([[0, 0, 0]], [0]), ([[1, 2, 3]], [1])])
def test_invalid_embedding_response_rejected(vectors, indices):
    with pytest.raises(ValueError):
        checked_vectors(response(vectors, indices), 1, 3)


def test_embedding_response_sorted_by_original_input():
    assert checked_vectors(response([[3, 2, 1], [1, 2, 3]], [1, 0]), 2, 3) == [[1, 2, 3], [3, 2, 1]]


class FakeEmbedder:
    def __init__(self, settings, fail=False):
        self.settings = settings
        self.inputs = 0
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("simulated failure")
        self.inputs += len(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def record(identifier, code, text="수익과 사업 내용"):
    return {"id": identifier, "text": text, "metadata": {"corp_code": code,
        "receipt_no": "receipt", "content_sha256": digest(text), "title": "사업 내용",
        "source_url": "https://example.test/report"}}


def test_persistent_index_rerun_no_duplicate_or_new_embedding(settings, tmp_path):
    _, meta, docs = open_store(settings, "testprofile", tmp_path)
    data = [record("a", "1"), record("b", "2", "다른 회사")]
    embedder = FakeEmbedder(settings)
    assert sync_records(meta, data, embedder)["updated"] == 2
    _, again, _ = open_store(settings, "testprofile", tmp_path)
    assert sync_records(again, data, embedder)["updated"] == 0
    assert again.count() == 2
    assert embedder.inputs == 2
    assert docs.count() == 0


def test_failed_replacement_keeps_old_records(settings, tmp_path):
    _, _, docs = open_store(settings, "testprofile", tmp_path)
    sync_records(docs, [record("old", "1"), record("unrelated", "2")], FakeEmbedder(settings))
    with pytest.raises(RuntimeError):
        sync_records(docs, [record("new", "1")], FakeEmbedder(settings, fail=True), where={"corp_code": "1"})
    assert set(docs.get()["ids"]) == {"old", "unrelated"}
    sync_records(docs, [record("new", "1")], FakeEmbedder(settings), where={"corp_code": "1"})
    assert set(docs.get()["ids"]) == {"new", "unrelated"}


@pytest.mark.parametrize("search_mode", ["similarity", "hybrid"])
def test_query_isolates_company_and_pins_own_metadata(settings, tmp_path, search_mode):
    settings["search_mode"] = search_mode
    _, meta, docs = open_store(settings, "testprofile", tmp_path)
    embedder = FakeEmbedder(settings)
    sync_records(meta, [record("1", "1"), record("2", "2")], embedder)
    sync_records(docs, [record("doc1", "1"), record("doc2", "2")], embedder)
    state = {"documents": {"1": {"complete": True, "receipt_no": "receipt", "chunk_count": 1, "chunk_file_sha256": "hash"}}}
    company = {"canonical_receipt_no": "receipt", "index_text": "수익과 사업 내용", "corp_name": "테스트은행"}
    result = query_evidence("1", "주요 사업", settings, embedder, meta, docs, state, company, "hash")
    assert [r["id"] for r in result["evidence"]] == ["doc1"]
    assert result["company_metadata"]["metadata"]["corp_code"] == "1"
    with pytest.raises(ValueError):
        query_evidence("1", "주요 사업", settings, embedder, meta, docs, state, company, "changed-hash")


def test_embedding_disk_cache_reuses_requests_and_duplicate_inputs(settings, tmp_path):
    calls = []
    class Endpoint:
        def create(self, **kwargs):
            calls.append(kwargs["input"])
            return response([[1., 0., 0.] for _ in kwargs["input"]])
    def create_embedder():
        obj = GmsEmbeddings.__new__(GmsEmbeddings)
        obj.settings, obj.dimensions, obj.cache = settings, 3, tmp_path / "cache"
        obj.profile, obj.model = "testprofile", "test-model"
        obj.api_calls = obj.input_count = obj.reported_tokens = obj.cache_hits = 0
        obj.last_call = 0
        obj.client = SimpleNamespace(embeddings=Endpoint())
        return obj
    first = create_embedder()
    first.embed(["은행", "은행", "결제"])
    second = create_embedder()
    assert second.embed(["결제", "은행"]) == [[1., 0., 0.], [1., 0., 0.]]
    assert calls == [["은행", "결제"]]
    assert second.api_calls == 0


def test_lexical_ranking_uses_body_and_korean_particles():
    data = [{"id": "irrelevant", "text": "예금과 대출 제목\n[본문]\n해외 송금 서비스"},
            {"id": "deposit", "text": "[본문]\n정기예금 상품"},
            {"id": "both", "text": "[본문]\n예금 및 대출 서비스"}]
    assert lexical_ranking("카카오뱅크의 예금과 대출은?", data, "카카오뱅크")[0] == "both"


def test_credit_protection_never_calls_provider_for_missing_embedding(settings, tmp_path):
    obj = GmsEmbeddings.__new__(GmsEmbeddings)
    obj.settings = {**settings, "allow_new_embeddings": False}
    obj.dimensions, obj.cache = 3, tmp_path / "cache"
    obj.profile, obj.model = "testprofile", "test-model"
    obj.api_calls = obj.input_count = obj.reported_tokens = obj.cache_hits = 0
    obj.last_call = 0
    class Endpoint:
        def create(self, **kwargs):
            pytest.fail("크레딧 보호 모드에서 외부 API를 호출했습니다.")
    obj.client = SimpleNamespace(embeddings=Endpoint())
    with pytest.raises(ValueError, match="크레딧 보호 모드"):
        obj.embed(["새로운 전문 청크"])
    assert obj.api_calls == obj.input_count == 0


def test_embedding_model_override_has_distinct_cache_profile(settings, tmp_path, monkeypatch):
    from src import vector_store
    monkeypatch.setenv("GMS_API_KEY", "offline-test")
    monkeypatch.setenv("GMS_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("GMS_EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setattr(vector_store, "OpenAI", lambda **kwargs: SimpleNamespace(close=lambda: None))
    large = GmsEmbeddings({**settings, "embedding_dimensions": 3072}, tmp_path)
    small = GmsEmbeddings({**settings, "embedding_model": "text-embedding-3-small", "embedding_dimensions": 1536}, tmp_path)
    try:
        assert large.model == "text-embedding-3-large" and large.dimensions == 3072
        assert small.model == "text-embedding-3-small" and small.dimensions == 1536
        assert large.profile != small.profile and large.cache != small.cache
    finally:
        large.close()
        small.close()


def test_empty_expansion_does_not_call_query_embedding(settings, tmp_path):
    _, meta, docs = open_store(settings, "testprofile", tmp_path)
    embedder = FakeEmbedder(settings)
    sync_records(meta, [record("1", "1")], embedder)
    inputs = embedder.inputs
    state = {"documents": {"1": {"complete": True, "receipt_no": "receipt", "chunk_count": 0, "chunk_file_sha256": "hash"}}}
    company = {"canonical_receipt_no": "receipt", "index_text": "수익과 사업 내용", "corp_name": "테스트은행"}
    result = query_evidence("1", "주요 사업", settings, embedder, meta, docs, state, company, "hash")
    assert result["evidence"] == [] and inputs == embedder.inputs
