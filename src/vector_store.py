"""GMS 임베딩 캐시와 두 개의 Chroma 컬렉션을 관리합니다."""
import math
import argparse
import hashlib
import os
import re
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import APIError, OpenAI

from src.common import ROOT, digest, read_json, write_json, utc_now
from src.chunking import (chunk_profile, chunk_paths, corpus_path, expansion_paths, expansion_settings,
                          load_company_chunks, load_settings)


@contextmanager
def index_lock(root=ROOT):
    """프로세스 종료 시 해제되는 OS 잠금으로 색인 상태 파일의 동시 갱신을 막습니다."""
    path = root / "data/manifests/index.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("다른 색인 작업이 진행 중입니다. 완료 후 다시 질문해 주세요.") from None
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


def checked_vectors(response, expected, dimensions):
    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(expected)):
        raise ValueError("임베딩 응답의 입력 순서 또는 개수가 잘못되었습니다.")
    vectors = [item.embedding for item in items]
    if any(len(v) != dimensions or not all(math.isfinite(x) for x in v) or not any(v) for v in vectors):
        raise ValueError("임베딩 차원·수치 검증 실패")
    return vectors


def embedding_profile(provider_url, model, dimensions):
    return digest({"provider_url": provider_url.rstrip("/"), "model": model, "dimensions": dimensions})[:12]


class GmsEmbeddings:
    def __init__(self, settings, root=ROOT):
        load_dotenv(root / ".env")
        key = os.getenv("GMS_API_KEY", "").strip()
        self.url = os.getenv("GMS_BASE_URL", "").strip().rstrip("/")
        self.model = settings.get("embedding_model") or os.getenv("GMS_EMBEDDING_MODEL", "").strip()
        if not all([key, self.url, self.model]):
            raise ValueError(".env의 GMS_API_KEY, GMS_BASE_URL, GMS_EMBEDDING_MODEL을 확인하세요.")
        self.settings = settings
        self.dimensions = settings["embedding_dimensions"]
        self.profile = embedding_profile(self.url, self.model, self.dimensions)
        self.cache = root / settings["embedding_cache_path"] / self.profile
        self.client = OpenAI(api_key=key, base_url=self.url, timeout=60.0, max_retries=2)
        self.api_calls = self.input_count = self.reported_tokens = self.cache_hits = 0
        self.last_call = 0.0

    def close(self):
        self.client.close()

    def stats(self):
        return {"embedding_model": self.model, "dimensions": self.dimensions,
            "api_requests_this_run": self.api_calls, "new_embedding_inputs": self.input_count,
            "cache_hits": self.cache_hits, "provider_reported_tokens": self.reported_tokens}

    def embed(self, texts):
        if any(not t.strip() or len(t.encode("utf-8")) > self.settings["max_input_utf8_bytes"] for t in texts):
            raise ValueError("빈 입력 또는 임베딩 입력 크기 초과")
        output, missing = {}, {}
        for index, text in enumerate(texts):
            checksum = digest(text)
            path = self.cache / f"{checksum}.json"
            if path.exists():
                item = read_json(path)
                vector = item["embedding"]
                if item["text_sha256"] != checksum or len(vector) != self.dimensions or not all(math.isfinite(x) for x in vector) or not any(vector):
                    raise ValueError("임베딩 캐시 무결성 오류")
                output[checksum] = vector
                self.cache_hits += 1
            elif checksum not in output:
                missing[checksum] = text
        pairs = list(missing.items())
        if pairs and not self.settings.get("allow_new_embeddings", True):
            raise ValueError("새 문서의 임베딩이 필요하지만 크레딧 보호 모드로 중지되어 있습니다. 기존에 색인된 자료로 질문해 주세요.")
        batch_size = self.settings["embedding_batch_size"]
        gate = Lock()
        def request_batch(batch):
            with gate:
                time.sleep(max(0, self.settings["request_gap_seconds"] - (time.monotonic() - self.last_call)))
                self.last_call = time.monotonic()
                self.api_calls += 1
            try:
                response = self.client.embeddings.create(
                    model=self.model, input=[text for _, text in batch], encoding_format="float")
            except APIError as error:
                raise RuntimeError(f"GMS 임베딩 실패: {type(error).__name__}, HTTP {getattr(error, 'status_code', None)}") from None
            vectors = checked_vectors(response, len(batch), self.dimensions)
            with gate:
                self.input_count += len(batch)
                self.reported_tokens += getattr(response.usage, "total_tokens", 0) or 0
            values = {}
            for (checksum, _), vector in zip(batch, vectors):
                values[checksum] = vector
                write_json(self.cache / f"{checksum}.json", {"text_sha256": checksum,
                    "model": response.model, "profile": self.profile, "embedding": vector})
            return values
        batches = [pairs[start:start + batch_size] for start in range(0, len(pairs), batch_size)]
        workers = self.settings.get("embedding_workers", 1)
        if workers > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
                for values in executor.map(request_batch, batches):
                    output.update(values)
        else:
            for batch in batches:
                output.update(request_batch(batch))
        return [output[digest(text)] for text in texts]


def open_store(settings, embedding_profile, root=ROOT, *, documents_only=False):
    client = chromadb.PersistentClient(path=str(root / settings["database_path"]),
                                      settings=Settings(anonymized_telemetry=False))
    company_name = f"company_meta_{embedding_profile}"
    document_name = f"disclosure_docs_{embedding_profile}_{chunk_profile(settings)}"
    common = {"hnsw:space": "cosine", "embedding_profile": embedding_profile}
    companies = None if documents_only else client.get_or_create_collection(company_name, metadata=common, embedding_function=None)
    documents = client.get_or_create_collection(document_name,
        metadata={**common, "chunk_profile": chunk_profile(settings)}, embedding_function=None)
    return client, companies, documents


def company_records(companies):
    records = []
    for c in companies:
        metadata = {key: c[key] for key in ["corp_code", "corp_name", "stock_code", "industry_code",
            "selection_group", "canonical_receipt_no", "report_period", "collected_at"]}
        metadata.update({"source_type": "company_metadata", "source_url": c["report_source_url"],
            "company_api_source_url": c["source_url"], "content_sha256": digest(c["index_text"])})
        records.append({"id": c["corp_code"], "text": c["index_text"], "metadata": metadata})
    return records


def sync_records(collection, records, embedder, where=None):
    """변경분을 먼저 저장한 뒤, 성공한 범위에서 이전 레코드만 제거합니다."""
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("빈 입력 또는 중복 ID")
    existing = collection.get(where=where, include=["metadatas"])
    old = dict(zip(existing["ids"], existing["metadatas"]))
    changed = [r for r in records if old.get(r["id"], {}).get("content_sha256") != r["metadata"]["content_sha256"]]
    size = embedder.settings["embedding_batch_size"] * embedder.settings.get("embedding_workers", 1)
    for start in range(0, len(changed), size):
        batch = changed[start:start + size]
        vectors = embedder.embed([r["text"] for r in batch])
        collection.upsert(ids=[r["id"] for r in batch], embeddings=vectors,
            documents=[r["text"] for r in batch], metadatas=[r["metadata"] for r in batch])
        print(f"  {collection.name}: {min(start + size, len(changed))}/{len(changed)} 갱신", flush=True)
    # 수집시각 같은 속성만 바뀐 경우에는 임베딩을 다시 만들지 않습니다.
    changed_ids = {r["id"] for r in changed}
    metadata_updates = [r for r in records if r["id"] not in changed_ids and old[r["id"]] != r["metadata"]]
    for start in range(0, len(metadata_updates), size):
        batch = metadata_updates[start:start + size]
        collection.update(ids=[r["id"] for r in batch], metadatas=[r["metadata"] for r in batch])
    expected = {r["id"] for r in records}
    stale = sorted(set(old) - expected)
    for start in range(0, len(stale), 500):
        collection.delete(ids=stale[start:start + 500])
    actual = collection.get(where=where, include=["metadatas"])["ids"]
    if set(actual) != expected:
        raise ValueError("색인 완료 후 ID 검증 실패")
    return {"expected": len(records), "updated": len(changed), "unchanged": len(records) - len(changed),
            "removed_obsolete": len(stale), "actual": len(actual)}


def state_path(settings, profile, root=ROOT):
    return root / "data/manifests" / f"index_state_{profile}_{chunk_profile(settings)}.json"


def lexical_ranking(question, records, company_name=""):
    """공백을 제거한 본문에서 질문 핵심어 포함 여부와 희소성으로 순위를 만듭니다."""
    stopwords = {"어떻게", "무엇인가요", "무엇", "얼마인가요", "알려주세요", "설명해주세요", "있나요", "해주세요"}
    terms = []
    for term in re.findall(r"[가-힣A-Za-z0-9]{2,}", question):
        term = re.sub(r"(?:에서|에게|으로|은|는|이|가|의|과|와|을|를|로)$", "", term)
        if len(term) >= 2 and term not in stopwords and term not in company_name and term not in terms:
            terms.append(term)
    compact = [re.sub(r"\s+", "", r["text"].split("\n[본문]\n", 1)[-1]).lower() for r in records]
    scores = [0.0 for _ in records]
    for term in terms:
        matches = [i for i, text in enumerate(compact) if term.lower() in text]
        weight = math.log(1 + len(records) / (1 + len(matches)))
        for i in matches:
            scores[i] += weight
    return [records[i]["id"] for i in sorted(range(len(records)), key=lambda i: (-scores[i], records[i]["id"])) if scores[i] > 0]


def query_evidence(code, question, settings, embedder, companies, documents, state, current_company, expected_chunk_sha, receipt_nos=None, expected_report_shas=None):
    entry = state.get("documents", {}).get(code, {})
    receipt = current_company["canonical_receipt_no"]
    if not entry.get("complete") or entry.get("receipt_no") != receipt or entry.get("chunk_file_sha256") != expected_chunk_sha:
        raise ValueError("선택 기업의 현재 보고서 색인이 없습니다. index_data.py --corp-code로 먼저 색인하세요.")
    receipts = list(dict.fromkeys(receipt_nos or [receipt]))
    if receipt not in receipts:
        raise ValueError("필수 자체 사업보고서가 검색 범위에서 빠졌습니다.")
    chunk_count = entry["chunk_count"]
    for additional in receipts:
        if additional == receipt:
            continue
        extra = state.get("extra_documents", {}).get(code + ":" + additional, {})
        expected = (expected_report_shas or {}).get(additional)
        if not extra.get("complete") or extra.get("corp_code") != code or extra.get("receipt_no") != additional or not expected or extra.get("chunk_file_sha256") != expected:
            raise ValueError("선택한 추가 보고서의 색인 또는 청크가 준비되지 않았습니다.")
        chunk_count += extra["chunk_count"]
    scope = {"$and": [{"corp_code": code}, {"receipt_no": {"$in": receipts}}]}
    fixed = companies.get(ids=[code], include=["documents", "metadatas"])
    if fixed["ids"] != [code]:
        raise ValueError("선택 기업의 메타데이터 색인이 없습니다.")
    if fixed["metadatas"][0]["content_sha256"] != digest(current_company["index_text"]):
        raise ValueError("기업 메타데이터가 변경되었습니다. 색인을 갱신하세요.")
    if chunk_count == 0:
        return {"question": question, "corp_code": code, "receipt_no": receipt, "receipt_nos": receipts,
            "company_metadata": {"text": fixed["documents"][0], "metadata": fixed["metadatas"][0]},
            "evidence": [], "is_generated_answer": False, "searched_at": utc_now()}
    vector = embedder.embed([question])[0]
    hybrid = settings.get("search_mode", "similarity") == "hybrid"
    count = min(settings.get("candidate_k", settings["top_k"]) if hybrid else settings["top_k"], chunk_count)
    result = documents.query(query_embeddings=[vector],
        where=scope,
        n_results=count,
        include=["documents", "metadatas", "distances"])
    evidence = []
    for identifier, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                               result["metadatas"][0], result["distances"][0]):
        if meta["corp_code"] != code or meta["receipt_no"] not in receipts:
            raise ValueError("다른 기업 또는 보고서가 검색되었습니다.")
        evidence.append({"id": identifier, "text": text, "metadata": meta, "distance": distance})
    if hybrid:
        saved = documents.get(where=scope,
                              include=["documents", "metadatas"])
        records = [{"id": i, "text": text, "metadata": meta, "distance": None}
                   for i, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"])]
        keyword_ids = lexical_ranking(question, records, current_company["corp_name"])[:count]
        by_id = {r["id"]: r for r in records}
        by_id.update({r["id"]: r for r in evidence})
        scores = {}
        # Reciprocal Rank Fusion: 벡터·핵심어 순위를 합칩니다. 거리 점수와 섞지 않습니다.
        for ranking in [[r["id"] for r in evidence], keyword_ids]:
            for rank, identifier in enumerate(ranking, 1):
                scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank)
        evidence = [{**by_id[i], "retrieval_score": scores[i]} for i in
                    sorted(scores, key=lambda i: (-scores[i], i))[:settings["top_k"]]]
    return {"question": question, "corp_code": code, "receipt_no": receipt, "receipt_nos": receipts,
        "company_metadata": {"text": fixed["documents"][0], "metadata": fixed["metadatas"][0]},
        "evidence": evidence, "search_mode": settings.get("search_mode", "similarity"),
        "is_generated_answer": False, "searched_at": utc_now()}


def available_reports(root, settings, provider_url, embedding_model):
    if not provider_url or not embedding_model or not (root / settings["database_path"]).exists():
        return {}
    profile = embedding_profile(provider_url, embedding_model, settings["embedding_dimensions"])
    state_path = root / "data/manifests" / f"index_state_{profile}_{chunk_profile(settings)}.json"
    _, manifest_path = chunk_paths(settings, root)
    if not state_path.exists() or not manifest_path.exists():
        return {}
    state, manifest = read_json(state_path), read_json(manifest_path)
    if not state.get("company_metadata_complete"):
        return {}
    mixed = settings.get("full_text_expansion", {}).get("enabled", False)
    small_state, expansion = {}, {}
    if mixed:
        small = expansion_settings(settings)
        small_profile = embedding_profile(provider_url, small["embedding_model"], small["embedding_dimensions"])
        small_path = root / "data/manifests" / f"index_state_{small_profile}_{chunk_profile(small)}.json"
        _, expansion_manifest = expansion_paths(settings, root)
        if not small_path.exists() or not expansion_manifest.exists():
            return {}
        small_state, expansion = read_json(small_path), read_json(expansion_manifest)
    ready = {}
    for code, entry in state.get("documents", {}).items():
        chunk = manifest["companies"].get(code, {})
        if (entry.get("complete") and entry.get("receipt_no") == chunk.get("receipt_no")
                and entry.get("chunk_file_sha256") == chunk.get("sha256")):
            fingerprint = profile + ":" + entry["chunk_file_sha256"]
            if mixed:
                extra, info = small_state.get("documents", {}).get(code, {}), expansion.get("companies", {}).get(code, {})
                if not extra.get("complete") or extra.get("receipt_no") != entry.get("receipt_no") or extra.get("chunk_file_sha256") != info.get("sha256") or info.get("base_sha256") != chunk.get("sha256"):
                    continue
                fingerprint += ":" + small_profile + ":" + info["sha256"]
            ready[code] = {**entry, "fingerprint": fingerprint}
    return ready

def _index_data_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corp-code", action="append", help="추가 색인할 기업 고유번호. 여러 번 지정 가능")
    parser.add_argument("--all", action="store_true", help="전체 기업의 공시까지 색인 (추가 GMS 사용량 발생)")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--full-corpus", action="store_true", help="전체 본문 프로필을 색인")
    parser.add_argument("--expansion", action="store_true", help="기존 large를 유지하며 전문 확장분을 small로 색인")
    parser.add_argument("--batch-size", type=int, choices=range(1,129), help="이번 실행의 임베딩 배치 크기")
    parser.add_argument("--allow-new-embeddings", action="store_true", help="새 임베딩을 GMS에 요청하도록 명시적으로 허용")
    parser.add_argument("--workers", type=int, choices=[1, 2, 4], default=1, help="호출 간격을 공유하는 임베딩 동시 요청 수")
    args = parser.parse_args()
    if args.expansion and (args.full_corpus or args.metadata_only):
        raise ValueError("--expansion은 --full-corpus, --metadata-only와 함께 쓸 수 없습니다.")
    if args.all and (args.corp_code or args.metadata_only):
        raise ValueError("--all은 --corp-code, --metadata-only와 함께 쓸 수 없습니다.")
    settings = load_settings()
    base_settings = settings
    if args.expansion:
        settings = expansion_settings(settings)
    if args.full_corpus:
        settings = {**settings, "corpus_scope": "full"}
    settings = {**settings, "embedding_workers": args.workers,
                "allow_new_embeddings": args.allow_new_embeddings}
    if args.batch_size:
        settings = {**settings, "embedding_batch_size": args.batch_size}
    companies = read_json(ROOT / "data/companies.json")
    by_code = {c["corp_code"]: c for c in companies}
    codes = [] if args.metadata_only else list(by_code) if args.all else args.corp_code or [settings["pilot_corp_code"]]
    codes = list(dict.fromkeys(codes))
    if any(c not in by_code for c in codes):
        raise ValueError("data/companies.json에 없는 기업 고유번호입니다.")
    chunk_manifest = None
    if codes:
        _, manifest_path = expansion_paths(settings) if args.expansion else chunk_paths(settings)
        chunk_manifest = read_json(manifest_path)
        source = corpus_path(settings)
        if hashlib.sha256(source.read_bytes()).hexdigest() != chunk_manifest["source_sha256"]:
            raise ValueError("원문 데이터가 바뀌었습니다. prepare_chunks.py를 다시 실행하세요.")
        for code in codes:
            if chunk_manifest["companies"].get(code, {}).get("receipt_no") != by_code[code]["canonical_receipt_no"]:
                raise ValueError("기업 목록의 대표 보고서가 바뀌었습니다. 청크를 다시 준비하세요.")
    embedder = GmsEmbeddings(settings)
    try:
        _, company_store, document_store = open_store(settings, embedder.profile, documents_only=args.expansion)
        if args.expansion:
            base_embedder = GmsEmbeddings(base_settings)
            try:
                _, company_store, _ = open_store(base_settings, base_embedder.profile)
            finally:
                base_embedder.close()
            if company_store.count() != len(companies):
                raise ValueError("기존 large 기업 메타데이터 색인을 먼저 준비하세요.")
        path = state_path(settings, embedder.profile)
        state = read_json(path) if path.exists() else {"documents": {}}
        state.update({"embedding_profile": embedder.profile,
            "company_collection": company_store.name, "document_collection": document_store.name,
            "company_metadata_complete": False})
        write_json(path, state)
        meta_result = ({"reused_existing_large": company_store.count()} if args.expansion else
                       sync_records(company_store, company_records(companies), embedder))
        state["company_metadata_complete"] = True
        state["company_count"] = company_store.count()
        state["company_metadata_result"] = meta_result
        write_json(path, state)
        results = {}
        for code in codes:
            records = load_company_chunks(code, settings, expansion=args.expansion)
            info = chunk_manifest["companies"][code]
            state["documents"][code] = {"complete": False, "receipt_no": info["receipt_no"],
                "chunk_file_sha256": info["sha256"], "chunk_count": len(records),
                "corp_name": by_code[code]["corp_name"]}
            write_json(path, state)
            print(f"[문서 색인] {by_code[code]['corp_name']} / {len(records)}개 청크", flush=True)
            results[code] = (sync_records(document_store, records, embedder,
                where={"$and": [{"corp_code": code}, {"receipt_no": info["receipt_no"]}]})
                if records else {"expected": 0, "updated": 0, "actual": 0})
            state["documents"][code].update({"complete": True, "indexed_at": utc_now(), "last_sync": results[code]})
            write_json(path, state)
        state["updated_at"] = utc_now()
        state["last_run_embedding_usage"] = embedder.stats()
        state["document_chunk_count"] = document_store.count()
        write_json(path, state)
        report = {"completed_at": utc_now(), "company_metadata": meta_result, "selected_documents": results,
            "company_collection": company_store.name, "document_collection": document_store.name,
            "indexed_company_reports": sum(bool(x.get("complete")) for x in state["documents"].values()),
            "document_chunk_count": document_store.count(), "embedding_usage": embedder.stats()}
        write_json(ROOT / "data/processed/indexing_result.json", report)
        print(f"INDEX_READY companies={company_store.count()} document_chunks={document_store.count()}")
        print(f"NEW_EMBEDDING_INPUTS={embedder.input_count} API_REQUESTS={embedder.api_calls}")
        return 0
    finally:
        embedder.close()


def index_data_main():
    with index_lock():
        return _index_data_main()
