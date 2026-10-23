"""기업별 large 기존 색인과 small 전문 확장 색인을 각각 검색해 순위를 병합합니다."""
import hashlib
import re
from src.common import ROOT, read_json
from src.chunking import (chunk_paths, chunk_profile, corpus_path, expansion_paths, expansion_settings,
                          load_company_chunks, load_settings)
from src.vector_store import GmsEmbeddings, open_store, query_evidence, state_path


def additional_chunk_hashes(code, receipts, canonical, settings):
    result = {}
    for receipt in receipts:
        if receipt == canonical:
            continue
        if not re.fullmatch(r"[0-9]{14}", receipt):
            raise ValueError("검색 접수번호 형식이 올바르지 않습니다.")
        folder = ROOT / "data/processed/report_chunks" / chunk_profile(settings) / code
        info = read_json(folder / f"{receipt}.json")
        if info["corp_code"] != code or info["receipt_no"] != receipt:
            raise ValueError("추가 보고서 청크의 기업 또는 접수번호가 다릅니다.")
        if hashlib.sha256((ROOT / info["filename"]).read_bytes()).hexdigest() != info["sha256"] or hashlib.sha256((ROOT / info["parsed_json_path"]).read_bytes()).hexdigest() != info["source_sha256"]:
            raise ValueError("추가 보고서 본문 또는 청크가 변경됐습니다. 색인을 갱신하세요.")
        result[receipt] = info["sha256"]
    return result


def merge_evidence(rankings, top_k):
    """서로 다른 임베딩 모델의 거리를 직접 비교하지 않고 RRF 순위로 병합합니다."""
    scores, records = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            meta = item["metadata"]
            key = (meta["corp_code"], meta["receipt_no"], meta.get("section_id", ""),
                   meta.get("start_index", item["id"]), meta.get("end_index", ""), meta.get("content_sha256", ""))
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            records.setdefault(key, item)
    return [{**records[key], "fusion_score": scores[key]} for key in
            sorted(scores, key=lambda key: (-scores[key], str(key)))[:top_k]]


def search_company(question, corp_code, receipt_nos=None):
    settings = load_settings()
    companies = read_json(ROOT / "data/companies.json")
    company = next((item for item in companies if item["corp_code"] == corp_code), None)
    if company is None:
        raise ValueError("기업 목록에 없는 기업입니다.")
    canonical = company["canonical_receipt_no"]
    receipts = list(dict.fromkeys(receipt_nos or [canonical]))
    if canonical not in receipts:
        raise ValueError("필수 자체 보고서가 검색 범위에서 빠졌습니다.")
    _, manifest_path = chunk_paths(settings)
    manifest = read_json(manifest_path)
    if hashlib.sha256(corpus_path(settings).read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("원문이 변경됐습니다. 청킹과 색인을 갱신하세요.")
    load_company_chunks(corp_code, settings)
    mixed = settings.get("full_text_expansion", {}).get("enabled", False)
    rank_settings = {**settings, "top_k": settings.get("candidate_k", 20)} if mixed else settings
    embedder = GmsEmbeddings(rank_settings)
    extra_embedder = None
    try:
        _, company_store, document_store = open_store(settings, embedder.profile)
        index_state = read_json(state_path(settings, embedder.profile))
        base_receipts = [canonical] if mixed else receipts
        result = query_evidence(corp_code, question, rank_settings, embedder, company_store, document_store,
            index_state, company, manifest["companies"][corp_code]["sha256"], receipt_nos=base_receipts,
            expected_report_shas=additional_chunk_hashes(corp_code, base_receipts, canonical, settings))
        usage = [embedder.stats()]
        for item in result["evidence"]:
            item["embedding_model"] = embedder.model
        rankings = [result["evidence"]]
        if mixed:
            small = expansion_settings(rank_settings)
            _, path = expansion_paths(settings)
            expansion = read_json(path)
            info = expansion["companies"][corp_code]
            if info["base_sha256"] != manifest["companies"][corp_code]["sha256"] or hashlib.sha256(corpus_path(small).read_bytes()).hexdigest() != expansion["source_sha256"]:
                raise ValueError("전문 확장 색인의 기준 코퍼스가 변경됐습니다.")
            load_company_chunks(corp_code, small, expansion=True)
            extra_embedder = GmsEmbeddings(small)
            _, _, small_store = open_store(small, extra_embedder.profile, documents_only=True)
            small_state = read_json(state_path(small, extra_embedder.profile))
            expanded = query_evidence(corp_code, question, small, extra_embedder, company_store, small_store,
                small_state, company, info["sha256"], receipt_nos=receipts,
                expected_report_shas=additional_chunk_hashes(corp_code, receipts, canonical, small))
            for item in expanded["evidence"]:
                item["embedding_model"] = extra_embedder.model
            rankings.append(expanded["evidence"])
            usage.append(extra_embedder.stats())
        result["evidence"] = merge_evidence(rankings, settings["top_k"])
        result["receipt_nos"] = receipts
        result["embedding_usage"] = {"profiles": usage, "api_requests_this_run": sum(s["api_requests_this_run"] for s in usage)}
        result["search_mode"] = "hybrid_model_rrf" if mixed else settings.get("search_mode", "similarity")
        return result
    finally:
        embedder.close()
        if extra_embedder is not None:
            extra_embedder.close()
