"""수집 데이터·청크·실제 검색 결과의 점검 도구."""
import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from src.common import ROOT, read_json, write_json, digest, utc_now
from src.disclosure_parser import normalize_xml, normalized_text
from src.chunking import (report_sections, corpus_path, chunk_paths, expansion_paths, expansion_settings,
                          load_company_chunks, load_settings)
from src.vector_store import GmsEmbeddings, open_store, query_evidence, state_path, embedding_profile

def validation_settings():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--expansion", action="store_true")
    parser.add_argument("--verify-store", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    if args.full_corpus:
        settings = {**settings, "corpus_scope": "full"}
    return {**settings, "validate_expansion": args.expansion, "verify_store": args.verify_store}


def validate_dataset_main():
    settings = validation_settings()
    companies = read_json(ROOT / "data/companies.json")
    reports = read_json(ROOT / "data/processed/report_candidates.json")["reports"]
    documents = read_json(ROOT / "data/processed/disclosure_manifest.json")["documents"]
    summary = read_json(ROOT / "data/processed/collection_summary.json")
    problems, source_checks = [], []
    report_by_company = {r["corp_code"]: r for r in reports}
    document_by_company = {d["corp_code"]: d for d in documents}
    company_ids = [c["corp_code"] for c in companies]
    if not (len(companies) >= 100 and len(company_ids) == len(set(company_ids))):
        problems.append("기업 100개 이상 또는 고유 ID 검증 실패")
    if not (len(companies) == len(reports) == len(documents)):
        problems.append("기업·보고서·문서 수 불일치")
    section_ids = set()
    rag_count, rag_chars, all_chars, all_tables = 0, 0, 0, 0
    source_groups = Counter()
    rag_path = corpus_path(settings)
    temporary = rag_path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as corpus:
        for company in companies:
            code = company["corp_code"]
            report = report_by_company.get(code)
            document = document_by_company.get(code)
            if not report or not document:
                problems.append(f"{code}: 보고서 연결 없음")
                continue
            try:
                zip_path = ROOT / document["zip_path"]
                parsed = read_json(ROOT / document["processed_path"])
                text = (ROOT / document["text_path"]).read_text(encoding="utf-8")
                if hashlib.sha256(zip_path.read_bytes()).hexdigest() != document["zip_sha256"]:
                    problems.append(f"{code}: ZIP 해시 불일치")
                if company["canonical_receipt_no"] != document["receipt_no"] or report["receipt_no"] != document["receipt_no"]:
                    problems.append(f"{code}: 접수번호 연결 불일치")
                if document["report_period"] != "2025.12" or document["receipt_date"] > summary["search_end"]:
                    problems.append(f"{code}: 기간 범위 불일치")
                if document["version_status"] != "latest_available_in_search_snapshot":
                    problems.append(f"{code}: 최종 보고서 상태 불일치")
                if parsed["section_count"] != len(parsed["sections"]) or len(text) != parsed["text_characters"]:
                    problems.append(f"{code}: 추출 본문 길이·섹션 수 불일치")
                if "\ufffd" in text:
                    problems.append(f"{code}: 문자 디코딩 대체문자 발견")
                if not document["source_url"].endswith(document["receipt_no"]):
                    problems.append(f"{code}: 원문 링크 접수번호 불일치")
                excerpt_section = next(s for s in parsed["sections"] if s["section_id"] == company["business_overview_section_id"])
                expected_excerpt = re.sub(r"\s+", " ", excerpt_section["text"]).strip()[:1800]
                if expected_excerpt != company["business_overview_excerpt"]:
                    problems.append(f"{code}: 사업 개요 발췌가 원문과 불일치")
                all_chars += len(text)
                all_tables += parsed["table_count"]
                group = company["selection_group"]
                # 각 선정 그룹의 두 기업을 원본 XML과 추가 대조합니다.
                if source_groups[group] < 2:
                    with zipfile.ZipFile(zip_path) as archive:
                        raw = archive.read(parsed["main_xml_member"])
                    source_root, _ = normalize_xml(raw)
                    source_company = source_root.find("COMPANY-NAME")
                    if source_company is None or source_company.get("AREGCIK") != code:
                        problems.append(f"{code}: 원본 XML 회사 ID 불일치")
                    paragraphs = [normalized_text(p) for p in source_root.findall(".//BODY//P")]
                    paragraphs = [p for p in paragraphs if len(p) > 25 and "|" not in p]
                    flat_text = re.sub(r"\s+", " ", text)
                    checked = 0
                    for index in sorted(set([0, len(paragraphs)//4, len(paragraphs)//2, 3*len(paragraphs)//4, len(paragraphs)-1])):
                        if paragraphs and paragraphs[index] not in flat_text:
                            problems.append(f"{code}: 원문 문단 표본 누락 {index}")
                        elif paragraphs:
                            checked += 1
                    source_checks.append({"corp_code":code,"company":company["corp_name"],"paragraphs_checked":checked})
                    source_groups[group] += 1
                full_records = {r["section_id"]: r for r in report_sections(report, parsed)} if settings.get("corpus_scope") == "full" else {}
                for section in parsed["sections"]:
                    identifier = code + ":" + section["section_id"]
                    if identifier in section_ids:
                        problems.append(f"{code}: 섹션 ID 중복")
                    section_ids.add(identifier)
                    path = section["section_path"]
                    title = section["title"]
                    selected = (
                        settings.get("corpus_scope") == "full"
                        or "사업의 내용" in title
                        or "경영진단" in title
                        or ("회사의 개요" in title and len(path) > 1 and "1. 회사의 개요" in path[-1])
                        or ("재무에 관한 사항" in title and len(path) > 1 and re.match(r"(?:1\.|2\.|4\.)", path[1]))
                        or ("임원 및 직원" in title and "1. 임원 및 직원" in title)
                    )
                    if not selected:
                        continue
                    record = {"corp_code":code,"corp_name":company["corp_name"],
                        "document_id":parsed["document_id"],"receipt_no":report["receipt_no"],
                        "report_name":report["report_name"],"report_period":report["report_period"],
                        "receipt_date":report["receipt_date"],"source_type":"disclosure_xml",
                        "source_url":report["source_url"],"version_status":report["version_status"],
                        **section}
                    if full_records:
                        record = full_records[section["section_id"]]
                    corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rag_count += 1
                    rag_chars += len(section["text"])
            except Exception as error:
                problems.append(f"{code}: {type(error).__name__}")
    if not problems:
        temporary.replace(rag_path)
    result = {"validated_at":datetime.now(timezone.utc).isoformat(),
        "passed":not problems,"company_count":len(companies),"report_count":len(reports),
        "document_count":len(documents),"section_count":len(section_ids),
        "table_count":all_tables,"full_text_characters":all_chars,
        "rag_section_count":rag_count,"rag_text_characters":rag_chars,
        "raw_source_sample_checks":source_checks,"problems":problems,
        "corpus_scope":settings.get("corpus_scope", "selected"),
        "corpus_is_chunked":False,"embedding_or_indexing_performed":False}
    (ROOT / "data/processed/dataset_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


def validate_chunks_main():
    settings = validation_settings()
    if settings.get("validate_expansion"):
        return validate_expansion(settings)
    _, path = chunk_paths(settings)
    manifest = read_json(path)
    source = corpus_path(settings)
    issues, identifiers, total = [], set(), 0
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        issues.append("원문과 청크 매니페스트 해시 불일치")
    sections = {}
    with source.open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            sections[(item["corp_code"], item["section_id"])] = item
    section_hashes = {key: digest(section["text"]) for key, section in sections.items()}
    intervals = defaultdict(list)
    omitted_whitespace = 0
    for code, info in manifest["companies"].items():
        for item in load_company_chunks(code, settings):
            total += 1
            meta = item["metadata"]
            key = (code, meta["section_id"])
            source_section = sections[key]
            body = item["text"].split("\n[본문]\n", 1)[1]
            start, end = meta["start_index"], meta["end_index"]
            if source_section["text"][start:end] != body:
                issues.append(f"{item['id']}: 본문·원문 위치 불일치")
            if meta["content_sha256"] != digest(item["text"]):
                issues.append(f"{item['id']}: 청크 내용 해시 불일치")
            if meta["source_section_sha256"] != section_hashes[key]:
                issues.append(f"{item['id']}: 원문 섹션 해시 불일치")
            if meta["receipt_no"] != info["receipt_no"] or meta["source_url"] != source_section["source_url"]:
                issues.append(f"{item['id']}: 접수번호·출처 불일치")
            if len(body) > settings["chunk_size"] or len(item["text"].encode("utf-8")) > settings["max_input_utf8_bytes"]:
                issues.append(f"{item['id']}: 길이 제한 초과")
            if item["id"] in identifiers:
                issues.append(f"{item['id']}: ID 중복")
            identifiers.add(item["id"])
            intervals[key].append((start, end))
    for key, section in sections.items():
        end = 0
        for start, stop in sorted(intervals[key]):
            if start > end:
                gap = section["text"][end:start]
                if gap.strip():
                    issues.append(f"{key}: 청크 사이 원문 누락")
                else:
                    omitted_whitespace += len(gap)
            end = max(end, stop)
        if end != len(section["text"]):
            tail = section["text"][end:]
            if tail.strip():
                issues.append(f"{key}: 원문 끝부분 누락")
            else:
                omitted_whitespace += len(tail)
    result = {"passed": not issues, "company_count": len(manifest["companies"]),
        "section_count": len(sections), "chunk_count": total,
        "chunk_size": settings["chunk_size"], "chunk_overlap": settings["chunk_overlap"],
        "omitted_boundary_whitespace_characters": omitted_whitespace,
        "api_calls": 0, "issues": issues}
    write_json(ROOT / "data/processed/chunk_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


def validate_expansion(settings):
    """large + small의 합집합이 전문 청크 전체를 포함하는지 API 없이 대조합니다."""
    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    small = expansion_settings(settings)
    _, path = expansion_paths(settings)
    manifest = read_json(path)
    _, base_path = chunk_paths(settings)
    base_manifest = read_json(base_path)
    _, full_path = chunk_paths(small)
    full_manifest = read_json(full_path)
    problems, checked = [], []
    if hashlib.sha256(corpus_path(small).read_bytes()).hexdigest() != manifest["source_sha256"]:
        problems.append("전문 원본 해시 불일치")
    def identity(record):
        meta = record["metadata"]
        return tuple(meta[k] for k in ["corp_code", "receipt_no", "section_id", "start_index", "end_index", "content_sha256"])
    stores = []
    if settings.get("verify_store"):
        for config, model in [(settings, os.getenv("GMS_EMBEDDING_MODEL", "")), (small, small["embedding_model"])]:
            profile = embedding_profile(os.getenv("GMS_BASE_URL", "").strip(), model, config["embedding_dimensions"])
            _, _, store = open_store(config, profile)
            state = read_json(state_path(config, profile))
            sample = store.get(limit=1, include=["embeddings"])["embeddings"]
            if len(sample) and len(sample[0]) != config["embedding_dimensions"]:
                problems.append(f"{model}: 벡터 차원 불일치")
            stores.append((store, state))
    large_total = small_total = 0
    section_keys = set()
    for code, info in manifest["companies"].items():
        large = load_company_chunks(code, settings)
        extra = load_company_chunks(code, small, expansion=True)
        full = load_company_chunks(code, small)
        section_keys.update((code, r["metadata"]["section_id"]) for r in full)
        if info["base_sha256"] != base_manifest["companies"][code]["sha256"]:
            problems.append(f"{code}: 기존 large 청크 해시 불일치")
        union = {identity(r) for r in large + extra}
        if union != {identity(r) for r in full}:
            problems.append(f"{code}: 두 모델의 합집합과 전문 범위 불일치")
        if {r["metadata"]["content_sha256"] for r in large} & {r["metadata"]["content_sha256"] for r in extra}:
            problems.append(f"{code}: large와 small의 본문 중복")
        for records, (store, state) in zip([large, extra], stores):
            scope = {"$and": [{"corp_code": code}, {"receipt_no": info["receipt_no"]}]}
            actual = store.get(where=scope, include=["metadatas", "documents"])
            expected = {r["id"]: r for r in records}
            if set(actual["ids"]) != set(expected) or not state.get("documents", {}).get(code, {}).get("complete"):
                problems.append(f"{code}: {store.name} ID·완료 상태 불일치")
            for identifier, text, meta in zip(actual["ids"], actual["documents"], actual["metadatas"]):
                record = expected.get(identifier)
                if record is None or text != record["text"] or meta != record["metadata"]:
                    problems.append(f"{code}: 색인 본문·메타데이터 불일치")
                    break
        large_total += len(large)
        small_total += len(extra)
        checked.append({"corp_code": code, "large": len(large), "small": len(extra), "full": len(full)})
    result = {"passed": not problems, "company_count": len(checked), "section_count": len(section_keys),
        "large_existing_chunks": large_total, "small_expansion_chunks": small_total,
        "full_chunk_count": full_manifest["chunk_count"], "api_calls": 0,
        "store_verified": bool(settings.get("verify_store")), "companies": checked, "issues": problems}
    write_json(ROOT / "data/processed/expansion_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


QUESTIONS = [
    ("카카오뱅크가 제공하는 예금과 대출 서비스는 무엇인가요?", "사업의 내용", ["예금", "대출"]),
    ("카카오뱅크의 직원 수와 평균 근속연수는 어떻게 되나요?", "임원 및 직원", ["직원수", "근속연수"]),
    ("카카오뱅크의 2025년 이자수익과 이자비용은 얼마인가요?", "재무", ["이자수익", "이자비용"]),
]


def verify_retrieval_main():
    settings = load_settings()
    code = settings["pilot_corp_code"]
    companies = read_json(ROOT / "data/companies.json")
    company = next(c for c in companies if c["corp_code"] == code)
    records = load_company_chunks(code, settings)
    _, manifest_path = chunk_paths(settings)
    chunk_manifest = read_json(manifest_path)
    issues, results = [], []
    embedder = GmsEmbeddings(settings)
    try:
        _, meta_store, doc_store = open_store(settings, embedder.profile)
        state = read_json(state_path(settings, embedder.profile))
        if set(meta_store.get()["ids"]) != {c["corp_code"] for c in companies}:
            issues.append("기업 메타데이터 ID 목록 불일치")
        saved = doc_store.get(where={"corp_code": code}, include=["documents", "metadatas"])
        expected = {r["id"]: r for r in records}
        if set(saved["ids"]) != set(expected):
            issues.append("기업별 공시 청크 ID 목록 불일치")
        for identifier, text, meta in zip(saved["ids"], saved["documents"], saved["metadatas"]):
            if identifier not in expected or digest(text) != meta["content_sha256"] or text != expected[identifier]["text"]:
                issues.append("저장 본문·해시 불일치")
        for question, expected_topic, expected_terms in QUESTIONS:
            result = query_evidence(code, question, settings, embedder, meta_store, doc_store,
                state, company, chunk_manifest["companies"][code]["sha256"])
            topic_found = any(expected_topic in item["metadata"]["title"] for item in result["evidence"])
            body = re.sub(r"\s+", "", "\n".join(item["text"].split("\n[본문]\n", 1)[-1] for item in result["evidence"]))
            term_checks = {term: term in body for term in expected_terms}
            if not result["evidence"] or not topic_found or not all(term_checks.values()):
                issues.append(f"대표 질문의 관련 섹션 검색 실패: {question}")
            for item in result["evidence"]:
                meta = item["metadata"]
                if meta["corp_code"] != code or meta["receipt_no"] != company["canonical_receipt_no"]:
                    issues.append("검색 기업·기간 필터 오류")
                if not meta["source_url"].endswith(meta["receipt_no"]):
                    issues.append("검색 근거의 원문 링크 불일치")
            result["expected_topic_found"] = topic_found
            result["expected_terms_in_body"] = term_checks
            results.append(result)
            print(f"[{topic_found}] {question}")
            for item in result["evidence"]:
                print("  " + item["metadata"]["title"])
        report = {"verified_at": utc_now(), "passed": not issues,
            "company_metadata_count": meta_store.count(), "pilot_corp_code": code,
            "pilot_chunk_count": len(saved["ids"]), "top_k": settings["top_k"],
            "query_checks": results, "issues": issues, "embedding_usage": embedder.stats(),
            "answer_generation_tested": False}
        write_json(ROOT / "data/processed/retrieval_verification.json", report)
        print(f"RETRIEVAL_VERIFIED passed={not issues} questions={len(results)}")
        return 0 if not issues else 1
    finally:
        embedder.close()
