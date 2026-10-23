"""선택 보고서의 원문·본문 준비와 기존 색인 일치 검사."""
import os
import re
import hashlib
from src.common import ROOT, read_json, write_json, digest, utc_now
from src.collection import DartClient, download_zip, parse_and_save
from src.chunking import chunk_paths, load_company_chunks, load_settings, prepare_report_chunks, expansion_settings
from src.vector_store import GmsEmbeddings, open_store, state_path, sync_records, index_lock, embedding_profile

REPORT_FIELDS = [
    "corp_code",
    "corp_name",
    "report_name",
    "report_period",
    "receipt_no",
    "receipt_date",
    "source",
    "source_url",
    "collected_at",
]


def prepare_reports(data):
    company = data["company"]
    selected = data["selected_reports"]

    if not selected:
        raise ValueError("선택된 보고서가 없습니다.")

    receipt_ids = [item["receipt_no"] for item in selected]

    if len(receipt_ids) != len(set(receipt_ids)):
        raise ValueError("선택 결과에 중복 보고서가 있습니다.")

    if company["canonical_receipt_no"] not in receipt_ids:
        raise ValueError("필수 자체 사업보고서가 빠졌습니다.")

    # 파일 경로에 사용할 식별자와 기업·기간을 확인합니다.
    for item in selected:
        if not re.fullmatch(r"[0-9]{8}", item["corp_code"]):
            raise ValueError("기업 코드 형식이 올바르지 않습니다.")

        if not re.fullmatch(r"[0-9]{14}", item["receipt_no"]):
            raise ValueError("접수번호 형식이 올바르지 않습니다.")

        if (
            item["corp_code"] != company["corp_code"]
            or not re.fullmatch(r"[0-9]{4}\.(?:03|06|09|12)", item["report_period"])
        ):
            raise ValueError("선택 보고서의 기업 또는 기간이 다릅니다.")

    api_key = os.getenv("DART_API_KEY", "").strip()

    if not api_key:
        raise ValueError("DART_API_KEY 설정을 확인하세요.")

    client = DartClient(api_key)
    prepared = []

    try:
        for item in selected:
            report = {
                field: item[field]
                for field in REPORT_FIELDS
            }

            print(f"\n[{report['report_name']}]")
            print(f"접수번호: {report['receipt_no']}")

            expected_zip = (
                ROOT
                / "data/raw/disclosures"
                / report["corp_code"]
                / f"{report['receipt_no']}.zip"
            )
            zip_existed = expected_zip.exists()

            # 기존 ZIP이 있으면 무결성을 확인하고 재사용합니다.
            zip_path = download_zip(client, report)

            print(
                "원문: 기존 ZIP 검증 완료"
                if zip_existed
                else "원문: 새 ZIP 다운로드 완료"
            )

            # 유효한 파싱 캐시가 있으면 재사용합니다.
            parsed, json_path, text_path = parse_and_save(
                zip_path,
                report,
            )

            if (
                parsed["corp_code"] != report["corp_code"]
                or parsed["receipt_no"] != report["receipt_no"]
                or parsed["report_period"] != report["report_period"]
            ):
                raise ValueError("파싱 결과의 기업·접수번호·기간이 다릅니다.")

            if not parsed["sections"] or parsed["text_characters"] <= 0:
                raise ValueError("파싱된 본문이 비어 있습니다.")

            prepared.append({
                **report,
                "zip_path": zip_path.relative_to(ROOT).as_posix(),
                "parsed_json_path": json_path.relative_to(ROOT).as_posix(),
                "text_path": text_path.relative_to(ROOT).as_posix(),
                "zip_sha256": parsed["zip_sha256"],
                "parser_version": parsed["parser_version"],
                "section_count": parsed["section_count"],
                "text_characters": parsed["text_characters"],
                "selection_reason": item["selection_reason"],
                "version_status": item.get("version_status", "selected_from_search_snapshot"),
            })

            print("본문: 파싱 결과 준비 완료")
            print(f"섹션 수: {parsed['section_count']}개")
            print(f"본문 글자 수: {parsed['text_characters']:,}자")

    finally:
        client.session.close()

    return prepared, client.calls



def report_index_records(report, settings, root=ROOT):
    """대표 보고서는 전체 코퍼스 청크를 재사용하고 나머지는 문서별 전문 캐시를 씁니다."""
    _, manifest_path = chunk_paths(settings, root)
    manifest = read_json(manifest_path) if manifest_path.exists() else {"companies": {}}
    info = manifest["companies"].get(report["corp_code"], {})
    canonical = info.get("receipt_no") == report["receipt_no"]
    if canonical:
        records = load_company_chunks(report["corp_code"], settings, root)
    else:
        records, info = prepare_report_chunks(report, settings, root)
    parsed = read_json(root / report["parsed_json_path"])
    if hashlib.sha256((root / report["zip_path"]).read_bytes()).hexdigest() != report["zip_sha256"] or parsed["zip_sha256"] != report["zip_sha256"]:
        raise ValueError("보고서 ZIP과 파싱 결과의 해시가 다릅니다.")
    if (parsed["corp_code"], parsed["receipt_no"], parsed["report_period"], parsed["parser_version"]) != (
            report["corp_code"], report["receipt_no"], report["report_period"], report["parser_version"]):
        raise ValueError("보고서 준비 기록과 파싱 결과가 다릅니다.")
    sections = {section["section_id"]: section["text"] for section in parsed["sections"]}
    section_hashes = {key: digest(text) for key, text in sections.items()}
    intervals = {}
    for record in records:
        meta = record["metadata"]
        text = sections.get(meta["section_id"])
        if text is None or meta["corp_code"] != report["corp_code"] or meta["receipt_no"] != report["receipt_no"] or meta["report_period"] != report["report_period"]:
            raise ValueError("청크의 기업·접수번호·기간·섹션이 다릅니다.")
        body = record["text"].split("\n[본문]\n", 1)[1]
        if section_hashes[meta["section_id"]] != meta["source_section_sha256"] or digest(record["text"]) != meta["content_sha256"] or text[meta["start_index"]:meta["end_index"]] != body:
            raise ValueError("청크와 현재 파싱 본문이 일치하지 않습니다.")
        intervals.setdefault(meta["section_id"], []).append((meta["start_index"], meta["end_index"]))
    if settings.get("corpus_scope") == "full" or not canonical:
        for identifier, text in sections.items():
            end = 0
            for start, stop in sorted(intervals.get(identifier, [])):
                if start > end and text[end:start].strip():
                    raise ValueError("보고서 전문 청크 사이에 본문이 누락되었습니다.")
                end = max(end, stop)
            if text[end:].strip():
                raise ValueError("보고서 전문에 색인되지 않은 섹션 또는 끝부분이 있습니다.")
    return records, info, canonical


def _ensure_index_group(data, settings):
    embedder = GmsEmbeddings(settings)
    try:
        _, _, store = open_store(settings, embedder.profile, ROOT, documents_only=True)
        path = state_path(settings, embedder.profile, ROOT)
        state = read_json(path) if path.exists() else {}
        if not state.get("company_metadata_complete"):
            raise ValueError("기본 기업 메타데이터와 전문 확장 색인을 먼저 준비하세요.")
        state.setdefault("extra_documents", {})
        result = []
        for report in data["reports"]:
            if report["corp_code"] != data["company"]["corp_code"]:
                raise ValueError("다른 기업의 선택 문서는 색인할 수 없습니다.")
            records, info, canonical = report_index_records(report, settings, ROOT)
            bucket = state.setdefault("documents" if canonical else "extra_documents", {})
            key = report["corp_code"] if canonical else report["corp_code"] + ":" + report["receipt_no"]
            entry = {"complete": False, "receipt_no": report["receipt_no"],
                     "corp_code": report["corp_code"], "report_period": report["report_period"],
                     "chunk_count": len(records), "chunk_file_sha256": info["sha256"],
                     "corp_name": report["corp_name"]}
            if not canonical:
                entry["chunk_manifest"] = info
            bucket[key] = entry
            write_json(path, state)
            synced = sync_records(store, records, embedder, where={"$and": [
                {"corp_code": report["corp_code"]}, {"receipt_no": report["receipt_no"]}]})
            entry.update({"complete": True, "indexed_at": utc_now(), "last_sync": synced})
            state["document_chunk_count"] = store.count()
            write_json(path, state)
            result.append({"corp_code": report["corp_code"], "receipt_no": report["receipt_no"],
                "chunk_count": len(records), "chunk_file_sha256": info["sha256"],
                "status": "matched", "sync": synced})
        return result, embedder.stats()
    finally:
        embedder.close()


def ensure_prepared_index(data):
    """기존 자체 보고서는 large 재사용, 추가 보고서 전문은 small로 색인합니다."""
    settings = load_settings()
    mixed = settings.get("full_text_expansion", {}).get("enabled", False)
    groups = [(settings, data["reports"])]
    if mixed:
        canonical = data["company"]["canonical_receipt_no"]
        groups = [({**settings, "allow_new_embeddings": False}, [r for r in data["reports"] if r["receipt_no"] == canonical]),
                  ({**expansion_settings(settings), "allow_new_embeddings": settings.get("allow_document_embedding", False)},
                   [r for r in data["reports"] if r["receipt_no"] != canonical])]
    else:
        groups = [({**settings, "allow_new_embeddings": settings.get("allow_document_embedding", False)}, data["reports"])]
    results, usages = [], []
    with index_lock(ROOT):
        for config, reports in groups:
            if reports:
                checked, usage = _ensure_index_group({**data, "reports": reports}, config)
                results.extend(checked)
                usages.append(usage)
    return results, {"profiles": usages,
        "api_requests_this_run": sum(u.get("api_requests_this_run", 0) for u in usages),
        "new_embedding_inputs": sum(u.get("new_embedding_inputs", 0) for u in usages)}


def check_prepared_index(data):
    settings = load_settings()
    url, model = os.getenv("GMS_BASE_URL", "").strip().rstrip("/"), os.getenv("GMS_EMBEDDING_MODEL", "").strip()
    if not url or not model:
        raise ValueError("GMS 연결 주소와 임베딩 모델 설정을 확인하세요.")
    if not data["reports"]:
        raise ValueError("준비된 보고서가 없습니다.")
    checked = []
    for report in data["reports"]:
        if report["corp_code"] != data["company"]["corp_code"]:
            raise ValueError("준비 결과의 기업이 다릅니다.")
        config = expansion_settings(settings) if settings.get("full_text_expansion", {}).get("enabled") and report["receipt_no"] != data["company"]["canonical_receipt_no"] else settings
        report_profile = embedding_profile(url, config.get("embedding_model", model), config["embedding_dimensions"])
        state = read_json(state_path(config, report_profile, ROOT))
        records, info, canonical = report_index_records(report, config, ROOT)
        key = report["corp_code"] if canonical else report["corp_code"] + ":" + report["receipt_no"]
        entry = state.get("documents" if canonical else "extra_documents", {}).get(key, {})
        if not entry.get("complete") or entry.get("receipt_no") != report["receipt_no"] or entry.get("chunk_file_sha256") != info["sha256"] or entry.get("chunk_count") != len(records):
            raise ValueError("준비한 보고서와 완료된 색인 기록이 다릅니다.")
        checked.append({"corp_code": report["corp_code"], "receipt_no": report["receipt_no"],
            "chunk_count": len(records), "chunk_file_sha256": info["sha256"], "status": "matched"})
        print(f"{report['corp_name']} / {report['receipt_no']}: 전문·청크·색인 일치 ({len(records):,}개)")
    return checked
