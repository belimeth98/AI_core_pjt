"""기업·섹션 경계를 지키며 청크를 만들고 표의 앞부분을 문맥으로 보존합니다."""
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common import ROOT, digest, read_json, write_json
CHUNKER_VERSION = "section-table-context-v1"


def load_settings(root=ROOT):
    settings = read_json(root / "config/retrieval.json")
    size, overlap = settings["chunk_size"], settings["chunk_overlap"]
    if not (100 <= size <= 1500 and 0 <= overlap < size):
        raise ValueError("청크 크기는 100~1500, 중복 길이는 청크 크기 미만이어야 합니다.")
    if not (1 <= settings["embedding_batch_size"] <= 128):
        raise ValueError("임베딩 배치는 1~128개로 설정하세요.")
    if not (1 <= settings["top_k"] <= 20 and settings.get("candidate_k", 20) >= settings["top_k"]):
        raise ValueError("Top-k는 1~20, 후보 수는 Top-k 이상이어야 합니다.")
    if settings.get("search_mode", "similarity") not in {"similarity", "hybrid"}:
        raise ValueError("검색 방식은 similarity 또는 hybrid로 설정하세요.")
    if not (1 <= settings["max_input_utf8_bytes"] <= 7500 and settings["request_gap_seconds"] >= 0):
        raise ValueError("입력 바이트 제한은 7500 이하, 호출 간격은 0 이상이어야 합니다.")
    return settings


def chunk_profile(settings):
    identity = {"version": CHUNKER_VERSION, "size": settings["chunk_size"],
                "overlap": settings["chunk_overlap"]}
    if settings.get("corpus_scope") == "full":
        identity["scope"] = "full-report-v2-gap-coverage"
    return digest(identity)[:12]


def corpus_path(settings, root=ROOT):
    name = "full_sections.jsonl" if settings.get("corpus_scope") == "full" else "rag_sections.jsonl"
    return root / "data/processed" / name


def expansion_settings(settings):
    config = settings.get("full_text_expansion", {})
    return {**settings, "corpus_scope": "full", "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
            "embedding_dimensions": config.get("embedding_dimensions", 1536),
            "embedding_batch_size": config.get("embedding_batch_size", 64),
            "embedding_workers": config.get("embedding_workers", 2)}


def expansion_paths(settings, root=ROOT):
    folder = root / "data/processed/expansion_chunks" / chunk_profile(expansion_settings(settings))
    return folder, folder / "manifest.json"


def report_sections(report, parsed):
    """파싱된 본문의 모든 섹션을 출처가 있는 색인 입력으로 구성합니다."""
    for section in parsed["sections"]:
        yield {**{key: report[key] for key in ["corp_code", "corp_name", "receipt_no",
            "report_name", "report_period", "receipt_date", "source_url", "collected_at"]},
            "document_id": parsed["document_id"], "source_type": "disclosure_xml",
            "version_status": report.get("version_status", "selected_from_search_snapshot"), **section}


def table_contexts(text):
    contexts = []
    for match in re.finditer(r"(?m)^\[표:.*?\n", text):
        start = match.start()
        end = text.find("\n\n", start)
        if end < 0:
            end = len(text)
        lines = text[start:end].splitlines()[1:4]
        # 값이나 제목을 추측하지 않고 실제 표 앞부분과 직전 단위 문장만 붙입니다.
        heading = "\n".join(lines)
        if len(heading) > 450:
            heading = heading[:450] + " … [긴 표 앞부분 발췌]"
        preceding = text[max(0, start - 400):start].split("\n\n")
        unit = next((p.strip() for p in reversed(preceding) if "단위" in p), "")[:150]
        context = "\n".join(x for x in [unit, heading] if x)
        contexts.append((start, end, context))
    return contexts


def build_chunks(section, settings):
    text = section["text"]
    profile = chunk_profile(settings)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["chunk_size"], chunk_overlap=settings["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""], add_start_index=True,
        strip_whitespace=False,
    )
    tables = table_contexts(text)
    docs = splitter.create_documents([text])
    pieces = [(doc.page_content, doc.metadata["start_index"]) for doc in docs]
    if settings.get("corpus_scope") == "full":
        # 반복되는 긴 표 구분문에서는 LangChain의 위치 추적이 다음 동일 문구로
        # 넘어갈 수 있습니다. 원문 위치의 비공백 누락 구간을 그대로 보충합니다.
        end, gaps = 0, []
        for content, start in sorted(pieces, key=lambda item: item[1]):
            if start > end and text[end:start].strip():
                gaps.append((end, start))
            end = max(end, start + len(content))
        if text[end:].strip():
            gaps.append((end, len(text)))
        for first, last in gaps:
            start = first
            while start < last:
                stop = min(last, start + settings["chunk_size"])
                pieces.append((text[start:stop], start))
                if stop == last:
                    break
                start = stop - settings["chunk_overlap"]
    output = []
    section_sha = digest(text)
    for index, (content, start) in enumerate(pieces):
        end = start + len(content)
        if start < 0 or text[start:end] != content:
            raise ValueError("청크와 원문 위치가 일치하지 않습니다.")
        table = next((item for item in tables if item[0] <= start < item[1]), None)
        frequency = "반기" if "반기보고서" in section["report_name"] else "분기" if "분기보고서" in section["report_name"] else "연간"
        prefix = (f"기업: {section['corp_name']}\n보고서 기준: {section['report_period']} ({frequency})\n"
                  f"섹션: {section['title']}\n")
        if table:
            prefix += "표 앞부분 원문 참고(현재 청크 수치와의 대응은 원문 확인):\n" + table[2] + "\n"
        document = prefix + "\n[본문]\n" + content
        if len(document.encode("utf-8")) > settings["max_input_utf8_bytes"]:
            raise ValueError("임베딩 입력 크기 초과: 청크 크기를 줄이세요.")
        record = {"id": f"{section['corp_code']}:{section['section_id']}:{profile}:{index:05d}:{digest(document)[:12]}",
                  "text": document, "metadata": {
            key: section[key] for key in ["corp_code", "corp_name", "document_id", "receipt_no",
                "report_name", "report_period", "receipt_date", "source_type", "source_url",
                "section_id", "title", "version_status"]}}
        record["metadata"].update({"chunk_index": index, "chunk_profile": profile,
            "source_section_sha256": section_sha, "content_sha256": digest(document),
            "start_index": start, "end_index": end, "table_context_included": bool(table)})
        record["metadata"]["collected_at"] = section.get("collected_at", "")
        output.append(record)
    return output


def chunk_paths(settings, root=ROOT):
    folder = root / "data/processed/chunks" / chunk_profile(settings)
    return folder, folder / "manifest.json"


def load_company_chunks(code, settings, root=ROOT, *, expansion=False):
    folder, manifest_path = expansion_paths(settings, root) if expansion else chunk_paths(settings, root)
    manifest = read_json(manifest_path)
    item = manifest["companies"].get(code)
    if item is None:
        raise ValueError(f"기업 {code}의 청크가 없습니다. prepare_chunks.py를 실행하세요.")
    path = folder / item["filename"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("청크 파일 해시 불일치: prepare_chunks.py를 다시 실행하세요.")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != item["chunk_count"] or not all(r["metadata"]["corp_code"] == code for r in records):
        raise ValueError("청크 개수 또는 기업 ID가 잘못되었습니다.")
    return records


def prepare_expansion_chunks(settings, root=ROOT):
    """기존 large 문서와 본문이 같은 청크는 제외해 새 본문만 small 색인에 보냅니다."""
    full = expansion_settings(settings)
    _, full_path = chunk_paths(full, root)
    full_manifest = read_json(full_path)
    folder, manifest_path = expansion_paths(settings, root)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"chunk_profile": chunk_profile(full), "base_chunk_profile": chunk_profile(settings),
                "source_sha256": full_manifest["source_sha256"], "companies": {}, "scope": "full-minus-existing-large"}
    excluded = 0
    for code, info in full_manifest["companies"].items():
        base = load_company_chunks(code, settings, root)
        large_hashes = {r["metadata"]["content_sha256"] for r in base}
        all_records = load_company_chunks(code, full, root)
        records = [r for r in all_records if r["metadata"]["content_sha256"] not in large_hashes]
        excluded += len(all_records) - len(records)
        path = folder / f"{code}.jsonl"
        temp = path.with_suffix(".jsonl.part")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        temp.replace(path)
        manifest["companies"][code] = {**info, "filename": path.name, "chunk_count": len(records),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "base_sha256": hashlib.sha256((chunk_paths(settings, root)[0] / f"{code}.jsonl").read_bytes()).hexdigest()}
    manifest.update({"company_count": len(manifest["companies"]), "chunk_count": sum(v["chunk_count"] for v in manifest["companies"].values()),
                     "excluded_existing_large": excluded, "created_at": datetime.now(timezone.utc).isoformat()})
    write_json(manifest_path, manifest)
    print(f"EXPANSION_READY companies={manifest['company_count']} new_chunks={manifest['chunk_count']} reused_large={excluded}")
    return manifest


def prepare_report_chunks(report, settings, root=ROOT):
    """추가 선택 문서 전문을 접수번호별로 캐시하며 다른 문서를 덮어쓰지 않습니다."""
    code, receipt = report["corp_code"], report["receipt_no"]
    if not re.fullmatch(r"[0-9]{8}", code) or not re.fullmatch(r"[0-9]{14}", receipt):
        raise ValueError("보고서 식별자 형식이 올바르지 않습니다.")
    parsed_path = root / report["parsed_json_path"]
    parsed = read_json(parsed_path)
    if (parsed["corp_code"], parsed["receipt_no"], parsed["report_period"], parsed["zip_sha256"]) != (
        code, receipt, report["report_period"], report["zip_sha256"]):
        raise ValueError("추가 보고서의 파싱 결과와 출처가 다릅니다.")
    folder = root / "data/processed/report_chunks" / chunk_profile(settings) / code
    path, manifest_path = folder / f"{receipt}.jsonl", folder / f"{receipt}.json"
    source_sha = hashlib.sha256(parsed_path.read_bytes()).hexdigest()
    if manifest_path.exists() and path.exists():
        info = read_json(manifest_path)
        if info.get("source_sha256") == source_sha and info.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(records) == info["chunk_count"] and all(r["metadata"]["receipt_no"] == receipt and r["metadata"]["corp_code"] == code for r in records):
                return records, info
    records = [chunk for section in report_sections(report, parsed) for chunk in build_chunks(section, settings)]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("추가 문서 청크가 비었거나 ID가 중복됩니다.")
    folder.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.part")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    temp.replace(path)
    info = {"corp_code": code, "receipt_no": receipt, "report_period": report["report_period"],
            "chunk_count": len(records), "section_count": len(parsed["sections"]),
            "scope": "full", "source_sha256": source_sha,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "filename": path.relative_to(root).as_posix(), "parsed_json_path": report["parsed_json_path"],
            "zip_path": report["zip_path"], "text_path": report["text_path"],
            "zip_sha256": report["zip_sha256"]}
    write_json(manifest_path, info)
    return records, info


def prepare_chunks_main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-corpus", action="store_true", help="전체 본문을 별도 프로필로 청킹")
    parser.add_argument("--expansion", action="store_true", help="준비된 전문 청크에서 기존 large 본문을 제외")
    args = parser.parse_args()
    settings = load_settings()
    if args.expansion:
        prepare_expansion_chunks(settings)
        return 0
    if args.full_corpus:
        settings = {**settings, "corpus_scope": "full"}
    source = corpus_path(settings)
    companies = read_json(ROOT / "data/companies.json")
    reports = read_json(ROOT / "data/processed/report_candidates.json")["reports"]
    dates = {r["receipt_no"]: r["collected_at"] for r in reports}
    if settings.get("corpus_scope") == "full":
        documents = read_json(ROOT / "data/processed/disclosure_manifest.json")["documents"]
        by_receipt = {r["receipt_no"]: r for r in reports}
        temp = source.with_suffix(".jsonl.part")
        with temp.open("w", encoding="utf-8") as file:
            for document in documents:
                parsed = read_json(ROOT / document["processed_path"])
                for section in report_sections(by_receipt[document["receipt_no"]], parsed):
                    file.write(json.dumps(section, ensure_ascii=False) + "\n")
        temp.replace(source)
    grouped = defaultdict(list)
    with source.open(encoding="utf-8") as file:
        for line in file:
            section = json.loads(line)
            section["collected_at"] = dates[section["receipt_no"]]
            grouped[section["corp_code"]].extend(build_chunks(section, settings))
    if set(grouped) != {c["corp_code"] for c in companies}:
        raise ValueError("기업 목록과 본문 기업 ID가 다릅니다.")
    folder, manifest_path = chunk_paths(settings)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_profile": chunk_profile(settings), "chunk_size": settings["chunk_size"],
        "chunk_overlap": settings["chunk_overlap"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "companies": {}}
    for company in companies:
        code = company["corp_code"]
        records = grouped[code]
        if not records or len({r["id"] for r in records}) != len(records):
            raise ValueError("빈 청크 또는 중복 ID가 있습니다.")
        if {r["metadata"]["receipt_no"] for r in records} != {company["canonical_receipt_no"]}:
            raise ValueError("메타데이터와 보고서 접수번호가 다릅니다.")
        target = folder / f"{code}.jsonl"
        temp = target.with_suffix(".jsonl.part")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        temp.replace(target)
        manifest["companies"][code] = {"corp_name": company["corp_name"],
            "filename": target.name, "receipt_no": company["canonical_receipt_no"],
            "chunk_count": len(records), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    manifest["company_count"] = len(companies)
    manifest["chunk_count"] = sum(len(x) for x in grouped.values())
    write_json(manifest_path, manifest)
    print(f"CHUNKS_READY companies={len(companies)} chunks={manifest['chunk_count']} profile={chunk_profile(settings)}")
    pilot = manifest["companies"][settings["pilot_corp_code"]]
    print(f"PILOT {pilot['corp_name']} chunks={pilot['chunk_count']}")
    return 0
