"""취준공시용 고유 기업·최종 사업보고서·원문 본문을 수집합니다.

실행: python scripts/collect_dataset.py --target 100
LLM 호출, 임베딩, Vector DB 색인은 이 단계에서 수행하지 않습니다.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv

from src.disclosure_parser import PARSER_VERSION, parse_report_archive

from src.common import ROOT, read_json, write_json, utc_now

API = "https://opendart.fss.or.kr/api/"
GROUPS = {
    "금융": "105560 055550 086790 316140 024110 138930 139130 175330 323410 377300 000810 005830 001450 032830 088350 082640 005940 006800 016360 039490 003540 071050".split(),
    "IT·통신·게임": "005930 000660 035420 035720 036570 251270 263750 259960 293490 030520 012510 030200 017670 032640 018260 022100 307950 060250 058850 078340 053800 034220 011070 009150 066570 402340".split(),
    "제조·에너지·인프라": "005380 000270 012330 086520 247540 051910 006400 373220 096770 010950 009830 011170 010130 005490 003670 004020 010140 009540 042660 329180 034020 047810 012450 079550 064350 272210 010120 267260 000720 006360 047040 375500 052690 036540 058470 240810 039030 084370 095340".split(),
    "유통·소비재·콘텐츠": "003230 004370 271560 097950 005300 280360 007310 000080 033780 090430 051900 161890 069960 004170 023530 139480 282330 007070 008770 035760 041510 035900 122870 352820".split(),
    "바이오·헬스케어": "207940 068270 128940 006280 000100 326030 145020 196170 028300 214150".split(),
    "운송·지주·서비스": "003490 089590 020560 028670 000120 086280 047050 028260 034730 003550".split(),
}


class CollectionError(Exception):
    def __init__(self, stage, code, detail=""):
        self.stage, self.code, self.detail = stage, str(code), detail
        super().__init__(f"{stage}:{code}")


class DartClient:
    def __init__(self, api_key):
        self.key = api_key
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "ChwijunGongsi-EducationalCollector/1.0"
        self.calls = 0
        self.last_call = 0.0

    def get(self, endpoint, params, binary=False):
        for attempt in range(3):
            time.sleep(max(0, .35 - (time.monotonic() - self.last_call)))
            self.last_call = time.monotonic()
            self.calls += 1
            try:
                response = self.session.get(API + endpoint, params={"crtfc_key": self.key, **params}, timeout=(15, 90))
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    delay = response.headers.get("Retry-After", "")
                    time.sleep(min(float(delay), 30) if delay.isdigit() else 2 ** attempt)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise CollectionError(endpoint, "HTTP_OR_CONNECTION_ERROR") from None
            if binary:
                if zipfile.is_zipfile(BytesIO(response.content)):
                    return response.content
                try:
                    code = ET.fromstring(response.content).findtext("status", "INVALID_BINARY")
                except ET.ParseError:
                    code = "INVALID_BINARY"
                raise CollectionError(endpoint, code)
            try:
                result = response.json()
            except ValueError:
                raise CollectionError(endpoint, "INVALID_JSON") from None
            status = result.get("status")
            if status in {"000", "013"}:
                return result
            if status in {"800", "900"} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise CollectionError(endpoint, status)
        raise CollectionError(endpoint, "RETRY_EXHAUSTED")


def company_detail(client, candidate):
    code = candidate["corp_code"]
    path = ROOT / f"data/raw/company_details/{code}.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("data", {}).get("status") == "000":
            return cached
    data = client.get("company.json", {"corp_code": code})
    if data.get("status") != "000":
        raise CollectionError("company.json", data.get("status"))
    cached = {"collected_at": utc_now(), "data": data}
    write_json(path, cached)
    return cached


def report_list(client, code, start, end, final_only="Y", report_type="A001"):
    if report_type not in {"A001", "A002", "A003"}:
        raise ValueError("지원하지 않는 정기보고서 유형입니다.")
    path = ROOT / f"data/raw/report_lists/{code}_{start}_{end}_{report_type}_{final_only}.json"
    if path.exists():
        cached = read_json(path)
        if isinstance(cached.get("reports"), list):
            return cached
    reports, page = [], 1
    while True:
        data = client.get("list.json", {"corp_code": code, "bgn_de": start, "end_de": end,
            "pblntf_detail_ty": report_type, "last_reprt_at": final_only, "sort": "date", "sort_mth": "desc",
            "page_count": 100, "page_no": page})
        if data.get("status") == "013":
            break
        if data.get("status") != "000":
            raise CollectionError("list.json", data.get("status", "INVALID_STATUS"))
        reports.extend(data.get("list", []))
        if page >= int(data.get("total_page", 1)):
            break
        page += 1
    result = {"corp_code": code, "search_start": start, "search_end": end,
              "collected_at": utc_now(), "final_only": final_only, "reports": reports}
    write_json(path, result)
    return result


def choose_report(cached, company, period):
    matching = [r for r in cached["reports"]
        if r.get("corp_code") == company["corp_code"]
        and "사업보고서" in r.get("report_nm", "")
        and f"({period})" in re.sub(r"\s+", "", r.get("report_nm", ""))]
    if not matching:
        raise CollectionError("report_selection", "TARGET_REPORT_NOT_FOUND")
    latest = max(matching, key=lambda r: (r.get("rcept_dt", ""), r["rcept_no"]))
    receipt = latest["rcept_no"]
    return {"corp_code": company["corp_code"], "corp_name": company["corp_name"],
        "report_name": latest["report_nm"], "report_period": period, "receipt_no": receipt,
        "receipt_date": latest["rcept_dt"], "source": "금융감독원 DART",
        "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
        "collected_at": cached["collected_at"], "version_status": "latest_available_in_search_snapshot"}


def download_zip(client, report):
    path = ROOT / f"data/raw/disclosures/{report['corp_code']}/{report['receipt_no']}.zip"
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise CollectionError("zip_validation", "CORRUPT_CACHE")
        return path
    payload = client.get("document.xml", {"rcept_no": report["receipt_no"]}, binary=True)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if archive.testzip() is not None:
            raise CollectionError("zip_validation", "CORRUPT_DOWNLOAD")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".zip.part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def parse_and_save(path, report):
    processed = ROOT / f"data/processed/disclosures/{report['corp_code']}/{report['receipt_no']}.json"
    text_path = processed.with_suffix(".txt")
    if processed.exists() and text_path.exists():
        cached = read_json(processed)
        if cached.get("parser_version") == PARSER_VERSION and cached.get("zip_sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
            return cached, processed, text_path
    data, full_text = parse_report_archive(path, report, ROOT)
    write_json(processed, data)
    text_path.write_text(full_text, encoding="utf-8")
    return data, processed, text_path


def candidate_order(candidates):
    lookup = {c["stock_code"]: c for c in candidates if c.get("stock_code")}
    seen = set()
    longest = max(map(len, GROUPS.values()))
    for index in range(longest):
        for group, codes in GROUPS.items():
            if index < len(codes) and codes[index] in lookup:
                candidate = lookup[codes[index]]
                if candidate["corp_code"] not in seen:
                    seen.add(candidate["corp_code"])
                    yield {**candidate, "selection_group": group}
    # 편집상 선정 그룹이 있는 후보가 부족할 때만 나머지 후보를 확인합니다.
    for candidate in sorted(candidates, key=lambda c: c["corp_code"]):
        if candidate["corp_code"] not in seen and candidate.get("stock_code"):
            seen.add(candidate["corp_code"])
            yield {**candidate, "selection_group": "추가 후보"}


def collect_dataset_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--period", default="2025.12")
    parser.add_argument("--as-of", default="20261022")
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    if not 1 <= args.target <= args.max_candidates or not re.fullmatch(r"\d{4}\.12", args.period):
        raise SystemExit("대상 수와 12월 결산 보고서 기간을 확인하세요.")
    if not re.fullmatch(r"\d{8}", args.as_of) or (args.prefix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix)):
        raise SystemExit("수집 기준일 또는 출력 접두사가 잘못되었습니다.")
    load_dotenv(ROOT / ".env")
    key = os.getenv("DART_API_KEY", "").strip()
    if not key:
        raise SystemExit("DART_API_KEY 미설정")
    client = DartClient(key)
    start = str(int(args.period[:4]) + 1) + "0101"
    candidate_path = ROOT / "data/processed/company_candidates.json"
    if not candidate_path.exists():
        collect_candidates(key)
    candidates = read_json(candidate_path)["companies"]
    companies, reports, documents, failures = [], [], [], []
    run_started = utc_now()
    processed_root = ROOT / "data/processed"
    for attempted, candidate in enumerate(candidate_order(candidates), 1):
        if attempted > args.max_candidates or len(companies) >= args.target:
            break
        print(f"[{attempted}] {candidate['corp_name']} — 완료 {len(companies)}/{args.target}", flush=True)
        try:
            detail = company_detail(client, candidate)
            raw_company = detail["data"]
            if raw_company.get("corp_cls") not in {"Y", "K", "N"}:
                raise CollectionError("company_validation", "NOT_CURRENT_LISTED_CLASS")
            if raw_company.get("acc_mt") != "12":
                raise CollectionError("company_validation", "NOT_DECEMBER_FISCAL_YEAR")
            company = {"corp_code": candidate["corp_code"], "corp_name": raw_company["corp_name"],
                "stock_name": raw_company.get("stock_name", ""), "stock_code": raw_company.get("stock_code", ""),
                "corp_cls": raw_company.get("corp_cls", ""), "industry_code": raw_company.get("induty_code", ""),
                "established_date": raw_company.get("est_dt", ""), "accounting_month": raw_company.get("acc_mt", ""),
                "homepage_url": raw_company.get("hm_url", ""), "ir_url": raw_company.get("ir_url", ""),
                "selection_group": candidate["selection_group"], "selection_group_is_editorial": True,
                "source": "금융감독원 OpenDART 기업개황", "source_url": API + "company.json",
                "collected_at": detail["collected_at"]}
            listed = report_list(client, company["corp_code"], start, args.as_of)
            report = choose_report(listed, company, args.period)
            path = download_zip(client, report)
            parsed, parsed_path, text_path = parse_and_save(path, report)
            company.update({"business_overview_excerpt": parsed["business_overview_excerpt"],
                "business_overview_section_id": parsed["business_overview_section_id"],
                "canonical_receipt_no": report["receipt_no"], "report_period": args.period,
                "report_source_url": report["source_url"]})
            company["index_text"] = "\n".join([
                "기업명: " + company["corp_name"], "종목명: " + company["stock_name"],
                "업종코드: " + company["industry_code"], "사업보고서 기준: " + args.period,
                "사업 개요 원문 발췌: " + company["business_overview_excerpt"]])
            document = {k: v for k, v in parsed.items() if k not in {"sections", "business_overview_excerpt"}}
            document.update({"zip_path": path.relative_to(ROOT).as_posix(),
                "processed_path": parsed_path.relative_to(ROOT).as_posix(),
                "text_path": text_path.relative_to(ROOT).as_posix(),
                "downloaded_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "zip_size_bytes": path.stat().st_size})
            companies.append(company); reports.append(report); documents.append(document)
            print(f"  확보: {parsed['section_count']}개 섹션 / {parsed['text_characters']:,}자", flush=True)
        except CollectionError as error:
            failure = {"corp_code": candidate["corp_code"], "corp_name": candidate["corp_name"],
                "stage": error.stage, "code": error.code, "recorded_at": utc_now()}
            failures.append(failure)
            print(f"  보류: {error.stage}/{error.code}", flush=True)
            if error.code in {"010", "011", "012", "020", "901"}:
                print("  인증·한도 문제로 중단합니다.", flush=True)
                break
        except Exception as error:
            # 인증키가 포함될 수 있는 네트워크 오류 전문은 저장하지 않습니다.
            reason = str(error) if isinstance(error, ValueError) and re.fullmatch(r"[a-z_]+", str(error)) else ""
            failures.append({"corp_code": candidate["corp_code"], "corp_name": candidate["corp_name"],
                "stage": "parse_or_file", "code": type(error).__name__, "detail": reason, "recorded_at": utc_now()})
            print(f"  보류: {type(error).__name__} {reason}", flush=True)
        write_json(processed_root / (args.prefix + "collection_progress.json"), {
            "target": args.target, "completed": len(companies), "failures": failures, "updated_at": utc_now()})

    summary = {"started_at": run_started, "completed_at": utc_now(), "target": args.target,
        "company_count": len(companies), "document_count": len(documents),
        "period": args.period, "search_start": start, "search_end": args.as_of,
        "api_calls_this_run": client.calls, "parser_version": PARSER_VERSION,
        "selection_groups": dict(Counter(c["selection_group"] for c in companies)),
        "total_sections": sum(d["section_count"] for d in documents),
        "total_tables": sum(d["table_count"] for d in documents),
        "total_text_characters": sum(d["text_characters"] for d in documents),
        "failure_count": len(failures), "failures": failures,
        "complete": len(companies) >= args.target, "embedding_or_indexing_performed": False}
    write_json(processed_root / (args.prefix + "collection_summary.json"), summary)
    if len(companies) < args.target:
        print("목표 미달: 기존 최종 데이터는 덮어쓰지 않았습니다.", flush=True)
        return 2
    artifacts = {
        "report_candidates.json": {"target_period": args.period, "search_start": start, "search_end": args.as_of,
            "count": len(reports), "reports": reports, "missing_companies": [], "failures": []},
        "disclosure_manifest.json": {"count": len(documents), "documents": documents, "failures": []},
    }
    snapshot = ROOT / "data/raw/snapshots" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    for name, data in artifacts.items():
        target = processed_root / (args.prefix + name)
        if target.exists() and not args.prefix:
            snapshot.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, snapshot / name)
        write_json(target, data)
    write_json(ROOT / "data" / (args.prefix + "companies.json"), companies)
    print(f"COLLECTION_COMPLETE companies={len(companies)} documents={len(documents)} calls={client.calls}", flush=True)
    return 0


def load_company_xml(api_key):
    # 이미 받은 원문은 재사용합니다.
    if (ROOT / "data/raw/corp_codes.xml").exists():
        print("저장된 기업 목록을 사용합니다.")
        return (ROOT / "data/raw/corp_codes.xml").read_bytes()

    print("DART 기업 목록을 내려받습니다.")

    response = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": api_key},
        timeout=60,
    )
    response.raise_for_status()

    archive_buffer = BytesIO(response.content)

    # 인증 오류 등은 ZIP 대신 오류 XML로 올 수 있습니다.
    if not zipfile.is_zipfile(archive_buffer):
        try:
            error_root = ET.fromstring(response.content)
            status = error_root.findtext("status", default="알 수 없음")
        except ET.ParseError:
            status = "응답 형식 오류"

        raise ValueError(f"기업 목록 다운로드 실패 — 상태: {status}")

    archive_buffer.seek(0)

    with zipfile.ZipFile(archive_buffer) as archive:
        xml_names = [
            name
            for name in archive.namelist()
            if Path(name).name.upper() == "CORPCODE.XML"
        ]

        if len(xml_names) != 1:
            raise ValueError("ZIP에서 기업 목록 XML을 찾지 못했습니다.")

        xml_bytes = archive.read(xml_names[0])

    # 올바른 목록인지 확인한 뒤 캐시로 저장합니다.
    root = ET.fromstring(xml_bytes)

    if not root.findall("list"):
        raise ValueError("기업 목록이 비어 있습니다.")

    (ROOT / "data/raw/corp_codes.xml").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/raw/corp_codes.xml").write_bytes(xml_bytes)

    return xml_bytes


def parse_candidates(xml_bytes):
    root = ET.fromstring(xml_bytes)
    companies = {}

    for item in root.findall("list"):
        corp_code = item.findtext("corp_code", default="").strip()
        corp_name = item.findtext("corp_name", default="").strip()
        stock_code = item.findtext("stock_code", default="").strip()

        # 종목코드가 있는 기업만 후보로 저장합니다.
        if not corp_code or not corp_name or not stock_code:
            continue

        # 기업 고유번호를 키로 사용해 중복을 방지합니다.
        companies[corp_code] = {
            "corp_code": corp_code,
            "corp_name": corp_name,
            "stock_code": stock_code,
            "modify_date": item.findtext(
                "modify_date", default=""
            ).strip(),
        }

    return sorted(
        companies.values(),
        key=lambda company: company["corp_name"],
    )


def collect_candidates(api_key):
    """전체 수집에 필요한 기업 목록을 최초 실행 시 준비합니다."""
    companies = parse_candidates(load_company_xml(api_key))
    if not companies:
        raise ValueError("기업 후보가 비어 있습니다.")
    write_json(ROOT / "data/processed/company_candidates.json", {
        "source": "금융감독원 OpenDART", "downloaded_at": utc_now(),
        "count": len(companies), "companies": companies,
    })
