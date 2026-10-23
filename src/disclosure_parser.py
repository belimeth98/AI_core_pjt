"""DART 원문은 보존하고, 검색에 쓸 본문과 표를 추출합니다."""
from __future__ import annotations

import hashlib
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

from lxml import etree

PARSER_VERSION = "dart-text-v1"
CELL_TAGS = {"TD", "TH", "TE", "TU"}


def normalized_text(node):
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def normalize_xml(raw):
    """누락된 & 이스케이프, 꺾쇠 본문, 손상된 표시용 속성을 정규화합니다.

    XML recover=True로 본문을 조용히 버리지 않습니다. 원본 ZIP은 수정하지 않습니다.
    회사 식별자·문서 코드·표 병합 속성은 보존합니다.
    """
    encoding = re.search(br'encoding=["\']([^"\']+)', raw[:150])
    encoding_name = encoding.group(1).decode("ascii") if encoding else "utf-8"
    text = raw.decode(encoding_name)
    text, amp_count = re.subn(
        r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", text
    )
    text, angle_count = re.subn(
        r"<(?!(?:/?[A-Z][A-Z0-9_-]*(?=[\s>/])|!--|\?|!DOCTYPE))", "&lt;", text
    )
    changed_attributes = 0

    def clean_tag(match):
        nonlocal changed_attributes
        name, attributes = match.group(1), match.group(2) or ""
        kept = "".join(
            f' {key}="{value}"'
            for key, value in re.findall(
                r'\b(ROWSPAN|COLSPAN|AREGCIK|ACODE)="([A-Za-z0-9._:-]+)"', attributes
            )
        )
        result = "<" + name + kept + ("/>" if attributes.rstrip().endswith("/") else ">")
        changed_attributes += result != match.group(0)
        return result

    text = re.sub(r"<([A-Z][A-Z0-9_-]*)(\s[^<>]*?)?>", clean_tag, text)
    text = re.sub(r'^\s*<\?xml[^>]*\?>', '', text)
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, recover=False, huge_tree=True
    )
    root = etree.fromstring(text.encode("utf-8"), parser)
    return root, {
        "bare_ampersands_escaped": amp_count,
        "literal_angle_brackets_escaped": angle_count,
        "display_attribute_tags_normalized": changed_attributes,
        "strict_parse_after_normalization": True,
    }


def render_table(table):
    rows = [
        row for row in table.iter("TR")
        if next(row.iterancestors("TABLE"), None) is table
    ]
    grid = {}
    max_column = 0
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row:
            if cell.tag not in CELL_TAGS:
                continue
            while (row_index, column) in grid:
                column += 1
            rowspan = max(1, int(cell.get("ROWSPAN", "1")))
            colspan = max(1, int(cell.get("COLSPAN", "1")))
            if colspan > 500 or column + colspan > 2000:
                raise ValueError("unsupported_table_width")
            value = normalized_text(cell).replace("|", "｜")
            for target_row in range(row_index, min(len(rows), row_index + rowspan)):
                for target_col in range(column, column + colspan):
                    grid[target_row, target_col] = value
            column += colspan
            max_column = max(max_column, column)
    if not rows or not max_column:
        return normalized_text(table), 0
    lines = [
        " | ".join(grid.get((row, col), "") for col in range(max_column)).rstrip()
        for row in range(len(rows))
    ]
    return "[표: 병합 셀의 값은 해당 행·열에 반복 표시]\n" + "\n".join(lines), len(rows)


def parse_report_archive(zip_path, report, project_root):
    """접수번호.xml을 우선하고 DOCUMENT-NAME ACODE와 회사 ID도 확인합니다."""
    receipt = report["receipt_no"]
    name = report.get("report_name", "사업보고서")
    expected_codes = ({"11012"} if "반기보고서" in name else
                      {"11013", "11014"} if "분기보고서" in name else {"11011"})
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt_zip")
        xml_names = [n for n in archive.namelist() if n.lower().endswith(".xml")]
        exact = [n for n in xml_names if Path(n).name.lower() == receipt + ".xml"]
        candidate_names = exact or xml_names
        choices = []
        for member in candidate_names:
            raw = archive.read(member)
            root, fixes = normalize_xml(raw)
            document_name = root.find("DOCUMENT-NAME")
            company_name = root.find("COMPANY-NAME")
            if document_name is None or document_name.get("ACODE") not in expected_codes:
                continue
            if company_name is None or company_name.get("AREGCIK") != report["corp_code"]:
                continue
            choices.append((member, raw, root, fixes))
    if len(choices) != 1:
        raise ValueError("main_document_not_uniquely_identified")
    member, raw, root, fixes = choices[0]
    sections = OrderedDict()
    block_count = 0
    table_count = 0
    table_row_count = 0

    def append(path, kind, text):
        nonlocal block_count
        if not text.strip():
            return
        key = tuple(path) or ("표지 및 기타",)
        sections.setdefault(key, []).append({"type": kind, "text": text})
        block_count += 1

    def walk(node, path):
        nonlocal table_count, table_row_count
        tag = node.tag
        if not isinstance(tag, str):
            return
        if tag.startswith("SECTION-"):
            title = node.find("TITLE")
            new_path = path + ([normalized_text(title)] if title is not None else [])
            for child in node:
                if child is not title:
                    walk(child, new_path)
            return
        if tag == "TABLE":
            text, row_count = render_table(node)
            append(path, "table", text)
            table_count += 1
            table_row_count += row_count
            return
        if tag in {"P", "TITLE", "COVER-TITLE", "SUBTITLE"}:
            append(path, "paragraph", normalized_text(node))
            return
        if node.text and node.text.strip() and tag not in {"SUMMARY", "EXTRACTION"}:
            append(path, "paragraph", node.text.strip())
        for child in node:
            walk(child, path)
            if child.tail and child.tail.strip():
                append(path, "paragraph", child.tail.strip())

    body = root.find("BODY")
    if body is None:
        raise ValueError("body_missing")
    walk(body, [])
    processed_sections = []
    for index, (path, blocks) in enumerate(sections.items()):
        processed_sections.append({
            "section_id": f"{receipt}:s{index:03d}",
            "section_path": list(path),
            "title": " > ".join(path),
            "text": "\n\n".join(block["text"] for block in blocks),
            "block_count": len(blocks),
            "table_count": sum(block["type"] == "table" for block in blocks),
        })
    if not any("사업의 내용" in s["title"] for s in processed_sections):
        raise ValueError("business_section_missing")
    full_text = "\n\n".join(
        "# " + s["title"] + "\n" + s["text"] for s in processed_sections
    )
    if len(full_text) < 10000:
        raise ValueError("document_text_unexpectedly_short")
    overview_section = next(
        (s for s in processed_sections if "사업의 내용" in s["title"] and "사업의 개요" in s["title"]),
        next(s for s in processed_sections if "사업의 내용" in s["title"]),
    )
    excerpt = re.sub(r"\s+", " ", overview_section["text"]).strip()[:1800]
    cover = root.find(".//COVER")
    cover_text = normalized_text(cover) if cover is not None else ""
    year = report["report_period"][:4]
    if year not in cover_text:
        raise ValueError("target_year_not_found_in_cover")
    if "사업보고서" not in name:
        month = int(report["report_period"][5:7])
        if not re.search(rf"{year}\s*(?:년|[./-])\s*0?{month}\s*(?:월|[./-])", cover_text):
            raise ValueError("target_period_not_found_in_cover")
    result = {
        **report,
        "document_id": f"{report['corp_code']}:{receipt}",
        "source_type": "disclosure_xml",
        "main_xml_member": member,
        "original_xml_sha256": hashlib.sha256(raw).hexdigest(),
        "zip_sha256": hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),
        "parser_version": PARSER_VERSION,
        "normalization": fixes,
        "section_count": len(processed_sections),
        "block_count": block_count,
        "table_count": table_count,
        "table_row_count": table_row_count,
        "text_characters": len(full_text),
        "business_overview_excerpt": excerpt,
        "business_overview_section_id": overview_section["section_id"],
        "sections": processed_sections,
    }
    return result, full_text
