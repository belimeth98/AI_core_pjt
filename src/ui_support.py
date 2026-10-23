"""화면에서 쓰는 기업 준비 상태·질문 기록·다운로드 텍스트를 관리합니다."""


def valid_result(result, company):
    if not isinstance(result, dict) or result.get("corp_code") != company["corp_code"]:
        return False
    if result.get("receipt_no") != company["canonical_receipt_no"]:
        return False
    if not isinstance(result.get("answer"), str) or not isinstance(result.get("sources"), list):
        return False
    receipts = result.get("receipt_nos", [company["canonical_receipt_no"]])
    if not isinstance(receipts, list) or company["canonical_receipt_no"] not in receipts or not all(isinstance(r, str) for r in receipts):
        return False
    return all(s.get("corp_code") == company["corp_code"] and s.get("receipt_no") in receipts
               for s in result["sources"])


def used_sources(result, *, include_questions=True):
    ids = set(result.get("cited_ids", []))
    if include_questions:
        for item in result.get("questions", []):
            ids.update(item.get("source_ids", []))
    return [source for source in result.get("sources", []) if source["source_id"] in ids]


def split_answer(answer):
    import re
    # 한글 조사 앞의 굵은 글씨가 원시 **로 보이지 않도록 표시 공백만 보정합니다.
    answer = re.sub(r"(\*\*[^*\n]+\*\*)(?=[가-힣])", r"\1 ", answer)
    sections = re.split(r"(?m)^#{1,6}\s+", answer)
    summary = ""
    rest = []
    for section in sections:
        if not section.strip():
            continue
        title, _, body = section.partition("\n")
        if "쉬운 설명" in title and not summary:
            summary = body.strip()
        else:
            rest.append("### " + section.strip())
    if summary:
        return summary, "\n\n".join(rest)
    return answer, ""


def matching_turn(turns, question, fingerprint):
    return next((index for index, turn in enumerate(turns)
                 if turn.get("question", "").strip() == question.strip()
                 and turn.get("ui_source_fingerprint") == fingerprint), None)


def export_markdown(result, company_name):
    lines = [f"# {company_name} | 공시 기반 기업 이해", "", "## 질문", result["question"], "", result["answer"], "", "## 추천 질문"]
    for item in result.get("questions", []):
        lines.append("- " + item["question"])
    lines.extend(["", "## 사용한 근거"])
    for source in used_sources(result):
        lines.append(f"- [{source['source_id']}] {source['title']} ({source['report_period']}) — {source['source_url']}")
    lines.extend(["", "공시 기준시점의 학습·취업 준비용 정보입니다. 현재 상품·채용 조건은 공식 안내를 확인하세요."])
    return "\n".join(lines)
