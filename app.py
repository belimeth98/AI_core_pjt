"""취준공시 화면. src.pipeline의 공통 파이프라인을 실행합니다."""
import os
import sys
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIError
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import read_json
from src.chunking import load_settings
from src.vector_store import available_reports
from src.ui_support import export_markdown, matching_turn, split_answer, used_sources, valid_result
from src.pipeline import run_pipeline
from src.briefing import load_saved_briefing
from src.company_search import find_companies


def queue_question(question, code):
    st.session_state["ui_pending_request"] = {"question": question, "corp_code": code}


def choose_company(code):
    # 버튼 콜백은 다음 실행의 기업 선택 위젯이 만들어지기 전에 실행됩니다.
    st.session_state["ui_company"] = code


def show_company_search(ready):
    result = st.session_state.get("ui_company_search")
    error = st.session_state.get("ui_company_search_error")
    with st.expander("조건으로 기업 찾기", expanded=bool(result or error)):
        st.caption("사업·제품·서비스 조건을 입력하면 기업 개요에서 근거를 확인해 찾아드립니다.")
        with st.form("ui_company_search_form"):
            query = st.text_input(
                "찾고 싶은 기업의 조건", max_chars=200,
                placeholder="예: 모바일 금융 서비스를 제공하는 기업",
                key="ui_company_search_query",
            )
            submitted = st.form_submit_button("조건으로 검색", use_container_width=True)
        if submitted:
            try:
                with st.spinner("기업 개요를 검색하고 조건을 확인하고 있어요."):
                    result = find_companies(query)
                st.session_state["ui_company_search"] = result
                st.session_state.pop("ui_company_search_error", None)
                error = None
            except Exception as exception:
                result = None
                st.session_state.pop("ui_company_search", None)
                error = (
                    "검색 조건을 2~200자로 입력해 주세요."
                    if not 2 <= len(query.strip()) <= 200
                    else safe_error(exception)
                )
                st.session_state["ui_company_search_error"] = error
        if error:
            st.warning(error)
        if not result:
            return
        st.caption(
            f"검색 조건: {result['query']} · 기업 개요 {result['company_count']}개에서 "
            f"후보 {result['candidate_count']}개를 확인했습니다."
        )
        if not result["matches"]:
            st.info("현재 기업 개요에서 모든 조건을 확인한 기업을 찾지 못했습니다. 조건을 줄이거나 바꿔 보세요.")
        for item in result["matches"]:
            with st.container(border=True):
                name = item.get("stock_name") or item["corp_name"]
                st.markdown(f"**{name} · {item['stock_code']}**")
                st.write(item["reason"])
                for support in item["supports"]:
                    st.caption(f"조건: {support['condition']}")
                    st.text(support["quote"])
                st.link_button("근거 사업보고서 보기 ↗", item["report_source_url"])
                state = "공시 질문 가능" if item["corp_code"] in ready else "기업 정보 확인 가능"
                st.caption(f"{item['report_period']} 연간 · {state}")
                st.button(
                    f"{name} 화면으로 이동 →", key=f"ui_search_select_{item['corp_code']}",
                    on_click=choose_company, args=(item["corp_code"],), use_container_width=True,
                )
        st.caption("검색은 저장된 기업 개요의 일부를 대상으로 합니다. 모든 기업의 조건 충족 여부나 현재 채용 정보는 보장하지 않습니다.")


def safe_error(error):
    if isinstance(error, APIError):
        code = getattr(error, "status_code", None)
        if code in {500, 502, 503, 504}:
            return "답변 서비스가 잠시 불안정합니다. 잠시 후 다시 질문해 주세요."
        if code == 429:
            return "모델 사용 한도에 도달했거나 요청이 몰렸습니다. 사용량을 확인한 뒤 다시 시도해 주세요."
        if code in {401, 403}:
            return "모델 연결 권한을 확인하지 못했습니다. GMS 설정을 확인해 주세요."
        return "모델 서비스에 연결하지 못했습니다. 네트워크와 GMS 설정을 확인해 주세요."
    if isinstance(error, ValidationError):
        return (
            "답변 준비 중 응답 형식을 확인하지 못했습니다. "
            "기존 답변은 유지됩니다. 다시 시도해 주세요."
        )
    if isinstance(error, ValueError) and "크레딧 보호 모드" in str(error):
        return "새 공시를 분석하려면 추가 색인이 필요합니다. 현재 자동 색인이 중지되어 있으니 기존 연간 자료로 질문해 주세요."
    if isinstance(error, ValueError) and "다른 색인 작업" in str(error):
        return "자료 색인이 진행 중입니다. 완료 후 다시 질문해 주세요."
    if isinstance(error, (ValueError, KeyError)):
        return "공시 근거 또는 답변의 인용을 확인하지 못했습니다. 자료 준비 상태를 확인한 뒤 다시 시도해 주세요."
    return "답변을 완성하지 못했습니다. 기존 질문 기록은 유지됩니다. 잠시 후 다시 시도해 주세요."


def execute_question(question, code, company, ready):
    question = question.strip()
    if not 2 <= len(question) <= 1500:
        st.session_state["ui_errors"][code] = "질문은 2~1500자로 입력해 주세요."
        return
    if code not in ready:
        st.session_state["ui_errors"][code] = "이 기업의 질문 기능은 자료 준비 후 사용할 수 있습니다."
        return
    if not all(os.getenv(key, "").strip() for key in ["GMS_API_KEY", "GMS_BASE_URL", "GMS_CHAT_MODEL", "GMS_EMBEDDING_MODEL"]):
        st.session_state["ui_errors"][code] = "GMS 연결 설정이 필요합니다. 프로젝트의 환경변수를 확인해 주세요."
        return
    turns = st.session_state["ui_turns"].setdefault(code, [])
    found = matching_turn(turns, question, ready[code]["fingerprint"])
    if found is not None:
        st.session_state["ui_active_turn"][code] = found
        versions = st.session_state["ui_history_version"]
        versions[code] = versions.get(code, 0) + 1
        st.session_state["ui_errors"].pop(code, None)
        st.toast("이전에 확인한 답변을 불러왔습니다.")
        return
    try:
        with st.status(
            "질문에 필요한 공시를 확인하고 답변을 준비하고 있어요.",
            expanded=False,
        ) as status:
            # P1 → 후보 조회 → P2 → 원문 준비 → 검사 → RAG
            result = run_pipeline(question, code)

            if not valid_result(result, company):
                raise ValueError("invalid_company_or_report")

            result["ui_source_fingerprint"] = ready[code]["fingerprint"]

            turns.append(result)
            st.session_state["ui_active_turn"][code] = len(turns) - 1

            versions = st.session_state["ui_history_version"]
            versions[code] = versions.get(code, 0) + 1

            st.session_state["ui_errors"].pop(code, None)

            status.update(
                label="공시 근거와 함께 답변을 준비했습니다.",
                state="complete",
            )

    except Exception as error:
        st.session_state["ui_errors"][code] = safe_error(error)


def seed_history(companies, ready):
    if st.session_state.get("ui_seed_loaded"):
        return
    st.session_state["ui_seed_loaded"] = True
    path = ROOT / "data/processed/latest_pipeline_result.json"
    if not path.exists():
        return
    try:
        result = read_json(path)
        code = result.get("corp_code")
        company = next((c for c in companies if c["corp_code"] == code), None)
        if company and code in ready and valid_result(result, company):
            result["ui_source_fingerprint"] = ready[code]["fingerprint"]
            st.session_state["ui_turns"][code] = [result]
            st.session_state["ui_active_turn"][code] = 0
    except (OSError, ValueError, KeyError):
        pass


def show_saved_briefing(company):
    with st.container(border=True, key="ui_briefing_card"):
        st.subheader("취업 준비 브리핑")
        try:
            saved = load_saved_briefing(company)
        except (OSError, ValueError, KeyError, TypeError):
            st.warning("브리핑의 형식과 공시 근거를 확인하지 못했습니다. 저장된 자료를 확인해 주세요.")
            return

        if saved is None:
            st.caption("선택한 기업·보고기간에 맞는 브리핑이 아직 준비되지 않았습니다.")
            return

        st.caption(f"{company['report_period']} 연간 공시 기반 · 면접 질문은 연습용입니다.")
        with st.expander("기업 요약 · 핵심 포인트 · 면접 연습 질문 보기"):
            st.markdown(saved["markdown"])
            st.download_button(
                "브리핑을 노트로 저장 ↓",
                data=saved["markdown"],
                file_name=f"취준공시_브리핑_{company['stock_code']}_{company['report_period']}.md",
                mime="text/markdown",
                key=f"ui_briefing_download_{company['corp_code']}",
            )


def main():
    st.set_page_config(page_title="취준공시 | 공시로 준비하는 첫 질문", page_icon="📘", layout="wide")
    load_dotenv(ROOT / ".env")
    st.markdown("<style>" + (ROOT / "assets/app.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
    try:
        companies = read_json(ROOT / "data/companies.json")
        settings = load_settings()
        ready = available_reports(ROOT, settings, os.getenv("GMS_BASE_URL", ""), os.getenv("GMS_EMBEDDING_MODEL", ""))
        receipts = {c["corp_code"]: c["canonical_receipt_no"] for c in companies}
        ready = {code: entry for code, entry in ready.items() if entry["receipt_no"] == receipts.get(code)}
    except (OSError, ValueError, KeyError):
        st.error("기업 자료를 불러오지 못했습니다. 데이터 준비 상태를 확인해 주세요.")
        st.stop()
    for key in ["ui_turns", "ui_active_turn", "ui_errors", "ui_history_version"]:
        st.session_state.setdefault(key, {})
    seed_history(companies, ready)
    by_code = {c["corp_code"]: c for c in companies}
    choices = sorted(by_code, key=lambda c: (c not in ready, by_code[c]["selection_group"] != "금융", by_code[c].get("stock_name", by_code[c]["corp_name"])))
    with st.sidebar:
        st.markdown('<div class="career-brand"><div class="career-wordmark">취준공시</div><div class="career-byline">공시에서 찾는, 나의 첫 기업 이해</div></div>', unsafe_allow_html=True)
        st.caption("지원 전에, 회사를 먼저 이해하세요.")
        st.divider()
        code = st.selectbox("분석할 기업", choices, key="ui_company",
            format_func=lambda c: (by_code[c].get("stock_name") or by_code[c]["corp_name"]) + " · " + by_code[c]["stock_code"])
        company = by_code[code]
        st.caption("기업명이나 종목코드로 목록에서 검색할 수 있습니다.")
        st.markdown(f'<div class="career-stats"><span>기업 정보</span><strong>{len(companies)}곳</strong></div><div class="career-stats"><span>지금 질문 가능한 기업</span><strong>{len(ready)}곳</strong></div>', unsafe_allow_html=True)
        turns = st.session_state["ui_turns"].setdefault(code, [])
        st.divider()
        st.subheader("이 기업에 물어본 질문")
        if turns:
            latest = st.session_state["ui_active_turn"].get(code, len(turns) - 1)
            selected = st.selectbox("질문 기록", list(range(len(turns))), index=min(latest, len(turns)-1),
                format_func=lambda i: turns[i]["question"], key=f"ui_history_{code}_{len(turns)}_{st.session_state['ui_history_version'].get(code, 0)}", label_visibility="collapsed")
            st.session_state["ui_active_turn"][code] = selected
            if st.button("이 기업의 기록 비우기", key="ui_clear", use_container_width=True):
                st.session_state["ui_turns"][code] = []
                st.session_state["ui_active_turn"].pop(code, None)
                st.session_state["ui_errors"].pop(code, None)
                st.rerun()
        else:
            st.caption("첫 질문을 남겨 보세요. 기업별로 기록이 구분됩니다.")
        st.divider()
        st.caption("기록은 현재 브라우저 세션에서 유지됩니다.")
    name = company.get("stock_name") or company["corp_name"]
    is_ready = code in ready and ready[code]["receipt_no"] == company["canonical_receipt_no"]
    show_company_search(ready)
    st.markdown('<div class="career-eyebrow">공시로 시작하는 기업 리서치</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="career-header"><h1>{escape(name)}</h1><div class="career-intro">어려운 공시를 쉬운 말로. 기업 이해를 면접의 첫 답변으로 연결해 보세요.</div></div>', unsafe_allow_html=True)
    state_label = "공시 기반 질문 가능" if is_ready else "기업 정보 확인 가능"
    st.markdown(f'<div class="career-badges"><span class="career-badge">{escape(company["selection_group"])}</span><span class="career-badge">{escape(company["stock_code"])}</span><span class="career-badge">{escape(company["report_period"])} 연간</span><span class="career-badge ready">{state_label}</span></div>', unsafe_allow_html=True)
    if not is_ready:
        names = [by_code[c].get("stock_name") or by_code[c]["corp_name"] for c in ready]
        help_text = " 지금 질문 가능한 기업: " + ", ".join(names) + "." if names else " 답변 서비스와 자료 준비 상태를 확인해 주세요."
        st.info("이 기업의 사업보고서는 수집되어 있습니다. 질문 기능은 자료 준비 후 사용할 수 있습니다." + help_text)
        with st.expander("수집된 사업 개요 살펴보기", expanded=True):
            st.write(company["business_overview_excerpt"])
            st.link_button("DART 사업보고서 보기", company["report_source_url"])
    content, guide = st.columns([3.5, 1.25], gap="large")
    with content:
        show_saved_briefing(company)
        with st.container(border=True, key="ui_question_card"):
            st.markdown('<div class="career-question-label">어떤 점이 궁금한가요?</div>', unsafe_allow_html=True)
            with st.form(f"ui_question_form_{code}", clear_on_submit=True):
                question = st.text_area("기업에 대해 궁금한 점", placeholder="예금과 대출 사업을 초보자도 이해할 수 있게 설명해 줘.",
                    height=90, max_chars=1500, label_visibility="collapsed", disabled=not is_ready, key=f"ui_question_{code}")
                submitted = st.form_submit_button("근거로 답변 받기 →", type="primary", disabled=not is_ready, use_container_width=True)
            st.caption("질문을 실행하면 선택한 기업의 공시를 검색해 답변합니다. 처음 조회하는 보고서는 자료 준비 시간과 모델 사용량이 추가될 수 있습니다.")
        if submitted:
            queue_question(question, code)
        pending = st.session_state.pop("ui_pending_request", None)
        if pending and pending["corp_code"] == code:
            execute_question(pending["question"], code, company, ready)
        if code in st.session_state["ui_errors"]:
            st.error(st.session_state["ui_errors"][code])
        turns = st.session_state["ui_turns"][code]
        if turns:
            current = turns[min(st.session_state["ui_active_turn"].get(code, len(turns)-1), len(turns)-1)]
            st.markdown('<div class="career-answer-label">공시 기반 쉬운 해설</div>', unsafe_allow_html=True)
            with st.container(border=True, key="ui_answer_card"):
                st.markdown(f'<div class="career-question">{escape(current["question"])}</div>', unsafe_allow_html=True)
                if len(current.get("receipt_nos", [])) > 1:
                    periods = sorted({source["report_period"] for source in current["sources"]})
                    st.caption("여러 보고서 조회 · 기준기간: " + ", ".join(periods) + " · 각 근거의 기간을 확인해 주세요.")
                summary, detail = split_answer(current["answer"])
                st.markdown(summary)
                if detail:
                    with st.expander("상품·용어·확인 범위 자세히 보기"):
                        st.markdown(detail)
                st.caption("인용 번호는 아래 공시 근거에서 확인할 수 있습니다.")
            st.subheader("이어서 알아보면 좋아요")
            for i, item in enumerate(current.get("questions", [])):
                st.button(item["question"] + "  ↗", key=f"ui_follow_{code}_{i}", use_container_width=True,
                    on_click=queue_question, args=(item["question"], code), disabled=not is_ready)
            st.subheader("답변에 사용한 공시 근거")
            if not current.get("cited_ids"):
                st.info("답변에 인용된 근거가 없습니다. 현재 자료로 확인되지 않은 내용은 사실로 단정하지 마세요.")
            for source in used_sources(current, include_questions=False):
                with st.expander(f"[{source['source_id']}] {source['title']}"):
                    report_name = source.get("report_name", "사업보고서")
                    kind = "반기보고서" if "반기보고서" in report_name else "분기보고서" if "분기보고서" in report_name else "사업보고서"
                    st.caption(f"{source['report_period']} · " + ("기업 메타데이터" if source["source_type"] == "company_metadata" else kind + " 본문"))
                    st.link_button("DART 원문 확인 ↗", source["source_url"])
                    st.text(source["text"])
            answer_ids = set(current.get("cited_ids", []))
            recommendation_sources = [source for source in used_sources(current) if source["source_id"] not in answer_ids]
            if recommendation_sources:
                with st.expander("추천 질문에 연결된 추가 근거"):
                    for source in recommendation_sources:
                        st.markdown(f"**[{source['source_id']}] {escape(source['title'])}**")
                        st.link_button("DART 원문 확인 ↗", source["source_url"])
                        st.text(source["text"])
            st.download_button("이 답변을 노트로 저장 ↓", export_markdown(current, name),
                file_name=f"취준공시_{company['stock_code']}.md", mime="text/markdown", key="ui_download")
        elif is_ready:
            st.subheader("첫 질문이 어렵다면")
            for i, (label, prompt) in enumerate([
                ("회사가 하는 일부터", "이 회사가 하는 일과 주요 사업을 비전공자에게 쉽게 설명해 줘."),
                ("어려운 재무용어 이해하기", "보고서의 주요 실적과 재무용어를 단위와 기간을 포함해 쉽게 설명해 줘."),
                ("면접 준비 포인트 찾기", "지원 전에 알아둘 사업 특징과 공시 기반 면접 연습 질문을 정리해 줘.")]):
                st.button(label + "  ↗", key=f"ui_starter_{i}", on_click=queue_question, args=(prompt, code), use_container_width=True)
    with guide:
        with st.container(border=True, key="ui_guide_card"):
            st.markdown("### 면접 전, 세 가지만")
            st.caption("답변을 읽고 내 말로 정리해 보세요.")
            for i, label in enumerate(["주요 사업을 한 문장으로 설명하기", "어려운 용어 하나 이해하기", "직무와 연결할 질문 하나 남기기"]):
                st.checkbox(label, key=f"ui_check_{code}_{i}")
        st.markdown("### 공시는 이렇게 읽어요")
        st.caption("01  회사가 하는 일부터\n\n02  숫자의 기간과 단위 확인\n\n03  원문으로 한 번 더 확인")
        st.caption("공시에서 확인한 사실과 취업 준비를 위한 해석·제안을 구분해 읽어 주세요.")
    st.divider()
    st.caption("공시 기준시점의 학습·취업 준비용 정보입니다. 현재 금리·상품 조건·채용 정보는 기업의 공식 안내를 확인하세요. 금융 의사결정은 전문가 조언을 확인하세요.")
    with st.expander("글꼴 출처 및 라이선스", key="ui_font_license"):
        st.markdown(
            "본문·보조 문구: **SUITE(스위트) SemiBold 600**, SUNN YOUN. "
            "[공식 소개](https://sun.fo/suite/) · "
            "[눈누](https://noonnu.cc/font_page/1150) · "
            "[SIL OFL 1.1·저작권](https://github.com/sun-typeface/SUITE/blob/main/LICENSE). "
            "상업적 웹 사용 가능. 폰트 단독 판매 금지 및 재배포 시 OFL 조건 준수.\n\n"
            "제목: **KCC간판체**, 한국저작권위원회. "
            "© 2022 Korea Copyright Commission. All rights reserved. "
            "[공식 배포처·가이드](https://gongu.copyright.or.kr/gongu/wrt/wrt/view.do?menuNo=200023&wrtSn=13333397) · "
            "[SIL Open Font License 1.1](https://openfontlicense.org/open-font-license-official-text/).\n\n"
            "상세 출처와 확인 기록: 프로젝트의 `docs/FONT_LICENSES.md`."
        )


if __name__ == "__main__":
    main()
