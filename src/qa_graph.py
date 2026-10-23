"""공시 검색 → 답변·추천 질문 생성의 단일 LangGraph 경로."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from src.retrieval import search_company
from src.answering import run_rag

class RagState(TypedDict, total=False):
    question: str
    corp_code: str
    receipt_no: str
    receipt_nos: list[str]
    search_result: dict
    sources: list[dict]
    answer: str
    cited_ids: list[int]
    questions: list[dict]


def build_graph(searcher=None, answerer=None):
    searcher = searcher or search_company
    answerer = answerer or run_rag

    def retrieve(state):
        print("\n[retrieve] 질문에 맞는 공시 근거를 검색합니다.")
        result = (searcher(state["question"], state["corp_code"], receipt_nos=state["receipt_nos"])
                  if state.get("receipt_nos") else searcher(state["question"], state["corp_code"]))
        return {"search_result": result}

    def generate(state):
        print("\n[generate] 답변과 추천 질문을 생성합니다.")
        return answerer(state["search_result"])

    builder = StateGraph(RagState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
