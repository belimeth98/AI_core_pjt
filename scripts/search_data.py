import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.common import ROOT, read_json, write_json
from src.chunking import load_settings
from src.retrieval import search_company
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corp-code")
    parser.add_argument("--question", default="카카오뱅크의 주요 사업과 수익원은 무엇인가요?")
    args = parser.parse_args()
    settings = load_settings()
    code = args.corp_code or settings["pilot_corp_code"]
    companies = read_json(ROOT / "data/companies.json")
    company = next((c for c in companies if c["corp_code"] == code), None)
    if not company:
        raise ValueError("기업 목록에 없는 기업입니다.")
    result = search_company(args.question, code)
    write_json(ROOT / "data/processed/retrieval_result.json", result)
    print(f"기업: {company['corp_name']} / {company['report_period']} 연간")
    print(f"질문: {args.question}\n검색 근거: {len(result['evidence'])}개")
    for i, item in enumerate(result["evidence"], 1):
        meta = item["metadata"]
        score = f"cosine distance {item['distance']:.4f}" if item["distance"] is not None else "핵심어 검색 후보"
        print(f"\n[{i}] {meta['title']} ({item.get('embedding_model', '')} / {score})")
        print(meta["source_url"])
        print(item["text"][:650])

    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
