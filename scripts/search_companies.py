"""자연어 사업 조건으로 기업을 찾고 선택 근거를 출력합니다."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.company_search import find_companies
from src.common import ROOT, write_json
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    result = find_companies(args.query, limit=args.limit)
    for item in result["matches"]:
        print(f"\n{item.get('stock_name') or item['corp_name']} ({item['stock_code']})")
        print(item["reason"])
        for support in item["supports"]:
            print(f"조건: {support['condition']}\n근거: {support['quote']}")
        print(item["report_source_url"])
    if not result["matches"]:
        print("현재 기업 개요에서 모든 조건을 확인한 기업을 찾지 못했습니다.")
    print(f"\n후보 {result['candidate_count']}개 확인 / 채팅 API 요청 {result['chat_api_requests']}회 / 캐시 {result['cache_hit']}")
    write_json(ROOT / "data/processed/company_search_result.json", result)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
