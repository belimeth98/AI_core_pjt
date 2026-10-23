import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.briefing import prepare_briefing_context, save_briefing_context
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corp-code", default="01133217")
    args = parser.parse_args()

    print("취업 준비 브리핑에 필요한 공시 근거를 검색합니다.")

    context = prepare_briefing_context(args.corp_code)

    output_path = save_briefing_context(context)

    print(f"\n기업: {context['company']['corp_name']}")
    print(f"보고기간: {context['company']['report_period']}")
    print(f"중복 제거 후 근거: {len(context['sources'])}개")

    for source in context["sources"]:
        print(f"\n[{source['source_id']}] {source['title']}")
        print(source["source_url"])
        print(source["text"][:180])

    print(f"\n저장 위치: {output_path}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
