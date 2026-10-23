import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.pipeline import run_pipeline
from src.ui_support import used_sources
from src.common import ROOT, write_json
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corp-code", default="01133217")
    parser.add_argument(
        "--question",
        default="카카오뱅크가 제공하는 예금과 대출 서비스는 무엇인가요?",
    )
    args = parser.parse_args()

    result = run_pipeline(args.question, args.corp_code)

    output_path = ROOT / "data/processed/latest_pipeline_result.json"
    write_json(output_path, result)

    print(f"\n질문: {result['question']}")
    print(f"\n{result['answer']}")

    print("\n추천 질문")

    for number, item in enumerate(result["questions"], start=1):
        references = " ".join(
            f"[{source_id}]"
            for source_id in sorted(set(item["source_ids"]))
        )
        print(f"{number}. {item['question']} {references}")

    print("\n사용한 근거")

    for source in used_sources(result):
        print(f"\n[{source['source_id']}] {source['title']}")
        print(f"보고기간: {source['report_period']}")
        print(source["source_url"])

    print("\n전체 파이프라인 실행 완료!")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
