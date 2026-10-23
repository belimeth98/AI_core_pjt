import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.briefing import create_company_briefing
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corp-code", default="01133217")
    parser.add_argument("--refresh", action="store_true", help="새로 생성하면 GMS 사용량이 발생합니다.")
    args = parser.parse_args()
    saved, output_path, reused = create_company_briefing(args.corp_code, refresh=args.refresh)
    print("저장된 브리핑을 재사용합니다." if reused else "새 브리핑을 생성했습니다.")
    print(saved["markdown"])
    print("\n브리핑 형식·출처 번호·원문 발췌 검사 통과!")
    print(f"저장 위치: {output_path}")
    print(f"읽기용 파일: {output_path.with_suffix('.md')}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
