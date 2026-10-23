"""선택 심화 F2015. 새 비교는 GMS 채팅·임베딩 사용량이 발생합니다."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_rag
from src.cli import run_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    result = evaluate_rag(refresh=args.refresh)
    print(f'RAG 전후 비교 {result["case_count"]}문항 확보. 사실 정확성은 원문 대조 검토가 필요합니다.')
    print('data/processed/evaluation/comparison.json')


if __name__ == '__main__':
    raise SystemExit(run_cli(main))
