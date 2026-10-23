import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vector_store import index_data_main
from src.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(index_data_main))
