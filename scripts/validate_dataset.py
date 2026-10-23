import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import validate_dataset_main
from src.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(validate_dataset_main))
