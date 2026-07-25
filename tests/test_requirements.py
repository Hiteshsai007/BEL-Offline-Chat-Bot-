"""
tests/test_requirements.py

Verifies that requirements.txt uses strict pinning (==) instead of ranges (>=).
This is critical for offline air-gapped deployments to ensure consistency.
Also verifies that the wheels/ folder exists and contains files.
"""
from pathlib import Path


def test_requirements_strict_pinning() -> None:
    """Ensure no dependencies in requirements.txt use >= or ~=."""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    assert req_file.exists(), "requirements.txt not found in project root"

    with open(req_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith("#"):
                continue

            assert ">=" not in line, f"Found '>=' on line {line_num}: {line}. Use '=='."
            assert "~=" not in line, f"Found '~=' on line {line_num}: {line}. Use '=='."
            assert "==" in line, f"Missing '==' on line {line_num}: {line}. Pin exactly."


def test_wheels_folder_exists() -> None:
    """Ensure the offline wheels folder exists and has at least some files."""
    wheels_dir = Path(__file__).parent.parent / "wheels"
    assert wheels_dir.exists(), "wheels/ directory not found. Run pip wheel -w wheels -r requirements.txt"
    assert wheels_dir.is_dir(), "wheels/ is not a directory"

    dists = list(wheels_dir.glob("*"))
    assert len(dists) > 0, "wheels/ directory is empty. No offline dependencies found."
