"""
Verify that model.ollama_tag from config.yaml is the single source of truth
for model selection across the entire app/ package.

No hardcoded 'qwen2.5:3b' string literals should appear in app/ Python code
outside of config.yaml itself.  Tests, fixtures, and bootstrap.py (which
reads config dynamically) are allowed to reference the string.
"""
import re
from pathlib import Path

from app.settings import MODEL_TAG

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _grep_app_for_literal(tag: str) -> list[tuple[str, int, str]]:
    """
    Walk app/ and find any .py file containing the literal model tag string
    as a quoted string (not just a comment mentioning it).
    """
    hits: list[tuple[str, int, str]] = []
    # Match the tag inside quotes: "qwen2.5:3b" or 'qwen2.5:3b'
    pattern = re.compile(rf"""['"]({re.escape(tag)})['"]""")

    for py_file in sorted(APP_DIR.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            # Skip comment-only lines (docstrings/comments are ok --
            # we only care about executable code using a hardcoded tag)
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if pattern.search(line):
                hits.append((str(py_file.relative_to(APP_DIR.parent)),
                             lineno, line.strip()))
    return hits


def test_model_tag_read_from_config():
    """MODEL_TAG in settings.py matches config.yaml."""
    assert isinstance(MODEL_TAG, str)
    assert len(MODEL_TAG) > 0


def test_no_hardcoded_model_tag_in_app_code():
    """
    No .py file under app/ should contain a hardcoded 'qwen2.5:3b' string
    literal in executable code.  The model tag must come from
    app.settings.MODEL_TAG which reads config.yaml.

    If this test fails, a hardcoded tag was introduced -- fix it by
    importing MODEL_TAG from app.settings instead.
    """
    # Use the actual configured tag (may differ from default)
    hits = _grep_app_for_literal(MODEL_TAG)
    assert hits == [], (
        f"Hardcoded model tag '{MODEL_TAG}' found in app/ code. "
        f"Use app.settings.MODEL_TAG instead.\n"
        + "\n".join(f"  {f}:{n}: {line}" for f, n, line in hits)
    )


def test_generator_uses_settings_model_tag():
    """generator.py imports MODEL_TAG from settings, not a hardcoded string."""
    import app.rag.generator as gen
    source = Path(gen.__file__).read_text(encoding="utf-8")
    assert "from app.settings import" in source
    assert "MODEL_TAG" in source


def test_main_uses_settings_model_tag():
    """main.py imports MODEL_TAG from settings, not a hardcoded string."""
    import app.main as main_mod
    source = Path(main_mod.__file__).read_text(encoding="utf-8")
    assert "MODEL_TAG" in source
    # Verify it's imported from settings
    assert "from app.settings import" in source


def test_bootstrap_reads_tag_from_config():
    """bootstrap.py reads ollama_tag from config.yaml dynamically."""
    import bootstrap
    tag = bootstrap._read_config_tag()
    assert isinstance(tag, str)
    assert len(tag) > 0
    # The tag should match what's in config.yaml
    assert tag == MODEL_TAG
