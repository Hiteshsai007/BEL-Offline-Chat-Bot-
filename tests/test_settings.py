"""
Tests for app/settings.py — config loading and error handling (H-6).
"""
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.settings import _get, _load


def test_get_missing_key_raises_descriptive_error() -> None:
    """A missing nested key must raise a descriptive KeyError naming the path."""
    cfg = {"model": {"ollama_tag": "qwen2.5:3b"}}
    with pytest.raises(KeyError, match="Missing required config key: .*model -> ollama_url"):
        _get(cfg, "model", "ollama_url")


def test_get_non_dict_intermediate_raises_descriptive_error() -> None:
    """A non-dict intermediate value must raise a descriptive TypeError."""
    cfg = {"model": "not_a_dict"}
    with pytest.raises(TypeError, match="expected dict at 'ollama_tag'"):
        _get(cfg, "model", "ollama_tag")


def test_load_missing_file_raises_descriptive_error(tmp_path: Path) -> None:
    """Missing config file must raise a descriptive FileNotFoundError."""
    missing = tmp_path / "nonexistent.yaml"
    with patch("app.settings._CONFIG_PATH", missing):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load()


def test_load_invalid_yaml_raises_descriptive_error(tmp_path: Path) -> None:
    """Invalid YAML must raise a descriptive yaml.YAMLError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("{ invalid yaml: [", encoding="utf-8")
    with patch("app.settings._CONFIG_PATH", bad):
        with pytest.raises(yaml.YAMLError, match="not valid YAML"):
            _load()


def test_load_non_dict_top_level_raises_descriptive_error(tmp_path: Path) -> None:
    """A top-level scalar in config.yaml must raise TypeError."""
    bad = tmp_path / "scalar.yaml"
    bad.write_text("just_a_string", encoding="utf-8")
    with patch("app.settings._CONFIG_PATH", bad):
        with pytest.raises(TypeError, match="must contain a top-level mapping"):
            _load()
