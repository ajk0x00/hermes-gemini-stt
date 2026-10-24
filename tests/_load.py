"""Shared test helpers: load the plugin as a package and provide a settings ctx."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
_HERMES_REPO = Path(os.environ.get("HERMES_REPO") or (Path.home() / ".hermes" / "hermes-agent"))


def _ensure_repo_on_path() -> None:
    if (_HERMES_REPO / "agent" / "transcription_provider.py").is_file():
        if str(_HERMES_REPO) not in sys.path:
            sys.path.insert(0, str(_HERMES_REPO))


def load_plugin():
    """Import the plugin directory the way the Hermes loader would (relative imports intact)."""
    _ensure_repo_on_path()
    name = "gemini_stt_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not build an import spec for the plugin")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeContext:
    """Minimal stand-in for ``PluginContext``; only ``get_config`` is used by the plugin."""

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = dict(settings or {})
        self.reads: list[str] = []

    def get_config(self, key: str, default=None):
        self.reads.append(key)
        return self.settings.get(key, default)
