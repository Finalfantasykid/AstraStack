"""Tests for Preferences.py, using a temporary config directory."""
import json
import os
import pathlib

import pytest

import Preferences as PreferencesModule
from Globals import g
from Preferences import Preferences


@pytest.fixture
def configDir(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Points the user config directory at ``tmp_path`` and enables saving."""
    monkeypatch.setattr(PreferencesModule, "user_config_dir", lambda: str(tmp_path))
    monkeypatch.setattr(g, "TESTING", False)
    return tmp_path


def test_file_location(configDir: pathlib.Path) -> None:
    """Preferences live in an ``AstraStack`` folder of the user config directory, created if missing."""
    preferences = Preferences()
    assert preferences.file == os.path.join(configDir, "AstraStack", "preferences.json")
    assert os.path.isdir(os.path.join(configDir, "AstraStack"))


def test_defaults(configDir: pathlib.Path) -> None:
    """Missing fields give the default, or an empty string without one."""
    preferences = Preferences()
    assert preferences.get("cpus") == ""
    assert preferences.get("cpus", 4) == 4


def test_set_get_round_trip(configDir: pathlib.Path) -> None:
    """Set values are returned by ``get`` and by a new ``Preferences`` reading the saved file."""
    preferences = Preferences()
    preferences.set("cpus", 3)
    preferences.set("darkTheme", True)
    assert preferences.get("cpus", 4) == 3
    with open(preferences.file) as f:
        assert json.load(f) == {"cpus": 3, "darkTheme": True}
    reloaded = Preferences()
    assert reloaded.get("cpus", 4) == 3
    assert reloaded.get("darkTheme") is True


def test_corrupt_file_uses_defaults(configDir: pathlib.Path) -> None:
    """An unreadable preferences file is ignored."""
    os.makedirs(os.path.join(configDir, "AstraStack"))
    with open(os.path.join(configDir, "AstraStack", "preferences.json"), "w") as f:
        f.write("{not json")
    assert Preferences().get("cpus", 4) == 4


def test_testing_mode_does_not_touch_files(configDir: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """While testing, preferences are kept in memory only."""
    monkeypatch.setattr(g, "TESTING", True)
    preferences = Preferences()
    preferences.set("cpus", 3)
    assert preferences.get("cpus") == 3
    assert os.listdir(configDir) == []
