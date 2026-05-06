import json
from pathlib import Path

import pytest

import omni_infer
from profile_setup import ensure_profile, resolve_profile_name


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(omni_infer, "PROFILES_DIR", profiles_root)
    monkeypatch.setattr("profile_setup.PROFILES_DIR", profiles_root)
    yield profiles_root


def test_ensure_profile_creates_meta_and_wav(isolated_profiles: Path):
    name, created = ensure_profile("  New_User_Profile  ")
    assert name == "new_user_profile"
    assert created is True
    d = isolated_profiles / name
    assert (d / "meta.json").exists()
    assert (d / "ref_audio.wav").exists()
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == name


def test_ensure_profile_idempotent(isolated_profiles: Path):
    n1, c1 = ensure_profile("foo")
    n2, c2 = ensure_profile("foo")
    assert n1 == n2 == "foo"
    assert c1 is True
    assert c2 is False


def test_resolve_creates_unknown_profile(isolated_profiles: Path):
    n = resolve_profile_name("pytest_profile")
    assert n == "pytest_profile"
    assert (isolated_profiles / n / "meta.json").exists()


def test_resolve_empty_defaults_to_first_or_default(isolated_profiles: Path):
    n = resolve_profile_name("")
    assert n == "default"
    assert (isolated_profiles / "default").is_dir()
