import os
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from core import env
from core.env_file import ConfigWriteError, atomic_write_bytes, set_env_value
from core.models import PROVIDER_KEY_VARIABLES


FAKE_KEYS = {
    variable
    for variables in PROVIDER_KEY_VARIABLES.values()
    for variable in variables
}


def _clear_keys(monkeypatch):
    for variable in FAKE_KEYS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("MERCY_SYSTEM_KEYS", raising=False)


def test_env_sources_and_empty_baseline(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=file-fake-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)

    env.load_env()

    assert env.key_source("DEEPSEEK_API_KEY") == "env_file"
    assert env.provider_key_source("deepseek") == "env_file"
    assert os.environ["MERCY_SYSTEM_KEYS"] == "[]"
    assert env.key_source("OPENAI_API_KEY") is None


def test_system_key_is_not_overridden(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "system-fake-key")
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=file-fake-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)

    env.load_env()

    assert os.environ["OPENAI_API_KEY"] == "system-fake-key"
    assert env.key_source("OPENAI_API_KEY") == "system"
    assert env.provider_key_source("openai") == "system"


def test_process_key_matching_env_value_is_not_system(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", " file-fake-key ")
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=file-fake-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)

    env.load_env()

    assert env.key_source("OPENAI_API_KEY") == "env_file"
    assert env.provider_key_source("openai") == "env_file"


def test_child_reload_uses_new_env_file_value(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=old-fake-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    env.load_env()
    inherited = os.environ.copy()
    path.write_text("DEEPSEEK_API_KEY=new-fake-key\n", encoding="utf-8")
    code = (
        "import os, sys; from pathlib import Path; from core import env; "
        "env.ENV_PATH=Path(sys.argv[1]); env.load_env(); "
        "assert os.environ['DEEPSEEK_API_KEY']=='new-fake-key'; "
        "print(env.key_source('DEEPSEEK_API_KEY'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=inherited,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "env_file"


def test_utf16_env_sets_error_and_continues(tmp_path, monkeypatch, caplog):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "system-fake-key")
    path = tmp_path / ".env"
    path.write_bytes("DEEPSEEK_API_KEY=file-fake".encode("utf-16"))
    monkeypatch.setattr(env, "ENV_PATH", path)

    env.load_env()

    assert env.env_error() == "backend/.env 不是 UTF-8 编码，请另存为 UTF-8 后重启"
    assert os.environ["OPENAI_API_KEY"] == "system-fake-key"
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert caplog.text.count(env.env_error()) == 1


@pytest.mark.parametrize(
    ("source", "name", "value"),
    [
        ("# comment\n\nOPENAI_API_KEY='old'\nOTHER=keep\n", "OPENAI_API_KEY", "new.fake-key"),
        ("export OPENAI_API_KEY=\"old\"\r\nOTHER=keep\r\n", "OPENAI_API_KEY", "new.fake-key"),
        ("OTHER=keep", "OPENAI_API_KEY", "new.fake-key"),
        ("", "OPENAI_API_KEY", "new.fake-key"),
    ],
)
def test_set_env_value_preserves_dotenv_values(tmp_path, source, name, value):
    path = tmp_path / ".env"
    path.write_bytes(source.encode())
    before = dotenv_values(path)

    set_env_value(path, name, value)

    after = dotenv_values(path)
    assert after[name] == value
    for other in before.keys() - {name}:
        assert after[other] == before[other]
    result = path.read_bytes()
    assert (b"\r\n" in result) == ("\r\n" in source)
    if source.startswith("export"):
        assert b"export OPENAI_API_KEY=new.fake-key\r\n" in result
    if source == "OTHER=keep":
        assert result == b"OTHER=keep\nOPENAI_API_KEY=new.fake-key\n"


def test_set_env_value_appends_new_file_with_lf(tmp_path):
    path = tmp_path / ".env"

    set_env_value(path, "DEEPSEEK_API_KEY", "fake-deepseek-key")

    assert path.read_bytes() == b"DEEPSEEK_API_KEY=fake-deepseek-key\n"


def test_set_env_value_appends_with_crlf_and_preserves_existing_bytes(tmp_path):
    path = tmp_path / ".env"
    original = b"OTHER=keep\r\n"
    path.write_bytes(original)

    set_env_value(path, "DEEPSEEK_API_KEY", "fake-deepseek-key")

    assert path.read_bytes() == original + b"DEEPSEEK_API_KEY=fake-deepseek-key\r\n"


def test_set_env_value_deduplicates_and_preserves_bom(tmp_path):
    path = tmp_path / ".env"
    original = (
        b"\xef\xbb\xbf# comment\r\n"
        b"OPENAI_API_KEY='first'\r\n"
        b"OTHER=keep\r\n"
        b"export OPENAI_API_KEY=\"second\"\r\n"
    )
    path.write_bytes(original)

    set_env_value(path, "OPENAI_API_KEY", "fake-new-key")

    result = path.read_bytes()
    assert result.startswith(b"\xef\xbb\xbf")
    assert result.count(b"OPENAI_API_KEY=") == 1
    assert result == (
        b"\xef\xbb\xbf# comment\r\n"
        b"OPENAI_API_KEY=fake-new-key\r\n"
        b"OTHER=keep\r\n"
    )


def test_multiline_quoted_value_is_not_treated_as_binding(tmp_path):
    path = tmp_path / ".env"
    source = (
        'OTHER="first line\nOPENAI_API_KEY=inside\nlast line"\n'
        "OPENAI_API_KEY=real-binding\n"
    )
    path.write_text(source, encoding="utf-8")

    set_env_value(path, "OPENAI_API_KEY", "fake-new-key")

    values = dotenv_values(path)
    assert values["OTHER"] == "first line\nOPENAI_API_KEY=inside\nlast line"
    assert values["OPENAI_API_KEY"] == "fake-new-key"
    assert path.read_text(encoding="utf-8").startswith('OTHER="first line\n')


def test_utf16_write_is_rejected_unchanged(tmp_path):
    path = tmp_path / ".env"
    original = "OPENAI_API_KEY=fake".encode("utf-16")
    path.write_bytes(original)

    with pytest.raises(ConfigWriteError, match="配置文件不是 UTF-8 编码，未保存"):
        set_env_value(path, "OPENAI_API_KEY", "fake-new-key")

    assert path.read_bytes() == original


def test_symlink_write_is_rejected(tmp_path):
    target = tmp_path / "target.env"
    target.write_text("OPENAI_API_KEY=unchanged\n", encoding="utf-8")
    path = tmp_path / ".env"
    try:
        path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ConfigWriteError, match="配置文件是符号链接，未保存"):
        set_env_value(path, "OPENAI_API_KEY", "fake-new-key")
    assert target.read_text(encoding="utf-8") == "OPENAI_API_KEY=unchanged\n"


def test_atomic_write_ignores_chmod_failure(tmp_path, monkeypatch):
    path = tmp_path / "config"
    monkeypatch.setattr("core.env_file.os.chmod", lambda *_args: (_ for _ in ()).throw(OSError()))

    atomic_write_bytes(path, b"written")

    assert path.read_bytes() == b"written"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode assertion")
def test_atomic_write_uses_private_permissions(tmp_path):
    path = tmp_path / "config"

    atomic_write_bytes(path, b"written")

    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_write_retries_replace_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "config"
    path.write_bytes(b"original")
    real_replace = os.replace
    attempts = []

    def fail_replace(source, target):
        attempts.append((source, target))
        raise PermissionError("busy")

    monkeypatch.setattr("core.env_file.os.replace", fail_replace)
    monkeypatch.setattr("core.env_file.time.sleep", lambda _delay: None)

    with pytest.raises(ConfigWriteError, match="无法写入配置文件"):
        atomic_write_bytes(path, b"replacement")

    assert len(attempts) == 3
    assert path.read_bytes() == b"original"
    assert list(tmp_path.glob(".mercy-*.tmp")) == []
    monkeypatch.setattr("core.env_file.os.replace", real_replace)
