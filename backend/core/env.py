from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path

from dotenv import dotenv_values

from core.env_file import config_lock
from core.models import PROVIDER_KEY_VARIABLES, provider_key


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
KEY_VARS = frozenset(
    variable
    for variables in PROVIDER_KEY_VARIABLES.values()
    for variable in variables
)
_ENCODING_ERROR = "backend/.env 不是 UTF-8 编码，请另存为 UTF-8 后重启"
ENV_READ_ERROR = "无法读取 backend/.env（可能被占用）"
_logger = logging.getLogger(__name__)
_env_error: str | None = None
_encoding_error_logged = False
_read_error_logged = False


def _system_baseline(values: dict | None = None) -> frozenset[str]:
    marker = os.environ.get("MERCY_SYSTEM_KEYS")
    if marker is not None:
        try:
            parsed = json.loads(marker)
            if isinstance(parsed, list) and all(
                isinstance(item, str) and item in KEY_VARS for item in parsed
            ):
                return frozenset(parsed)
        except (json.JSONDecodeError, TypeError):
            pass
    if values is None:
        try:
            values = _read_values()
        except OSError:
            values = {}
    baseline = frozenset(
        variable
        for variable in KEY_VARS
        if os.environ.get(variable, "").strip()
        and os.environ.get(variable, "").strip()
        != (
            values.get(variable).strip()
            if isinstance(values.get(variable), str)
            else None
        )
    )
    os.environ["MERCY_SYSTEM_KEYS"] = json.dumps(sorted(baseline))
    return baseline


def _read_values() -> dict:
    global _env_error, _encoding_error_logged, _read_error_logged
    try:
        with config_lock(ENV_PATH):
            data = Path(ENV_PATH).read_bytes()
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise UnicodeDecodeError(
                "utf-8", data, 0, min(len(data), 2), "UTF-16 BOM"
            )
        text = data.decode("utf-8-sig")
    except FileNotFoundError:
        _env_error = None
        _encoding_error_logged = False
        _read_error_logged = False
        return {}
    except UnicodeDecodeError:
        _env_error = _ENCODING_ERROR
        _read_error_logged = False
        if not _encoding_error_logged:
            _logger.error(_ENCODING_ERROR)
            _encoding_error_logged = True
        return {}
    except OSError:
        _env_error = ENV_READ_ERROR
        _encoding_error_logged = False
        if not _read_error_logged:
            _logger.error(ENV_READ_ERROR)
            _read_error_logged = True
        return {}
    _env_error = None
    _encoding_error_logged = False
    _read_error_logged = False
    return dotenv_values(stream=io.StringIO(text))


def load_env() -> None:
    values = _read_values()
    baseline = _system_baseline(values)
    for variable in KEY_VARS - baseline:
        value = values.get(variable)
        if isinstance(value, str) and value.strip():
            os.environ[variable] = value
        else:
            os.environ.pop(variable, None)
    for variable, value in values.items():
        if (
            variable not in KEY_VARS
            and value is not None
            and variable not in os.environ
        ):
            os.environ[variable] = value


def env_error() -> str | None:
    return _env_error


def key_source(variable: str) -> str | None:
    if variable not in KEY_VARS:
        return None
    baseline = _system_baseline()
    if variable in baseline:
        return "system"
    return "env_file" if os.environ.get(variable, "").strip() else None


def provider_key_source(provider: str) -> str | None:
    variables = PROVIDER_KEY_VARIABLES.get(provider, ())
    baseline = _system_baseline()
    if any(variable in baseline for variable in variables):
        return "system"
    resolved = provider_key(provider)
    if resolved is None:
        return None
    return key_source(resolved[1])
