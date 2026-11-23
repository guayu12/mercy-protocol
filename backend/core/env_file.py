from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

from dotenv.parser import parse_stream


class ConfigWriteError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


_locks_guard = threading.Lock()
_path_locks: dict[str, threading.Lock] = {}
_SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.-]{1,512}$")


@contextmanager
def config_lock(path: str | os.PathLike[str]):
    resolved = str(Path(path).resolve())
    with _locks_guard:
        lock = _path_locks.setdefault(resolved, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ConfigWriteError("配置文件是符号链接，未保存")
    tmp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=".mercy-", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass

        for attempt in range(3):
            try:
                os.replace(tmp_path, path)
                tmp_path = None
                return
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    except OSError as error:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ConfigWriteError("无法写入配置文件（可能被占用或只读）") from error


def _decode_env_file(path: Path) -> tuple[str, bool]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return "", False
    except OSError as error:
        raise ConfigWriteError("无法写入配置文件（可能被占用或只读）") from error
    has_bom = data.startswith(b"\xef\xbb\xbf")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ConfigWriteError("配置文件不是 UTF-8 编码，未保存")
    try:
        return data.decode("utf-8-sig"), has_bom
    except UnicodeDecodeError as error:
        raise ConfigWriteError("配置文件不是 UTF-8 编码，未保存") from error


def set_env_value(
    path: str | os.PathLike[str],
    name: str,
    value: str,
) -> None:
    if not _SAFE_NAME.fullmatch(name) or not _SAFE_VALUE.fullmatch(value):
        raise ConfigWriteError("key 格式不正确")
    path = Path(path)
    with config_lock(path):
        text, has_bom = _decode_env_file(path)
        newline = "\r\n" if "\r\n" in text else "\n"
        bindings = list(parse_stream(StringIO(text)))
        if "".join(binding.original.string for binding in bindings) != text:
            raise ConfigWriteError("配置文件不是 UTF-8 编码，未保存")

        offsets = []
        offset = 0
        for binding in bindings:
            raw = binding.original.string
            offsets.append((offset, offset + len(raw), binding))
            offset += len(raw)

        first = True
        parts = []
        offset = 0
        for start, end, binding in offsets:
            if binding.key != name:
                continue
            if first:
                raw = binding.original.string
                leading = len(raw) - len(raw.lstrip())
                prefix = raw[:leading]
                if raw[leading:].startswith("export") and (
                    len(raw[leading:]) == 6
                    or raw[leading + 6 : leading + 7].isspace()
                ):
                    prefix += "export "
                trailing = ""
                if raw.endswith("\r\n"):
                    trailing = "\r\n"
                elif raw.endswith("\n") or raw.endswith("\r"):
                    trailing = raw[-1]
                parts.append((start, end, f"{prefix}{name}={value}{trailing}"))
                first = False
            else:
                parts.append((start, end, ""))

        output = text
        for start, end, replacement in reversed(parts):
            output = output[:start] + replacement + output[end:]
        if first:
            if output and not output.endswith(("\n", "\r")):
                output += newline
            output += f"{name}={value}"
        encoded = output.encode("utf-8")
        if has_bom:
            encoded = b"\xef\xbb\xbf" + encoded
        atomic_write_bytes(path, encoded)
