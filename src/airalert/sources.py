"""Завантаження сирих даних із маніфестом: джерело, дата, sha256 (AC-1.1)."""

import hashlib
import json
import shutil
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

from airalert import config

MANIFEST = config.RAW / "manifest.json"

# Python з python.org на macOS не бачить системних сертифікатів,
# тому явно беремо набір із certifi
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path, refresh: bool = False) -> Path:
    """Завантажує файл, якщо його ще немає (або refresh=True), і записує його в маніфест.

    Без refresh повторний запуск працює з тим самим знімком даних (AC-1.4).
    """
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = Request(url, headers={"User-Agent": "airalert/0.1"})
    with urlopen(request, context=_SSL_CONTEXT) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f)
    part.replace(dest)
    _record(url, dest)
    return dest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(url: str, path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[path.relative_to(config.RAW).as_posix()] = {
        "url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
