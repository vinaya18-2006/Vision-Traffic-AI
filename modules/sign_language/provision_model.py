"""Explicitly provision and verify the pinned static ASL landmark checkpoint."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import default_model_path
from .errors import ModelLoadError

MODEL_REVISION = "9e5cde38db66ae889322aeea3c547848676fa96e"
MODEL_SIZE_BYTES = 245_970
MODEL_SHA256 = "9A73CD144A41C2CA92BD4D07ECEC2FE32F5F4DDE5E28C8BCC0B0ADA013502F11"
MODEL_URL = (
    "https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier/resolve/"
    f"{MODEL_REVISION}/model.keras?download=true"
)


def provision_model(destination: Path | str | None = None) -> Path:
    """Download only the pinned small model file after explicit user invocation."""
    model_path = Path(destination).expanduser().resolve() if destination else default_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        if _sha256_file(model_path) == MODEL_SHA256:
            return model_path
        raise ModelLoadError(
            f"Refusing to overwrite an existing checkpoint with a different checksum: "
            f"'{model_path}'."
        )

    request = Request(MODEL_URL, headers={"User-Agent": "RoadSight-Sign-Language/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read(MODEL_SIZE_BYTES + 1)
    except (OSError, URLError) as exc:
        raise ModelLoadError(f"Could not retrieve the pinned ASL checkpoint: {exc}") from exc
    if len(payload) != MODEL_SIZE_BYTES:
        raise ModelLoadError(
            f"Pinned ASL checkpoint has unexpected size {len(payload)} bytes; "
            f"expected {MODEL_SIZE_BYTES}."
        )
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != MODEL_SHA256:
        raise ModelLoadError(
            f"Pinned ASL checkpoint checksum mismatch; expected {MODEL_SHA256}, got {digest}."
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{model_path.name}.",
            suffix=".tmp",
            dir=model_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, model_path)
    except OSError as exc:
        raise ModelLoadError(
            f"Could not install the verified ASL checkpoint at '{model_path}': {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return model_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    print(provision_model())
