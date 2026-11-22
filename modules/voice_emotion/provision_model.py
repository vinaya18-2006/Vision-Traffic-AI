"""Explicit provisioning command for the pinned local SUPERB emotion model."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

from .detector import MODEL_ID, default_model_path

MODEL_REVISION = "441a7599c3b22107314dcbd9166621c5c83f2cc5"
_MODEL_FILES = {
    "config.json": {
        "size": 2_153,
        "git_blob_sha1": "7a277348ce4faf10d8573a7c21b861f25729a397",
    },
    "preprocessor_config.json": {
        "size": 215,
        "git_blob_sha1": "10f6def8c83d70a2b087a567dcf523b75152a80b",
    },
    "pytorch_model.bin": {
        "size": 378_361_105,
        "sha256": "dac0f46128deec048751ce33eb1a9caecb570904d3d6cf732d6ec345cd6c2e1b",
    },
}


def _verify_file(filename: str, path: Path) -> None:
    """Check a model artifact against the pinned Hub revision metadata."""
    expected = _MODEL_FILES[filename]
    size = path.stat().st_size
    if size != expected["size"]:
        raise ValueError(
            f"Unexpected size for {filename}: received {size} bytes; "
            f"expected {expected['size']}."
        )

    if "sha256" in expected:
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected["sha256"]:
            raise ValueError(f"SHA-256 verification failed for {filename}.")
    else:
        digest = hashlib.sha1()
        digest.update(b"blob " + str(size).encode("ascii") + b"\0")
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected["git_blob_sha1"]:
            raise ValueError(f"Git blob verification failed for {filename}.")


def provision_model(model_directory: Path | None = None) -> tuple[Path, ...]:
    """Download and verify only the three inference files for the pinned model."""
    destination = model_directory or default_model_path()
    destination.mkdir(parents=True, exist_ok=True)
    installed_files: list[Path] = []

    for filename in _MODEL_FILES:
        url = (
            f"https://huggingface.co/{MODEL_ID}/resolve/"
            f"{MODEL_REVISION}/{filename}?download=true"
        )
        target = destination / filename
        partial_path = destination / f".{filename}.part"
        try:
            if target.is_file():
                try:
                    _verify_file(filename, target)
                except (OSError, ValueError):
                    if partial_path.exists():
                        raise ValueError(
                            f"Both {filename} and its resumable partial file exist; "
                            "neither will be overwritten."
                        )
                    os.replace(target, partial_path)
                else:
                    installed_files.append(target)
                    print(f"Already verified {target} ({target.stat().st_size:,} bytes).")
                    continue

            expected_size = _MODEL_FILES[filename]["size"]
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            if offset > expected_size:
                partial_path.unlink()
                offset = 0
            elif offset == expected_size:
                _verify_file(filename, partial_path)
                os.replace(partial_path, target)
                installed_files.append(target)
                print(f"Provisioned {target} ({target.stat().st_size:,} bytes).")
                continue

            headers = {"User-Agent": "RoadSight-Voice-Emotion-Provisioner/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not (
                        content_range.endswith(f"/{expected_size}")
                    ):
                        raise ValueError(
                            f"Unexpected Content-Range for {filename}: "
                            f"{content_range or 'missing'}."
                        )
                    mode = "ab"
                elif status == 200:
                    mode = "wb"
                    offset = 0
                else:
                    raise ValueError(
                        f"Unexpected HTTP status for {filename}: {status}."
                    )

                total_bytes = offset
                with partial_path.open(mode) as partial_file:
                    while chunk := response.read(1024 * 1024):
                        total_bytes += len(chunk)
                        if total_bytes > expected_size:
                            raise ValueError(
                                f"Download for {filename} exceeded the expected size."
                            )
                        partial_file.write(chunk)

            _verify_file(filename, partial_path)
            os.replace(partial_path, target)
            installed_files.append(target)
            print(f"Provisioned {target} ({target.stat().st_size:,} bytes).")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise RuntimeError(f"Could not provision {filename} from {url}: {exc}") from exc

    return tuple(installed_files)


def main() -> None:
    """Run the explicit local model provisioning operation."""
    paths = provision_model()
    print(f"Provisioned {len(paths)} files from {MODEL_ID} at revision {MODEL_REVISION}.")


if __name__ == "__main__":
    main()
