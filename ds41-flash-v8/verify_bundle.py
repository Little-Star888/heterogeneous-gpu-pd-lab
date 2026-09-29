"""Verify the self-contained V8 Docker bundle without role images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
BUNDLE = BASE / "bundle"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((BUNDLE / "baseline.json").read_text(encoding="utf-8-sig"))
    assert manifest["version"] == "V8"
    assert manifest["architecture"]["P"]["tp"] == 2
    assert manifest["architecture"]["D"]["tp"] == 4
    assert manifest["architecture"]["D"]["ep"] == 4
    assert manifest["effective_d"]["speculative_algorithm"] == "DSPARK"
    assert manifest["effective_d"]["speculative_dspark_block_size"] == 5
    assert manifest["effective_d"]["chunked_prefill_size"] == 2048
    assert manifest["tail_tokens_configured"] == 256

    checks = 0
    for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative.startswith("../"):
            continue
        path = BUNDLE / relative
        assert path.is_file(), relative
        assert digest(path) == expected, relative
        checks += 1

    packages = manifest["package_order"]
    assert packages == [
        "packages/xyvllm-overlay-20260920.tgz",
        "packages/xyvllm-dspark-compute-delta-20260920.tgz",
    ]
    for package in packages:
        assert (BUNDLE / package).is_file(), package
    runtime = (BUNDLE / "runtime.env").read_text(encoding="utf-8-sig")
    for required in ("SPEC_ALGO=DSPARK", "DSPARK_BLOCK_SIZE=5", "flashinfer_cutlass"):
        assert required in runtime, required
    print(f"PASS: V8 bundle checks={checks}; TP2->TP4/EP4; package order=2; runtime=OK")


if __name__ == "__main__":
    main()
