#!/usr/bin/env python3
"""Apply the XYVLLM transfer-backend branch to the *already patched* SGLang
overlay utils.py on a Spark host (idempotent, text-exact, asserts single hits).

Usage:  python3 apply_xy_patch.py /var/tmp/dsv41-d40/overlay/utils.py
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/dsv41-d40/overlay/utils.py"

OLD_ENUM = '    FAKE = "fake"\n'
NEW_ENUM = (
    '    FAKE = "fake"\n'
    '    # XY cross-engine PD: SGLang decode pulls KV from a vLLM NIXL producer.\n'
    '    XYVLLM = "xyvllm"\n'
)

OLD_FACTORY = "    elif transfer_backend == TransferBackend.FAKE:\n"
NEW_FACTORY = (
    "    elif transfer_backend == TransferBackend.XYVLLM:\n"
    "        from sglang.srt.disaggregation.xyvllm import (\n"
    "            XYVLLMKVManager,\n"
    "            XYVLLMKVReceiver,\n"
    "        )\n"
    "\n"
    "        # No bootstrap server: the prefill side is a vLLM NIXL producer.\n"
    "        class_mapping = {\n"
    "            KVClassType.MANAGER: XYVLLMKVManager,\n"
    "            KVClassType.RECEIVER: XYVLLMKVReceiver,\n"
    "        }\n"
    "    elif transfer_backend == TransferBackend.FAKE:\n"
)

with open(PATH, encoding="utf-8") as fh:
    src = fh.read()

if "XYVLLM = \"xyvllm\"" in src:
    print("ALREADY_PATCHED")
    raise SystemExit(0)

assert src.count(OLD_ENUM) == 1, "enum anchor hit %d times" % src.count(OLD_ENUM)
assert src.count(OLD_FACTORY) == 1, "factory anchor hit %d times" % src.count(OLD_FACTORY)

src = src.replace(OLD_ENUM, NEW_ENUM, 1)
src = src.replace(OLD_FACTORY, NEW_FACTORY, 1)

with open(PATH, "w", encoding="utf-8") as fh:
    fh.write(src)

print("PATCHED_OK bytes=%d" % len(src))
