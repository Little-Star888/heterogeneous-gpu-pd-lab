#!/usr/bin/env python3
"""Dump the vLLM producer's NIXL metadata (region names / block lens / strides).

Run inside the SGLang decode container:
    python3 /tmp/xy_dump_meta.py
"""
import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

# Import the nixl package first: it pulls in xyvllm.conn, which in turn imports
# sglang...nixl.conn. Going straight to xyvllm.conn triggers a circular import.
import sglang.srt.disaggregation.nixl  # noqa: F401,E402

from sglang.srt.disaggregation.xyvllm.conn import XYVLLMMeta  # noqa: E402

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "192.168.177.101:5600"

meta = XYVLLMMeta.fetch(ENDPOINT).meta
print("block_size   =", meta.get("block_size"))
print("num_blocks   =", meta.get("num_blocks"))
print("device_id    =", meta.get("device_id"))
lens = meta.get("block_lens")
strides = meta.get("block_strides")
print("block_lens   =", list(lens) if lens is not None else None)
print("block_strides=", list(strides) if strides is not None else None)
layer_map = meta.get("xy_layer_map") or {}
print("layer_map entries =", len(layer_map))
for i, (key, value) in enumerate(layer_map.items()):
    if i >= 48:
        print("  ...")
        break
    name = key.decode() if isinstance(key, bytes) else str(key)
    print("  %-3d %-58s %s" % (i, name, value))
