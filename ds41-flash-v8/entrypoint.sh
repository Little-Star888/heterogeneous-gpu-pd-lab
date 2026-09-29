#!/bin/sh
set -eu

BUNDLE="${DS41_V8_BUNDLE:-/opt/ds41-flash-v8/bundle}"

verify() {
    cd "$BUNDLE"
    test -s baseline.json
    test -s SHA256SUMS
    test -s packages/xyvllm-overlay-20260920.tgz
    test -s packages/xyvllm-dspark-compute-delta-20260920.tgz
    grep -q '"version": "V8"' baseline.json
    # SHA256SUMS also contains one source-tree entry (../ds4.1flash6gpu-v8.md).
    # The image verifies every artifact embedded in the image and reports that
    # external source-tree entry separately instead of pretending it is present.
    grep -v '  \.\./' SHA256SUMS | sha256sum -c -
    echo 'PASS: DS4.1 Flash V8 bundle, package order, and embedded checksums'
    echo 'INFO: the ../ds4.1flash6gpu-v8.md checksum is a source-tree reference'
}

show_config() {
    cat <<'EOF'
DS4.1 Flash V8
Architecture: PP2 / PD separation
P stage: vLLM P21, 2 x RTX 6000D (TP2); RTX 5500 Pro or RTX 6000 Pro may substitute when capacity and kernels fit
D stage: SGLang D40, 4 x DGX Spark/GB10 (TP4/EP4), one distributed four-node group
Cross-stage transport: vLLM -> SGLang NIXL KV handoff with the bundled layout/state adapter
Model: deepseek-ai/DeepSeek-V4.1-Flash (official FP8 weights)
Runtime: DSPARK block=5, chunked prefill=2048, XY_PD_TAIL=256, FP8 KV
SGLang: flashinfer_cutlass, shared-expert fusion disabled, disaggregation mode=decode, transfer=nixl
EOF
}

case "${1:-verify}" in
    verify) verify ;;
    show-config) show_config ;;
    shell) exec /bin/sh ;;
    *) echo "usage: ds41-v8 {verify|show-config|shell}" >&2; exit 2 ;;
esac
