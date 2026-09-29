#!/bin/sh
set -eu

BASE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BUNDLE="${DS41_V8_BUNDLE:-$BASE/bundle}"
MANIFEST_ADD="$BUNDLE/patch/manifest.add"
OVERLAY_DIR="/var/tmp/dsv41-d40/overlay"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: apply-bundle.sh [--overlay-dir DIR] [--dry-run]

Apply the frozen V8 cross-engine overlay and the final DSpark/compute delta.
The target must be an existing SGLang D40 overlay directory from the role image.
Model weights, Engram dictionaries, role images, and host networking are never
copied or changed by this script.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --overlay-dir) [ "$#" -ge 2 ] || { usage >&2; exit 2; }; OVERLAY_DIR=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

test -s "$BUNDLE/packages/xyvllm-overlay-20260920.tgz"
test -s "$BUNDLE/packages/xyvllm-dspark-compute-delta-20260920.tgz"

echo "V8 package order: xyvllm-overlay-20260920.tgz -> xyvllm-dspark-compute-delta-20260920.tgz"
echo "target overlay: $OVERLAY_DIR"
if [ "$DRY_RUN" -eq 1 ]; then
    echo 'DRY-RUN: no files changed'
    exit 0
fi

test -d "$OVERLAY_DIR" || { echo "overlay directory does not exist: $OVERLAY_DIR" >&2; exit 1; }
tar -xzf "$BUNDLE/packages/xyvllm-overlay-20260920.tgz" -C "$OVERLAY_DIR"
tar -xzf "$BUNDLE/packages/xyvllm-dspark-compute-delta-20260920.tgz" -C "$OVERLAY_DIR"

if [ -f "$OVERLAY_DIR/manifest" ]; then
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        grep -Fqx "$line" "$OVERLAY_DIR/manifest" || printf '%s\n' "$line" >> "$OVERLAY_DIR/manifest"
    done < "$MANIFEST_ADD"
fi

echo 'PASS: V8 overlay and final compute delta extracted'
echo 'NEXT: source runtime.env on the D nodes and restart the role image; start the P proxy with xy_pd_proxy.py from this bundle.'
