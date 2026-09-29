"""Validated byte plans for SM120 vLLM pages and SGLang DSV4.1 pools.

P pages contain 64 stored states. Their logical token count is 64 * ratio;
the top-level metadata.block_size is not the block size of every cache group.
Values and scales occupy separate spans within a page on both engines.
"""
import os
import zlib
from array import array
from dataclasses import dataclass

REQUEST_LAYOUTS = {}
# bootstrap_room -> (token_count, digest, head, tail) of the decode-side prompt
# sequence. Recorded by _xy_pd_truncate and compared against the producer's own
# fingerprint before a single KV byte is pulled: equal lengths alone do not
# prove the two engines are talking about the same prompt.
PROMPT_DIGESTS = {}


def prompt_digest(token_ids):
    """Fingerprint a prompt token sequence for the cross-engine check."""
    values = [int(token) for token in token_ids]
    packed = array("i", values).tobytes() if values else b""
    return "%d:%08x" % (len(values), zlib.crc32(packed) & 0xFFFFFFFF)


# Enforcement switch. While the flag file is absent a sequence mismatch is
# reported and the request continues (so a first rollout cannot break the
# service); creating the file turns the same check into a hard refusal and no
# decode worker has to be restarted to flip it.
DIGEST_ENFORCE_FLAG = os.environ.get(
    "XY_PROMPT_DIGEST_ENFORCE_FLAG", "/var/tmp/dsv41-d40/xy-digest-enforce"
)


def prompt_digest_enforced() -> bool:
    return bool(DIGEST_ENFORCE_FLAG) and os.path.exists(DIGEST_ENFORCE_FLAG)


def prefix_length(n, tail=256, page_size=256):
    if n < 3:
        raise ValueError("cross-engine tail replay requires at least 3 prompt tokens")
    # Pair compression must restart on an even token. Long prefixes also align
    # to the destination page so no uncomputed bytes become a cached prefix.
    return max(2, ((n - max(tail, 128)) // page_size) * page_size)


@dataclass(frozen=True)
class Span:
    local: int
    remote: int
    size: int


def segregated_spans(*, remote_base, remote_stride, remote_len, remote_blocks,
                     remote_num_blocks, local_base, local_len, local_stride,
                     local_pages, local_states, start_state, count,
                     value_bytes, scale_bytes, label, remote_page_offset=0):
    """Copy logical rows, independently tiling 64-state source pages.

    local_pages starts at the destination page containing start_state. Caller
    supplies an aligned start_state and checks the request's token geometry.
    Every byte span is checked before any transfer is submitted.
    """
    if count < 0 or start_state < 0 or local_states <= 0:
        raise ValueError(f"{label}: invalid row range")
    if local_stride < local_states * (value_bytes + scale_bytes):
        raise ValueError(f"{label}: destination page is too small")
    if remote_len < 64 * (value_bytes + scale_bytes):
        raise ValueError(f"{label}: incompatible source row format")
    spans = []
    done = 0
    while done < count:
        src_page, src_off = divmod(start_state + done, 64)
        src_page -= remote_page_offset
        dst_page_pos, dst_off = divmod(done, local_states)
        if src_page < 0 or src_page >= len(remote_blocks) or dst_page_pos >= len(local_pages):
            raise ValueError(f"{label}: missing source or destination page at row {done}")
        block = int(remote_blocks[src_page])
        page = int(local_pages[dst_page_pos])
        if not 0 < block < remote_num_blocks:
            raise ValueError(f"{label}: null, expired, or invalid source block {block}")
        if page < 0:
            raise ValueError(f"{label}: invalid destination page {page}")
        rows = min(count - done, 64 - src_off, local_states - dst_off)
        for width, src_plane, dst_plane in (
            (value_bytes, 0, 0),
            (scale_bytes, 64 * value_bytes, local_states * value_bytes),
        ):
            if width == 0:
                continue
            src_offset = src_plane + src_off * width
            dst_offset = page * local_stride + dst_plane + dst_off * width
            length = rows * width
            if src_offset + length > remote_len or dst_offset + length > local_len:
                raise ValueError(f"{label}: descriptor exceeds registered allocation")
            spans.append(Span(local_base + dst_offset,
                              remote_base + block * remote_stride + src_offset,
                              length))
        done += rows
    return spans


def region_entry(meta, suffix):
    matches = []
    for name, entry in (meta.get("xy_layer_map") or {}).items():
        name = name.decode() if isinstance(name, bytes) else name
        if name.endswith(suffix):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f"{suffix}: missing cache group/region pair")
            matches.append(tuple(map(int, entry)))
    if len(matches) != 1:
        raise ValueError(f"{suffix}: expected exactly one producer region")
    return matches[0]
