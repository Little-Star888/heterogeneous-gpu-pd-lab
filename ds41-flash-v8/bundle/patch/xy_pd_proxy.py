#!/usr/bin/env python3
"""xy_pd_proxy.py - cross-engine PD front door (vLLM P21 prefill -> SGLang D40 decode).

Architecture (see hetero-v2/PLAN-xyvllm-pd-20260919.md):

* P = vLLM P21 ``kv_producer`` (2xRTX 6000D, TP2, layers 0-20) on ysy101:8000.
  It only publishes *where* its KV lives (``transfer_mode=pull``) and only
  starts its ZMQ side-channel listener once it has been driven by a PD request.
* D = SGLang D40 decode worker (4xSpark, TP4) - its ``XYVLLMKVReceiver`` pulls
  the producer's KV over NIXL instead of waiting for a push.

Per request:

  1. copy of the body -> ``max_tokens=1``, ``stream=false``, ``kv_transfer_params={do_remote_decode: true, ...}``
     POST to P; P prefills layers 0-20 and answers with ``kv_transfer_params``
     (``remote_block_ids`` / ``remote_engine_id`` / ``remote_host`` / ``remote_port`` / ...);
  2. allocate a ``bootstrap_room`` and park that mapping in :class:`RoomTable`;
  3. original body + ``bootstrap_host``/``bootstrap_port``/``bootstrap_room`` -> POST to D.
     The SGLang-side receiver looks the mapping up via ``GET /xyblocks?room=N``
     (``XY_VLLM_BLOCKS_URL``) and issues the NIXL READs;
  4. relay D's response (SSE stream or JSON) back chunk by chunk.

Endpoints: ``POST /v1/chat/completions``, ``POST /v1/completions``,
``GET /xyblocks?room=N``, ``GET /health``, ``GET /live``, ``GET /stats``.

  python3 xy_pd_proxy.py --port 5701 \\
      --prefill-url http://192.168.177.101:8000 \\
      --decode-url  http://192.168.177.47:8200

Runs inside the vllm image (aiohttp is a vLLM dependency), same as pd_proxy.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
import zlib
from array import array
from contextlib import suppress
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
except ImportError:  # pragma: no cover
    sys.stderr.write("xy_pd_proxy.py needs aiohttp (pip install aiohttp) - run it inside the vllm image\n")
    sys.exit(2)

log = logging.getLogger("xy_pd_proxy")

P_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}

ROOM_TTL = 900.0

# Reserved placeholder / grounding tokens at the top of the vocabulary. The
# decode engine copies these out of the prompt (an answer once came back with
# ``<|place_holder_mm_span_0442|>`` in it), so keep them out of the answer while
# leaving the prompt untouched - the prompt still needs the producer's image pad.
RESERVED_TOKEN_IDS = range(128847, 129280)


def normalize_image_token_text(body: Dict[str, Any]) -> Dict[str, Any]:
    """Quote literal image markers in text before BOTH engines see the prompt.

    D's chat encoder rejects this literal in any message, including old assistant
    replies and tool output. P accepts it, so normalizing only at D would produce
    different token sequences for the transferred KV. Real image blocks remain
    intact; their actual placeholders are inserted later by each image processor.
    Use visible spaces rather than silently dropping text or hiding characters.
    """
    marker = "<\uff5cdeepseek_image\uff5c>"
    quoted = "< \uff5cdeepseek_image\uff5c >"

    def text(value):
        return value.replace(marker, quoted) if isinstance(value, str) else value

    def blocks(value):
        if not isinstance(value, list):
            return text(value)
        result = []
        for block in value:
            if not isinstance(block, dict):
                result.append(block)
                continue
            item = dict(block)
            if item.get("type") in ("text", "input_text", "output_text"):
                item["text"] = text(item.get("text"))
            # The decoder supports nested text/refusal content blocks.
            if item.get("type") not in ("image", "image_url", "input_image"):
                if "content" in item:
                    item["content"] = blocks(item["content"])
            result.append(item)
        return result

    result = dict(body)
    messages = body.get("messages")
    if isinstance(messages, list):
        result["messages"] = []
        for message in messages:
            if not isinstance(message, dict):
                result["messages"].append(message)
                continue
            item = dict(message)
            for field in ("content", "content_blocks"):
                if field in item:
                    item[field] = blocks(item[field])
            if "reasoning_content" in item:
                item["reasoning_content"] = text(item["reasoning_content"])
            result["messages"].append(item)
    return result


def canonical_prompt_token_ids(value: Any) -> Optional[List[int]]:
    """Return a safe prompt id list, or None when the value is not one.

    Only a non-empty list of ints is canonical. Booleans are rejected because
    they are ints in Python and would silently change the sequence length.
    """
    if not isinstance(value, list) or not value:
        return None
    ids: List[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            return None
        ids.append(int(token))
    return ids


def producer_prompt_token_ids(payload: Dict[str, Any]) -> tuple:
    """Read the producer's prompt_token_ids without inventing a count.

    Returns ``(ids, None)`` when one safe list is present, ``(None, None)``
    when the field is absent (caller keeps the previous path), and
    ``(None, message)`` when a field is present but not a safe int list or
    the two known locations disagree.
    """
    if not isinstance(payload, dict):
        return None, "prefill response is not an object"
    top = payload.get("prompt_token_ids", None)
    choice = None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0].get("prompt_token_ids", None)
    has_top = "prompt_token_ids" in payload and top is not None
    has_choice = choice is not None
    if not has_top and not has_choice:
        return None, None
    top_ids = canonical_prompt_token_ids(top) if has_top else None
    choice_ids = canonical_prompt_token_ids(choice) if has_choice else None
    if has_top and top_ids is None:
        return None, "prompt_token_ids is not a non-empty int list"
    if has_choice and choice_ids is None:
        return None, "choices[0].prompt_token_ids is not a non-empty int list"
    if top_ids is not None and choice_ids is not None and top_ids != choice_ids:
        return None, "prompt_token_ids locations disagree"
    return top_ids or choice_ids, None


def sequence_digest(token_ids: List[int]) -> str:
    """Fingerprint the producer's real prompt sequence.

    The decode side computes the same fingerprint over the ids it replays
    from; matching lengths alone cannot catch a same-length mismatch, which
    installed the producer's KV in front of a different prompt.
    """
    values = [int(token) for token in token_ids]
    packed = array("i", values).tobytes() if values else b""
    return "%d:%08x" % (len(values), zlib.crc32(packed) & 0xFFFFFFFF)


def flatten_blocks(kv: Dict[str, Any]) -> List[int]:
    """``remote_block_ids`` is nested when the producer batches: [[1, 2, ...]]."""
    blocks = kv.get("remote_block_ids") or []
    if blocks and isinstance(blocks[0], (list, tuple)):
        blocks = blocks[0]
    return [int(b) for b in blocks]


class RoomTable:
    """``bootstrap_room`` -> producer block mapping, read back by the D receiver."""

    def __init__(self, ttl: float = ROOM_TTL):
        self.ttl = ttl
        self.rooms: Dict[int, Dict[str, Any]] = {}

    def new_room(self) -> int:
        while True:
            room = random.randint(1, 2 ** 31 - 1)
            if room not in self.rooms:
                return room

    def put(self, room: int, kv: Dict[str, Any]) -> None:
        entry = dict(kv)
        entry["at"] = time.time()
        self.rooms[room] = entry

    def get(self, room: int) -> Optional[Dict[str, Any]]:
        self.sweep()
        return self.rooms.get(room)

    def sweep(self) -> None:
        if not self.rooms:
            return
        cutoff = time.time() - self.ttl
        for room in [r for r, e in self.rooms.items() if e.get("at", 0) < cutoff]:
            self.rooms.pop(room, None)


class Proxy:
    def __init__(self, prefill_url: str, decode_url: str, timeout: float,
                 kv_host: Optional[str] = None, kv_port: Optional[int] = None):
        self.p = prefill_url.rstrip("/")
        self.d = decode_url.rstrip("/")
        self.timeout = ClientTimeout(total=timeout, sock_read=timeout)
        # Optional override: where D should reach the producer side channel.
        self.kv_host = kv_host
        self.kv_port = kv_port
        self.session: Optional[ClientSession] = None
        self.table = RoomTable()
        self.n = 0
        self.fail = 0
        self.active = 0
        self.p_lat: List[float] = []
        self.d_lat: List[float] = []
        self.last_room: Optional[int] = None

    async def start(self, app):
        # A producer call can outlast D's keep-alive window. Reusing that idle
        # socket races D's close and drops a POST before it reaches the engine.
        # Use a fresh transport; never replay a possibly accepted inference POST.
        self.session = ClientSession(connector=TCPConnector(limit=64, force_close=True), timeout=self.timeout)

    async def stop(self, app):
        if self.session is not None:
            await self.session.close()

    # ------------------------------------------------------------------ health
    async def backend_health(self) -> Dict[str, Any]:
        async def one(base):
            try:
                async with self.session.get(base + "/health", timeout=ClientTimeout(total=5)) as r:
                    return r.status
            except Exception:
                return "unreachable"
        values = await asyncio.gather(one(self.p), one(self.d))
        return dict(zip(("prefill", "decode"), values))

    async def health(self, request):
        status = await self.backend_health()
        ok = all(v == 200 for v in status.values())
        return web.json_response(
            {"ok": ok, **status, "rooms": len(self.table.rooms)},
            status=200 if ok else 503,
        )

    async def live(self, request):
        return web.json_response({"alive": True})

    async def stats(self, request):
        return web.json_response({
            "requests": self.n, "failures": self.fail, "active_requests": self.active,
            "rooms": len(self.table.rooms), "last_room": self.last_room,
            "prefill_s": self.p_lat[-20:], "decode_s": self.d_lat[-20:],
        })

    async def xyblocks(self, request):
        """Producer block mapping for one bootstrap room (D-side receiver lookup)."""
        raw = request.query.get("room")
        try:
            room = int(raw)
        except (TypeError, ValueError):
            return web.json_response({"error": "bad_room", "room": raw}, status=400)
        entry = self.table.get(room)
        if entry is None:
            return web.json_response({"error": "unknown_room", "room": room}, status=404)
        blocks = flatten_blocks(entry)
        canonical_ids = entry.get("_xy_prompt_token_ids")
        if isinstance(canonical_ids, list) and canonical_ids:
            # The producer's real sequence wins over remote_num_tokens / usage,
            # which can count the handoff token or a different expansion.
            prompt_tokens = len(canonical_ids)
        else:
            prompt_tokens = entry.get("remote_num_tokens",
                                      entry.get("_xy_prompt_ids_count",
                                                entry.get("_xy_prompt_tokens")))
        return web.json_response({
            "room": room,
            "block_ids": blocks,
            "block_groups": entry.get("remote_block_ids"),
            "prompt_tokens": prompt_tokens,
            "prompt_token_ids": canonical_ids,
            "prompt_digest": entry.get("_xy_prompt_digest"),
            "prompt_ids_head": entry.get("_xy_prompt_head"),
            "prompt_ids_tail": entry.get("_xy_prompt_tail"),
            "producer_tp_size": entry.get("tp_size", 2),
            "engine_id": entry.get("remote_engine_id"),
            "host": self.kv_host or entry.get("remote_host"),
            "port": self.kv_port or entry.get("remote_port"),
            "request_id": entry.get("remote_request_id"),
            "num_blocks": len(blocks),
            "first_token": entry.get("_xy_first_token"),
        })

    # ------------------------------------------------------------- completions
    async def completions(self, request):
        self.active += 1
        try:
            return await self.relay(request)
        finally:
            self.active -= 1

    async def relay(self, request):
        self.n += 1
        rid = self.n
        path = "/" + request.path.lstrip("/")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            self.fail += 1
            return web.json_response({"error": "invalid JSON body"}, status=400)

        body = normalize_image_token_text(body)

        if path == "/v1/completions":
            prompt = body.get("prompt")
            if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
                count = len(prompt)
            elif isinstance(prompt, str):
                async with self.session.post(self.p + "/tokenize", json={
                    "model": body.get("model"), "prompt": prompt,
                }) as response:
                    if response.status != 200:
                        return web.json_response({"error": "prompt tokenization failed"}, status=400)
                    tokenized = await response.json()
                    count = tokenized.get("count", len(tokenized.get("tokens", [])))
            else:
                return web.json_response({"error": "one prompt or one token-id sequence is required"}, status=400)
            if count < 8:
                return web.json_response({"error": "this producer requires at least 8 completion prompt tokens; use chat for short messages"}, status=400)

        # --- 1. prefill on P (this also primes the producer side channel) -------
        p_body = dict(body)
        p_body["max_tokens"] = 1
        p_body["stream"] = False
        p_body.pop("stream_options", None)
        p_body["kv_transfer_params"] = dict(P_PARAMS)
        # Ask the producer for the sampled token ids: the decode side needs the
        # handoff token and a vLLM producer never RDMA-writes it.
        p_body["return_token_ids"] = True
        # A prefix-cache hit can retain compressed KV but omit earlier SWA
        # pages needed at D's replay boundary. This cross-engine path needs a
        # fresh producer prefix until SWA-aware prefix export is implemented.
        p_body["cache_salt"] = "xyvllm-%d-%d" % (time.time_ns(), rid)
        t0 = time.perf_counter()
        try:
            async with self.session.post(self.p + path, json=p_body) as pr:
                p_raw = await pr.read()
                if pr.status != 200:
                    self.fail += 1
                    log.warning("req %d %s P status %d: %s", rid, path, pr.status, p_raw[:300])
                    return web.Response(body=p_raw, status=502, content_type="application/json")
        except Exception as e:  # noqa: BLE001
            self.fail += 1
            log.warning("req %d %s P error: %s", rid, path, e)
            return web.json_response({"error": "prefill request failed: %s" % e}, status=502)
        t1 = time.perf_counter()
        try:
            p_json = json.loads(p_raw)
        except Exception:  # noqa: BLE001
            p_json = {}
        kv = p_json.get("kv_transfer_params")
        if not kv:
            self.fail += 1
            log.warning("req %d %s: P response without kv_transfer_params: %s", rid, path, p_raw[:300])
            return web.Response(body=p_raw, status=502, content_type="application/json")
        first_token = None
        prompt_ids, prompt_id_error = producer_prompt_token_ids(p_json)
        if prompt_id_error:
            self.fail += 1
            log.warning("req %d %s: %s", rid, path, prompt_id_error)
            return web.json_response({"error": prompt_id_error}, status=400)
        try:
            tokens = p_json["choices"][0].get("token_ids") or []
        except Exception:  # noqa: BLE001
            tokens = []
        if tokens:
            first_token = int(tokens[0])
        else:
            log.warning("req %d %s: no token_ids in P response (return_token_ids unsupported?)",
                        rid, path)
        self.p_lat.append(t1 - t0)

        room = self.table.new_room()
        kv = dict(kv)
        kv["_xy_first_token"] = first_token
        kv["_xy_prompt_tokens"] = p_json.get("usage", {}).get("prompt_tokens")
        if prompt_ids:
            # The producer tells us what it really prefilled; publish its
            # fingerprint so the decode side can prove it replays the same
            # sequence before any KV is pulled. The count is len(ids), never
            # usage.prompt_tokens or remote_num_tokens.
            kv["_xy_prompt_token_ids"] = prompt_ids
            kv["_xy_prompt_ids_count"] = len(prompt_ids)
            kv["_xy_prompt_digest"] = sequence_digest(prompt_ids)
            kv["_xy_prompt_head"] = [int(token) for token in prompt_ids[:8]]
            kv["_xy_prompt_tail"] = [int(token) for token in prompt_ids[-8:]]
        self.table.put(room, kv)
        self.last_room = room

        # --- 2. decode on D (relay) --------------------------------------------
        d_body = dict(body)
        d_body["cache_salt"] = p_body["cache_salt"]
        if prompt_ids:
            # Same list P prefilled. Decode applies it only on the cross-engine
            # path; other channels never see this proxy.
            d_body["prompt_token_ids"] = prompt_ids
        # A caller-supplied logit_bias wins over this reserved-token ban.
        bias = {str(i): -100.0 for i in RESERVED_TOKEN_IDS}
        bias.update(d_body.get("logit_bias") or {})
        d_body["logit_bias"] = bias
        d_body["bootstrap_host"] = self.kv_host or kv.get("remote_host")
        d_body["bootstrap_port"] = int(self.kv_port or kv.get("remote_port") or 5600)
        d_body["bootstrap_room"] = room
        blocks = flatten_blocks(kv)
        log.info("req %d %s room=%d blocks=%d host=%s:%s",
                 rid, path, room, len(blocks), d_body["bootstrap_host"], d_body["bootstrap_port"])
        try:
            async with self.session.post(self.d + path, json=d_body) as dr:
                ctype = dr.headers.get("Content-Type", "application/json")
                if dr.status != 200:
                    raw = await dr.read()
                    self.fail += 1
                    log.warning("req %d %s P=%.3fs D status %d: %s",
                                rid, path, t1 - t0, dr.status, raw[:300])
                    return web.Response(body=raw, status=dr.status,
                                        content_type=ctype.split(";")[0])
                resp = web.StreamResponse(status=200, headers={"Content-Type": ctype})
                await resp.prepare(request)
                async for chunk in dr.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
        except Exception as e:  # noqa: BLE001
            self.fail += 1
            log.warning("req %d %s P=%.3fs D error: %s", rid, path, t1 - t0, e)
            return web.json_response({"error": "decode request failed: %s" % e}, status=502)
        t2 = time.perf_counter()
        self.d_lat.append(t2 - t1)
        log.info("req %d %s room=%d P=%.3fs D=%.3fs", rid, path, room, t1 - t0, t2 - t1)
        return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5701)
    ap.add_argument("--prefill-url", default="http://192.168.177.101:8000")
    ap.add_argument("--decode-url", default="http://192.168.177.47:8200")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--kv-host", default=None,
                    help="override the producer side-channel host handed to D")
    ap.add_argument("--kv-port", type=int, default=None,
                    help="override the producer side-channel port handed to D")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    px = Proxy(a.prefill_url, a.decode_url, a.timeout, a.kv_host, a.kv_port)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.on_startup.append(px.start)
    app.on_cleanup.append(px.stop)
    app.router.add_get("/health", px.health)
    app.router.add_get("/live", px.live)
    app.router.add_get("/stats", px.stats)
    app.router.add_get("/xyblocks", px.xyblocks)
    app.router.add_post("/v1/completions", px.completions)
    app.router.add_post("/v1/chat/completions", px.completions)
    log.info("xy_pd_proxy on %s:%d  P=%s  D=%s", a.host, a.port, a.prefill_url, a.decode_url)
    web.run_app(app, host=a.host, port=a.port, access_log=None)


if __name__ == "__main__":
    main()
