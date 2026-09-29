"""XY cross-engine PD backend: SGLang decode pulling KV straight out of a vLLM
``NixlPullConnector`` producer.

Direction is the mirror image of SGLang's own NIXL backend: SGLang prefill pushes,
vLLM producer only publishes *where* its KV lives, and the decode side issues the
NIXL READs (``transfer_mode=pull``).  So this backend keeps SGLang's manager (it
already registers this rank's KV pool with a local NIXL agent) and swaps the
receiver for a pull implementation:

1. ``init()``            - talk to the producer's ZMQ side channel (port 5600+tp_rank)
                           and fetch ``NixlHandshakePayload``/``NixlAgentMetadata``;
                           there is no SGLang bootstrap server on the vLLM side.
2. ``send_metadata()``   - instead of telling a prefill to push, map this request's
                           local page indices to producer regions and issue READs.
3. ``poll()``            - report KVPoll.Success once the NIXL transfers land.

Remote block geometry (which producer blocks hold this request's prompt) comes
from the XY router/proxy: ``XY_VLLM_REMOTE_BLOCKS`` (static JSON, debug only) or
``XY_VLLM_BLOCKS_URL`` (template containing ``{room}``; the normal path - the
router answers ``GET /xyblocks?room=N``).  With neither set the receiver only
caches the handshake, so a decode worker can still start and dump its layout.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
import urllib.request
from typing import Any, Dict, List, Optional

import msgpack
import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, PrefillServerInfo
from sglang.srt.disaggregation.nixl.conn import NixlKVManager

logger = logging.getLogger(__name__)

# NixlAgentMetadata field order (msgspec encodes the dataclass as a positional array).
VLLM_META_FIELDS = [
    "engine_id",
    "agent_metadata",
    "kv_caches_base_addr",
    "device_id",
    "num_blocks",
    "block_lens",
    "block_strides",
    "kv_cache_layout",
    "block_size",
    "ssm_sizes",
    "attn_backend_name",
    "physical_blocks_per_logical_kv_block",
    "dcp_size",
    "pcp_size",
    "xy_layer_map",
    "xy_layer_mode",
]

# vLLM side-channel message id for "give me your agent metadata".
GET_META_MSG = b"get_meta_msg"

# Decode-side kv group index -> producer layer suffix.
#
# SGLang's pools are grouped by compression ratio (``get_contiguous_buf_infos``
# walks ``for ratio in (4, 128, 1, 2)``), and the measured group order in this
# deployment is item_lens=[149760, 17408, 74880, 74880, 74880, 8704, 8704, 8704]
# with page_size=256, i.e.
#   0      : ratio1 compressed KV        (layers 20-39 share layer 20's)
#   1      : ratio1 indexer K             (layer 20)
#   2,3,4  : ratio2 compressed KV         (kv-sharing groups 2-7 / 8-13 / 14-19)
#   5,6,7  : ratio2 indexer K             (same three groups)
# which lines up exactly with the producer's four "compressed MLA" regions
# (layers.{2,8,14,20}.attn) and four indexer regions.
XY_GROUP_LAYER_SUFFIX = [
    "layers.20.attn",
    "layers.20.attn.indexer.k_cache",
    "layers.2.attn",
    "layers.8.attn",
    "layers.14.attn",
    "layers.2.attn.indexer.k_cache",
    "layers.8.attn.indexer.k_cache",
    "layers.14.attn.indexer.k_cache",
]


def _as_str_dict(raw: Any) -> Dict[str, Any]:
    """msgspec may decode structs as dicts with bytes keys or as arrays."""
    if isinstance(raw, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in raw.items()
        }
    return dict(zip(VLLM_META_FIELDS, raw))


class XYVLLMMeta:
    """Producer handshake, fetched once per side-channel endpoint."""

    _cache: Dict[str, "XYVLLMMeta"] = {}

    def __init__(self, endpoint: str, tp_rank=0):
        self.endpoint = endpoint
        self.tp_rank = int(tp_rank)
        self.meta: Dict[str, Any] = {}
        self.local_agent_name: Optional[str] = None
        self.compat_hash: Optional[str] = None
        self.remote_agent_name: Optional[str] = None
        self._registrations = {}
        self.fetched_at: float = 0.0

    @classmethod
    def fetch(cls, endpoint: str, engine_id=None, tp_rank=0) -> "XYVLLMMeta":
        key = (endpoint, int(tp_rank))
        cached = cls._cache.get(key)
        if cached is None or (engine_id and cached.meta.get("engine_id") != engine_id):
            cached = cls(endpoint, tp_rank)
            cached._handshake()
            cls._cache[key] = cached
        return cached

    def _handshake(self) -> None:
        host, _, port = self.endpoint.rpartition(":")
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        try:
            sock.setsockopt(zmq.RCVTIMEO, 15000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect("tcp://%s:%s" % (host, port or 5600))
            sock.send(msgpack.packb([GET_META_MSG, 0, self.tp_rank]))
            parts = sock.recv_multipart()
            assert len(parts) == 2, "unexpected handshake reply: %d parts" % len(parts)
            head = _as_str_dict(msgpack.unpackb(parts[0], strict_map_key=False))
            self.compat_hash = head.get("compatibility_hash")
            if isinstance(self.compat_hash, bytes):
                self.compat_hash = self.compat_hash.decode()
            self.meta = _as_str_dict(
                msgpack.unpackb(head["agent_metadata_bytes"], strict_map_key=False)
            )
            self.fetched_at = time.time()
        finally:
            sock.close()
            ctx.term()

    def register_with(self, agent) -> str:
        """Load the producer's NIXL agent into the local agent (READ peer)."""
        key = id(agent)
        if key not in self._registrations:
            self.remote_agent_name = agent.add_remote_agent(self.meta["agent_metadata"])
            self._registrations[key] = (agent, self.remote_agent_name)
            logger.info(
                "XY: vLLM producer agent registered: %s (engine_id=%s, regions=%d)",
                self.remote_agent_name,
                self.meta.get("engine_id"),
                len(self.meta.get("kv_caches_base_addr") or []),
            )
        return self._registrations[key][1]

    def regions_for_layer(self, layer_name: str) -> Optional[int]:
        """Producer KV region index holding ``layer_name`` (layer-region mode)."""
        layer_map = self.meta.get("xy_layer_map") or {}
        entry = None
        for key in (layer_name, layer_name.encode()):
            if key in layer_map:
                entry = layer_map[key]
                break
        if entry is None:
            return None
        if isinstance(entry, (list, tuple)):
            return int(entry[-1])
        return int(entry)

    def region_for_suffix(self, suffix: str) -> Optional[int]:
        """Producer region whose layer name *ends with* ``suffix``.

        Producer keys are fully qualified (``language_model.model.layers.2.attn``),
        while the decode side names its pools by much shorter aliases.
        """
        layer_map = self.meta.get("xy_layer_map") or {}
        best: Optional[int] = None
        for key, entry in layer_map.items():
            name = key.decode() if isinstance(key, bytes) else key
            if not name.endswith(suffix):
                continue
            idx = int(entry[-1]) if isinstance(entry, (list, tuple)) else int(entry)
            # ``layers.2.attn`` must not swallow ``layers.2.attn.indexer.k_cache``
            # (whose suffix it also ends with) - keep the shortest match.
            if best is None or len(name) < best[0]:
                best = (len(name), idx)
        return best[1] if best else None


def _dump_local_layout(mgr: NixlKVManager) -> None:
    """One-shot debug dump of this rank's KV pool so P/D layouts can be aligned."""
    if os.environ.get("XY_DUMP_LAYOUT", "1") != "1":
        return
    kv_args = mgr.kv_args
    item_lens = list(kv_args.kv_item_lens or [])
    layer_ids = list(kv_args.kv_layer_ids or [])
    logger.info(
        "XY-LAYOUT decode rank=%s pp=%s groups=%d page_size=%s dtype=%s "
        "state_types=%s",
        kv_args.engine_rank,
        getattr(kv_args, "pp_rank", None),
        len(item_lens),
        getattr(kv_args, "page_size", None),
        getattr(kv_args, "kv_cache_dtype_str", None),
        [getattr(st, "value", st) for st in (kv_args.state_types or [])],
    )
    logger.info(
        "XY-LAYOUT kv groups: layer_ids=%s item_lens=%s data_lens=%s",
        layer_ids,
        item_lens,
        [int(x) for x in (kv_args.kv_data_lens or [])],
    )
    logger.info(
        "XY-LAYOUT ratios=%s prefill_layers=%s..%s kv_heads=%s",
        list(getattr(kv_args, "mla_compression_ratios", None) or []),
        getattr(kv_args, "prefill_start_layer", None),
        getattr(kv_args, "prefill_end_layer", None),
        getattr(kv_args, "total_kv_head_num", None),
    )
    state_types = list(kv_args.state_types or [])
    for idx, st in enumerate(state_types):
        ptrs = (kv_args.state_data_ptrs or [[]])[idx]
        lens = (kv_args.state_item_lens or [[]])[idx]
        layers = (kv_args.state_layer_ids or [[]])[idx]
        dims = (getattr(kv_args, "state_dim_per_tensor", None) or [[]])[idx]
        logger.info(
            "XY-LAYOUT state[%d] %s: layers=%s item_lens=%s dim_per_tensor=%s ptrs=%d",
            idx,
            getattr(st, "value", st),
            list(layers) if layers is not None else None,
            list(lens) if lens is not None else None,
            list(dims) if dims is not None else None,
            len(ptrs) if ptrs is not None else 0,
        )


class XYVLLMKVManager(NixlKVManager):
    """SGLang's NIXL manager; the peer happens to be a vLLM producer."""

    def __init__(
        self,
        args,
        disaggregation_mode,
        server_args,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        _dump_local_layout(self)

    def _start_heartbeat_checker_thread(self):
        # The inherited checker sends HTTP /health to bootstrap_addr, which
        # is vLLM's ZMQ port here. It falsely removes healthy producers and
        # can abort an unrelated transfer. The front door checks the real
        # HTTP health endpoint; every request checks engine identity and NIXL
        # completion. There is no SGLang bootstrap server to heartbeat.
        logger.info("XY: vLLM liveness uses HTTP front-door health and NIXL transfer status")

    def try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
        """Publish a synthetic single-DP prefill entry for the producer.

        A vLLM producer has no SGLang bootstrap server, and the inherited
        implementation would issue ``GET /route`` over HTTP to the producer's
        ZMQ side-channel port.  That HTTP text arrives in the ROUTER socket of
        the producer's handshake listener, where ``msgspec.msgpack.decode``
        raises ``trailing characters`` and kills the listener thread (observed
        2026-09-19 13:57, which is why the first end-to-end request failed with
        "Resource temporarily unavailable").

        ``dp_size=1`` makes ``_resolve_prefill_dp_rank`` short-circuit to rank 0,
        so neither this method nor ``query_prefill_dp_ranks`` ever performs a
        network lookup and ``init()`` runs against the producer address.
        """
        if bootstrap_addr in self.prefill_info_table:
            return True
        self.prefill_info_table[bootstrap_addr] = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=None,
            kv_cache_dtype=getattr(self.kv_args, "kv_cache_dtype_str", None),
            follow_bootstrap_room=False,
        )
        logger.info(
            "XY: synthetic prefill info for %s (dp_size=1, no /route query)",
            bootstrap_addr,
        )
        return True


class XYVLLMKVReceiver(CommonKVReceiver):
    """Pull KV out of the vLLM producer instead of waiting for a push."""

    def __init__(
        self,
        mgr: XYVLLMKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.xy_started = False
        self.xy_meta: Optional[XYVLLMMeta] = None
        self.xy_handles: List[Any] = []
        self.xy_bytes = 0
        self.xy_plan = None
        self.xy_decode_prefix_len = 0
        self.xy_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self.xy_state_indices: Optional[List] = None
        # Fetched lazily in send_metadata(): the lookup key is bootstrap_room,
        # which only exists once super().__init__ has run.
        self.xy_remote: Optional[Dict[str, Any]] = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    @staticmethod
    def _xy_parse_remote() -> Optional[Dict[str, Any]]:
        raw = os.environ.get("XY_VLLM_REMOTE_BLOCKS")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception as e:  # pragma: no cover - malformed operator input
            logger.error("XY: bad XY_VLLM_REMOTE_BLOCKS: %s", e)
            return None

    def _xy_remember_first_token(self, payload: Dict[str, Any]) -> None:
        """Park the producer's handoff token for the decode-side gate.

        A vLLM producer never RDMA-writes the decode side's ``MetadataBuffers``,
        so the token it sampled has to reach ``_xy_fill_metadata_buffers``
        through this registry once the front door forwards it.
        """
        first_token = payload.get("first_token")
        if first_token is not None:
            XY_FIRST_TOKENS[int(self.bootstrap_room)] = int(first_token)

    def _xy_fetch_remote(self) -> Optional[Dict[str, Any]]:
        """Producer block mapping for this room: static env first, then router."""
        remote = self._xy_parse_remote()
        if remote is not None:
            self._xy_remember_first_token(remote)
            return remote
        template = os.environ.get("XY_VLLM_BLOCKS_URL")
        if not template:
            return None
        url = template.replace("{room}", str(self.bootstrap_room))
        timeout = float(os.environ.get("XY_VLLM_BLOCKS_TIMEOUT", "60"))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            logger.error("XY: block lookup %s failed: %s", url, e)
            return None
        if not payload or not payload.get("block_ids"):
            logger.warning("XY: room=%s lookup returned no blocks: %s",
                           self.bootstrap_room, payload)
            return None
        logger.info("XY: room=%s remote mapping: %d blocks from %s:%s",
                    self.bootstrap_room, len(payload["block_ids"]),
                    payload.get("host"), payload.get("port"))
        self._xy_remember_first_token(payload)
        return payload

    # ------------------------------------------------------------------ setup
    def init(self, prefill_dp_rank: int):
        """No SGLang bootstrap server exists on a vLLM producer: handshake directly."""
        endpoint = self.bootstrap_addr
        try:
            self.xy_meta = XYVLLMMeta.fetch(endpoint)
        except Exception as e:
            logger.error("XY: handshake with vLLM producer %s failed: %s", endpoint, e)
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        self.prefill_dp_rank = prefill_dp_rank
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)
        logger.info(
            "XY: handshake OK with %s hash=%s layers=%d",
            endpoint,
            self.xy_meta.compat_hash,
            len(self.xy_meta.meta.get("xy_layer_map") or {}),
        )

    def _register_remote(self) -> str:
        assert self.xy_meta is not None
        return self.xy_meta.register_with(self.kv_mgr.agent)

    # -------------------------------------------------------------- transfer
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.xy_meta is None:
            # The vLLM side has no bootstrap server, so SGLang cannot resolve a
            # prefill dp rank and may never call init(); handshake lazily here.
            self.init(0)
        if self.xy_meta is None:
            return
        self.xy_kv_indices = np.asarray(kv_indices, dtype=np.int64)
        self.xy_state_indices = state_indices
        self.xy_decode_prefix_len = int(decode_prefix_len or 0)
        if self.xy_remote is None:
            self.xy_remote = self._xy_fetch_remote()
        if self.xy_remote is None:
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            # No producer block mapping yet (bootstrap-only run): stay in
            # WaitingForInput so the debug layout dump can be collected.
            logger.warning(
                "XY: room=%s no remote block mapping (XY_VLLM_REMOTE_BLOCKS / "
                "XY_VLLM_BLOCKS_URL unset or empty); skipping transfer",
                self.bootstrap_room,
            )
            self.xy_started = True
            self.init_time = time.time()
            return
        try:
            self._issue_reads()
        except Exception as e:
            logger.error("XY: read from producer failed: %s", e, exc_info=True)
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        self.xy_started = True
        self.init_time = time.time()

    def _issue_reads(self) -> None:
        from .transfer import TransferPlan
        assert self.xy_remote is not None
        self.xy_meta = XYVLLMMeta.fetch(self.bootstrap_addr, self.xy_remote.get("engine_id"))
        if self.xy_meta.meta.get("engine_id") != self.xy_remote.get("engine_id"):
            raise ValueError("producer restarted between prefill and transfer")
        peer = self._register_remote()
        self.xy_plan = TransferPlan(self)
        try:
            self.xy_plan.build()
        except Exception:
            self.xy_plan.close()
            raise
        spans = self.xy_plan.spans
        agent = self.kv_mgr.agent
        local_device = int(self.kv_mgr.kv_args.gpu_id)
        remote_device = int(self.xy_meta.meta["device_id"])
        local = agent.get_xfer_descs([(s.local, s.size, local_device) for s in spans], "VRAM")
        remote = agent.get_xfer_descs([(s.remote, s.size, remote_device) for s in spans], "VRAM")
        handle = agent.initialize_xfer("READ", local, remote, peer, b"")
        if handle is None:
            self.xy_plan.close()
            raise RuntimeError("NIXL did not create a transfer handle")
        self.xy_handles = [handle]
        agent.transfer(handle)
        self.xy_bytes = sum(s.size for s in spans)
        logger.info("XY: room=%s issued %d bounded spans (%d bytes), prefix=%d tail=%d",
                    self.bootstrap_room, len(spans), self.xy_bytes,
                    self.xy_plan.prefix_n, self.xy_plan.full_n - self.xy_plan.prefix_n)

    def _notify_producer(self):
        request_id = self.xy_remote.get("request_id")
        if not request_id:
            raise ValueError("producer request id is required for block release")
        host, _, port = self.bootstrap_addr.rpartition(":")
        consumers = int(os.environ.get("XY_VLLM_CONSUMERS", "4"))
        message = (str(request_id) + ":" + str(consumers)).encode()
        # Every producer TP worker holds its replica until all D ranks finish.
        # Follow vLLM's req_id:consumer_count protocol, never arbitrary payloads.
        for rank in range(int(self.xy_remote.get("producer_tp_size", 2))):
            meta = XYVLLMMeta.fetch(self.bootstrap_addr,
                                   self.xy_remote.get("engine_id"), tp_rank=rank)
            peer = meta.register_with(self.kv_mgr.agent)
            self.kv_mgr.agent.send_notif(peer, message)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        if not self.xy_started:
            return KVPoll.WaitingForInput
        if not self.xy_handles:
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return self.conclude_state
        for handle in list(self.xy_handles):
            state = self.kv_mgr.agent.check_xfer_state(handle)
            if state == "ERR":
                self.xy_handles.remove(handle)
                self.kv_mgr.record_failure(
                    self.bootstrap_room, "NIXL READ from vLLM producer failed"
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return self.conclude_state
            if state == "DONE":
                self.xy_handles.remove(handle)
                self.kv_mgr.agent.release_xfer_handle(handle)
        if not self.xy_handles:
            try:
                self.xy_plan.finish()
                self._notify_producer()
            except Exception:
                logger.exception("XY: post-transfer index conversion failed")
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return self.conclude_state
            from .layout import PROMPT_DIGESTS, REQUEST_LAYOUTS
            REQUEST_LAYOUTS.pop(int(self.bootstrap_room), None)
            PROMPT_DIGESTS.pop(int(self.bootstrap_room), None)
            self.conclude_state = KVPoll.Success
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
            logger.info(
                "XY: room=%s transfer complete (%d bytes)", self.bootstrap_room, self.xy_bytes
            )
            return self.conclude_state
        return KVPoll.Transferring

    def failure_exception(self):
        raise RuntimeError(
            "XY vLLM pull from %s for room %s failed" % (self.bootstrap_addr, self.bootstrap_room)
        )


# --------------------------------------------------------------------------
# Handoff-token injection
# --------------------------------------------------------------------------
# SGLang's decode path expects the PREFILL engine to RDMA the handoff token and
# its bootstrap_room into ``MetadataBuffers``; ``_apply_metadata_gate`` then
# downgrades Success -> Transferring for any request whose slot still reads
# room 0, and ``_commit_transfer_to_req`` aborts on a room mismatch. A vLLM
# producer never writes those buffers, so every request stalled forever in the
# decode transfer queue (observed 2026-09-19 15:21, "num_waiting_reqs=1" while
# all 8 NIXL READs had completed). Fill them in-process from the registry that
# ``_xy_fetch_remote`` populates.
XY_FIRST_TOKENS: Dict[int, int] = {}


def _xy_fill_metadata_buffers(decode_reqs, metadata_buffers) -> int:
    """Return how many slots were filled from the pull-side handoff registry."""
    filled = 0
    for decode_req in decode_reqs:
        idx = getattr(decode_req, "metadata_buffer_index", -1)
        if idx is None or idx < 0:
            continue
        room = getattr(decode_req.req, "bootstrap_room", None)
        if room is None:
            continue
        room = int(room)
        if int(metadata_buffers.bootstrap_room[idx, 0].item()) == room:
            continue
        token = XY_FIRST_TOKENS.get(room)
        if token is None:
            continue
        metadata_buffers.output_ids[idx, 0] = int(token)
        metadata_buffers.bootstrap_room[idx, 0] = room
        filled += 1
    return filled


def install_metadata_fill() -> bool:
    """Idempotently wrap ``_apply_metadata_gate`` so pulled rooms pass the gate."""
    try:
        from sglang.srt.disaggregation import utils as disagg_utils
    except Exception:  # pragma: no cover - non-SGLang import context
        return False
    if getattr(disagg_utils, "_xy_metadata_fill_installed", False):
        return True
    original = disagg_utils._apply_metadata_gate

    def _gate(polls, decode_reqs, metadata_buffers) -> None:
        try:
            filled = _xy_fill_metadata_buffers(decode_reqs, metadata_buffers)
            if filled:
                logger.info("XY: filled %d handoff metadata buffer(s)", filled)
        except Exception as e:  # noqa: BLE001 - never break the scheduler loop
            logger.warning("XY: metadata fill failed: %s", e)
        original(polls, decode_reqs, metadata_buffers)

    disagg_utils._apply_metadata_gate = _gate
    disagg_utils._xy_metadata_fill_installed = True
    return True


install_metadata_fill()
