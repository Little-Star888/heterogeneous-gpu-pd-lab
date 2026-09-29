"""DSV4.1 transfer plan, FP8 index staging, and destination FP4 conversion."""
import logging

logger = logging.getLogger(__name__)

from .layout import (
    PROMPT_DIGESTS,
    REQUEST_LAYOUTS,
    region_entry,
    segregated_spans,
)


def borrowed_bytes(ptr, length, device):
    import torch
    storage = torch._C._construct_storage_from_data_pointer(int(ptr), device, int(length))
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (int(length),), (1,))


class TransferPlan:
    def __init__(self, receiver):
        import torch
        self.receiver = receiver
        self.args = receiver.kv_mgr.kv_args
        self.meta = receiver.xy_meta.meta
        self.payload = receiver.xy_remote
        self.device = torch.device("cuda", int(self.args.gpu_id))
        self.spans = []
        self.index_jobs = []
        self.buffers = []
        self.registrations = []
        self.full_n, self.prefix_n = REQUEST_LAYOUTS[int(receiver.bootstrap_room)]
        if int(self.payload.get("prompt_tokens", -1)) != self.full_n:
            raise ValueError("P and D token counts differ; refusing incompatible prompt KV")
        self._check_sequence()
        if self.prefix_n % 2 or self.prefix_n >= self.full_n:
            raise ValueError("tail replay requires an even, non-final prefix")
        self.groups = self.payload.get("block_groups")
        if not isinstance(self.groups, list) or not self.groups:
            raise ValueError("producer must publish every cache group without flattening")
        self.start = int(receiver.xy_decode_prefix_len or 0)
        if self.start % int(self.args.page_size):
            raise ValueError("destination cached prefix is not page aligned")
        if self.start >= self.prefix_n:
            raise ValueError("cross-engine prefix is already cached; local resume required")

    def _check_sequence(self) -> None:
        """Refuse a producer KV that was built from a different prompt.

        The producer reports prompt_token_ids; the decode side recorded the ids
        it is about to replay from in PROMPT_DIGESTS. Matching lengths is not
        enough: a same-length mismatch installed the producer's KV in front of
        another sequence and every later token was generated from a corrupted
        context (observed as reserved-token noise in the answer). Fail loudly
        instead; a producer/proxy that publishes no fingerprint keeps the old
        length-only gate.
        """
        room = int(self.receiver.bootstrap_room)
        local = PROMPT_DIGESTS.get(room)
        remote = self.payload.get("prompt_digest")
        if local is None or not remote:
            return
        if int(local[0]) != self.full_n:
            raise ValueError("P and D token counts differ; refusing incompatible prompt KV")
        if str(remote) == str(local[1]):
            logger.info(
                "XY: room=%s prompt sequence verified (%s) against the producer",
                room, remote,
            )
            return
        from .layout import prompt_digest_enforced

        detail = ("P=%s D=%s; P head=%s tail=%s D head=%s tail=%s"
                  % (remote, local[1], self.payload.get("prompt_ids_head"),
                     self.payload.get("prompt_ids_tail"), local[2], local[3]))
        if not prompt_digest_enforced():
            logger.warning(
                "XY: room=%s prompt token sequences differ (warn mode, flag file "
                "absent): %s", room, detail,
            )
            return
        raise ValueError(
            "P and D prompt token sequences differ (%s); refusing "
            "incompatible prompt KV" % detail
        )

    def add(self, suffix, **kwargs):
        group, region = region_entry(self.meta, suffix)
        if group >= len(self.groups):
            raise ValueError(f"{suffix}: cache group {group} is absent")
        if suffix.endswith(".swa_cache"):
            # vLLM clips sliding-window block lists to the retained suffix.
            # Their first entry is not logical page zero for long prompts.
            kwargs["remote_page_offset"] = max(0, (self.full_n + 63) // 64 - len(self.groups[group]))
        self.spans.extend(segregated_spans(
            remote_base=int(self.meta["kv_caches_base_addr"][region]),
            remote_stride=int(self.meta["block_strides"][region]),
            remote_len=int(self.meta["block_lens"][region]),
            remote_blocks=self.groups[group],
            remote_num_blocks=int(self.meta["num_blocks"]), label=suffix, **kwargs))

    def build(self):
        import torch
        args = self.args
        page_size = int(args.page_size)
        pages = self.receiver.xy_kv_indices.tolist()
        if len(pages) != (self.prefix_n - self.start + page_size - 1) // page_size:
            raise ValueError("decode page count differs from the truncated prompt length")
        if len(args.kv_data_ptrs) != 8:
            raise ValueError("expected DSV4.1 four KV source pools and four index pools")
        for layer, ratio, kv_group, idx_group in ((20,1,0,1), (2,2,2,5), (8,2,3,6), (14,2,4,7)):
            self.add(f"layers.{layer}.attn",
                local_base=int(args.kv_data_ptrs[kv_group]),
                local_len=int(args.kv_data_lens[kv_group]),
                local_stride=int(args.kv_item_lens[kv_group]), local_pages=pages,
                local_states=page_size // ratio, start_state=self.start // ratio,
                count=(self.prefix_n - self.start) // ratio, value_bytes=576, scale_bytes=8)
            count = (self.prefix_n - self.start) // ratio
            staging = torch.empty(count * 132, dtype=torch.uint8, device=self.device)
            self.buffers.append(staging)
            reg = self.receiver.kv_mgr.agent.register_memory(
                [(staging.data_ptr(), staging.numel(), int(args.gpu_id), "")], "VRAM")
            if reg is None:
                raise RuntimeError("index staging registration failed")
            self.registrations.append(reg)
            self.add(f"layers.{layer}.attn.indexer.k_cache",
                local_base=staging.data_ptr(), local_len=staging.numel(),
                local_stride=count * 132, local_pages=[0], local_states=count,
                start_state=self.start // ratio, count=count, value_bytes=128, scale_bytes=4)
            slots_per_page = page_size // ratio
            slots = torch.arange(count, device=self.device, dtype=torch.int64)
            page_tensor = torch.tensor(pages, device=self.device, dtype=torch.int64)
            locations = page_tensor[slots // slots_per_page] * slots_per_page + slots % slots_per_page
            if int(args.kv_item_lens[idx_group]) != slots_per_page * 68:
                raise ValueError("destination indexer is not the verified MXFP4 layout")
            target = borrowed_bytes(args.kv_data_ptrs[idx_group], args.kv_data_lens[idx_group], self.device)
            self.index_jobs.append((staging, target.view(-1, 64 * 68), locations, count))
        # SWA transfer uses its own allocator's page ids, not full-KV page ids.
        # Missing layers 21..39 are reconstructed by the existing bounded replay.
        for component, kind in enumerate(args.state_types):
            kind = getattr(kind, "value", kind)
            state_ids = self.receiver.xy_state_indices[component]
            if state_ids is None:
                raise ValueError(f"missing destination {kind} state indices")
            state_ids = [int(x) for x in state_ids]
            if kind == "swa":
                start = max(0, self.prefix_n - 128) // page_size * page_size
                for layer, ptr in enumerate(args.state_data_ptrs[component]):
                    length = int(args.state_data_lens[component][layer])
                    stride = int(args.state_item_lens[component][layer])
                    target = borrowed_bytes(ptr, length, self.device)
                    for page in state_ids:
                        if page < 0 or (page + 1) * stride > length:
                            raise ValueError("SWA page exceeds destination allocation")
                        target[page * stride:(page + 1) * stride].zero_()
                    if layer <= 20:
                        self.add(f"layers.{layer}.attn.swa_cache", local_base=int(ptr),
                            local_len=length, local_stride=stride, local_pages=state_ids,
                            local_states=page_size, start_state=start,
                            count=self.prefix_n - start, value_bytes=576, scale_bytes=8)
            elif kind == "c128_state":
                # This model's three components are ratio-2 pending-pair rings.
                # At an even prefix there is no pending token. Replaying from
                # that boundary overwrites the pair; copying the P-end state
                # would incorrectly import a different sequence position.
                if len(args.state_data_ptrs[component]) != 3:
                    raise ValueError("unexpected compressor state geometry")
                for idx, ptr in enumerate(args.state_data_ptrs[component]):
                    stride = int(args.state_item_lens[component][idx])
                    target = borrowed_bytes(ptr, args.state_data_lens[component][idx], self.device)
                    for slot in state_ids:
                        if slot < 0 or (slot + 1) * stride > target.numel():
                            raise ValueError("compressor slot exceeds allocation")
                        target[slot * stride:(slot + 1) * stride].zero_()
            else:
                raise ValueError(f"unimplemented destination state {kind}")
        torch.cuda.synchronize(self.device)
        if not self.spans:
            raise ValueError("empty transfer plan")
        return self

    def finish(self):
        import torch
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache
        for staging, target, locations, count in self.index_jobs:
            values = staging[:count * 128].view(torch.float8_e4m3fn).view(count, 128).float()
            scales = staging[count * 128:].view(torch.float32).view(count, 1)
            decoded = (values * scales).contiguous()
            store_fp4_index_k_cache(input=decoded, cache=target, loc=locations, page_size=64, rne=True)
        torch.cuda.synchronize(self.device)
        self.close()

    def close(self):
        for reg in self.registrations:
            self.receiver.kv_mgr.agent.deregister_memory(reg)
        self.registrations.clear()
        self.index_jobs.clear()
        self.buffers.clear()
