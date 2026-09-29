"""XY cross-engine PD backend: SGLang decode side pulling KV from a vLLM
NIXL producer (NixlPullConnector).

Drop-in for ``--disaggregation-transfer-backend xyvllm``.  The manager reuses
SGLang's own ``NixlKVManager`` (local GPU pool registration, NIXL agent) and the
receiver replaces SGLang's "wait for the prefill to push" flow with a pull:
it fetches the vLLM producer's ``NixlAgentMetadata`` over the producer's ZMQ
side channel, then issues NIXL READs from the vLLM KV regions into this rank's
own KV pool.
"""

from sglang.srt.disaggregation.xyvllm.conn import (
    XYVLLMKVManager,
    XYVLLMKVReceiver,
)

__all__ = ["XYVLLMKVManager", "XYVLLMKVReceiver"]
