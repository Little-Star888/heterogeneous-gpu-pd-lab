# DeepSeek-V4.1-Flash V8 Docker bundle

This directory publishes the **DS4.1 Flash V8 reproducibility layer**. The
image contains the exact cross-engine KV/NIXL patches, the final DSpark compute
delta, runtime environment, package order, and SHA256 evidence used by the
2026-09-20 V8 baseline.

The image is intentionally a small patch/configuration image. It does **not**
contain the 510 GB official weights, private P21/D40 role images, Engram
dictionaries, or a network fabric. Those assets stay on the target GB10 hosts
and are mounted or selected by the role launchers. This keeps the image
redistributable while preserving the patches that make the two stages work.

## Pull and verify

```bash
docker pull ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest show-config
docker run --rm ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest verify
```

The same files are available in `bundle/` for an air-gapped build. Build the
image from this directory with `docker build -t ds41-flash-v8:local .`.
The source-tree self-check is `python ds41-flash-v8/verify_bundle.py`.

## Hardware and topology

The measured arrangement is **PP2 with PD separation**:

```text
P21: 2 x RTX 6000D (TP2, vLLM)  -- vLLM -> SGLang NIXL KV handoff -->
D40: 4 x DGX Spark/GB10 (TP4/EP4, SGLang, one distributed group)
```

The two 6000D cards jointly run the P21 Prefill stage. An RTX 5500 Pro or RTX
6000 Pro can be substituted only if its memory, FP8/NVFP4 kernels, CUDA/NCCL,
PCIe topology, and host power budget are validated; the substitution does not
turn the system into a single-machine six-GPU PP runtime. The D stage remains
four Spark nodes and must be launched with ranks 0–3 on their real hosts.

The fixed V8 settings are DSPARK block 5, chunked Prefill 2048, FP8 KV,
`XY_PD_TAIL=256`, `flashinfer_cutlass`, shared-expert fusion disabled, and
SGLang decode disaggregation over NIXL. The official
`deepseek-ai/DeepSeek-V4.1-Flash` FP8 model, the matching P21 vLLM and D40
SGLang role images, Engram data, and host networking are prerequisites.

## Apply the patches to a role image

On a D40 host with the matching role image overlay mounted:

```bash
docker run --rm -v /var/tmp/dsv41-d40:/var/tmp/dsv41-d40 \
  ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/ds41-flash-v8:latest \
  shell
# Or copy ds41-flash-v8/apply-bundle.sh and bundle/ to the host:
./apply-bundle.sh --overlay-dir /var/tmp/dsv41-d40/overlay
```

The script extracts `xyvllm-overlay-20260920.tgz` first and
`xyvllm-dspark-compute-delta-20260920.tgz` second, then merges the bundled
SGLang manifest entries. It never copies model weights or changes Docker
images.
Source `runtime.env` on all D ranks and restart them. Run the P-side
`patch/xy_pd_proxy.py` in the vLLM P21 image with the P and D URLs reachable
over the host network; use the existing validated launcher for the role image.

`compose.yml` is a parameterized template. It deliberately does not create
four fake Spark replicas on one host. Start one D rank per Spark node, set
`D_NODE_RANK=0..3` and `D_MASTER_ADDR`, and provide the tested role launch
commands through `P_COMMAND` and `D_COMMAND`.

## Evidence and limits

`bundle/baseline.json`, `bundle/SHA256SUMS`, and `bundle/evidence/` are the
frozen evidence. The image packages the patch layer; it is not a claim that a
fresh machine can cold-restore the private role images without their original
base images, model, Engram data, and NIXL-capable network. Run the bundled
baseline verifier in the source tree before publishing a new tag.
