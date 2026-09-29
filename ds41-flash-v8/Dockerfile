FROM debian:bookworm-slim

ARG V8_DATE=20260920
LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash V8 PD bundle" \
      org.opencontainers.image.description="Reproducibility bundle for the V8 vLLM TP2 to SGLang TP4/EP4 PD deployment" \
      org.opencontainers.image.version="V8-${V8_DATE}" \
      org.opencontainers.image.source="https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab"

ENV DS41_V8_BUNDLE=/opt/ds41-flash-v8/bundle

COPY bundle/ ${DS41_V8_BUNDLE}/
COPY entrypoint.sh /usr/local/bin/ds41-v8
COPY apply-bundle.sh /opt/ds41-flash-v8/apply-bundle.sh

RUN chmod 0755 /usr/local/bin/ds41-v8 /opt/ds41-flash-v8/apply-bundle.sh \
    && printf '%s\n' 'V8 bundle image installed' > /opt/ds41-flash-v8/IMAGE-MARKER

ENTRYPOINT ["/usr/local/bin/ds41-v8"]
CMD ["verify"]
