# syntax=docker/dockerfile:1.6
# FlashQLA HKV-output integration plus Qwen3.6 DFlash on DGX Spark / GB10.
FROM ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest

ENV VLLM_NVFP4_GEMM_BACKEND=cutlass     TORCH_CUDA_ARCH_LIST=12.1a     FLASHINFER_CUDA_ARCH_LIST=12.1a     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     MAX_JOBS=2     NVCC_THREADS=1     FLASHINFER_MAX_JOBS=2     FLASHINFER_NVCC_THREADS=1

WORKDIR /repro
COPY scripts /repro/scripts
COPY patches /repro/patches
RUN chmod +x /repro/scripts/*.sh

# DFlash aux layer tap compatibility fix used by the proven Qwen3.6 recipe.
RUN bash /repro/patches/apply_dflash_off_by_one.sh \
    && grep -q "DFlash layer-tap off-by-one fix applied" \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py

# Long-prompt switch: PP2048 runs on AR/FlashQLA; short requests keep DFlash.
RUN bash /repro/patches/apply_dflash_prompt_threshold.sh \
    && grep -q "Combined DFlash prompt-threshold AR path active" \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py

# FlashQLA source pin used by the measured image.
RUN git clone https://github.com/my-other-github-account/FlashQLA.git /opt/flashqla &&     cd /opt/flashqla &&     git checkout 827fdd88e0829646e3c90be0c76158a9be62ab37 &&     git apply /repro/patches/flashqla-source-diff-827fdd88-hkv.patch

RUN pip install --no-cache-dir tilelang==0.1.8 apache-tvm-ffi==0.1.9 flash_linear_attention==0.5.0 &&     cd /opt/flashqla && pip install --no-cache-dir -e . --no-build-isolation

COPY patches/flashqla_hkv_o.py /opt/flashqla_hkv_o.py
ENV PYTHONPATH=/opt:${PYTHONPATH}
COPY patches/sitecustomize.py /usr/lib/python3.12/sitecustomize.py

CMD ["bash", "/repro/scripts/launch_server.sh"]
