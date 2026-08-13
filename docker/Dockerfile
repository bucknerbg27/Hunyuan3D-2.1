FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

LABEL name="hunyuan3d21" maintainer="hunyuan3d21"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl \
    CUDA_HOME=/usr/local/cuda \
    CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

ENV PATH="${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# 1. System Dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git git-lfs wget curl vim unzip cmake pkg-config \
    libegl1-mesa-dev libglib2.0-0 libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libglvnd-dev libgl1-mesa-dev libgles2-mesa-dev mesa-utils-extra \
    libxrender1 libxrender-dev libeigen3-dev python3-dev python3-setuptools \
    libcgal-dev libxi6 libxkbcommon-x11-0 libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Blender 4.0
# -----------------------------------------------------------------------------
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && \
    tar -xvf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm /tmp/blender.tar.xz

ENV PATH="/opt/blender:${PATH}"

# -----------------------------------------------------------------------------
# 3. Conda + Python 3.12
# -----------------------------------------------------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /workspace/miniconda3 && \
    rm miniconda.sh

ENV PATH="/workspace/miniconda3/bin:${PATH}"

RUN conda config --add channels conda-forge && conda config --set channel_priority strict
RUN conda create -n hunyuan3d21 python=3.12 -c conda-forge -y && conda clean -a -y

ENV PATH="/workspace/miniconda3/envs/hunyuan3d21/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:/opt/blender/4.0/python/lib:${LD_LIBRARY_PATH}"

RUN echo "/opt/blender/4.0/scripts/modules" > /workspace/miniconda3/envs/hunyuan3d21/lib/python3.12/site-packages/blender.pth && \
    echo "/opt/blender/4.0/python/lib/python3.10" >> /workspace/miniconda3/envs/hunyuan3d21/lib/python3.12/site-packages/blender.pth && \
    echo "/opt/blender/4.0/python/lib/python3.10/site-packages" >> /workspace/miniconda3/envs/hunyuan3d21/lib/python3.12/site-packages/blender.pth

# -----------------------------------------------------------------------------
# 4. Build Tools
# -----------------------------------------------------------------------------
RUN conda install -n hunyuan3d21 -c conda-forge ninja libstdcxx-ng -y && conda clean -a -y

# -----------------------------------------------------------------------------
# 5. Custom PyTorch wheels (Blackwell-compatible, from ComfyUI)
# -----------------------------------------------------------------------------
COPY torch-*.whl torchvision-*.whl torchaudio-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/torch-*.whl /tmp/torchvision-*.whl /tmp/torchaudio-*.whl

# -----------------------------------------------------------------------------
# 6. Clone Repository
# -----------------------------------------------------------------------------
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /workspace/Hunyuan3D-2.1
WORKDIR /workspace/Hunyuan3D-2.1

# Fix MAX_SEED float -> int for Python 3.12
RUN sed -i 's/MAX_SEED = 1e7/MAX_SEED = 10000000/' gradio_app.py

# -----------------------------------------------------------------------------
# 8. Install Python Dependencies
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir wheel setuptools pybind11
RUN sed -i '/^bpy/d' requirements.txt
RUN sed -i 's/numpy==1.24.4/numpy>=1.26.0/' requirements.txt
RUN sed -i 's/open3d==0.18.0/open3d>=0.19.0/' requirements.txt
RUN sed -i '/^pymeshlab/d' requirements.txt
RUN sed -i 's/[<>=!]=.*//' requirements.txt
RUN sed -i '/^--extra-index-url/d' requirements.txt
RUN sed -i '/^$/d' requirements.txt
RUN pip install --no-cache-dir ninja
RUN pip install --no-cache-dir einops
RUN pip install --no-cache-dir rembg
RUN pip install --no-cache-dir realesrgan
RUN pip install --no-cache-dir gradio
RUN pip install --no-cache-dir cupy-cuda12x
RUN pip install --no-cache-dir onnxruntime

# Install pymeshlab from pre-built wheel
COPY pymeshlab-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/pymeshlab-*.whl

# Install basicsr
RUN pip install --no-cache-dir --no-build-isolation basicsr==1.4.2

# Install all remaining deps from local wheels (fast, no network)
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --find-links /tmp/wheels/ -r requirements.txt

# -----------------------------------------------------------------------------
# 9. Pre-built CUDA extensions (from ComfyUI-3D-Pack)
# -----------------------------------------------------------------------------
# Build custom_rasterizer from source (against CUDA 13)
RUN cd hy3dpaint/custom_rasterizer && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
    pip install --no-build-isolation --no-cache-dir -e .

# Build DifferentiableRenderer from source
RUN cd hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh

# -----------------------------------------------------------------------------
# 10. Patch realesrgan to skip fp16
# -----------------------------------------------------------------------------
RUN <<'PYEOF' python3
path = '/workspace/miniconda3/envs/hunyuan3d21/lib/python3.12/site-packages/realesrgan/utils.py'
with open(path, 'r') as f:
    content = f.read()
content = content.replace('self.model = self.model.half()', 'pass  # disabled for Blackwell')
with open(path, 'w') as f:
    f.write(content)
print('Patched realesrgan/utils.py successfully')
PYEOF

# -----------------------------------------------------------------------------
# 11. RealESRGAN Weights
# -----------------------------------------------------------------------------
RUN mkdir -p hy3dpaint/ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

# -----------------------------------------------------------------------------
# 12. Patch Configuration Paths
# -----------------------------------------------------------------------------
RUN sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' hy3dpaint/textureGenPipeline.py && \
    sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' hy3dpaint/utils/multiview_utils.py

# -----------------------------------------------------------------------------
# 13. Final Setup
# -----------------------------------------------------------------------------
RUN mkdir -p gradio_cache && echo "conda activate hunyuan3d21" >> ~/.bashrc
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.13.0 /usr/local/cuda/lib64/libcudart.so.12 && \
    ldconfig

EXPOSE 8080
CMD ["/bin/bash", "-c", "source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate hunyuan3d21 && exec python gradio_app.py"]