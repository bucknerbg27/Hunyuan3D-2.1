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
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"

# Blender's Python 3.10 is incompatible with Python 3.12 (pkg_resources conflict).
# bpy was replaced with trimesh, so Blender is not needed at all.
# Remove it entirely to prevent sys.path contamination.
RUN rm -rf /opt/blender

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
ARG CACHEBUST=0
RUN echo "Cache bust: ${CACHEBUST}" && git clone -b dev/docker-fixes https://github.com/bucknerbg27/Hunyuan3D-2.1.git /workspace/Hunyuan3D-2.1
WORKDIR /workspace/Hunyuan3D-2.1

# -----------------------------------------------------------------------------
# 7. Install Python Dependencies
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir wheel setuptools pybind11

# setuptools>=70 removed pkg_resources, but lightning_fabric (pytorch-lightning 1.9.5) imports it.
# Provide a minimal shim that exposes declare_namespace.
COPY pkg_resources_shim.py /workspace/miniconda3/envs/hunyuan3d21/lib/python3.12/site-packages/pkg_resources.py
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
# 8. Pre-built CUDA extensions (from ComfyUI-3D-Pack)
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
# 9. RealESRGAN Weights
# -----------------------------------------------------------------------------
RUN mkdir -p hy3dpaint/ckpt && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

# -----------------------------------------------------------------------------
# 11. Final Setup
# -----------------------------------------------------------------------------
RUN mkdir -p gradio_cache && echo "conda activate hunyuan3d21" >> ~/.bashrc
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.13.0 /usr/local/cuda/lib64/libcudart.so.12 && \
    ldconfig

EXPOSE 8080
CMD ["/bin/bash", "-c", "source /workspace/miniconda3/etc/profile.d/conda.sh && conda activate hunyuan3d21 && exec python gradio_app.py"]
