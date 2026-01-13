# hash:sha256:4d448f835961c9c304deff24b35dc2351b4076e9e4d173407b01ed9e9081b44b
FROM registry.codeocean.com/codeocean/lightning-jupyterlab-py39:1.0

ARG DEBIAN_FRONTEND=noninteractive

ENV HDF5_USE_FILE_LOCKING=FALSE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.4.0-1 \
        libglib2.0-0=2.72.4-0ubuntu2.3 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
        libxrender1=1:0.9.10-1build4 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    argparse==1.4.0 \
    climage==0.2.2 \
    einops==0.6.1 \
    h5py==3.9.0 \
    matplotlib==3.7.2 \
    mrcfile==1.4.3 \
    numpy==1.24.1 \
    opencv-python==4.10.0.84 \
    pandas==2.0.3 \
    pillow==9.3.0 \
    scikit-image==0.21.0 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    tifffile==2023.8.12 \
    tqdm==4.66.1
