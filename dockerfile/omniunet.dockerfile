FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# For xorg graphics
ENV QT_X11_NO_MITSHM 1

ARG USERNAME=nonroot
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Contact info
LABEL maintainer="Raul Castilla Arquillo <raulcastar@uma.es>"
LABEL authors="Raul Castilla Arquillo <raulcastar@uma.es>,\
Carlos Jesus Perez del Pulgar Mancebo <carlosperez@uma.es>"
LABEL organization="Space Robotics Laboratory (University of Malaga)"
LABEL url="https://www.uma.es/space-robotics"
LABEL version="1.0"
LABEL license="MIT License"
LABEL description=""
LABEL created=""

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $USER_GID $USERNAME && \
    adduser --uid $USER_UID --gid $USER_GID --disabled-password --gecos "" $USERNAME && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Set the working directory
WORKDIR /home/$USERNAME/

# Visual and dev packages
RUN apt-get update && \
      apt-get -y install python3.8  \
      libopencv-dev python3-opencv wget \
      libcanberra-gtk-module libcanberra-gtk3-module \
      tmux xorg nano vim curl python3-gi-cairo python3-pip \
      && rm -rf /var/lib/apt/lists/*

# Packages to run jupyter
RUN apt-get update && \
      apt-get -y install jupyter firefox ffmpeg python3-tk git-all chafa \
      && rm -rf /var/lib/apt/lists/*

# Set the non-root user as the default user
USER $USERNAME

# Packages needed to run neural networks
RUN pip3 install torch==1.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torchvision torchaudio
RUN pip3 install einops pytorchvideo timm
RUN pip3 install matplotlib
RUN pip3 install hydra ipywidgets==7.5.1
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip3 install wandb scipy pycocotools scikit-image
RUN pip3 pandas
RUN pip3 install --upgrade requests