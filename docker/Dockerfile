FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG user
ARG uid

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video,utility,compute

ENV DEBIAN_FRONTEND noninteractive

# Before any libraries are installed, make sure everyone knows about cuda-aware ompi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

##### PACKAGES
RUN apt-get -y update && apt-get install -y libassimp-dev libjpeg-dev libpng-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libgl1-mesa-dev libegl1-mesa-dev


# Dependencies for glvnd and X11. -> https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1
# libboost-all-dev is already installed when installing libpcl-dev
RUN apt-get -y update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libglx0 \
    libxext6 \
    libx11-6 \
    libglfw3 \
    libglfw3-dev \
    libpcl-dev \
    tmux \
    zip\
    locate\
    libboost-program-options-dev\
    libboost-filesystem-dev\
    libboost-graph-dev\
    libboost-system-dev\
    libboost-test-dev\
    libeigen3-dev\
    libsuitesparse-dev\
    libfreeimage-dev\
    libmetis-dev\
    libgoogle-glog-dev\
    libgflags-dev\
    libglew-dev\
    qtbase5-dev\
    libqt5opengl5-dev\
    libcgal-dev\
    libatlas-base-dev\
    libsuitesparse-dev\
    libomp-dev\
    libomp5\
    imagemagick
    

##### PIP
# Install common packages
RUN pip install ipdb tensorboard pymcubes imageio PyOpenGL glfw xatlas gdown wandb ninja hsluv moviepy seaborn opencv-python==4.5.5.64 scikit-image
RUN pip install --no-deps lpips

##### CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /repos

RUN useradd -u ${uid} -ms /bin/bash ${user}
RUN chown -R ${user} /opt /repos /usr
USER ${user}
ENV DOCKERMODE="YES"

# interactive console
ENV DEBIAN_FRONTEND teletype
