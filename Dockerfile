# source: https://github.com/openai/mujoco-py/blob/master/Dockerfile
# build by: sudo docker build --network=host -t  nvcr.io/nvidian/nvr-rock/raml -f  Dockerfile .
# run by: sudo nvidia-docker run -it --rm --runtime=nvidia --gpus all  nvcr.io/nvidian/nvr-rock/raml:latest /bin/bash
FROM docker.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC  apt-get install -y \
    tzdata \
    curl \
    git \
    git-lfs \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.9 python3-pip cmake -y
# RUN virtualenv --python=python3.6 env
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN pip3 install torch torchvision torchaudio numpy gym[classic_control] wandb jupyterlab tensorboard seaborn scipy cross_entropy_method mujoco
RUN git config --global --add safe.directory /opt/project
ENV WANDB_API_KEY=54aa3773f37573aff0d7322ec7eaebbbf411a599
#RUN rm /usr/bin/python
#RUN ln -s /env/bin/python3.6 /usr/bin/python
#RUN ln -s /env/bin/pip3.6 /usr/bin/pip
#RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/
COPY ./requirements.dev.txt /mujoco_py/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY . /mujoco_py
RUN python setup.py install