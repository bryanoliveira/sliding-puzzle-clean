# 1. Test setup:
# docker run -it --rm --gpus all pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg WANDB_KEY=hash -f Dockerfile -t aluno_bryan-sldp_cleanrl:latest .
# docker run -it --rm --gpus '"device=0"' --cpus=16.0 --memory=16g -v /raid/bryan/sliding-puzzle-env:/sldp -v /raid/bryan/cleanrl/experiments:/sliding-puzzle/experiments -v /raid/bryan/cleanrl/runs:/sliding-puzzle/runs aluno_bryan-sldp_cleanrl

# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
    libglew2.1 libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    bc \
    && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV NUMBA_CACHE_DIR=/tmp

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install "gymnasium[atari, accept-rom-license]"

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT
ARG WANDB_KEY
ENV WANDB_API_KEY=$WANDB_KEY

RUN mkdir /sliding-puzzle
COPY ppo.py /sliding-puzzle
COPY entrypoint.sh /sliding-puzzle
RUN chown -R 1000:root /sliding-puzzle && chmod -R 775 /sliding-puzzle
WORKDIR /sliding-puzzle

RUN chmod +x /sliding-puzzle/entrypoint.sh
ENTRYPOINT ["/sliding-puzzle/entrypoint.sh"]
CMD ["bash", "experiments/daemon.sh"]