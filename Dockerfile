FROM nvidia/cuda:11.6.2-devel-ubuntu20.04 AS devbase

#WORKDIR "./extensions/stable-diffusion-grpcserver"

# Basic updates. Do super early so we can cache for a long time
RUN apt update
RUN apt install -y curl
RUN apt install -y git

# Set up core python environment
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

COPY /extensions/stable-diffusion-grpcserver/environment.yaml .
RUN /bin/micromamba -r /env -y create -f environment.yaml

# Install dependancies
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116
RUN /bin/micromamba -r /env -n sd-grpc-server run pip install torch~=1.12.1




FROM devbase AS regularbase

# Install dependancies
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116
ENV FLIT_ROOT_INSTALL=1

# We copy only the minimum for flit to run so avoid cache invalidation on code changes
COPY /extensions/stable-diffusion-grpcserver/pyproject.toml .
COPY /extensions/stable-diffusion-grpcserver/sdgrpcserver/__init__.py sdgrpcserver/
RUN touch README.md
RUN /bin/micromamba -r /env -n sd-grpc-server run flit install --pth-file
RUN /bin/micromamba -r /env -n sd-grpc-server run pip cache purge

# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

RUN mkdir -p $NVM_DIR

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default




# Build Xformers

FROM devbase AS xformersbase

RUN git clone https://github.com/facebookresearch/xformers.git
WORKDIR /xformers
RUN git submodule update --init --recursive
RUN /bin/micromamba -r /env -n sd-grpc-server run pip install -r requirements.txt

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"

RUN /bin/micromamba -r /env -n sd-grpc-server run pip install .

RUN tar cvjf /xformers.tbz /env/envs/sd-grpc-server/lib/python3.*/site-packages/xformers*




FROM nvidia/cuda:11.6.2-base-ubuntu20.04 AS main

COPY --from=regularbase /bin/micromamba /bin/
RUN mkdir -p /env/envs
COPY --from=regularbase /env/envs /env/envs/
RUN mkdir -p /nvm
COPY --from=regularbase /nvm /nvm/

# Setup NVM & Node for Localtunnel
ENV NVM_DIR=/nvm
ENV NODE_VERSION=16.18.0

ENV NODE_PATH $NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

RUN npm install -g localtunnel

# Now we can copy everything we need
COPY /extensions/stable-diffusion-grpcserver/sdgrpcserver /sdgrpcserver/
COPY /extensions/stable-diffusion-grpcserver/server.py .

# Set up some config files
RUN mkdir -p /huggingface
RUN mkdir -p /weights
RUN mkdir -p /config
COPY /extensions/stable-diffusion-grpcserver/engines.yaml /config/

# Set up some environment files

ENV HF_HOME=/huggingface
ENV HF_API_TOKEN=mustset
ENV SD_ENGINECFG=/config/engines.yaml
ENV SD_WEIGHT_ROOT=/weights
ENV SD_NSFW_BEHAVIOUR=flag
CMD [ "/bin/micromamba", "-r", "env", "-n", "sd-grpc-server", "run", "python", "./server.py" ]




FROM main as xformers

COPY --from=xformersbase /xformers/requirements.txt /
RUN /bin/micromamba -r /env -n sd-grpc-server run pip install -r requirements.txt
RUN rm requirements.txt
COPY --from=xformersbase /xformers.tbz /
RUN tar xvjf /xformers.tbz
RUN rm /xformers.tbz

CMD [ "/bin/micromamba", "-r", "env", "-n", "sd-grpc-server", "run", "python", "./server.py" ]
