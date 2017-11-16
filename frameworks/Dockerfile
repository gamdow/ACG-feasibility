FROM ubuntu:16.04
WORKDIR /root

RUN touch build_wrapper.stdout

RUN (apt-get update && apt-get install -y wget bzip2 make docker docker.io python-qt4) >> build_wrapper.stdout

# Conda
ARG CONDA_INSTALL_PATH=/opt/conda
RUN (wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && bash miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh) >> build_wrapper.stdout
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH

# Jupyter
ADD base/jupyter_config.sh .
RUN (conda install -y jupyter numpy progressbar2 matplotlib \
  && bash jupyter_config.sh) >> build_wrapper.stdout

ADD entrypoint.sh .
RUN chmod +x entrypoint.sh

RUN mkdir working
WORKDIR /root/working
ADD env.list .

EXPOSE 8888/tcp
# Start Jupyter Automatically
ENTRYPOINT ["../entrypoint.sh"]
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
