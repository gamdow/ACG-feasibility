FROM acg-env

RUN touch build_devito.stdout

RUN (apt-get update \
  && apt-get install -y wget git bzip2) >> build_devito.stdout

# CONDA
ARG CONDA_INSTALL_PATH=/opt/conda
RUN (wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && bash miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh) >> build_devito.stdout
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH

# JUPYTER
RUN (conda install -y jupyter progressbar2 gcc_linux-64 \
  && bash jupyter_config.sh) >> build_devito.stdout

# DEVITO
RUN (git clone https://github.com/opesci/devito.git \
  && cd devito \
  && conda env update -n root -f environment.yml \
  && pip install -e . \
  && python -c "import devito") >> build_devito.stdout
ENV DEVITO_OPENMP=1

ADD run.py .

# Start Jupyter Automatically
EXPOSE 8888/tcp
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
