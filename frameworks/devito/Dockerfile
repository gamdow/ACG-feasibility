FROM acg-env

RUN apt-get update \
  && apt-get install -qqy wget git bzip2

# CONDA
ARG CONDA_INSTALL_PATH=/opt/conda
RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && bash miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH

# JUPYTER
RUN conda install -qy jupyter progressbar2 gcc_linux-64 \
  && bash jupyter_config.sh

# DEVITO
RUN git clone https://github.com/opesci/devito.git \
  && cd devito \
  && conda env update -q -n root -f environment.yml \
  && pip install -qe . \
  && python -c "import devito"
ENV DEVITO_OPENMP=1

ADD run.py .

# Start Jupyter Automatically
EXPOSE 8888/tcp
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
