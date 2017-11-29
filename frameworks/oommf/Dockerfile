FROM acg-env

RUN touch build_oommf.stdout

RUN (apt-get update \
  && apt-get install -y wget make git tcl-dev tk-dev) >> build_oommf.stdout

# CONDA
ARG CONDA_INSTALL_PATH=/opt/conda
RUN (wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && bash miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh) >> build_oommf.stdout
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH

# JUPYTER
RUN (conda install -y jupyter progressbar2 gcc_linux-64 \
  && bash jupyter_config.sh) >> build_oommf.stdout

# Additional Deps
RUN (pip install discretisedfield sarge \
  && python -c "import discretisedfield") >> build_oommf.stdout

# OOMMF
ADD https://api.github.com/repos/gamdow/oommf/compare/master...HEAD /dev/null
RUN (git clone https://github.com/gamdow/oommf.git \
  && cd oommf \
  && make) >> build_oommf.stdout
ENV OOMMFTCL /root/oommf/oommf/oommf.tcl

ADD run.py .

# Start Jupyter Automatically
EXPOSE 8888/tcp
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
