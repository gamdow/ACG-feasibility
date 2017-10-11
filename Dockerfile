ARG RUNTEST="FALSE"

FROM ubuntu:16.04
WORKDIR /root


RUN apt-get update && apt-get install -y gcc g++ make wget git bzip2 \
  && gcc --version >> /root/versions.txt \
  && g++ --version >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt


# INSTALL CONDA
ENV CONDA_INSTALL_PATH=/opt/conda
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH
RUN apt-get update && apt-get install -y apt-utils \
  && wget -q https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh
RUN bash ./miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh \
  && echo "# Conda" >> /root/versions.txt \
  && conda --version >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt


# INSTALL JUPYTER
RUN conda install -y numpy matplotlib progressbar2 jupyter \
  && conda install -y ffmpeg -c menpo \
  && echo "# Jupyter Notebook" >> /root/versions.txt \
  && jupyter --version >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt


# Install OOMMF
ENV OOMMFTCL /root/oommf/oommf/oommf.tcl
RUN apt-get update && apt-get install -y tcl-dev tk-dev \
  && git clone https://github.com/fangohr/oommf.git \
  && cd oommf \
  && echo "# OOMMF" >> /root/versions.txt \
  && cat /root/oommf/oommf-version >> /root/versions.txt \
  && make
RUN pip install discretisedfield sarge \
  && python -c "import discretisedfield" \
  && pip list | grep "discretisedfield" >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt
#  && pip install oommfc \
#  && python -c "import oommfc"


# INSTALL DEVITO
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8
RUN apt-get update && apt-get install -y locales locales-all libgl1-mesa-glx \
  && git clone https://github.com/opesci/devito.git \
  && cd devito \
  && git reset --hard 8920ad2db1326f9e6ff480a3b0b46b98141c400e \
  && echo "# Devito" >> /root/versions.txt \
  && git config --get remote.origin.url >> /root/versions.txt \
  && git rev-parse HEAD >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt \
  && pip install -r requirements.txt sympy==1.0 \
  && pip install -e . \
  && (if test "$RUNTEST" = "TRUE"; then pytest; fi) \
  && python -c "import devito"


# INSTALL OpenSBLI

## OpenMPI
ENV MPI_INSTALL_PATH /usr/lib/openmpi
RUN apt-get update && apt-get install -y openmpi-bin libopenmpi-dev \
  && mkdir /usr/lib/openmpi/bin/ \
  && cd /usr/lib/openmpi/bin/ \
  && ln -s /usr/bin/mpiCC mpiCC \
  && ln -s /usr/bin/mpicc mpicc \
  && cd /root \
  && echo "# OpenMPI" >> /root/versions.txt \
  && mpicc --version >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt

## HDF5
ENV HDF5_VERSION=hdf5-1.8.19 HDF5_INSTALL_PATH=/usr/local
RUN wget -q https://support.hdfgroup.org/ftp/HDF5/current18/src/$HDF5_VERSION.tar.bz2 -O hdf5.tar.bz2 \
  && tar -xjf hdf5.tar.bz2 \
  && rm hdf5.tar.bz2 \
  && cd $HDF5_VERSION \
  && ./configure --enable-parallel --prefix=$HDF5_INSTALL_PATH \
  && make && (if test "$RUNTEST" = "TRUE"; then make test; fi) \
  && make install && if test "$RUNTEST" = "TRUE"; then make check-install; fi \
  && echo "# HDF5" >> /root/versions.txt \
  && echo $HDF5_VERSION >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt

## CUDA
ENV CUDA_VERSION=cuda_8.0.61_375.26_linux CUDA_INSTALL_PATH=/usr/local/cuda-8.0 OPENCL_INSTALL_PATH=/usr/local/cuda-8.0
RUN apt-get update && wget -q https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/$CUDA_VERSION-run -O cuda.run \
  && chmod -x cuda.run \
  && echo 'XKBMODEL="pc105"\nXKBLAYOUT="gb"\nXKBVARIANT=""\nXKBOPTIONS=""\n' > /etc/default/keyboard
# Split installation step here else CUDA does not install correctly (does not copy headers to install path)
RUN bash ./cuda.run --silent --toolkit \
  && rm cuda.run \
  && apt-get install -f \
  && echo "# CUDA" >> /root/versions.txt \
  && echo $CUDA_VERSION >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt

## OPS
# Editable install ("-e") doesn't work here (python import fails) for unknown reason.
ENV OPS_COMPILER=gnu OPS_INSTALL_PATH=/root/OPS/ops
RUN apt-get update && apt-get install -y python-pytools \
  && git clone https://github.com/gamdow/OPS.git \
  && cd OPS \
  && echo "# OPS" >> /root/versions.txt \
  && git config --get remote.origin.url >> /root/versions.txt \
  && git rev-parse HEAD >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt \
  && cd ops/c \
  && make \
  && cd ../.. \
  && pip install . \
  && python -c "import ops_translator.c.ops"

## OpenSBLI
#  && 2to3 -W -n opensbli \
RUN apt-get update && apt-get install -y python-pytools \
  && git clone https://github.com/gamdow/opensbli.git \
  && cd opensbli \
  && echo "# OpenSBLI" >> /root/versions.txt \
  && git config --get remote.origin.url >> /root/versions.txt \
  && echo "\n" >> /root/versions.txt \
  && git rev-parse HEAD >> /root/versions.txt \
  && pip install -r requirements.txt \
  && pip install -e . && (if test "$RUNTEST" = "TRUE"; then pytest; fi) \
  && python -c "import opensbli"

## libz
RUN apt-get update && apt-get install -y libz-dev


# Additional Evironment Variables
ENV DEVITO_OPENMP=1

# Jupyter Tweaks
RUN jupyter notebook --generate-config --allow-root \
  && echo "c.NotebookApp.iopub_data_rate_limit = 100000000" >> .jupyter/jupyter_notebook_config.py \
  && echo "c.NotebookApp.token = ''" >> .jupyter/jupyter_notebook_config.py \
  && mkdir .jupyter/custom \
  && echo ".container { width:100% !important; }" > .jupyter/custom/custom.css

# Jupyter Alias
#RUN echo 'alias notebook="jupyter notebook --allow-root --no-browser --ip=0.0.0.0"' >> ~/.bashrc

WORKDIR working
# Start Jupyter Automatically
ENTRYPOINT ["/bin/bash", "-c", "cp ../versions.txt versions.txt && jupyter notebook --allow-root --no-browser --ip=0.0.0.0"]
