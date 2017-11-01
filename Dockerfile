FROM ubuntu:16.04
RUN chown root:root /tmp
RUN chmod 1777 /tmp
WORKDIR /root


RUN apt-get update && apt-get install -y linux-headers-$(uname -r) cpio g++ g++-multilib make wget git bzip2


# CONDA
ARG CONDA_INSTALL_PATH=/opt/conda
RUN apt-get update && apt-get install -y apt-utils \
  && wget -q https://repo.continuum.io/miniconda/Miniconda3-4.3.21-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh
RUN bash ./miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh
  ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH


# JUPYTER
RUN conda install -y numpy matplotlib progressbar2 jupyter \
  && conda install -y ffmpeg -c menpo


# INTEL COMPILER
ARG INTEL_PATH=/opt/intel
ARG INTEL_YEAR=2018
ARG INTEL_VERSION=2018.0.128
ARG INTEL_DOWNLOAD=parallel_studio_xe_2018_cluster_edition_online
ADD $INTEL_DOWNLOAD.tgz .
ADD silent${INTEL_YEAR}.cfg $INTEL_DOWNLOAD/.
RUN cd $INTEL_DOWNLOAD && ./install.sh --silent=silent${INTEL_YEAR}.cfg

## Intel Environment
RUN echo COMPILERVARS_ARCHITECTURE=intel64 >> intelenv.sh \
  && echo . $(find /opt/intel/bin -iname compilervars.sh) >> intelenv.sh \
  && echo . $(find /opt/intel -iname mpivars.sh) >> intelenv.sh \
  && echo . $(find /opt/intel -iname mklvars.sh) >> intelenv.sh
RUN echo export CC=mpiicc CXX=mpiicpc MPICC=mpiicc MPICXX=mpiicpc MPICH_CC=icc MPICH_CXX=icpc "CFLAGS=\"-O3 -xHost -fno-alias -align\" CXXFLAGS=\"-O3 -xHost -fno-alias -align\"" >> intelcomp.sh
#RUN echo export FFlags="\"-I$(find /opt/intel -iname include64) -L$(find /opt/intel -iname lib64)\"" >> intelcomp.sh


# OOMMF
RUN export OOMMF_CPP='icpc -c' && . ./intelenv.sh && . ./intelcomp.sh \
  && apt-get update && apt-get install -y tcl-dev tk-dev \
  && git clone https://github.com/gamdow/oommf.git \
  && cd oommf \
  && tclsh oommf/oommf.tcl +platform \
  && make
ENV OOMMFTCL /root/oommf/oommf/oommf.tcl
RUN pip install discretisedfield sarge \
  && python -c "import discretisedfield"


# DEVITO
RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get update && apt-get install -y locales-all \
  && export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8 \
  && apt-get install -y locales locales-all libgl1-mesa-glx \
  && git clone https://github.com/opesci/devito.git \
  && cd devito \
  && git reset --hard 8920ad2db1326f9e6ff480a3b0b46b98141c400e \
  && pip install -r requirements.txt \
  && pip install -e . \
  && python -c "import devito"


# INSTALL OpenSBLI


ARG RUNTEST="FALSE"

## zlib
ARG ZLIB_VERSION=zlib-1.2.11
RUN wget -q https://zlib.net/$ZLIB_VERSION.tar.gz -O zlib.tar.gz \
  && tar -zxvf zlib.tar.gz \
  && rm zlib.tar.gz \
  && . ./intelenv.sh && export CC=icc CFLAGS="-O3 -xHost -ip" \
  && cd $ZLIB_VERSION \
  && ./configure --prefix=/usr/local/$ZLIB_VERSION \
  && make && (if test "$RUNTEST" = "TRUE"; then make check; fi) \
  && make install

## HDF5
ARG HDF5_VERSION=hdf5-1.10.1
ARG HDF5_INSTALL_PATH=/usr/local
RUN apt-get update && apt-get install -y file gcc \
  && wget -q https://support.hdfgroup.org/ftp/HDF5/current/src/$HDF5_VERSION.tar.gz -O hdf5.tar.gz \
  && tar -zxvf hdf5.tar.gz \
  && rm hdf5.tar.gz \
  && . ./intelenv.sh && . ./intelcomp.sh \
  && cd $HDF5_VERSION \
  && ./configure --prefix=$HDF5_INSTALL_PATH --enable-parallel \
  && make && (if test "$RUNTEST" = "TRUE"; then make check; fi) \
  && make install

## CUDA
ARG CUDA_VERSION=cuda_8.0.61_375.26_linux
RUN apt-get update && wget -q https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/$CUDA_VERSION-run -O cuda.run \
  && chmod -x cuda.run \
  && echo 'XKBMODEL="pc105"\nXKBLAYOUT="gb"\nXKBVARIANT=""\nXKBOPTIONS=""\n' > /etc/default/keyboard
# Split installation step here else CUDA does not install correctly (does not copy headers to install path)
RUN bash ./cuda.run --silent --toolkit \
  && rm cuda.run \
  && apt-get install -f

## OPS
ENV OPS_COMPILER=intel OPS_INSTALL_PATH=/root/OPS/ops MPI_INSTALL_PATH=$INTEL_PATH/compilers_and_libraries_${INTEL_VERSION}/linux/mpi/intel64 HDF5_INSTALL_PATH=$HDF5_INSTALL_PATH CUDA_INSTALL_PATH=/usr/local/cuda-8.0 OPENCL_INSTALL_PATH=/usr/local/cuda-8.0
RUN apt-get update && apt-get install -y python-pytools \
  && git clone https://github.com/gamdow/OPS.git \
  && . ./intelenv.sh && . ./intelcomp.sh \
  && cd OPS/ops/c \
  && make \
  && cd ../.. \
  && pip install . \
  && python -c "import ops_translator.c.ops"

## OpenSBLI
RUN apt-get update && apt-get install -y python-pytools \
  && git clone https://github.com/gamdow/opensbli.git \
  && cd opensbli \
  && pip install -r requirements.txt \
  && pip install -e . && (if test "$RUNTEST" = "TRUE"; then pytest; fi) \
  && python -c "import opensbli"

## libz
#RUN apt-get update && apt-get install -y libz-dev


# Additional Evironment Variables
ENV DEVITO_OPENMP=1
ENV DEVITO_ARCH=intel


# Jupyter Tweaks
RUN jupyter notebook --generate-config --allow-root \
  && echo "c.NotebookApp.iopub_data_rate_limit = 100000000" >> .jupyter/jupyter_notebook_config.py \
  && echo "c.NotebookApp.token = ''" >> .jupyter/jupyter_notebook_config.py \
  && mkdir .jupyter/custom \
  && echo ".container { width:100% !important; }" > .jupyter/custom/custom.css

# Jupyter Alias
RUN echo 'alias notebook="jupyter notebook --allow-root --no-browser --ip=0.0.0.0"' >> ~/.bashrc

#WORKDIR working
# Start Jupyter Automatically
CMD ["sh", "-c", ". ./intelenv.sh && . ./intelcomp.sh && jupyter notebook --allow-root --no-browser --ip=0.0.0.0"]
