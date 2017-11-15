FROM acg-env

RUN apt-get update \
  && apt-get install -qqy make wget git bzip2 file python-pytools libz-dev

# CONDA
ARG CONDA_INSTALL_PATH=/opt/conda
RUN wget -q https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh \
  && chmod +x miniconda.sh \
  && bash miniconda.sh -b -p $CONDA_INSTALL_PATH \
  && rm miniconda.sh
ENV PATH=$CONDA_INSTALL_PATH/bin:$PATH

# JUPYTER
RUN conda install -qy jupyter progressbar2 gcc_linux-64 \
  && bash jupyter_config.sh

# INSTALL OpenSBLI
ARG RUNTEST="FALSE"

## zlib
ARG ZLIB_VERSION=zlib-1.2.11
RUN wget -q https://zlib.net/$ZLIB_VERSION.tar.gz -O zlib.tar.gz \
  && tar -zxf zlib.tar.gz \
  && rm zlib.tar.gz \
  && export CC=icc CFLAGS="-O3 -xHost -ip" \
  && cd $ZLIB_VERSION \
  && ./configure --prefix=/usr/local/$ZLIB_VERSION \
  && make -s && (if test "$RUNTEST" = "TRUE"; then make check; fi) \
  && make -s install

## HDF5
ARG HDF5_VERSION=hdf5-1.10.1
ARG HDF5_INSTALL_PATH=/usr/local
RUN wget -q https://support.hdfgroup.org/ftp/HDF5/current/src/$HDF5_VERSION.tar.gz -O hdf5.tar.gz \
  && tar -zxf hdf5.tar.gz \
  && rm hdf5.tar.gz \
  && cd $HDF5_VERSION \
  && ./configure --prefix=$HDF5_INSTALL_PATH --enable-parallel \
  && make -s && (if test "$RUNTEST" = "TRUE"; then make check; fi) \
  && make -s install

## CUDA
ARG CUDA_VERSION=cuda_8.0.61_375.26_linux
RUN wget -q https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/$CUDA_VERSION-run -O cuda.run \
  && chmod -x cuda.run \
  && echo 'XKBMODEL="pc105"\nXKBLAYOUT="gb"\nXKBVARIANT=""\nXKBOPTIONS=""\n' > /etc/default/keyboard
# Split installation step here else CUDA does not install correctly (does not copy headers to install path)
RUN bash ./cuda.run --silent --toolkit \
  && rm cuda.run \
  && apt-get install -f

## OPS
ENV OPS_INSTALL_PATH=/root/OPS/ops HDF5_INSTALL_PATH=$HDF5_INSTALL_PATH CUDA_INSTALL_PATH=/usr/local/cuda-8.0 OPENCL_INSTALL_PATH=/usr/local/cuda-8.0
RUN git clone https://github.com/gamdow/OPS.git \
  && cd OPS/ops/c \
  && make \
  && cd ../.. \
  && pip install -qe . \
  && python -c "import ops_translator.c.ops"

## OpenSBLI
RUN git clone https://github.com/opensbli/opensbli.git \
  && cd opensbli \
  && pip install -q sympy==1.0 h5py pytest flake8 Sphinx \
  && pip install -qe . && (if test "$RUNTEST" = "TRUE"; then pytest; fi) \
  && python -c "import opensbli"

RUN ldconfig

ADD run.py .
ADD Makefile .

# Start Jupyter Automatically
EXPOSE 8888/tcp
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
