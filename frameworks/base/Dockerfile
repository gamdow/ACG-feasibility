FROM ubuntu:16.04
RUN chown root:root /tmp && chmod 1777 /tmp
WORKDIR /root
RUN apt-get update \
  && apt-get install -qqy cpio linux-headers-$(uname -r) g++ g++-multilib

# INTEL COMPILER
ARG INTEL_PATH=/opt/intel
ARG INTEL_DOWNLOAD=parallel_studio
ADD $INTEL_DOWNLOAD.tgz .
RUN mv $INTEL_DOWNLOAD*/ $INTEL_DOWNLOAD/
ADD silent.cfg $INTEL_DOWNLOAD
RUN cd $INTEL_DOWNLOAD && ./install.sh --silent=silent.cfg
RUN rm -r $INTEL_DOWNLOAD

## Intel Environment
ENV CC=mpiicc CXX=mpiicpc MPICC=mpiicc MPICXX=mpiicpc MPICH_CC=icc MPICH_CXX=icpc CFLAGS=-O3\ -xHost\ -fno-alias\ -align CXXFLAGS=-O3\ -xHost\ -fno-alias\ -align
ENV DEVITO_ARCH=intel OOMMF_CPP='icpc -c' OPS_COMPILER=intel

# Jupyter Config Script
ADD jupyter_config.sh .

RUN echo '#!/bin/bash' >> entrypoint.sh \
  && echo . $(find $INTEL_PATH/bin -iname compilervars.sh) -arch intel64 >> entrypoint.sh \
  && echo . $(find $INTEL_PATH -iname mpivars.sh) -arch intel64 >> entrypoint.sh \
  && echo . $(find $INTEL_PATH -iname mklvars.sh) intel64 >> entrypoint.sh \
  && echo 'export MPI_INSTALL_PATH=$I_MPI_ROOT/intel64' >> entrypoint.sh \
  && echo 'exec "$@"' >> entrypoint.sh \
  && chmod +x entrypoint.sh
EXPOSE 8888/tcp
ENTRYPOINT ["./entrypoint.sh"]
CMD ["jupyter-notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
