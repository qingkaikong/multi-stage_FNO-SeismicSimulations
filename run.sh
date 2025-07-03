#!/bin/bash
  
source /etc/profile.d/z00_lmod.sh

module load rocm/6.2.1
module load cray-mpich/8.1.30
 
# get the hostname of the first node in the flux allocation
firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
echo "first host: $firsthost"

# set MASTER_ADDR to hostname of first compute node in allocation
# set MASTER_PORT to any unused port number
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456
echo "$MASTER_ADDR"

# the AWS-OFI-RCCL plugin lets RCCL use libfabric instead of TCP sockets
# settings below taken from:
#   https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl#running-rccl-perf-tests
# set LD_LIBRARY_PATH to point to /lib directory containing librccl-net.so of the aws-ofi-rccl plugin
export LD_LIBRARY_PATH=/collab/usr/global/tools/rccl/${SYS_TYPE}/rocm-6.2.1/install/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0

# if not using the AWS-OFI-RCCL plugin above ...
# HACK: for various reasons, RCCL fails when all NICs are available for TCP sockets
# this constrains RCCL to only use the first NIC with TCP sockets, which happens to work
export NCCL_SOCKET_IFNAME=hsi0

# Point to node-local storage to cache MIOpen performance DB files and pre-compiled kernels
# These otherwise default to user home directories on NFS like ~/.config/miopen/ and ~/.cache/miopen
#   https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
 
flux run -N 2 -n 8 --exclusive python 01_FNO-SingleFrequency-multistage.py