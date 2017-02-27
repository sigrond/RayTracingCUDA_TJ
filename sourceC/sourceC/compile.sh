#!/bin/bash
source /opt/intel/bin/compilervars.sh intel64
make clean
make
cp mrfin ../mrfin
