#!/bin/bash
make clean
make -j4 mrfin_timing_debug
make -j4 client
