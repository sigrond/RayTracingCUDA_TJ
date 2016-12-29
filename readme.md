RayTracingCUDA
author: Tomasz Jakubczyk

The aim of this project is improveing ray tracing in matlab by building CUDA code to mex file.
Instead of executing one raytracing function at a time they will be all executed at once.

These are relevant files (used by currrent build):
RayTracing.cu
RayTraceCUDA.cu
RayTraceCUDA_kernel.cu

IntensCalc.cu
IntensCalc_CUDA_kernel.cu
IntensCalc_CUDA.cu
CyclicBuffer.hpp
CyclicBuffer.cpp
MovingAverage_CUDA_kernel.cu

compilation line:
nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda


nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda -output RayTracingCUDA


Tip:
If you get an error while execution, first try reseting matlab.


IntensCalc

build in Matlab command line:
nvmex -f nvmexopts64.bat IntensCalc.cu IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu CyclicBuffer.cpp MovingAverage_CUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"





ReducedMean

nvmex -f nvmexopts64.bat ReducedMean.cu ReducedMean_CUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda

MovingAverage

nvmex -f nvmexopts64.bat MovingAverage.cu MovingAverage_CUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda
