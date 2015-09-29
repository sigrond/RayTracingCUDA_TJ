RayTracingCUDA
author: Tomasz Jakubczyk

This project is ought to improve ray tracing in matlab by building CUDA code to mex file.
Instead of executing one raytracing function at a time they will be all executed at once.

cmopilation line:
nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda