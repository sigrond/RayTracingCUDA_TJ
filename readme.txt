RayTracingCUDA
author: Tomasz Jakubczyk

The aim of this project is improveing ray tracing in matlab by building CUDA code to mex file.
Instead of executing one raytracing function at a time they will be all executed at once.

compilation line:
nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda


nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda -output RayTracingCUDA


Tip:
If you get an error while execution, first try reseting matlab.


TODO:
for each ray, if it hits pixel calculate average ray's y & z for that pixel
