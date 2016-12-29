# RayTracingCUDA
author: Tomasz Jakubczyk

This projects aim is to extract vector (700 values) of corrected and representative values of frame in theta angles (horizontal) from experiments movie.

Project is based on earleir created PikeReader Matlab app with relatively intuitive and easy to use Graphical User Interface.
On top of that due to very long computational time two CUDA C++ MEX subrutines were written: RayTracingCUDA and IntensCalc.

RayTracingCUDA calculates quite high quality correction image based on experiments parameters.
Correction image represents how much light should theoreticly land on CCD.
RayTracingCUDA thanks to GPU power allows to execute calculations for many rays at onece instead of one at a time.
Faster computaions also alows to calculate more rays in reasonable time and thus to have higher quality correction image.

IntensCalc reades movie from hard drive, decodes and demosaics frames, apllies masks and correction images, smooths data with moveing average and finaly selects and vector of representative values by theta angle.
Reading from hard drive is dane as fast as posible because it is one of the main bottlenecks but this has a drowback of having movie in blocks that do not fit frame size.
In current version of app frames are found and put together on CPU. The rest of movie processing and extracting data is moustly done on GPU.




### These are relevant files (used by currrent build):
- RayTracing.cu
- RayTraceCUDA.cu
- RayTraceCUDA_kernel.cu

- IntensCalc.cu
- IntensCalc_CUDA_kernel.cu
- IntensCalc_CUDA.cu
- CyclicBuffer.hpp
- CyclicBuffer.cpp
- MovingAverage_CUDA_kernel.cu

#### RayTraceCUDA

##### compilation line:
32bit:
nvmex -f nvmexopts.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\win32 -lcufft -lcudart -lcuda -output RayTracingCUDA
64bit:
nvmex -f nvmexopts64.bat RayTracing.cu RayTraceCUDA.cu RayTraceCUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda -output RayTracingCUDA


Tip:
If you get an error while execution, first try reseting matlab.


#### IntensCalc

##### build in Matlab command line:
nvmex -f nvmexopts64.bat IntensCalc.cu IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu CyclicBuffer.cpp MovingAverage_CUDA_kernel.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"


