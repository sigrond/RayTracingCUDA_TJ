/** \file OneshotAviread.cpp
 * \author Tomasz Jakubczyk
 * \brief
 * kompilacja w matlabie:
 * nvmex -f nvmexopts64.bat OneshotAviread.cpp IntensCalc_CUDA_kernel.cu IntensCalc_CUDA.cu -IC:\CUDA\include -IC:\CUDA\inc -LC:\cuda\lib\x64 -lcufft -lcudart -lcuda COMPFLAGS="$COMPFLAGS -std=c++11"
 */

#define WIN32
#include "mex.h"
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <string>
#include <thread>
#include <exception>


#include "IntensCalc_CUDA.cuh"

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    setupCUDA_IC();
    char name[]="E:\\DEG_clean402.avi\0";
    ifstream file (name, ios::in|ios::binary);
    const int skok = (640*480*2)+8;
    char* buff=new char[65535*10];/**< aktualny adres zapisu z dysku */

    int i=1;
    file.seekg((34824+(skok*(i))),ios::beg);

    for(int j=0;j<10;j++)
    {
        file.read(buff+j*65535,65535);/**< 64KB to optymalny rozmiar bloku czytanego z dysku */
    }
    //for(int i=0;i<640*480*2;i++)
    //    printf("%d ",buff[i]);
    copyBuff(buff);
    doIC(nullptr,nullptr,nullptr);
    freeCUDA_IC();
}
