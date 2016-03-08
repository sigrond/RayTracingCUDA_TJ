/** \file MovingAverage_CUDA_kernel.cuh
 * \author Tomasz Jakubczyk
 * \brief plik nag³ówkowy kernelu CUDA
 *
 *
 *
 */

extern "C"
{

    __global__
    void MovingAverageD(float* I, unsigned int I_size, float* I_S, float* sI, float step);

    __global__
    void DivD(unsigned int I_size, float* sI, float step);

}
