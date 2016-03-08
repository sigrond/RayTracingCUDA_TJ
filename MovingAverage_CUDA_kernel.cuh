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
    void MovingAverageD(float* Theta_S,unsigned int Theta_S_size, float* I, float* I_S, float* sTheta, float* sI);

    __global__
    void DivD(unsigned int Theta_S_size, float* sTheta, float* sI);

}
