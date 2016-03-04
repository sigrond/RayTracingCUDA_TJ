/** \file ReducedMean_CUDA_kernel.cuh
 * \author Tomasz Jakubczyk
 * \brief header for kernel functions
 *
 *
 *
 */


extern "C"
{
    __global__
    void ReducedMeanD(float* Theta_S, float deltaT, unsigned int max_nom, float* I, float* I_S, float* nTheta, float* nI, float* counter);
}
