#include "HandlesStructures.cuh"

extern "C"
{
    /** \brief compute grid and thread block size for a given number of elements
     *
     * \param n uint
     * \param blockSize uint
     * \param numBlocks uint&
     * \param numThreads uint&
     * \return void
     *
     */
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    /** \brief calculate ray tracing
     *
     * \param P2 float3*
     * \param VH_length int
     * \param Vb_length int
     * \param S HandlesStructures*
     * \param IM float3*
     * \return void
     *
     */
    void RayTrace(float3* P2, int VH_length, int Vb_length, HandlesStructures S, float3* IM)
    {
        uint numThreads, numBlocks;
        computeGridSize(VH_length*Vb_length, 256, numBlocks, numThreads);
        RayTraceD<<< numBlocks, numThreads >>>(P2,VH_length,Vb_length,S,IM);
    }
}
