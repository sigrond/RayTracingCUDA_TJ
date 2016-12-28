/** \file IntensCalc_CUDA.cuh
 * \author Tomasz Jakubczyk
 * \brief header file of functions calling CUDA
 *
 *
 *
 */


extern "C"
{


/** \brief prepare GPU
 *
 * \return void
 *
 */
void setupCUDA_IC();


void setMasksAndImagesAndSortedIndexes(
    int* ipR,int ipR_size,int* ipG,int ipG_size,int* ipB, int ipB_size,
    float* ICR_N, float* ICG_N, float* ICB_N,
    int* I_S_R, int* I_S_G, int* I_S_B,
    unsigned char* BgMaskR, float* BgMaskSizeR,
    unsigned char* BgMaskG, float* BgMaskSizeG,
    unsigned char* BgMaskB, float* BgMaskSizeB);



/** \brief copy buffer to GPU
 *
 * \param buff char* file buffer 640*480*2
 * \return void
 *
 */
void copyBuff(char* buff);

/** \brief load data to left side of DataSpace
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void loadLeft(char* buff);

/** \brief load data to right side of DataSpace
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void loadRight(char* buff);

/** \brief move to next part of data
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void cycleDataSpace(char* buff);

/** \brief find positions and sizes of JUNK and header sections
 * copy selected fragments to buffer for frame
 * \return void
 *
 */
void findJunkAndHeaders();

/** \brief returns intensity vectors by theta
 *
 * \param I_Red float* red intensity vector of 700 elements
 * \param I_Green float* green intensity vector of 700 elements
 * \param I_Blue float* blue intensity vector of 700 elements
 * \return void
 *
 */
void doIC(float* I_Red, float* I_Green, float* I_Blue);

/** \brief free used CUDA resources
 *
 * \return void
 *
 */
void freeCUDA_IC();

}
