/** \file IntensCalc_CUDA.cuh
 * \author Tomasz Jakubczyk
 * \brief plik nagłówkowy funkcji wywołujących CUDA'ę
 *
 *
 *
 */


extern "C"
{


/** \brief przygotowanie do obliczeń
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



/** \brief kopiuje bufor do GPU
 *
 * \param buff char* bufor z pliku 640*480*2
 * \return void
 *
 */
void copyBuff(char* buff);

/** \brief wczytaj dane do lewej strony przestrzeni danych
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void loadLeft(char* buff);

/** \brief wczytaj dane do prawej strony przestrzeni danych
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void loadRight(char* buff);

/** \brief przesuń dalej dane z filmu
 *
 * \param buff char* 640KB
 * \return void
 *
 */
void cycleDataSpace(char* buff);

/** \brief znajdź pozycje i rozmiary sekcji JUNK i header
 * skopiuj wybrane fragmenty do bufora na klatkę
 * \return void
 *
 */
void findJunkAndHeaders();

/** \brief zwraca wektor natężeń od theta
 *
 * \param I_Red float* wektor 700 czerwonego natężenia
 * \param I_Green float* wektor 700 zielonego natężenia
 * \param I_Blue float* wektor 700 niebieskiego natężenia
 * \return void
 *
 */
void doIC(float* I_Red, float* I_Green, float* I_Blue);

/** \brief zwalnianie zajętych zasobów CUDA
 *
 * \return void
 *
 */
void freeCUDA_IC();

}
