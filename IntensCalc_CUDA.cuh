/** \file IntensCalc_CUDA.cuh
 * \author Tomasz Jakubczyk
 * \brief plik nagłówkowy funkcji wywołujących CUDA'ę
 *
 *
 *
 */

/** \brief przygotowanie do obliczeń
 *
 * \return void
 *
 */
void setupCUDA_IC();

/** \brief kopiuje bufor do GPU
 *
 * \param buff char* bufor z pliku 640*48*2
 * \return void
 *
 */
void copyBuff(char* buff);

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
