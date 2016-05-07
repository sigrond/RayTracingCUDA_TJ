/** \file FrameReader.hpp
 * \author Tomasz Jakubczyk
 * \brief Plik nag³owkowy klasy FrameReader
 *
 */

#pragma once

#include "CyclicBuffer.hpp"
#include <queue>

/** \brief Klasa maj¹ca za zadanie czytaæ film
 * z bufora cyklicznego i zwracaæ jego klatki klatki
 */
class FrameReader : private queue
{
public:
    /** \brief Default constructor
     * \param CyclicBuffer* bufor cykliczny z którego bêdziemy pobieraæ film
     */
    FrameReader(CyclicBuffer*);
    /** Default destructor */
    ~FrameReader();
    /** \brief zwraca pierwszy element i usuwa go z kolejki
     * \return char* adres zwróconych danych
     */
    char* pop();
    /** \brief zwraca obecn¹ liczbê klatek gotowych do przetworzenia
     * \return int liczba czekaj¹cych elementów
     */
    int size();
    /** \brief wczytuje klatki z bufora cyklicznego
     * \return void
     */
    void readFrames();
private:
};
