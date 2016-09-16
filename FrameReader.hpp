/** \file FrameReader.hpp
 * \author Tomasz Jakubczyk
 * \brief Plik nag³owkowy klasy FrameReader
 *
 */

#pragma once

#include "CyclicBuffer.hpp"
#include <exception>
#include <string>

/** \brief klasa wyjątku dla FrameReader
 */
class FrameReaderException : public std::exception
{
private:
    std::string str;
public:
    FrameReaderException(std::string s) : str(s){}
    virtual char const * what() const
    {
        return str.c_str();
    }
};

/** \brief Klasa maj¹ca za zadanie czytaæ film
 * z bufora cyklicznego i zwracaæ jego klatki klatki
 *
 * Żeby zachować wydajność dane z bufora otrzymanego od bufora cyklicznego
 * należy skopiować tylko raz (raz z dysku do bufora cyklicznego, raz w FrameReader,
 * raz z FrameReader do pamięci karty graficznej) i ma to ułatwić znajdowanie
 * klatek właściwych, bo chcielibyśmy je mięć w jednym ciągłym fragmencie pamięci.
 * Wszystkie wskaźniki, iteratory i liczniki powinny być opakowane w struktury
 * lub klasy pozwalające dobrze je identyfikować, określać ich cel i zbierać
 * statystyki. Powinny też mieć mechanizm modyfikowania się przy zamianie buforów.
 * FrameReader powinien mieć minimalistyczny interfejs zewnętrzny.
 */
class FrameReader
{
public:
    /** \brief Default constructor
     * \param CyclicBuffer* bufor cykliczny z którego bêdziemy pobieraæ film
     */
    FrameReader(CyclicBuffer*);
    /** Default destructor */
    ~FrameReader();
    /** \brief zwraca wskaźnik do klatki filmu, dane pod wskaźnikiem zmienią się
     * przy następnym wywołaniu tej metody, więc należy je skopiować (do pamięci
     *  karty graficznej)
     * \return char* wskażnik na klatkę
     *
     */
    char* getFrame();
private:
    CyclicBuffer* cyclicBuffer;/**< wskaźnik na bufor cykliczny z którego pobierane są dane */
    /** \brief blok z danymi z miejscem na dane z dwóch buforów
     */
    struct DataSpace
    {
    public:
        DataSpace(unsigned long int);
        ~DataSpace();
        char* pt;/**< wskaźnik na dane */
        unsigned long int size;/**< rozmiar danych */
        char* ptLeft;
        char* ptRight;
        unsigned long int halfSize;
    } *dataSpace;
    struct Header
    {
        Header();
        char* pt;/**< adres nagłówka */
        unsigned long int position;/**< pozycja nagłówka w DataSpace */
        bool found;/**< czy nagłówkek jest znaleziony */
        unsigned long int number;/**< numer nagłówka */
        unsigned long int size;/**< rozmiar nagłówka */
    } header;
    struct Junk
    {
        Junk();
        char* pt;/**< wskaźnik na ostatno znalezioną sekcję JUNK */
        unsigned long int position;/**< pozycja ostatnio znalezionej sekcji JUNK */
        unsigned long int number;/**< numer sekcji JUNK */
        unsigned long int size;/**< całkowity rozmiar sekcji JUNK */
        bool found;/**< czy sekcja JUNK została znaleziona przed nagłówkiem */
        unsigned long int hSize;/**< rozmiar nagłówka sekcji JUNK */
    } junk;
    struct Frame
    {
        Frame();
        unsigned long int size;
        char* pt;
        unsigned long int position;
        bool found;
    } frame;
    bool emptyLeft;/**< czy lewa strona danych jest niezapełniona */
    bool emptyRight;/**< czy prawa strona danych jest niezapełniona */
    void findNextHeader();/**< wyszukanie następnego nagłówka */
    void cycleDataSpace();/**< przesunięcie danych w lewo i doczytanie nowych */
    void loadLeft();/**< wczytanie danych z bufora cyklicznego do lewej strony danych */
    void loadRight();/**< wczytanie danych z bufora cyklicznego do prawej strony danych */
};
