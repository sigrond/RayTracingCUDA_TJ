/** \file CyclicBuffer.hpp
 * \author Tomasz Jakubczyk
 * \brief plik nagłówkowy klasy monitora z buforem cyklicznym
 *
 *
 *
 */

#include <mutex>
#include <condition_variable>
#include <exception>
#include <string>

class CyclicBufferException : public std::exception
{
private:
    std::string str;
public:
    CyclicBufferException(std::string s) : str(s){}
    virtual char const * what() const
    {
        return str.c_str();
    }
};

struct buffId
{
    buffId(int id,char* pt):id(id),frameNo(-1),pt(pt) {};
    buffId(int id,char* pt,int frameNo):id(id),frameNo(frameNo),pt(pt) {};
    int id;
    int frameNo;
    char* pt;
};


#define CBUFFS 32
/** \brief monitor dla bufora cyklicznego
 */
class CyclicBuffer
{
private:
    const int cBuffS;/**< rozmiar bufora cyklicznego */
    int cBeg,cEnd;/**< początek i koniec bufora cyklicznego */
    int itemCount;
    std::condition_variable full;/**< bufor cykliczny pełny */
    std::condition_variable empty;/**< bufor cykliczny pusty */
    bool buffReady[CBUFFS];/**< czy bufor nie jest już używany */
    int frameNo[CBUFFS];/**< numery klatek pomogą znaleźć błędy */
    std::condition_variable buffReadyCond[CBUFFS];
    char* cBuff[CBUFFS];/**< bufor cykliczny z buforami odczytu z dysku */
    std::condition_variable monitorCond;
    std::mutex monitorMtx;
    int errorCount;
public:
    /** \brief konstruktor
     */
    CyclicBuffer();
    /** \brief destruktor
     */
    ~CyclicBuffer();
    /** \brief zajmij wskaźnik bufora do zapisu
     * \return char*
     */
    buffId* claimForWrite();
    /** \brief zwolnienie bufora po zapisaniu
     * \param id buffId*
     * \return void
     */
    void writeEnd(buffId* id);
    /** \brief zajmij wskaźnik bufora do odczytu
     * \return buffId*
     */
    buffId* claimForRead();
    /** \brief zwolnienie bufora po odczytaniu
     * \param id buffId*
     * \return void
     */
    void readEnd(buffId* id);
    /** \brief zwraca aktualną chwilową liczbę elementów
     * \return int itemCount
     */
    int tellItemCount();
};
