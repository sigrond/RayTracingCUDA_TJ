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

/** \brief klasa wyjątku dla bufora cyklicznego
 */
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

/** \brief struktura zawierajaca wskaźnik na bufor i opisująca ten bufor
 */
struct buffId
{
    buffId(int id,char* pt):id(id),frameNo(-1),pt(pt) {};
    buffId(int id,char* pt,int frameNo):id(id),frameNo(frameNo),pt(pt) {};
    int id;
    int frameNo;
    char* pt;
};


#define CBUFFS 32
#define ERRNUM 10
namespace ErrorCode
{
    /** \brief kody błędów bufora cyklicznego
     */
    enum ErrorCode
    {
        BufferOverflow=0,
        cBegIsNotcEndAtitemCount0=1,
        ClaimForReadOfNotExistingElement=2,
        ReadEndOfNotExistingElement=3,
        ReadEndWithNegativeNumberOfElements=4,
        ReadEndAndNextBuffIsNotBeingWriten=5,
        ReadEndCousedNegativeNumberOfElements=6,
    };
    /** \brief opisy błędów bufora cyklicznego
     */
    static const char* ErrorNames[ERRNUM]={
        "BufferOverflow",
        "cBegIsNotcEndAtitemCount0",
        "ClaimForReadOfNotExistingElement",
        "ReadEndOfNotExistingElement",
        "ReadEndWithNegativeNumberOfElements",
        "ReadEndAndNextBuffIsNotBeingWriten",
        "ReadEndCousedNegativeNumberOfElements",
        "",
        "",
        ""
        };
}
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
    int buffReady[CBUFFS];/**< czy bufor nie jest już używany
    1 - gotowy pusty
    2 - gotowy pełny
    -1 - używany do zapisu
    -2 - używany do odczytu */
    int frameNo[CBUFFS];/**< numery klatek pomogą znaleźć błędy */
    std::condition_variable buffReadyCond[CBUFFS];
    char* cBuff[CBUFFS];/**< bufor cykliczny z buforami odczytu z dysku */
    std::condition_variable monitorCond;
    std::mutex monitorMtx;
    int errorCount[ERRNUM];
    /** \brief wypisuje status bez blokowania
     * \return void
     */
    void _printStatus();
    float averageLoad;
    float loadCount;
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
    /** \brief blokuje i wypisuje informacje o buforze
     * \return void
     */
    void printStatus();
};
