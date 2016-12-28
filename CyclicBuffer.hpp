/** \file CyclicBuffer.hpp
 * \author Tomasz Jakubczyk
 * \brief header file for monitor class with cyclic buffer
 *
 *
 *
 */

#pragma once

#include <mutex>
#include <condition_variable>
#include <exception>
#include <string>

/** \brief exception class for cyclic buffer
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

/** \brief structure containing pointer to buffer and describing it
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
    /** \brief cyclic buffer error codes
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
    /** \brief cyclic bufer errors descriptions
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
/** \brief monitor for cyclic buffer
 */
class CyclicBuffer
{
private:
    const int cBuffS;/**< size of cyclic buffer */
    int cBeg,cEnd;/**< beginig and end of cyclic buffer */
    int itemCount;
    std::condition_variable full;/**< cyclic buffer full */
    std::condition_variable empty;/**< cyclic buffer empty */
    int buffReady[CBUFFS];/**< is buffer already used
    1 - ready empty
    2 - ready full
    -1 - used for writing
    -2 - used for reading */
    int frameNo[CBUFFS];/**< numbers of frames may help to find errors */
    std::condition_variable buffReadyCond[CBUFFS];
    char* cBuff[CBUFFS];/**< cyclic buffer of buffers for reading from hard drive */
    std::condition_variable monitorCond;
    std::mutex monitorMtx;
    int errorCount[ERRNUM];
    /** \brief prints status without blocking
     * \return void
     */
    void _printStatus();
    float averageLoad;
    float loadCount;
public:
    /** \brief constructor
     */
    CyclicBuffer();
    /** \brief destructor
     */
    ~CyclicBuffer();
    /** \brief claim buffer pointer for writing
     * \return char*
     */
    buffId* claimForWrite();
    /** \brief unclaim buffer after writing
     * \param id buffId*
     * \return void
     */
    void writeEnd(buffId* id);
    /** \brief claim buffer pointer for reading
     * \return buffId*
     */
    buffId* claimForRead();
    /** \brief unclaim buffer after reading
     * \param id buffId*
     * \return void
     */
    void readEnd(buffId* id);
    /** \brief return current temporary number of elements
     * \return int itemCount
     */
    int tellItemCount();
    /** \brief block and print information about buffer
     * \return void
     */
    void printStatus();
};
