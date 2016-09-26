/** \file FrameReader.cpp
 * \author Tomasz Jakubczyk
 * \brief Plik z implementacjami metod klasy FrameReader
 *
 */

#include "FrameReader.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif // MATLAB_MEX_FILE
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

using namespace std;

extern thread::id mainThreadId;

int reDecodedFrames=0;

int microSearchCount=0;

mutex printmMutex;
/** \brief wypisanie tekstu do matlaba
 *
 * \param c const char*
 * \return void
 *
 */
void printm(const char* c)
{
    try
    {
        //lock_guard<mutex> lck(printmMutex);//ma chronić przed zaklesczeniem przy wyjątku
        printf("%s\n",c);
        #ifdef MATLAB_MEX_FILE
        if(mainThreadId==this_thread::get_id())
        mexEvalString("pause(.001);");
        #endif // MATLAB_MEX_FILE
    }
    catch(const exception& e)
    {
        printf("wyjątek (exception) printm: %s !!!\n\n\n\n\n\n\n",e.what());
        throw;
    }
    catch(...)
    {
        printf("nieznany wyjątek printm !!!\n\n\n\n\n\n\n");
        throw;
    }
}

const char frameStartCode[8]={'0','0','d','b',0x00,0x60,0x09,0x00};
const char frameStartCodeS[8]={'0','0','d','c',0x00,0x60,0x09,0x00};
const char junkCode[]="JUNK";
unsigned long long int safetyCounter=0;

FrameReader::CorrectnessControl* FrameReader::correctnessControl=nullptr;

FrameReader::FrameReader(CyclicBuffer* c) :
    cyclicBuffer(c),dataSpace(nullptr),emptyLeft(true),emptyRight(true)
{
    #ifdef DEBUG
    printm("FrameReader::FrameReader(CyclicBuffer* c)");
    #endif // DEBUG
    dataSpace=new DataSpace(65535*10*2);
    correctnessControl=new CorrectnessControl();
}

FrameReader::~FrameReader()
{
    #ifdef DEBUG
    printm("FrameReader::~FrameReader()");
    #endif // DEBUG
    delete dataSpace;
    delete correctnessControl;
}

void FrameReader::loadLeft()
{
    #ifdef DEBUG
    printm("void FrameReader::loadLeft()");
    #endif // DEBUG
    buffId* tmpBuff=nullptr;
    #ifdef DEBUG
    printStatus();
    #endif // DEBUG
    if(cyclicBuffer==nullptr)
    {
        #ifdef DEBUG
        printm("cyclicBuffer==nullptr");
        #endif // DEBUG
        throw FrameReaderException("cyclicBuffer==nullptr");
    }
    tmpBuff=cyclicBuffer->claimForRead();
    if(tmpBuff==nullptr)
    {
        #ifdef DEBUG
        printm("tmpBuff==nullptr");
        #endif // DEBUG
        throw FrameReaderException("tmpBuff==nullptr");
    }
    #ifdef DEBUG
    printf("tmpBuff->frameNo: %d\n",tmpBuff->frameNo);
    printm("tmpBuff=cyclicBuffer->claimForRead()");
    #endif // DEBUG
    if(dataSpace->pt==nullptr)
    {
        #ifdef DEBUG
        printm("dataSpace->pt==nullptr");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->pt==nullptr");
    }
    if(tmpBuff->pt==nullptr)
    {
        #ifdef DEBUG
        printm("tmpBuff->pt==nullptr");
        #endif // DEBUG
        throw FrameReaderException("tmpBuff->pt==nullptr");
    }
    if(dataSpace->halfSize<=0)
    {
        #ifdef DEBUG
        printm("dataSpace->halfSize<=0");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->halfSize<=0");
    }
    if(dataSpace->halfSize>655350)
    {
        #ifdef DEBUG
        printm("dataSpace->halfSize>655350");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->halfSize>655350");
    }
    memcpy(dataSpace->pt,tmpBuff->pt,dataSpace->halfSize);
    #ifdef DEBUG
    printm("memcpy(dataSpace->pt,tmpBuff->pt,dataSpace->halfSize)");
    #endif // DEBUG
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyLeft=false;
}

void FrameReader::loadRight()
{
    #ifdef DEBUG2
    printm("void FrameReader::loadRight()");
    #endif // DEBUG
    buffId* tmpBuff=nullptr;
    tmpBuff=cyclicBuffer->claimForRead();
    if(dataSpace->ptRight==nullptr)
    {
        #ifdef DEBUG
        printm("dataSpace->ptRight==nullptr");
        #endif // DEBUG
        throw FrameReaderException("dataSpace->ptRight==nullptr");
    }
    memcpy(dataSpace->ptRight,tmpBuff->pt,dataSpace->halfSize);
    cyclicBuffer->readEnd(tmpBuff);
    tmpBuff=nullptr;
    emptyRight=false;
}

void FrameReader::cycleDataSpace()
{
    #ifdef DEBUG2
    printm("void FrameReader::cycleDataSpace()");
    #endif // DEBUG
    if(header.position>=dataSpace->halfSize)
    {
        header.position-=dataSpace->halfSize;
        header.pt-=dataSpace->halfSize;
        memcpy(dataSpace->ptLeft,dataSpace->ptRight,dataSpace->halfSize);
        loadRight();
    }
}

char* FrameReader::getFrame()
{
    #ifdef DEBUG2
    printm("char* FrameReader::getFrame()");
    #endif // DEBUG
    if(emptyRight && !emptyLeft)
    {
        loadRight();
    }
    if(emptyLeft)
    {
        loadLeft();
    }
    if(header.position>=dataSpace->halfSize)
    {
        cycleDataSpace();
    }
    if(!header.found)/**< jeśli jeszcze nie został znaleziony nagłówek */
    {
        findNextHeader();
        #ifdef DEBUG
        printStatus();
        #endif // DEBUG
    }
    if(header.position-junk.size<frame.size)/**< jeśli przed nagłówkiem nie ma miejsca na klatkę to trzeba znaleźć następny nagłówek */
    {
        findNextHeader();
    }
    else if(header.position+frame.size+junk.size+header.size<(emptyRight?dataSpace->halfSize:dataSpace->size))
    {
        findNextHeader();
    }
    else if(!emptyRight)
    {
        printm("to nie powinno się wydażyć #001");
        printStatus();
        system("PAUSE");
        throw FrameReaderException("to nie powinno się wydażyć #001");
    }
    frame.position=(junk.found?junk.position:header.position)-frame.size;
    if(frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size)
    {
        #ifdef DEBUG
        printm("frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size");
        #endif // DEBUG
        throw FrameReaderException("frame.position+frame.size+(junk.found?junk.size:0)+header.size>dataSpace->size");
    }
    frame.pt=dataSpace->pt+frame.position;
    frame.found=true;

    correctnessControl->addFrame(dataSpace,&header,&junk,&frame);

    #ifdef DEBUG
    if(safetyCounter++<=5)
    {
        printStatus();
        //system("pause");
    }
    #endif // DEBUG
    return frame.pt;
}

void FrameReader::findNextHeader()
{
    #ifdef DEBUG2
    printm("void FrameReader::findNextHeader()");
    #endif // DEBUG
    bool headerB=true;
    bool junkB=true;
    junk.found=false;
    if(header.found)
    {
        header.position+=frame.size;/**< najwcześniejsza możliwa pozycja następnego nagłówka */
    }

    for(int j=header.found?(header.position+header.size):header.position;
    j<(header.found?(dataSpace->size-header.size):dataSpace->halfSize-header.size);
    j++)
    {
        junkB=true;
        for(int i=0;i<junk.hSize;i++)
        {
            junkB&=junkCode[i]==dataSpace->pt[j+i];
            if(!junkB)
            {
                break;
            }
        }
        if(junkB)
        {
            junk.found=true;
            junk.number++;
            junk.position=j;
            junk.pt=dataSpace->pt+junk.position;
            junk.size=*(int*)(dataSpace->pt+j+junk.hSize);
            junk.size+=junk.hSize;
            j+=junk.size;
        }
        headerB=true;
        for(int i=0;i<header.size;i++)
        {
            headerB&=frameStartCode[i]==dataSpace->pt[j+i] || frameStartCodeS[i]==dataSpace->pt[j+i];
            if(!headerB)
            {
                break;
            }
        }
        if(headerB)
        {
            header.found=true;
            header.number++;
            header.position=j;
            header.pt=dataSpace->pt+header.position;
            break;
        }
    }
    if(!headerB)
    {
        header.found=false;
    }
}

FrameReader::DataSpace::DataSpace(unsigned long int s) :
    pt(nullptr), size(s), ptLeft(nullptr), ptRight(nullptr), halfSize(s/2)
{
    #ifdef DEBUG
    printm("FrameReader::DataSpace::DataSpace(unsigned long int s)");
    #endif // DEBUG
    pt=new char[size];
    ptLeft=pt;
    ptRight=pt+size/2;

}

FrameReader::DataSpace::DataSpace(const DataSpace& o) :
    pt(nullptr), size(o.size), ptLeft(nullptr), ptRight(nullptr), halfSize(o.halfSize)
{
    pt=new (nothrow) char[size];
    if(pt==nullptr)
    {
        printm("Low memory! Buy more RAM! :D");
        this_thread::sleep_for (chrono::seconds(1));
        pt=new (nothrow) char[size];
    }
    while(pt==nullptr)
    {
        //this_thread::sleep_for (chrono::seconds(1));
        pt=new (nothrow) char[size];
    }
    ptLeft=pt;
    ptRight=pt+size/2;
    memcpy(pt,o.pt,size);
}

FrameReader::DataSpace::~DataSpace()
{
    #ifdef DEBUG
    printm("FrameReader::DataSpace::~DataSpace()");
    #endif // DEBUG
    delete[] pt;
}

FrameReader::Header::Header() :
    pt(nullptr), position(0), found(false), number(0), size(8)
{
    #ifdef DEBUG
    printm("FrameReader::Header::Header()");
    #endif // DEBUG
}

FrameReader::Junk::Junk() :
    pt(nullptr), position(0), number(0), size(0), found(false), hSize(4)
{
    #ifdef DEBUG
    printm("FrameReader::Junk::Junk()");
    #endif // DEBUG
}

FrameReader::Frame::Frame() :
    pt(nullptr), position(0),found(false)
{
    #ifdef DEBUG
    printm("FrameReader::Frame::Frame()");
    #endif // DEBUG
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameReader::Frame::Frame()");
    #endif // DEBUG
}

void FrameReader::printStatus()
{
    printf("FrameReader this: %p\n",this);
    printm("void FrameReader::printStatus()");
    printf("cyclicBuffer: 0x%p\n",cyclicBuffer);
    printf("dataSpace: 0x%p\n",dataSpace);
    printf("dataSpace->pt: 0x%p\n",dataSpace->pt);
    printf("dataSpace->size: %lu\n",dataSpace->size);
    printf("dataSpace->ptLeft: 0x%p\n",dataSpace->ptLeft);
    printf("dataSpace->ptRight: 0x%p\n",dataSpace->ptRight);
    printf("dataSpace->halfSize: %lu\n",dataSpace->halfSize);
    printf("header.pt: 0x%p\n",header.pt);
    printf("header.position: %lu\n",header.position);
    printf(header.found?"header.found: true\n":"header.found: false\n");
    printf("header.number: %lu\n",header.number);
    printf("header.size: %lu\n",header.size);
    printf("junk.pt: 0x%p\n",junk.pt);
    printf("junk.position: %lu\n",junk.position);
    printf("junk.number: %lu\n",junk.number);
    printf("junk.size: %lu\n",junk.size);
    printf(junk.found?"junk.found: true\n":"junk.found: false\n");
    printf("junk.hSize: %lu\n",junk.hSize);
    printf("frame.size: %lu\n",frame.size);
    printf("frame.pt: 0x%p\n",frame.pt);
    printf("frame.position: %lu\n",frame.position);
    printf(frame.found?"frame.found: true\n":"frame.found: false\n");
    printf(emptyLeft?"emptyLeft: true\n":"emptyLeft: false\n");
    printf(emptyRight?"emptyRight: true\n":"emptyRight: false\n");
    printm("void FrameReader::printStatus() end");
}

FrameReader::CorrectnessControl::CorrectnessControl() :
    lastFrameCorrect(true), decodedFrame(nullptr)
{
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameReader::CorrectnessControl::CorrectnessControl(const FrameReader& parent)");
    printf("Frame::size: %d\n",Frame::size);
    printm("OK");
    #endif // DEBUG_CORRECTNESSCONTROL
    decodedFrame=new char[Frame::size];
    for(int i=0;i<ERRNUM2;i++)
    {
        errorCount[i]=0;
    }
}

FrameReader::CorrectnessControl::~CorrectnessControl()
{
    using namespace SetToReDecode;
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameReader::CorrectnessControl::~CorrectnessControl()");
    #endif // DEBUG_CORRECTNESSCONTROL
    if(!q.empty())
    {
        printm("kolejka korekcji poprawności nie jest pusta");
        while(!q.empty())
        {
            delete q.front();
            q.pop();
        }
        //throw FrameReaderException("kolejka korekcji poprawności nie jest pusta");
    }
    printf("reDecodedFrames: %d\n",reDecodedFrames);
    for(int i=0;i<ERRNUM;i++)
    {
        if(errorCount[i]!=0)
        printf("errorCount[%d]: %d - %s\n",i,errorCount[i],ErrorNames[i]);
    }
    printf("microSearchCount: %d\n",microSearchCount);
    delete[] decodedFrame;
}

FrameReader::CorrectnessControl::FrameData::FrameData(DataSpace* d,Header* h,Junk* j,Frame* f) :
    dataSpacePt(nullptr),headerPt(nullptr),junkPt(nullptr),framePt(nullptr)
{
    #ifdef DEBUG_CORRECTNESSCONTROL2
    printm("FrameReader::CorrectnessControl::FrameData::FrameData(DataSpace* d,Header* h,Junk* j,Frame* f)");
    #endif // DEBUG_CORRECTNESSCONTROL
    dataSpacePt=new DataSpace(*d);
    headerPt=new Header(*h);
    junkPt=new Junk(*j);
    framePt=new Frame(*f);
}

FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)
{
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)");
    #endif // DEBUG_CORRECTNESSCONTROL
    printm("FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)");
    throw FrameReaderException("FrameReader::CorrectnessControl::FrameData::FrameData(const FrameData& o)");
}

FrameReader::CorrectnessControl::FrameData::~FrameData()
{
    #ifdef DEBUG_CORRECTNESSCONTROL2
    printm("FrameReader::CorrectnessControl::FrameData::~FrameData()");
    #endif // DEBUG_CORRECTNESSCONTROL
    delete dataSpacePt;
    dataSpacePt=nullptr;
    delete headerPt;
    headerPt=nullptr;
    delete junkPt;
    junkPt=nullptr;
    delete framePt;
    framePt=nullptr;
}

void FrameReader::CorrectnessControl::addFrame(DataSpace* d,Header* h,Junk* j,Frame* f)
{
    #ifdef DEBUG_CORRECTNESSCONTROL2
    printm("void FrameReader::CorrectnessControl::addFrame(DataSpace* d,Header* h,Junk* j,Frame* f)");
    #endif // DEBUG_CORRECTNESSCONTROL
    while(!q.empty())
    this_thread::yield();
    FrameData* frameData=new FrameData(d,h,j,f);
    unique_lock<mutex> lck(m);
    q.push(frameData);
    lck.unlock();
    empty.notify_one();
}


bool FrameReader::CorrectnessControl::checkFrame()
{
    using namespace SetToReDecode;
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("bool FrameReader::CorrectnessControl::checkFrame()");
    #endif // DEBUG_CORRECTNESSCONTROL
    if(this==nullptr)
    {
        printm("FrameReader::CorrectnessControl::checkFrame() this==nullptr");
        throw FrameReaderException("FrameReader::CorrectnessControl::checkFrame() this==nullptr");
    }
    try
    {
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("checkFrame try lck0");
        #endif // DEBUG_CORRECTNESSCONTROL
        unique_lock<mutex> lck0(m);
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("lck0 locked");
        #endif // DEBUG_CORRECTNESSCONTROL
        lck0.unlock();
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("lck0 unlocked");
        #endif // DEBUG_CORRECTNESSCONTROL
        lck0.release();
    }
    catch(...)
    {
        printm("unique_lock<mutex> lck0(m) throwed smth");
        delete q.front();
        q.pop();
        return true;
    }
    unique_lock<mutex> lck(m);
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("unique_lock<mutex> lck(m);");
    #endif // DEBUG_CORRECTNESSCONTROL
    while(q.empty())
    {
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("while(q.empty())");
        #endif // DEBUG_CORRECTNESSCONTROL
        empty.wait(lck);
    }
    FrameData* frameData=q.front();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameData* frameData=q.front();");
    #endif // DEBUG_CORRECTNESSCONTROL
    lck.unlock();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("lck.unlock();");
    #endif // DEBUG_CORRECTNESSCONTROL
    bool headerB=true;
    bool junkB=true;
    bool microSearch=false;
    char* tmpPt=frameData->dataSpacePt->pt;
    for(int j=0;j<frameData->dataSpacePt->size;j++)
    {
        junkB=true;
        for(int i=0;i<frameData->junkPt->hSize;i++)
        {
            junkB&=junkCode[i]==tmpPt[j+i];
        }
        if(junkB)
        {
            frameData->junkV.emplace_back();
            frameData->junkV.back().position=j;
            frameData->junkV.back().size=*(int*)(tmpPt+j+4);
            frameData->junkV.back().size+=4;
            j+=frameData->junkV.back().size;
            microSearch=false;
        }
        headerB=true;
        for(int i=0;i<frameData->headerPt->size;i++)
        {
            headerB&=frameStartCode[i]==tmpPt[j+i] ||
             frameStartCodeS[i]==tmpPt[j+i];
        }
        if(headerB)
        {
            frameData->headerV.emplace_back();
            frameData->headerV.back().position=j;
            j+=frameData->headerV.back().size;
        }
    }

    if(frameData->headerV.empty())
    {
        if(frameData->headerPt->found)
        {
            frameData->headerV.emplace_back();
            frameData->headerV.back().position=frameData->framePt->position;
        }
    }

    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("przeglądanie dataSpace zakończone");
    #endif // DEBUG_CORRECTNESSCONTROL
    #ifdef DEBUG_CORRECTNESSCONTROL
    printf("frameData->headerV.size(): %d\n",frameData->headerV.size());
    printf("frameData->junkV.size(): %d\n",frameData->junkV.size());
    for(int i=0;i<frameData->headerV.size();i++)
    {
        printf("frameData->headerV.at(i).position: %d\n",frameData->headerV.at(i).position);
    }
    for(int j=0;j<frameData->junkV.size();j++)
    {
        printf("frameData->junkV.at(j).position: %d\n",frameData->junkV.at(j).position);
    }
    printm("znalezione znaczniki end");
    #endif // DEBUG_CORRECTNESSCONTROL
    for(int i=0;i<frameData->headerV.size();i++)
    {
        if(frameData->headerV.at(i).position==frameData->headerPt->position)
        {//nagłówek znaleziony na wcześniej ustalonej pozycji
            if(frameData->junkV.empty())
            {//nie znaleziono żadnego JUNK
                if(frameData->junkPt->found)
                {//było zaznaczone, że wcześniej znaleziono w danym segmncie JUNK
                    lastFrameCorrect=false;
                    errorCount[junkSetAsFoundAndThenNotFound]++;
                    return false;///dziwny przrypadek, raczej nie powinien się zdażyć
                }
                else
                {
                    if(frameData->headerV.at(i).position>=frameData->framePt->size)
                    {//przed nagłówkiem jest dość miejsca na klatkę
                        if(i==0)
                        {//jest to pierwszy nagłówek z kolei
                            lck.lock();
                            delete q.front();
                            q.pop();
                            lck.unlock();
                            lastFrameCorrect=true;
                            return true;///nie ma JUNK, nie ma wcześniejszego nagłówka, a klatka w całości mieści się przed nagłówkiem
                        }
                        else if(frameData->headerV.at(i-1).position+frameData->headerV.at(i-1).size+frameData->framePt->size<=frameData->headerV.at(i).position)
                        {//między nagłówkami jest dość miejsca na klatkę
                            lck.lock();
                            delete q.front();
                            q.pop();
                            lck.unlock();
                            lastFrameCorrect=true;
                            return true;///nie ma JUNK, a klatka w całości zmieśiła się mi,ędzy nagłówkami
                        }
                        else
                        {
                            lastFrameCorrect=false;
                            errorCount[notEnoughPlaceForFrameBetweenHeaders]++;
                            return false;///za mało miejsca na całą klatkę pomiędzy nagłówkami klatek, to było by dziwne
                        }
                    }
                    else
                    {
                        lastFrameCorrect=false;
                        errorCount[frameIsCut]++;
                        return false;///urżnięta klatka, raczej nie powinno sie wydażyć
                    }
                }
            }
            else
            {
                for(int j=0;j<frameData->junkV.size();j++)
                {
                    if(frameData->junkV.at(j).position==frameData->junkPt->position)
                    {//zaneziono wykożystany JUNK
                        #ifdef EXTRA_DEBUG1
                        printf("frameData->junkV.at(j).position: %d\n",frameData->junkV.at(j).position);
                        printf("frameData->junkV.at(j).size: %d\n",frameData->junkV.at(j).size);
                        printf("frameData->headerPt->position: %d\n",frameData->headerPt->position);
                        #endif // EXTRA_DEBUG1
                        if(frameData->junkV.at(j).position+frameData->junkV.at(j).size+frameData->junkV.at(j).hSize==frameData->headerPt->position)
                        {//JUNK jest przyklejony do odpowiadającego nagłówka
                            if(j==0)
                            {//jest to pierwszy JUNK z kolei
                                if(frameData->junkV.at(j).position>=frameData->framePt->size)
                                {//przed JUNK jest dość miejsca na klatkę
                                    if(i==0)
                                    {//i jest to pierwszy nagłówek z kolei
                                        lck.lock();
                                        delete q.front();
                                        q.pop();
                                        lck.unlock();
                                        lastFrameCorrect=true;
                                        return true;///pierwszy nagłowek i pierwszy JUNK sklejone, a przed nimi jest dość miejsca na całą klatkę
                                    }
                                    else
                                    {//wcześniej był inny nagłówek
                                        if(frameData->junkV.at(j).position-(frameData->headerV.at(i).position+frameData->headerPt->size)>=frameData->framePt->size)
                                        {//między JUNK a header jest dość miejsca na klatkę
                                            lck.lock();
                                            delete q.front();
                                            q.pop();
                                            lck.unlock();
                                            lastFrameCorrect=true;
                                            return true;///między JUNK(przyklejonym do nagłówka), a poprzednim nagłówkiem jest dość miejsca na klatkę i nie ma tam dodatkowego JUNK
                                        }
                                        else
                                        {
                                            lastFrameCorrect=false;
                                            errorCount[notEnoughPlaceForFrameBetweenJunkAndPreviosHeader]++;
                                            return false;///między JUNK, a poprzednim nagłówkiem jest za mało miejsca na całą klatkę, dziwny przypadek, raczej nie powinien się wydażyć
                                        }
                                    }
                                }
                                else
                                {
                                    lastFrameCorrect=false;
                                    errorCount[frameIsCutBecauseNotEnoughPlaceBeforeJunk]++;
                                    return false;///urżnięta klatka, bo przed JUNK jest za mało miejsca, raczej nie powinno się wydażyć
                                }
                            }
                            else
                            {//nie pierwszy JUNK z kolei
                                if(i==0)
                                {//pierwszy nagłówek z kolei
                                    if(frameData->junkV.at(j-1).position+frameData->junkV.at(j-1).size+frameData->framePt->size<=frameData->junkV.at(j).position)
                                    {//między JUNK przyklejonym do nagłówka a poprzednim JUNK jest dość miejsca na klatkę
                                        lck.lock();
                                        delete q.front();
                                        q.pop();
                                        lck.unlock();
                                        lastFrameCorrect=true;
                                        return true;///oznaczało by to, że bezpośrednio za nagłówkiem, który się nie zmieścił w bloku danych był JUNK i było dość miejsca na klatkę do następnego JUNK przyklejonego do nagłówka
                                    }
                                    else
                                    {
                                        lastFrameCorrect=false;
                                        errorCount[junkInsideFrameAndPreviousHeaderDidntFitInDataSpace]++;
                                        return false;///JUNK wystąpił w śrdoku klatki i wcześniejszy nagłówek nie zmieścił się w DataSpace
                                    }
                                }
                                else
                                {//nie pierwszy nagłówek z kolei
                                    if(frameData->junkV.at(j-1).position+frameData->junkV.at(j-1).size+frameData->framePt->size<=frameData->junkV.at(j).position)
                                    {
                                        lck.lock();
                                        delete q.front();
                                        q.pop();
                                        lck.unlock();
                                        lastFrameCorrect=true;
                                        return true;///przed JUNK przyklejonym do nagłówka było dość miejsca na klatkę
                                    }
                                    else
                                    {
                                        lastFrameCorrect=false;
                                        errorCount[junkInsideFrame]++;
                                        return false;///JUNK wystąpił w środku klatki
                                    }
                                }
                            }
                        }
                        else
                        {
                            lastFrameCorrect=false;
                            errorCount[junkDidntStickToHeader]++;
                            return false;///JUNK nie przyklejony do nagłówka klatki
                        }
                    }
                }
            }
        }
    }
    lastFrameCorrect=false;
    errorCount[didntFindPreviouslyFoundHeaderOrDidntFindPreviouslyFoundJunkOrOtherError]++;
    return false;///nie znalezion wcześniej znalezionego nagłówka, albo nie znaleziono wcześniej znalezionego JUNK, albo inny błąd
}

char* FrameReader::CorrectnessControl::decodeFrame()
{
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("char* FrameReader::CorrectnessControl::decodeFrame()");
    #endif // DEBUG_CORRECTNESSCONTROL
    if(lastFrameCorrect)
    {
        printm("próbujemy zdekodować ponownie klatkę oznaczoną jako poprawnie zdekodowaną");
        throw FrameReaderException("próbujemy zdekodować ponownie klatkę oznaczoną jako poprawnie zdekodowaną");
    }
    unique_lock<mutex> lck(m);
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("unique_lock<mutex> lck(m);");
    #endif // DEBUG_CORRECTNESSCONTROL
    while(q.empty())
    {
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("while(q.empty())");
        #endif // DEBUG_CORRECTNESSCONTROL
        empty.wait(lck);
    }
    FrameData* frameData=q.front();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("FrameData* frameData=q.front();");
    #endif // DEBUG_CORRECTNESSCONTROL
    lck.unlock();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printf("frameData->headerPt->position: %d\n",frameData->headerPt->position);
    #endif // DEBUG_CORRECTNESSCONTROL
    #ifdef DEBUG_CORRECTNESSCONTROL
    printf("lck.unlock();");
    #endif // DEBUG_CORRECTNESSCONTROL
    if(frameData->headerV.empty())
    {
        lck.lock();
        delete q.front();
        q.pop();
        lck.unlock();
        printm("duża przestrzeń bez klatek?");
        return nullptr;///duża przestrzeń bez klatek?
    }
    reDecodedFrames++;
    #ifdef EXTRA_DEBUG2
    printf("frameData->headerV.size(): %d\n",frameData->headerV.size());
    printf("frameData->junkV.size(): %d\n",frameData->junkV.size());
    for(int i=0;i<frameData->headerV.size();i++)
    {
        printf("frameData->headerV.at(i).position: %d\n",frameData->headerV.at(i).position);
    }
    for(int j=0;j<frameData->junkV.size();j++)
    {
        printf("frameData->junkV.at(j).position: %d\n",frameData->junkV.at(j).position);
    }
    #endif // EXTRA_DEBUG2
    if(frameData->junkV.empty())
    {
        if(frameData->headerPt->position>=frameData->framePt->size)
        {
            printm("klatka która powinna być dobrze zdekodowana, została oznaczona jako źle zdekodowana");
            memcpy(decodedFrame,frameData->dataSpacePt->pt,frameData->framePt->size);
            lck.lock();
            delete q.front();
            q.pop();
            lck.unlock();
            return decodedFrame;
        }
        else
        {
            printm("utrata części klatki?");
            memcpy(decodedFrame+(frameData->framePt->size-frameData->headerPt->position),frameData->dataSpacePt->pt,frameData->headerPt->position);
            lck.lock();
            delete q.front();
            q.pop();
            lck.unlock();
            return decodedFrame;
        }
    }
    int copiedFrame=0;/**< ile klatki zostało skopiowane */
    int lastSkipedPosition=frameData->headerPt->position;/**< ostatnio ominięty fragment */
    #ifdef DEBUG_CORRECTNESSCONTROL
    printf("lastSkipedPosition: %d\n",lastSkipedPosition);
    #endif // DEBUG_CORRECTNESSCONTROL
    int sizeToCopy=0;
    int dstOffset=0;
    int srcOffset=0;
    for(int i=0;i<frameData->headerV.size();i++)
    {
        if(frameData->headerV.at(i).position==frameData->headerPt->position)
        {//jest przynajmniej jeden JUNK i header
            #ifdef DEBUG_CORRECTNESSCONTROL
            printm("(1) jest przynajmniej jeden JUNK i header");
            #endif // DEBUG_CORRECTNESSCONTROL
            for(int j=frameData->junkV.size()-1;j>=0;j--)
            {
                if(frameData->junkV.at(j).position<frameData->headerPt->position)
                {//JUNK jest przed nagłówkiem
                    #ifdef DEBUG_CORRECTNESSCONTROL
                    printm("(2) JUNK jest przed nagłówkiem");
                    #endif // DEBUG_CORRECTNESSCONTROL
                    if(frameData->junkV.at(j).position+frameData->junkV.at(j).size>=frameData->headerPt->position)
                    {//JUNK przyklejony do nagłówka
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printm("(3) JUNK przyklejony do nagłówka");
                        #endif // DEBUG_CORRECTNESSCONTROL
                        lastSkipedPosition=frameData->junkV.at(j).position;
                    }
                    else
                    {
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printm("(4) JUNK nie przyklejony do nagłówka");
                        #endif // DEBUG_CORRECTNESSCONTROL
                        if(copiedFrame>=frameData->framePt->size)
                        {//klatka została już skopiowana
                            #ifdef DEBUG_CORRECTNESSCONTROL
                            printm("(5) klatka została już skopiowana");
                            #endif // DEBUG_CORRECTNESSCONTROL
                            lck.lock();
                            delete q.front();
                            q.pop();
                            lck.unlock();
                            return decodedFrame;
                        }
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printm("(6) kopiujemy fragment klatki");
                        #endif // DEBUG_CORRECTNESSCONTROL
                        sizeToCopy=(lastSkipedPosition-(frameData->junkV.at(j).position+frameData->junkV.at(j).size))<=(frameData->framePt->size-copiedFrame)?(lastSkipedPosition-(frameData->junkV.at(j).position+frameData->junkV.at(j).size)):(frameData->framePt->size-copiedFrame);
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printf("sizeToCopy: %d\n",sizeToCopy);
                        #endif // DEBUG_CORRECTNESSCONTROL
                        if(sizeToCopy<0)
                        {
                            printm("sizeToCopy<0");
                            throw FrameReaderException("sizeToCopy<0");
                        }
                        if(sizeToCopy>frameData->framePt->size)
                        {
                            printm("sizeToCopy>frameData->framePt->size");
                            throw FrameReaderException("sizeToCopy>frameData->framePt->size");
                        }
                        dstOffset=frameData->framePt->size-copiedFrame-sizeToCopy;
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printf("dstOffset: %d\n",dstOffset);
                        #endif // DEBUG_CORRECTNESSCONTROL
                        if(dstOffset<0)
                        {
                            printm("dstOffset<0");
                            throw FrameReaderException("dstOffset<0");
                        }
                        if(dstOffset>frameData->framePt->size)
                        {
                            printm("dstOffset>frameData->framePt->size");
                            throw FrameReaderException("dstOffset>frameData->framePt->size");
                        }
                        srcOffset=frameData->junkV.at(j).position+frameData->junkV.at(j).size;
                        #ifdef DEBUG_CORRECTNESSCONTROL
                        printf("srcOffset: %d\n",srcOffset);
                        #endif // DEBUG_CORRECTNESSCONTROL
                        memcpy(decodedFrame+dstOffset,
                               frameData->dataSpacePt->pt+srcOffset,
                               sizeToCopy);
                        lastSkipedPosition=frameData->junkV.at(j).position;
                        copiedFrame+=sizeToCopy;
                    }
                }
            }
        }
    }
    if(copiedFrame<frameData->framePt->size)
    {
        #ifdef DEBUG_CORRECTNESSCONTROL
        printm("(7) potrzebny jeszcze jeden fragment klatki");
        #endif // DEBUG_CORRECTNESSCONTROL
        if(lastSkipedPosition-frameData->framePt->size+copiedFrame>=0)
        {//jest z kąd kopiować
            #ifdef DEBUG_CORRECTNESSCONTROL
            printm("(8) jest z kąd kopiować");
            #endif // DEBUG_CORRECTNESSCONTROL
            memcpy(decodedFrame,frameData->dataSpacePt->pt+(lastSkipedPosition-frameData->framePt->size+copiedFrame),frameData->framePt->size-copiedFrame);
        }
        else
        {
            #ifdef DEBUG_CORRECTNESSCONTROL
            printm("(9) brakuje danych, żeby skopiować całą klatkę");
            #endif // DEBUG_CORRECTNESSCONTROL
            memcpy(decodedFrame+(frameData->framePt->size-copiedFrame-lastSkipedPosition),frameData->dataSpacePt->pt,lastSkipedPosition);
        }
    }
    lck.lock();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("(10) lck.lock()");
    #endif // DEBUG_CORRECTNESSCONTROL
    delete q.front();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("(11) delete q.front()");
    #endif // DEBUG_CORRECTNESSCONTROL
    q.pop();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("(12) q.pop()");
    #endif // DEBUG_CORRECTNESSCONTROL
    lck.unlock();
    #ifdef DEBUG_CORRECTNESSCONTROL
    printm("(12.5) lck.unlock()");
    #endif // DEBUG_CORRECTNESSCONTROL
    return decodedFrame;
}

void FrameReader::CorrectnessControl::test1()
{
    printm("void FrameReader::CorrectnessControl::test1()");
    printf("this: %p, q.size(): %d, decodedFrame: %p\n",this,q.size(),decodedFrame);
    printm("OK");
}

void FrameReader::test2()
{
    printm("void FrameReader::test2()");
    printf("correctnessControl: %p, correctnessControl->q.size(): %d, correctnessControl->decodedFrame: %p\n",correctnessControl,correctnessControl->q.size(),correctnessControl->decodedFrame);
    printm("OK");
}

void test3(FrameReader* f)
{
    printm("void test3(FrameReader* f)");
    printf("FrameReader* f: %p\n",f);
    //f->printStatus();
    printm("test3 end");
}

void test4(FrameReader* f)
{
    printm("void test4(FrameReader* f)");
    printf("FrameReader* f: %p\n",f);
    //f->printStatus();
    printm("test4 end");
}

void test5()
{
    printm("void test5()");
    printf("i=");
    for(int i=0;i<4;i++)
    printf("%d, ",i);
    printf("\n");
    printm("void test5() end");
}

void test6()
{
    printf("void test6()\n");
    printf("i=");
    for(int i=0;i<4;i++)
    printf("%d, ",i);
    printf("\n");
    printf("void test6() end\n");
}







