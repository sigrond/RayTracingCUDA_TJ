/** \file JunkStruct.h
 * \author Tomasz Jakubczyk
 * \brief plik zawieraj¹cy definicje struktury JunkStruct
 *
 */

#pragma once

extern "C"
{
    /** \brief struktura JunkStruct
     */
    struct JunkStruct
    {
        long int position;/**< pozycja JUNK */
        long int size;/**< rozmiar JUNK */
    };
}
