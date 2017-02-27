#pragma once
//#include <vector_types.h>
#include "float3.hpp"
/** \brief Structure with handles data used by raytracing.
 * Structure should be filled with data from matlab on initialization.
 */
typedef struct HandlesStructures
{
    //Parameters of lens:
    float D;//=15;/**< lens diametr. Set 15 instead of 12 (real aperture) is convenient for calculations */
    float efD;//=11.8;/**< effective diameter of lens */
    float R1;//=10.3;/**< radius of lens curvature first lens */
    float R2;//=10.3;/**< radius of lens curvature second lens */
    float g;//=4;/**< thickness of the whole lens along optical axes */
    //Distances:
    float l1;//=17;/**< Distance between center of the trap and  first lens */
    float ll;
    //CCD parameters
    float lCCD;// = 82.8;/**< Distance to CCD detector */
    float CCDPH;// = 480;/**< width of CCD [ Pix ] */
    float CCDPW;// = 640;/**< height of CCD [Pix ] */
    float PixSize;// = 7.4e-3;/**< Pixel size[ mm ] Pike */
    float CCDH;// = CCDPH * PixSize;/**< height of CCD */
    float CCDW;// = CCDPW * PixSize;/**< width  of CCD */
    //Droplet position
    float3 Pk;//=(float3){0.0f,0.0f,0.0f};/**< Position of droplet relativ to the origin of coordinat system */
	/*struct Pk
	{
		float x,y,z;
	} Pk;*/
	
    float Cs1;//=0.0f;
    float Cs2;//=0.0f;
    float m2;//=0.0f;
    float shX;//=0.0f;
    float shY;//=0.0f;
}HandlesStructures;
