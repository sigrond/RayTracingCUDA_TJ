#include "float3.hpp"
#include <cmath>



float3 cross(const struct float3 &a, const struct float3 &b)
{
	float3 v;
	v.x=a.y*b.z-b.y*a.z;
	v.y=b.x*a.z-a.x*b.z;
	v.z=a.x*b.y-b.x*a.y;
	return v;
}

float norm(const float3 &v)
{
	return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline float3 operator -(const float3 &a, const float3 &b)
{
	float3 v;
	v.x=a.x-b.x;
	v.y=a.y-b.y;
	v.z=a.z-b.z;
	return v;
}

inline float3 operator +(const float3 &a, const float3 &b)
{
	float3 v;
	v.x=a.x+b.x;
	v.y=a.y+b.y;
	v.z=a.z+b.z;
	return v;
}

inline float3 operator /(const float3 &a, const float &b)
{
	float3 v;
	v.x=a.x/b;
	v.y=a.y/b;
	v.z=a.z/b;
	return v;
}

/*inline float3 float3::operator -(const float3 &b)
{
	float3 v;
	v.x=this->x-b.x;
	v.y=this->y-b.y;
	v.z=this->z-b.z;
	return v;
}*/

float3 operator *(const float &a, const float3 &b)
{
	float3 v;
	v.x=a*b.x;
	v.y=a*b.y;
	v.z=a*b.z;
	return v;
}



