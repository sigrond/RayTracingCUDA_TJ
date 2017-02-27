#pragma once
#include <cmath>
struct float3
{
	float x,y,z;
	inline float3 operator -(const float3 &b)
	{
		float3 v;
		v.x=this->x-b.x;
		v.y=this->y-b.y;
		v.z=this->z-b.z;
		return v;
	};
	inline float3 operator +(const float3 &b)
	{
		float3 v;
		v.x=this->x+b.x;
		v.y=this->y+b.y;
		v.z=this->z+b.z;
		return v;
	};
	inline float3 operator /(const float &b)
	{
		float3 v;
		v.x=this->x/b;
		v.y=this->y/b;
		v.z=this->z/b;
		return v;
	};
	inline float3 operator *(const float &b)
	{
		float3 v;
		v.x=this->x*b;
		v.y=this->y*b;
		v.z=this->z*b;
		return v;
	};
	inline float3 operator -()
	{
		float3 v;
		v.x=-this->x;
		v.y=-this->y;
		v.z=-this->z;
		return v;
	};
};

float dot(const float3 &a, const float3 &b);

float3 cross(const float3 &a, const float3 &b);

float norm(const float3 &v);

float3 operator -(const float3 &a, const float3 &b);

float3 operator +(const float3 &a, const float3 &b);

float3 operator /(const float3 &a, const float &b);

float3 operator *(const float &a, const float3 &b);