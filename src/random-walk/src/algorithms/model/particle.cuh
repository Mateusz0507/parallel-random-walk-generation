#pragma once

#include <cmath>

#include "common/common.cuh"

#define DISTANCE 1.0
#define _FLOAT

#ifdef _DOUBLE
	typedef double real_t;
#elif defined _FLOAT
	typedef float real_t;
#endif


namespace algorithms
{
	namespace model
	{

#ifdef _DOUBLE
		typedef double3 particle;
#elif defined _FLOAT
		typedef float3 particle;
#endif

		__host__ __device__ real_t get_distance(const particle& a, const particle& b);
	}
}
