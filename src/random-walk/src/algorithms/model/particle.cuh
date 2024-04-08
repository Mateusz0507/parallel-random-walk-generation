#pragma once

#include <cmath>

#include "common/common.cuh"

#define DISTANCE 1.0
#define _FLOAT

#ifdef _FLOAT
	typedef float real_t;
#elif defined _DOUBLE
	typedef double real_t;
#endif


namespace algorithms
{
	namespace model
	{
		struct particle
		{
			real_t x, y, z;
		};

		__host__ __device__ real_t get_distance(const particle& a, const particle& b);
	}
}
