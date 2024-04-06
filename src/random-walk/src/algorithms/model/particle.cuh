#pragma once

#include "cmath"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define REAL_TYPE float
#define DISTANCE 1.0

namespace algorithms
{
	namespace model
	{
		struct particle
		{
			REAL_TYPE x, y, z;
		};

		__device__ float get_distance(particle& a, particle& b);
	}
}
