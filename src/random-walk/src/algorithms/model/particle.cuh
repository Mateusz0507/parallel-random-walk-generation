#pragma once

#include "cmath"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DISTANCE 1.0

typedef float real;

namespace algorithms
{
	namespace model
	{
		struct particle
		{
			real x, y, z;
		};

		__host__ __device__ real get_distance(const particle& a, const particle& b);
	}
}
