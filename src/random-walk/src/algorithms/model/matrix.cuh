#pragma once

#include "algorithms//constaces/math_constances.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/model/spherical_coordinates.cuh"


namespace algorithms
{
	namespace model
	{
		class matrix
		{
		public:
			real_t m[3][3];

			matrix(vector3 v1, vector3 v2, vector3 v3);
			__device__ matrix(spherical_coordinates coords);
			__device__ vector3 multiply(vector3 v);
		};
	}
}
