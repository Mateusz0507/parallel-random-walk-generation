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
			float m[3][3];

			matrix(float3 v1, float3 v2, float3 v3);
			__device__ matrix(spherical_coordinates coords);
			__device__ float3 multiply(float3 v);
		};
	}
}
