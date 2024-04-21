#pragma once

#include "algorithms/model/particle.cuh"


namespace algorithms
{
	namespace model
	{
		class spherical_coordinates
		{
		public:
			real_t alpha, beta;

			__device__ spherical_coordinates(real_t alpha, real_t beta);
			__device__ static vector3 get_vector(real_t alpha, real_t beta);
		};
	}
}
