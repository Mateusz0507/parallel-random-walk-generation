#pragma once

#include "algorithms/model/particle.cuh"


namespace algorithms
{
	namespace model
	{
		class spherical_coordinates
		{
		public:
			float alpha, beta;

			__device__ spherical_coordinates(float alpha, float beta);
			__device__ static float3 get_vector(float alpha, float beta);
		};
	}
}
