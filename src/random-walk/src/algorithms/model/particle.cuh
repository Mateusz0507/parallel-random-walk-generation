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

#ifdef _DOUBLE
	typedef double3 vector3;
#elif defined _FLOAT
	typedef float3 vector3;
#endif

namespace algorithms
{
	namespace model
	{
		struct particle_soa
		{
			real_t *dev_x, *dev_y, *dev_z;
		};

		__host__ __device__ real_t get_distance(const vector3& a, const vector3& b);

		__host__ __device__ real_t get_distance(real_t ax, real_t ay, real_t az, real_t bx, real_t by, real_t bz);


		class add_particles
		{
		public:
			__host__ __device__ vector3 operator()(const vector3& a, const vector3& b) const;
		};
	}
}
