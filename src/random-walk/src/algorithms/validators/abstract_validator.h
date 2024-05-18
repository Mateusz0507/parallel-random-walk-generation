#pragma once
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	namespace validators
	{
		class abstract_validator
		{
		public:
			virtual bool validate(vector3* dev_data, int N, real_t distance, real_t precision) = 0;
		};
	}
}
