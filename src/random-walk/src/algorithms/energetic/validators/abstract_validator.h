#pragma once
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	namespace energetic
	{
		namespace validators
		{
			class abstract_validator
			{
			public:
				virtual int validate(model::particle* dev_data, int N, float distance, float precision) = 0;
			};
		}
	}
}
