#pragma once

#include "common/common.h"
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	namespace cpu
	{
		bool validate_sequentially(vector3* result, int N, const real_t distance, const real_t precision, bool print = false);
	}
}