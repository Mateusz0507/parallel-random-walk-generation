#pragma once

#include "common/common.h"
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	namespace cpu
	{
		bool validate_sequentially(model::particle* result, int N, const float distance, const float precision, bool print = false);
	}
}