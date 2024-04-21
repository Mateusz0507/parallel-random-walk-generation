#pragma once
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	class abstract_method
	{
	public:
		virtual bool run(model::vec3** result, int N) = 0;
	};
}