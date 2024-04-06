#pragma once
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	class abstract_method
	{
	public:
		virtual bool run(model::particle** result, int N) = 0;
	};
}