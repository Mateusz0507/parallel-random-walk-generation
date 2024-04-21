#pragma once
#include "algorithms/model/particle.cuh"

namespace algorithms
{
	class abstract_method
	{
	public:
		virtual bool run(vector3** result, int N) = 0;
	};
}