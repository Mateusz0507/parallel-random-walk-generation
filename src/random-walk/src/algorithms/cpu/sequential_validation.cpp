#pragma once

#include "algorithms/cpu/sequential_validation.h"

bool algorithms::cpu::validate_sequentially(vector3* result, int N, const float distance, const float precision, const bool print)
{
	bool are_valid = true;
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			float dist = algorithms::model::get_distance(result[i], result[j]);
			if(print) std::cout << "(" << i << "," << j << "): " << dist << std::endl;
			are_valid = dist >= distance - precision;
		}
	}
	return are_valid;
}