#include "algorithms/model/particle.cuh"

__device__ float algorithms::model::get_distance(model::particle& a, model::particle& b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

