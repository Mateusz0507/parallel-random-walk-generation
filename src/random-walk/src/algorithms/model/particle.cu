#include "algorithms/model/particle.cuh"

__host__ __device__ real algorithms::model::get_distance(const model::particle& a, const model::particle& b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

