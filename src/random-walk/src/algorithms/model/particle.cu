#include "algorithms/model/particle.cuh"

__host__ __device__ real_t algorithms::model::get_distance(const model::particle& a, const model::particle& b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

__host__ __device__ real_t algorithms::model::get_distance(real_t ax, real_t ay, real_t az, real_t bx, real_t by, real_t bz)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz));
}

__host__ __device__ algorithms::model::particle algorithms::model::operator+(const particle& a, const particle& b)
{
	particle c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	c.z = a.z + b.z;
	return c;
}

