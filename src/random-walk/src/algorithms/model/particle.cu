#include "algorithms/model/particle.cuh"

__host__ __device__ real_t algorithms::model::get_distance(const model::vec3& a, const model::vec3& b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

__host__ __device__ real_t algorithms::model::get_distance(real_t ax, real_t ay, real_t az, real_t bx, real_t by, real_t bz)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz));
}

__host__ __device__ algorithms::model::vec3 algorithms::model::add_particles::operator()(const algorithms::model::vec3& a, const algorithms::model::vec3& b) const
{
	algorithms::model::vec3 c = a;
	c.x += b.x;
	c.y += b.y;
	c.z += b.z;
	return c;
}
