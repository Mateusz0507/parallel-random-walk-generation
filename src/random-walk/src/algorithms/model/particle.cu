#include "algorithms/model/particle.cuh"

__host__ __device__ real_t algorithms::model::get_distance(const vector3 &a, const vector3 &b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

__host__ __device__ real_t algorithms::model::get_distance(real_t ax, real_t ay, real_t az, real_t bx, real_t by, real_t bz)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz));
}

__host__ __device__ real_t algorithms::model::norm(const vector3& a)
{
	return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z));
}

__device__ real_t algorithms::model::dot(real_t ax, real_t ay, real_t az, real_t bx, real_t by, real_t bz)
{
	return ax * bx + ay * by + az * bz;
}

__device__ real_t algorithms::model::dot(const vector3& a, const vector3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}



__host__ __device__ vector3 algorithms::model::add_vector3::operator()(const vector3 &a, const vector3 &b) const
{
	vector3 c = a;
	c.x += b.x;
	c.y += b.y;
	c.z += b.z;
	return c;
}
