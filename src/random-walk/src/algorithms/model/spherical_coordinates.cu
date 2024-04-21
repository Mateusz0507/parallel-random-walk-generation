#include "algorithms/model/spherical_coordinates.cuh"

__device__ algorithms::model::spherical_coordinates::spherical_coordinates(real_t alpha, real_t beta)
{
	this->alpha = alpha;
	this->beta = beta;
}

__device__ vector3 algorithms::model::spherical_coordinates::get_vector(real_t alpha, real_t beta)
{
	real_t x = sin(alpha) * cos(beta);
	real_t y = sin(alpha) * sin(beta);
	real_t z = cos(alpha);

	vector3 vector;
	vector.x = x;
	vector.y = y;
	vector.z = z;

	return vector;
}
